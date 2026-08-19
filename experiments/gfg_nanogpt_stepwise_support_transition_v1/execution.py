from __future__ import annotations

from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_transition_v1.runtime import StateSnapshot

from .contracts import ComponentRegistry, ProbeContract
from .runtime import BatchEvidence, StepEvidence, StepwiseTrainingRuntime
from .storage import TensorStore, restorable_state_from_manifest


SUPPORT_ARRAY_KEYS = (
    "double_failure_slack",
    "effective_support",
    "necessity",
    "pair_backup",
    "single_failure_slack",
    "support_allocation",
    "support_concentration",
)


def _checked_result(path: Path, material: dict[str, Any]) -> dict[str, Any]:
    result = {**material, "result_sha256": payload_sha256(material)}
    if path.exists():
        existing = read_json(path)
        require(existing == result, f"SST_EXISTING_RESULT_DRIFT:{path}")
        return existing
    write_json(path, result)
    require(read_json(path) == result, "SST_RESULT_REREAD_MISMATCH")
    return result


def _read_checked(path: Path, schema: str) -> dict[str, Any]:
    value = read_json(path)
    require(value["schema"] == schema, f"SST_EXISTING_SCHEMA_INVALID:{path}")
    material = {key: child for key, child in value.items() if key != "result_sha256"}
    require(payload_sha256(material) == value["result_sha256"], f"SST_EXISTING_RESULT_INVALID:{path}")
    return value


def _encode_state(
    store: TensorStore,
    state: StateSnapshot,
    *,
    entry_id: str,
    window_id: str,
    optimizer_step: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    exp_avg = {name: child["exp_avg"] for name, child in state.optimizer.items()}
    exp_avg_sq = {name: child["exp_avg_sq"] for name, child in state.optimizer.items()}
    steps = {name: child["step"] for name, child in state.optimizer.items()}
    commitment = state.commitment()
    identity_material = {
        "entry_id": entry_id,
        "optimizer_step": optimizer_step,
        "protocol_sha256": protocol_sha256,
        "state_sha256": commitment["state_sha256"],
    }
    return {
        "state_id": "state-" + payload_sha256(identity_material)[:32],
        "entry_id": entry_id,
        "window_id": window_id,
        "optimizer_step": optimizer_step,
        "commitment": commitment,
        "parameters": store.put_named(state.parameters, representation="restorable_complete_named_parameter_state"),
        "optimizer_exp_avg": store.put_named(exp_avg, representation="restorable_complete_named_adam_exp_avg"),
        "optimizer_exp_avg_sq": store.put_named(exp_avg_sq, representation="restorable_complete_named_adam_exp_avg_sq"),
        "optimizer_steps": store.put_named(steps, representation="restorable_complete_named_adam_steps"),
        "restorable_without_training_reexecution": True,
        "training_continuation_rng_policy": "content-derived current-runtime seed; historical RNG payload unavailable",
    }


def _encode_probe(
    store: TensorStore,
    probe: dict[str, Any],
    *,
    state_id: str,
) -> dict[str, Any]:
    forwards = []
    for index, row in enumerate(probe["forward_rows"]):
        gate_label = "baseline" if not row["gate_components"] else "+".join(row["gate_components"])
        prefix = f"{probe['probe_contract_id']}:{state_id}:forward:{index}:{gate_label}"
        forwards.append(
            {
                "gate_components": row["gate_components"],
                "group_membership": store.put(row["group_membership"], representation=prefix + ":complete_group_membership"),
                "group_q10_margin": store.put(row["group_q10_margin"], representation=prefix + ":complete_target_group_q10_margin"),
                "logits": store.put(row["logits"], representation=prefix + ":complete_decision_logits"),
                "margins": store.put(row["margins"], representation=prefix + ":complete_per_example_margin"),
                "predictions": store.put(row["predictions"], representation=prefix + ":complete_predictions"),
            }
        )
    result = {
        "probe_observation_id": "probe-" + payload_sha256(
            {
                "probe_contract_id": probe["probe_contract_id"],
                "probe_contract_sha256": probe["probe_contract_sha256"],
                "state_id": state_id,
            }
        )[:32],
        "observed_state_id": state_id,
        "probe_contract_id": probe["probe_contract_id"],
        "probe_contract_sha256": probe["probe_contract_sha256"],
        "component_registry_id": probe["component_registry_id"],
        "component_registry_sha256": probe["component_registry_sha256"],
        "component_ids": probe["component_ids"],
        "pair_ids": probe["pair_ids"],
        "actual_forward_count": probe["actual_forward_count"],
        "baseline_byte_exact": probe["baseline_byte_exact"],
        "capability_accuracy": probe["capability_accuracy"],
        "component_loads": probe["component_loads"],
        "forwards": forwards,
        "undefined_effective_support_groups": probe["undefined_effective_support_groups"].tolist(),
        "append_only_observation_layer": True,
    }
    for key in SUPPORT_ARRAY_KEYS:
        result[key] = store.put(
            probe[key],
            representation=f"{probe['probe_contract_id']}:{state_id}:complete_{key}",
        )
    return result


def _probe_result_path(entry_root: Path, probe_contract_id: str, state_id: str) -> Path:
    return entry_root / "probe-observations" / probe_contract_id / f"{state_id}.json"


def ensure_probe_observation(
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    *,
    entry_root: Path,
    state_manifest: dict[str, Any],
) -> dict[str, Any]:
    state_id = str(state_manifest["state_id"])
    path = _probe_result_path(entry_root, runtime.probe_contract.probe_contract_id, state_id)
    if path.exists():
        return _read_checked(path, "nanogpt-stepwise-probe-observation-v1")
    state = restorable_state_from_manifest(entry_root, state_manifest)
    runtime.restore(state)
    material = {
        "schema": "nanogpt-stepwise-probe-observation-v1",
        "status": "PASS",
        **_encode_probe(store, runtime.support_probe(), state_id=state_id),
    }
    return _checked_result(path, material)


def _source_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": row["object_id"],
        "optimizer_step": int(row["optimizer_step"]),
        "role": row["role"],
        "semantic_key": row["semantic_key"],
        "content_sha256": row["content_sha256"],
        "materialized": bool(row["materialized"]),
        "locator": row["locator"],
    }


def _encode_batch(batch: BatchEvidence) -> dict[str, Any]:
    material = {
        "optimizer_step": batch.optimizer_step,
        "source_training_gfg_objects": {
            role: _source_ref(row) for role, row in sorted(batch.source_rows.items())
        },
        "batch_selection_order_availability": (
            {
                "status": "CAPTURED",
                "source_object_id": batch.source_rows["batch_selection_order"]["object_id"],
            }
            if batch.selection_order is not None
            else batch.selection_order_disposition
        ),
    }
    return {**material, "batch_evidence_sha256": payload_sha256(material)}


def _encode_step(
    store: TensorStore,
    evidence: StepEvidence,
    *,
    transition_id: str,
) -> dict[str, Any]:
    prefix = f"{transition_id}:training-step-{evidence.optimizer_step}"
    parameter_update: dict[str, Any]
    optimizer_deltas: dict[str, Any]
    if evidence.execute_optimizer:
        parameter_update = store.put_named(evidence.parameter_updates, representation=prefix + ":complete_named_parameter_update")
        optimizer_deltas = {
            "exp_avg": store.put_named(evidence.exp_avg_deltas, representation=prefix + ":complete_named_exp_avg_delta"),
            "exp_avg_sq": store.put_named(evidence.exp_avg_sq_deltas, representation=prefix + ":complete_named_exp_avg_sq_delta"),
            "adam_step": store.put_named(evidence.adam_step_deltas, representation=prefix + ":complete_named_adam_step_delta"),
            "post_preconditioned_direction": store.put_named(
                evidence.post_preconditioned_directions,
                representation=prefix + ":complete_named_post_preconditioned_direction",
            ),
        }
    else:
        parameter_update = {
            "outcome_kind": "ExplicitDisposition",
            "disposition": "OPTIMIZER_STEP_SKIPPED_BY_FROZEN_BRANCH",
            "all_parameter_updates_zero": True,
        }
        optimizer_deltas = {
            "outcome_kind": "ExplicitDisposition",
            "disposition": "ADAM_STATE_UPDATE_SKIPPED_BY_FROZEN_BRANCH",
            "all_optimizer_deltas_zero": True,
        }
    material = {
        "optimizer_step": evidence.optimizer_step,
        "execute_optimizer": evidence.execute_optimizer,
        "training_logits": store.put(evidence.training_logits, representation=prefix + ":complete_training_logits"),
        "registered_component_activations": store.put_named(
            evidence.activation_outputs,
            representation=prefix + ":complete_registered_component_activations",
        ),
        "loss": evidence.loss,
        "raw_gradients": store.put_named(evidence.raw_gradients, representation=prefix + ":complete_named_raw_gradients"),
        "clipped_gradients": store.put_named(evidence.clipped_gradients, representation=prefix + ":complete_named_clipped_gradients"),
        "total_gradient_norm": evidence.total_gradient_norm,
        "parameter_update": parameter_update,
        "nominal_weight_decay_update": (
            store.put_named(
                evidence.nominal_weight_decay_updates,
                representation=prefix + ":complete_named_nominal_weight_decay_update",
            )
            if evidence.execute_optimizer
            else {
                "outcome_kind": "ExplicitDisposition",
                "disposition": "WEIGHT_DECAY_NOT_APPLIED_BECAUSE_OPTIMIZER_STEP_SKIPPED",
            }
        ),
        "adaptive_update_residual": (
            store.put_named(
                evidence.adaptive_update_residuals,
                representation=prefix + ":complete_named_adaptive_update_residual",
            )
            if evidence.execute_optimizer
            else {
                "outcome_kind": "ExplicitDisposition",
                "disposition": "ADAPTIVE_PARAMETER_UPDATE_NOT_APPLIED_BECAUSE_OPTIMIZER_STEP_SKIPPED",
            }
        ),
        "optimizer_deltas": optimizer_deltas,
        "rng_before": evidence.rng_before,
        "rng_after": evidence.rng_after,
        "optimizer_config": evidence.optimizer_config,
    }
    return {**material, "step_evidence_sha256": payload_sha256(material)}


def _state_result_path(window_root: Path, step: int) -> Path:
    return window_root / "states" / f"step-{step:05d}.json"


def _transition_result_path(window_root: Path, step: int) -> Path:
    return window_root / "transitions" / f"step-{step:05d}-to-{step + 1:05d}.json"


def execute_window(
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    *,
    window: dict[str, Any],
    entry_root: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    window_root = entry_root / "windows" / str(window["window_id"])
    receipt_path = window_root / "window_receipt.json"
    if receipt_path.exists():
        return _read_checked(receipt_path, "nanogpt-stepwise-window-receipt-v1")
    start = int(window.get("capture_start_optimizer_step", window["start_optimizer_step"]))
    end = int(window.get("capture_end_optimizer_step", window["end_optimizer_step"]))
    scientific_start = int(window.get("scientific_start_optimizer_step", window["start_optimizer_step"]))
    scientific_end = int(window.get("scientific_end_optimizer_step", window["end_optimizer_step"]))
    restore_step = int(window.get("restore_optimizer_step", start))
    require(0 <= restore_step <= start <= scientific_start <= scientific_end <= end, "SST_WINDOW_PHASE_BOUNDARY_INVALID")
    entry_id = str(window["entry_id"])
    start_source: dict[str, Any] | None = None
    warmup_receipt: dict[str, Any] | None = None

    current_state: StateSnapshot
    contiguous = start - 1
    for step in range(start, end + 1):
        if _state_result_path(window_root, step).exists():
            _read_checked(_state_result_path(window_root, step), "nanogpt-stepwise-state-v1")
            contiguous = step
        else:
            break
    if contiguous >= start:
        existing = _read_checked(_state_result_path(window_root, contiguous), "nanogpt-stepwise-state-v1")
        current_state = restorable_state_from_manifest(entry_root, existing["state"])
        runtime.restore(current_state)
        first_captured = _read_checked(_state_result_path(window_root, start), "nanogpt-stepwise-state-v1")
        start_source = first_captured.get("source_start_state_objects")
        if isinstance(start_source, dict):
            warmup_receipt = start_source.get("replay_warmup")
    else:
        current_state, source = runtime.load_source_state(restore_step)
        runtime.restore(current_state)
        warmup_steps: list[dict[str, Any]] = []
        warmup_initial_commitment = current_state.commitment()
        for warmup_step in range(restore_step, start):
            pre_commitment = current_state.commitment()
            batch = runtime.load_batch(warmup_step)
            seed = runtime.derive_seed(protocol_sha256, entry_id, warmup_step, 1)
            evidence = runtime.train_actual_step(batch, execute_optimizer=True, seed=seed)
            current_state = runtime.snapshot()
            warmup_steps.append(
                {
                    "optimizer_step": warmup_step,
                    "batch": _encode_batch(batch),
                    "seed": seed,
                    "loss": evidence.loss,
                    "total_gradient_norm_before_clipping": evidence.total_gradient_norm,
                    "from_state_sha256": pre_commitment["state_sha256"],
                    "to_state_sha256": current_state.commitment()["state_sha256"],
                    "rng_before": evidence.rng_before,
                    "rng_after": evidence.rng_after,
                }
            )
        warmup_material = {
            "schema": "nanogpt-stepwise-replay-warmup-v1",
            "status": "PASS",
            "restore_optimizer_step": restore_step,
            "capture_start_optimizer_step": start,
            "transition_count": start - restore_step,
            "source_checkpoint": source,
            "initial_state_sha256": warmup_initial_commitment["state_sha256"],
            "terminal_state_sha256": current_state.commitment()["state_sha256"],
            "steps": warmup_steps,
            "captured_support_observation_claimed": False,
        }
        warmup_receipt = {**warmup_material, "warmup_sha256": payload_sha256(warmup_material)}
        start_source = {"source_checkpoint": source, "replay_warmup": warmup_receipt}

    completed_transitions: list[dict[str, Any]] = []
    completed_states: list[dict[str, Any]] = []
    for step in range(start, end + 1):
        state_path = _state_result_path(window_root, step)
        if state_path.exists():
            state_result = _read_checked(state_path, "nanogpt-stepwise-state-v1")
            current_state = restorable_state_from_manifest(entry_root, state_result["state"])
            runtime.restore(current_state)
        else:
            state_manifest = _encode_state(
                store,
                current_state,
                entry_id=entry_id,
                window_id=str(window["window_id"]),
                optimizer_step=step,
                protocol_sha256=protocol_sha256,
            )
            state_material = {
                "schema": "nanogpt-stepwise-state-v1",
                "status": "PASS",
                "entry_id": entry_id,
                "window_id": window["window_id"],
                "optimizer_step": step,
                "state": state_manifest,
                "state_summary": runtime.state_summary(current_state),
                "source_start_state_objects": start_source if step == start else None,
            }
            state_result = _checked_result(state_path, state_material)
        probe_result = ensure_probe_observation(
            runtime,
            store,
            entry_root=entry_root,
            state_manifest=state_result["state"],
        )
        completed_states.append(
            {
                "optimizer_step": step,
                "result_sha256": state_result["result_sha256"],
                "probe_contract_id": probe_result["probe_contract_id"],
                "probe_result_sha256": probe_result["result_sha256"],
            }
        )
        if step == end:
            break

        transition_path = _transition_result_path(window_root, step)
        if transition_path.exists() and _state_result_path(window_root, step + 1).exists():
            transition_result = _read_checked(transition_path, "nanogpt-stepwise-transition-v1")
            next_result = _read_checked(_state_result_path(window_root, step + 1), "nanogpt-stepwise-state-v1")
            current_state = restorable_state_from_manifest(entry_root, next_result["state"])
            runtime.restore(current_state)
        else:
            pre_commitment = current_state.commitment()
            batch = runtime.load_batch(step)
            seed = runtime.derive_seed(protocol_sha256, entry_id, step, 1)
            evidence = runtime.train_actual_step(batch, execute_optimizer=True, seed=seed)
            current_state = runtime.snapshot()
            transition_id = "transition-" + payload_sha256(
                {
                    "entry_id": entry_id,
                    "from_state_sha256": pre_commitment["state_sha256"],
                    "optimizer_step": step,
                    "to_state_sha256": current_state.commitment()["state_sha256"],
                    "window_id": window["window_id"],
                }
            )[:32]
            transition_material = {
                "schema": "nanogpt-stepwise-transition-v1",
                "status": "PASS",
                "transition_id": transition_id,
                "entry_id": entry_id,
                "window_id": window["window_id"],
                "optimizer_step": step,
                "from_state_sha256": pre_commitment["state_sha256"],
                "to_state_sha256": current_state.commitment()["state_sha256"],
                "batch": _encode_batch(batch),
                "step": _encode_step(store, evidence, transition_id=transition_id),
                "occurrence_decomposition": {
                    "native_forward_occurrence": True,
                    "loss_is_forward_outcome_not_separate_occurrence": True,
                    "native_backward_occurrence": True,
                    "native_gradient_clip_occurrence": True,
                    "native_optimizer_step_occurrence": True,
                    "adam_and_parameter_changes_are_joint_optimizer_step_outcomes": True,
                },
            }
            transition_result = _checked_result(transition_path, transition_material)
        completed_transitions.append({"optimizer_step": step, "result_sha256": transition_result["result_sha256"]})

    terminal = _read_checked(_state_result_path(window_root, end), "nanogpt-stepwise-state-v1")
    terminal_state = restorable_state_from_manifest(entry_root, terminal["state"])
    runtime.restore(terminal_state)
    comparison = runtime.compare_historical_state(terminal_state, end)
    material = {
        "schema": "nanogpt-stepwise-window-receipt-v1",
        "status": "PASS",
        "window": window,
        "protocol_sha256": protocol_sha256,
        "capture_interval": [start, end],
        "scientific_interval": [scientific_start, scientific_end],
        "lookback_prehistory_steps": scientific_start - start,
        "replay_warmup": warmup_receipt,
        "completed_states": completed_states,
        "completed_transitions": completed_transitions,
        "endpoint_comparison": comparison,
        "historical_raw_logits_claimed": False,
        "runtime_reexecution_observations_claimed": True,
    }
    return _checked_result(receipt_path, material)


def execute_entry_windows(
    *,
    entry_id: str,
    source_bundle: Path,
    trainer_root: Path,
    output_root: Path,
    protocol_path: Path,
    selection_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    max_windows: int | None = None,
) -> dict[str, Any]:
    protocol_sha = file_sha256(protocol_path)
    selection = read_json(selection_path)
    require(selection["protocol_sha256"] == protocol_sha, "SST_SELECTION_PROTOCOL_DRIFT")
    windows = [row for row in selection["windows"] if row["entry_id"] == entry_id]
    if max_windows is not None:
        windows = windows[:max_windows]
    require(bool(windows), f"SST_ENTRY_WINDOWS_EMPTY:{entry_id}")
    entry_root = output_root / entry_id
    store = TensorStore(entry_root / "tensor-objects")
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    runtime = StepwiseTrainingRuntime(source_bundle, trainer_root, registry, probe_contract)
    completed = []
    try:
        for ordinal, window in enumerate(windows, start=1):
            result = execute_window(
                runtime,
                store,
                window=window,
                entry_root=entry_root,
                protocol_sha256=protocol_sha,
            )
            completed.append({"window_id": window["window_id"], "result_sha256": result["result_sha256"]})
            print(
                {
                    "event": "SST_WINDOW_COMPLETE",
                    "entry_id": entry_id,
                    "ordinal": ordinal,
                    "window_count": len(windows),
                    "window_id": window["window_id"],
                },
                flush=True,
            )
    finally:
        runtime.close()
    material = {
        "schema": "nanogpt-stepwise-entry-receipt-v1",
        "status": "PASS",
        "entry_id": entry_id,
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection["selection_sha256"],
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "source_bundle_manifest": read_json(source_bundle / "manifest.json"),
        "source_bundle_manifest_file_sha256": file_sha256(source_bundle / "manifest.json"),
        "implementation": {
            "contracts_py_sha256": file_sha256(Path(__file__).with_name("contracts.py")),
            "execution_py_sha256": file_sha256(Path(__file__)),
            "runtime_py_sha256": file_sha256(Path(__file__).with_name("runtime.py")),
            "storage_py_sha256": file_sha256(Path(__file__).with_name("storage.py")),
            "trainer_model_py_sha256": file_sha256(trainer_root / "model.py"),
        },
        "runtime_environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_device_count": torch.cuda.device_count(),
        },
        "completed_windows": completed,
    }
    return _checked_result(entry_root / "entry_receipt.json", material)


def append_probe_layer_for_entry(
    *,
    entry_id: str,
    source_bundle: Path,
    trainer_root: Path,
    output_root: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
) -> dict[str, Any]:
    """Evaluate a new versioned probe contract on stored states without replaying training."""

    entry_root = output_root / entry_id
    require(entry_root.exists(), f"SST_ENTRY_RESULT_MISSING:{entry_id}")
    store = TensorStore(entry_root / "tensor-objects")
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    state_files = sorted((entry_root / "windows").glob("*/states/step-*.json"))
    require(bool(state_files), f"SST_STORED_STATE_EMPTY:{entry_id}")
    by_state_id: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, list[str]] = {}
    for path in state_files:
        result = _read_checked(path, "nanogpt-stepwise-state-v1")
        manifest = result["state"]
        state_id = str(manifest["state_id"])
        if state_id in by_state_id:
            require(
                by_state_id[state_id]["commitment"] == manifest["commitment"],
                f"SST_STATE_ID_COMMITMENT_COLLISION:{state_id}",
            )
        else:
            by_state_id[state_id] = manifest
        source_paths.setdefault(state_id, []).append(path.relative_to(entry_root).as_posix())

    runtime = StepwiseTrainingRuntime(source_bundle, trainer_root, registry, probe_contract)
    observations: list[dict[str, Any]] = []
    try:
        for ordinal, state_id in enumerate(sorted(by_state_id), start=1):
            result = ensure_probe_observation(
                runtime,
                store,
                entry_root=entry_root,
                state_manifest=by_state_id[state_id],
            )
            observations.append(
                {
                    "state_id": state_id,
                    "source_state_records": sorted(source_paths[state_id]),
                    "result_sha256": result["result_sha256"],
                }
            )
            print(
                {
                    "event": "SST_PROBE_OBSERVATION_COMPLETE",
                    "entry_id": entry_id,
                    "ordinal": ordinal,
                    "state_count": len(by_state_id),
                    "probe_contract_id": probe_contract.probe_contract_id,
                    "state_id": state_id,
                },
                flush=True,
            )
    finally:
        runtime.close()
    material = {
        "schema": "nanogpt-stepwise-probe-layer-receipt-v1",
        "status": "PASS",
        "entry_id": entry_id,
        "component_registry_id": registry.registry_id,
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_id": probe_contract.probe_contract_id,
        "probe_contract_sha256": probe_contract.source_sha256,
        "training_reexecution_count": 0,
        "unique_restored_state_count": len(by_state_id),
        "observations": observations,
    }
    receipt_path = entry_root / "probe-layer-receipts" / f"{probe_contract.probe_contract_id}.json"
    return _checked_result(receipt_path, material)
