from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import tensor_sha256

from .runtime import (
    CausalTrainingRuntime,
    StateSnapshot,
    StepEvidence,
    assert_snapshot_isolation,
)
from .storage import TensorStore


SUPPORT_ARRAY_KEYS = (
    "double_failure_slack",
    "effective_support",
    "necessity",
    "pair_backup",
    "single_failure_slack",
    "support_allocation",
    "support_concentration",
)


def _jsonable_hash(value: dict[str, Any]) -> dict[str, Any]:
    material = dict(value)
    return {**material, "payload_sha256": payload_sha256(material)}


def _encode_state(store: TensorStore, state: StateSnapshot, *, label: str) -> dict[str, Any]:
    exp_avg = {name: child["exp_avg"] for name, child in state.optimizer.items()}
    exp_avg_sq = {name: child["exp_avg_sq"] for name, child in state.optimizer.items()}
    steps = {name: child["step"].reshape(1) for name, child in state.optimizer.items()}
    result = {
        "commitment": state.commitment(),
        "optimizer_exp_avg": store.put_named(exp_avg, representation=f"{label}:complete_named_adam_exp_avg"),
        "optimizer_exp_avg_sq": store.put_named(exp_avg_sq, representation=f"{label}:complete_named_adam_exp_avg_sq"),
        "optimizer_steps": store.put_named(steps, representation=f"{label}:complete_named_adam_steps"),
        "parameters": store.put_named(state.parameters, representation=f"{label}:complete_named_parameter_state"),
    }
    return _jsonable_hash(result)


def _encode_step(store: TensorStore, evidence: StepEvidence, *, label: str) -> dict[str, Any]:
    result = {
        "execute_optimizer": evidence.execute_optimizer,
        "loss": evidence.loss,
        "parameter_update": store.put_named(
            evidence.parameter_updates,
            representation=f"{label}:complete_named_parameter_update",
        )
        if evidence.execute_optimizer
        else {
            "disposition": "OPTIMIZER_UPDATE_SKIPPED_BY_FROZEN_MATCHED_CONTROL",
            "all_parameter_updates_zero": True,
        },
        "raw_gradients": store.put_named(
            evidence.raw_gradients,
            representation=f"{label}:complete_named_raw_gradients",
        ),
        "clipped_gradients": store.put_named(
            evidence.clipped_gradients,
            representation=f"{label}:complete_named_clipped_gradients",
        ),
        "rng_after": evidence.rng_after,
        "rng_before": evidence.rng_before,
        "total_gradient_norm": evidence.total_gradient_norm,
    }
    return _jsonable_hash(result)


def _encode_probe(store: TensorStore, probe: dict[str, Any], *, label: str) -> dict[str, Any]:
    forwards = []
    for index, row in enumerate(probe["forward_rows"]):
        gate_label = "ungated" if not row["gate_components"] else "+".join(row["gate_components"])
        forwards.append(
            {
                "gate_components": row["gate_components"],
                "group_membership": store.put(
                    row["group_membership"],
                    representation=f"{label}:forward:{index}:{gate_label}:complete_group_membership",
                ),
                "group_q10_margin": store.put(
                    row["group_q10_margin"],
                    representation=f"{label}:forward:{index}:{gate_label}:complete_23_group_q10_margin",
                ),
                "logits": store.put(
                    row["logits"],
                    representation=f"{label}:forward:{index}:{gate_label}:complete_decision_logits",
                ),
                "margins": store.put(
                    row["margins"],
                    representation=f"{label}:forward:{index}:{gate_label}:complete_per_example_margin",
                ),
                "predictions": store.put(
                    row["predictions"],
                    representation=f"{label}:forward:{index}:{gate_label}:complete_predictions",
                ),
            }
        )
    result = {
        "actual_forward_count": probe["actual_forward_count"],
        "baseline_byte_exact": probe["baseline_byte_exact"],
        "capability_accuracy": probe["capability_accuracy"],
        "component_loads": probe["component_loads"],
        "forwards": forwards,
        "undefined_effective_support_groups": probe["undefined_effective_support_groups"].tolist(),
    }
    for key in SUPPORT_ARRAY_KEYS:
        result[key] = store.put(
            probe[key],
            representation=f"{label}:complete_{key}",
        )
    return _jsonable_hash(result)


def _probe_effect(
    store: TensorStore,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for key in SUPPORT_ARRAY_KEYS:
        arrays[key] = store.put(
            np.asarray(left[key]) - np.asarray(right[key]),
            representation=f"{label}:causal_difference:{key}",
        )
    left_baseline = left["forward_rows"][0]
    right_baseline = right["forward_rows"][0]
    arrays["decision_logits"] = store.put(
        left_baseline["logits"] - right_baseline["logits"],
        representation=f"{label}:causal_difference:complete_decision_logits",
    )
    arrays["per_example_margins"] = store.put(
        np.asarray(left_baseline["margins"]) - np.asarray(right_baseline["margins"]),
        representation=f"{label}:causal_difference:complete_per_example_margins",
    )
    load_effects = {
        component: {
            name: float(left["component_loads"][component][name])
            - float(right["component_loads"][component][name])
            for name in left["component_loads"][component]
        }
        for component in left["component_loads"]
    }
    return _jsonable_hash(
        {
            "capability_accuracy": float(left["capability_accuracy"] - right["capability_accuracy"]),
            "component_load_effects": load_effects,
            "tensor_effects": arrays,
        }
    )


def _four_way_interaction(
    store: TensorStore,
    full: dict[str, Any],
    parameter: dict[str, Any],
    optimizer: dict[str, Any],
    skip: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    arrays = {
        key: store.put(
            np.asarray(full[key])
            - np.asarray(parameter[key])
            - np.asarray(optimizer[key])
            + np.asarray(skip[key]),
            representation=f"{label}:interaction:{key}",
        )
        for key in SUPPORT_ARRAY_KEYS
    }
    full_baseline = full["forward_rows"][0]
    parameter_baseline = parameter["forward_rows"][0]
    optimizer_baseline = optimizer["forward_rows"][0]
    skip_baseline = skip["forward_rows"][0]
    arrays["decision_logits"] = store.put(
        full_baseline["logits"]
        - parameter_baseline["logits"]
        - optimizer_baseline["logits"]
        + skip_baseline["logits"],
        representation=f"{label}:interaction:complete_decision_logits",
    )
    return _jsonable_hash(
        {
            "capability_accuracy": float(
                full["capability_accuracy"]
                - parameter["capability_accuracy"]
                - optimizer["capability_accuracy"]
                + skip["capability_accuracy"]
            ),
            "tensor_effects": arrays,
        }
    )


def _write_completed(path: Path, material: dict[str, Any]) -> dict[str, Any]:
    result = {**material, "result_sha256": payload_sha256(material)}
    write_json(path, result)
    reread = read_json(path)
    require(reread == result, "CST_RESULT_REREAD_MISMATCH")
    return result


def run_scan_checkpoint(
    runtime: CausalTrainingRuntime,
    store: TensorStore,
    *,
    contract_sha256: str,
    entry_id: str,
    optimizer_step: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        existing = read_json(output_path)
        material = {key: value for key, value in existing.items() if key != "result_sha256"}
        require(payload_sha256(material) == existing["result_sha256"], "CST_EXISTING_SCAN_RESULT_INVALID")
        return existing
    prestate, source = runtime.load_source_state(optimizer_step)
    seed = runtime.derive_seed(contract_sha256, entry_id, optimizer_step, 1)
    runtime.restore(prestate)
    runtime.set_rng(seed)
    pre_first, pre_second = runtime.ungated_baseline_repeat()
    require(torch.equal(pre_first, pre_second), "CST_SCAN_PREBASELINE_NOT_EXACT")
    full_pre = prestate.clone()
    skip_pre = prestate.clone()
    assert_snapshot_isolation([full_pre, skip_pre])

    runtime.restore(full_pre)
    full_step = runtime.train_step(execute_optimizer=True, seed=seed)
    full_state = runtime.snapshot()
    full_probe = runtime.support_probe(full_step)

    runtime.restore(skip_pre)
    skip_step = runtime.train_step(execute_optimizer=False, seed=seed)
    skip_state = runtime.snapshot()
    require(skip_state.commitment()["state_sha256"] == prestate.commitment()["state_sha256"], "CST_SKIP_STATE_CHANGED")
    skip_probe = runtime.support_probe(skip_step)
    require(full_step.loss == skip_step.loss, "CST_MATCHED_BRANCH_LOSS_DIVERGED_BEFORE_UPDATE")
    require(
        _state_hashes_for_compare(full_step.raw_gradients) == _state_hashes_for_compare(skip_step.raw_gradients),
        "CST_MATCHED_BRANCH_RAW_GRADIENT_DIVERGED",
    )
    require(
        _state_hashes_for_compare(full_step.clipped_gradients) == _state_hashes_for_compare(skip_step.clipped_gradients),
        "CST_MATCHED_BRANCH_CLIPPED_GRADIENT_DIVERGED",
    )
    encoded = {
        "effect": _probe_effect(store, full_probe, skip_probe, label=f"scan:{entry_id}:{optimizer_step}:full-minus-skip"),
        "entry_id": entry_id,
        "full": {
            "probe": _encode_probe(store, full_probe, label=f"scan:{entry_id}:{optimizer_step}:full"),
            "state": _encode_state(store, full_state, label=f"scan:{entry_id}:{optimizer_step}:full"),
            "step": _encode_step(store, full_step, label=f"scan:{entry_id}:{optimizer_step}:full"),
        },
        "historical_next_comparison": runtime.compare_historical_next(full_state, optimizer_step + 1),
        "optimizer_step": optimizer_step,
        "prestate": {
            "baseline_logits": store.put(pre_first, representation=f"scan:{entry_id}:{optimizer_step}:prestate:baseline_logits"),
            "source_objects": source,
            "state": _encode_state(store, prestate, label=f"scan:{entry_id}:{optimizer_step}:prestate"),
        },
        "schema": "nanogpt-support-transition-scan-checkpoint-v1",
        "skip": {
            "probe": _encode_probe(store, skip_probe, label=f"scan:{entry_id}:{optimizer_step}:skip"),
            "state": _encode_state(store, skip_state, label=f"scan:{entry_id}:{optimizer_step}:skip"),
            "step": _encode_step(store, skip_step, label=f"scan:{entry_id}:{optimizer_step}:skip"),
        },
        "status": "PASS",
    }
    return _write_completed(output_path, encoded)


def _state_hashes_for_compare(values: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in sorted(values.items())}


def _hybrid_state(
    parameters_from: StateSnapshot,
    optimizer_from: StateSnapshot,
) -> StateSnapshot:
    return StateSnapshot(
        {name: value.clone() for name, value in parameters_from.parameters.items()},
        {
            name: {key: value.clone() for key, value in child.items()}
            for name, child in optimizer_from.optimizer.items()
        },
    )


def run_deep_anchor(
    runtime: CausalTrainingRuntime,
    store: TensorStore,
    *,
    anchor: dict[str, Any],
    contract_sha256: str,
    output_path: Path,
    horizons: tuple[int, ...] = (1, 5, 20, 100),
) -> dict[str, Any]:
    if output_path.exists():
        existing = read_json(output_path)
        material = {key: value for key, value in existing.items() if key != "result_sha256"}
        require(payload_sha256(material) == existing["result_sha256"], "CST_EXISTING_ANCHOR_RESULT_INVALID")
        return existing
    entry_id = str(anchor["entry_id"])
    optimizer_step = int(anchor["optimizer_step"])
    prestate, source = runtime.load_source_state(optimizer_step)
    seed = runtime.derive_seed(contract_sha256, entry_id, optimizer_step, 1)
    runtime.restore(prestate)
    runtime.set_rng(seed)
    pre_first, pre_second = runtime.ungated_baseline_repeat()
    require(torch.equal(pre_first, pre_second), "CST_ANCHOR_PREBASELINE_NOT_EXACT")

    initial_states: dict[str, StateSnapshot] = {}
    initial_evidence: dict[str, StepEvidence] = {}
    full_replays: list[tuple[StateSnapshot, StepEvidence]] = []
    for _index in range(3):
        replay_pre = prestate.clone()
        runtime.restore(replay_pre)
        evidence = runtime.train_step(execute_optimizer=True, seed=seed)
        full_replays.append((runtime.snapshot(), evidence))
    replay_commitments = [state.commitment()["state_sha256"] for state, _ in full_replays]
    require(len(set(replay_commitments)) == 1, "CST_FULL_BASIS_REPLAYS_NOT_BYTE_EXACT")
    initial_states["full_update"] = full_replays[0][0]
    initial_evidence["full_update"] = full_replays[0][1]

    runtime.restore(prestate.clone())
    skip_evidence = runtime.train_step(execute_optimizer=False, seed=seed)
    initial_states["skip_update"] = runtime.snapshot()
    initial_evidence["skip_update"] = skip_evidence
    require(initial_states["skip_update"].commitment()["state_sha256"] == prestate.commitment()["state_sha256"], "CST_ANCHOR_SKIP_CHANGED")

    initial_states["parameter_only"] = _hybrid_state(full_replays[1][0], prestate)
    initial_evidence["parameter_only"] = full_replays[1][1]
    initial_states["optimizer_state_only"] = _hybrid_state(prestate, full_replays[2][0])
    initial_evidence["optimizer_state_only"] = full_replays[2][1]
    assert_snapshot_isolation(initial_states.values())

    branch_outputs: dict[str, Any] = {}
    raw_probes: dict[int, dict[str, dict[str, Any]]] = {horizon: {} for horizon in horizons}
    for branch, initial_state in initial_states.items():
        runtime.restore(initial_state)
        current_evidence = initial_evidence[branch]
        horizon_rows: dict[str, Any] = {}
        for opportunity in range(1, max(horizons) + 1):
            if opportunity > 1:
                continuation_seed = runtime.derive_seed(
                    contract_sha256,
                    entry_id,
                    optimizer_step,
                    opportunity,
                )
                current_evidence = runtime.train_step(
                    execute_optimizer=True,
                    seed=continuation_seed,
                )
            if opportunity in horizons:
                state = runtime.snapshot()
                probe = runtime.support_probe(current_evidence)
                raw_probes[opportunity][branch] = probe
                horizon_rows[str(opportunity)] = {
                    "probe": _encode_probe(
                        store,
                        probe,
                        label=f"anchor:{entry_id}:{optimizer_step}:{branch}:h{opportunity}",
                    ),
                    "state": _encode_state(
                        store,
                        state,
                        label=f"anchor:{entry_id}:{optimizer_step}:{branch}:h{opportunity}",
                    ),
                    "terminal_step": _encode_step(
                        store,
                        current_evidence,
                        label=f"anchor:{entry_id}:{optimizer_step}:{branch}:h{opportunity}:terminal",
                    ),
                    "training_opportunities": opportunity,
                }
        branch_outputs[branch] = {
            "horizons": horizon_rows,
            "initial_step": _encode_step(
                store,
                initial_evidence[branch],
                label=f"anchor:{entry_id}:{optimizer_step}:{branch}:initial",
            ),
        }

    effects: dict[str, Any] = {}
    for horizon in horizons:
        probes = raw_probes[horizon]
        full = probes["full_update"]
        skip = probes["skip_update"]
        parameter = probes["parameter_only"]
        optimizer = probes["optimizer_state_only"]
        label = f"anchor:{entry_id}:{optimizer_step}:h{horizon}"
        effects[str(horizon)] = {
            "full": _probe_effect(store, full, skip, label=label + ":full-minus-skip"),
            "interaction": _four_way_interaction(store, full, parameter, optimizer, skip, label=label),
            "optimizer": _probe_effect(store, optimizer, skip, label=label + ":optimizer-minus-skip"),
            "parameter": _probe_effect(store, parameter, skip, label=label + ":parameter-minus-skip"),
        }
    result = {
        "anchor": anchor,
        "branches": branch_outputs,
        "effects": effects,
        "entry_id": entry_id,
        "full_basis_replay_state_sha256": replay_commitments,
        "historical_next_comparison": runtime.compare_historical_next(initial_states["full_update"], optimizer_step + 1),
        "optimizer_step": optimizer_step,
        "prestate": {
            "baseline_logits": store.put(pre_first, representation=f"anchor:{entry_id}:{optimizer_step}:prestate:baseline_logits"),
            "source_objects": source,
            "state": _encode_state(store, prestate, label=f"anchor:{entry_id}:{optimizer_step}:prestate"),
        },
        "schema": "nanogpt-support-transition-deep-anchor-v1",
        "status": "PASS",
    }
    return _write_completed(output_path, result)


def source_bundle_map(source_csrg_archive_manifest: Path) -> dict[str, str]:
    manifest = read_json(source_csrg_archive_manifest)
    require(manifest["status"] == "PASS", "CST_SOURCE_CSRG_ARCHIVE_NOT_VALID")
    result = {
        str(row["entry_id"]): str(row["gfg_bundle_id"])
        for row in manifest["support_bundles"]
    }
    require(len(result) == 13, "CST_SOURCE_BUNDLE_MAP_COUNT_INVALID")
    return result


def run_entry_scan(
    *,
    entry_id: str,
    bundle: Path,
    trainer_root: Path,
    output_root: Path,
    contract_path: Path,
    steps: Iterable[int] = range(100, 10001, 100),
) -> dict[str, Any]:
    output = output_root / entry_id
    store = TensorStore(output / "tensor-objects")
    contract_sha = file_sha256(contract_path)
    runtime = CausalTrainingRuntime(bundle, trainer_root)
    completed = []
    try:
        for ordinal, step in enumerate(steps, start=1):
            result = run_scan_checkpoint(
                runtime,
                store,
                contract_sha256=contract_sha,
                entry_id=entry_id,
                optimizer_step=int(step),
                output_path=output / "scan" / f"step-{int(step):05d}.json",
            )
            completed.append({"optimizer_step": int(step), "result_sha256": result["result_sha256"]})
            print(json.dumps({"event": "CST_SCAN_COMPLETE", "entry_id": entry_id, "ordinal": ordinal, "optimizer_step": int(step)}, sort_keys=True), flush=True)
    finally:
        runtime.close()
    material = {
        "completed": completed,
        "contract_sha256": contract_sha,
        "entry_id": entry_id,
        "schema": "nanogpt-support-transition-entry-scan-receipt-v1",
        "status": "PASS",
    }
    return _write_completed(output / "scan_receipt.json", material)


def run_entry_anchors(
    *,
    entry_id: str,
    bundle: Path,
    trainer_root: Path,
    output_root: Path,
    contract_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    anchors = [row for row in selection["anchors"] if row["entry_id"] == entry_id]
    require(len(anchors) == 4, f"CST_ENTRY_ANCHOR_COUNT_INVALID:{entry_id}")
    output = output_root / entry_id
    store = TensorStore(output / "tensor-objects")
    contract_sha = file_sha256(contract_path)
    require(selection["contract_sha256"] == contract_sha, "CST_SELECTION_CONTRACT_DRIFT")
    runtime = CausalTrainingRuntime(bundle, trainer_root)
    completed = []
    try:
        for anchor in anchors:
            category = str(anchor["anchor_category"])
            step = int(anchor["optimizer_step"])
            result = run_deep_anchor(
                runtime,
                store,
                anchor=anchor,
                contract_sha256=contract_sha,
                output_path=output / "anchors" / f"{category}--step-{step:05d}.json",
            )
            completed.append({"anchor_category": category, "optimizer_step": step, "result_sha256": result["result_sha256"]})
            print(json.dumps({"event": "CST_ANCHOR_COMPLETE", "entry_id": entry_id, "anchor_category": category, "optimizer_step": step}, sort_keys=True), flush=True)
    finally:
        runtime.close()
    material = {
        "completed": completed,
        "contract_sha256": contract_sha,
        "entry_id": entry_id,
        "schema": "nanogpt-support-transition-entry-anchor-receipt-v1",
        "selection_sha256": selection["selection_sha256"],
        "status": "PASS",
    }
    return _write_completed(output / "anchor_receipt.json", material)
