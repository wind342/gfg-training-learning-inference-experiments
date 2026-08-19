from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
)
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import tensor_sha256
from experiments.gfg_nanogpt_support_transition_v1.runtime import (
    StateSnapshot,
    assert_snapshot_isolation,
)

from .contracts import ComponentRegistry, ProbeContract
from .execution import (
    _checked_result,
    _encode_batch,
    _encode_state,
    _encode_step,
    _read_checked,
    ensure_probe_observation,
)
from .runtime import StepEvidence, StepwiseTrainingRuntime
from .storage import TensorStore, restorable_state_from_manifest


BRANCHES = ("full_step", "skip_step", "parameter_only", "optimizer_state_only")
HORIZONS = (1, 2, 5, 10, 20, 50, 100)


def _tensor_commitment(value: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(value.detach().cpu().numpy().dtype),
        "raw_tensor_sha256": tensor_sha256(value),
        "shape": list(value.shape),
        "materialized": False,
        "outcome_kind": "ExplicitDisposition",
        "disposition": "BRANCH_INTERMEDIATE_TENSOR_PAYLOAD_NOT_MATERIALIZED_UNDER_FROZEN_PROFILE",
    }


def _named_commitments(values: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _tensor_commitment(value) for name, value in sorted(values.items())}


def _lightweight_step(evidence: StepEvidence) -> dict[str, Any]:
    material = {
        "optimizer_step": evidence.optimizer_step,
        "execute_optimizer": evidence.execute_optimizer,
        "training_logits": _tensor_commitment(evidence.training_logits),
        "registered_component_activations": _named_commitments(evidence.activation_outputs),
        "loss": evidence.loss,
        "raw_gradients": _named_commitments(evidence.raw_gradients),
        "clipped_gradients": _named_commitments(evidence.clipped_gradients),
        "total_gradient_norm": evidence.total_gradient_norm,
        "parameter_updates": _named_commitments(evidence.parameter_updates),
        "nominal_weight_decay_updates": _named_commitments(evidence.nominal_weight_decay_updates),
        "adaptive_update_residuals": _named_commitments(evidence.adaptive_update_residuals),
        "exp_avg_deltas": _named_commitments(evidence.exp_avg_deltas),
        "exp_avg_sq_deltas": _named_commitments(evidence.exp_avg_sq_deltas),
        "adam_step_deltas": _named_commitments(evidence.adam_step_deltas),
        "post_preconditioned_directions": _named_commitments(evidence.post_preconditioned_directions),
        "rng_before": evidence.rng_before,
        "rng_after": evidence.rng_after,
        "optimizer_config": evidence.optimizer_config,
    }
    return {**material, "step_evidence_sha256": payload_sha256(material)}


def _maps_exact(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _state_from_parts(parameters: Mapping[str, torch.Tensor], optimizer: Mapping[str, Mapping[str, torch.Tensor]]) -> StateSnapshot:
    return StateSnapshot(
        parameters={name: value.detach().contiguous().cpu().clone() for name, value in parameters.items()},
        optimizer={
            name: {key: value.detach().contiguous().cpu().clone() for key, value in child.items()}
            for name, child in optimizer.items()
        },
    )


def _source_state_record(formal_root: Path, window: dict[str, Any], optimizer_step: int) -> dict[str, Any]:
    path = (
        formal_root
        / str(window["entry_id"])
        / "windows"
        / str(window["window_id"])
        / "states"
        / f"step-{optimizer_step:05d}.json"
    )
    return _read_checked(path, "nanogpt-stepwise-state-v1")


def _load_observation_arrays(entry_root: Path, observation: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}

    def load(reference: dict[str, Any]) -> np.ndarray:
        locator = str(reference["locator"])
        require(locator.startswith("tensor-objects/"), "SST_BRANCH_TENSOR_LOCATOR_INVALID")
        path = entry_root / locator
        require(file_sha256(path) == reference["file_sha256"], "SST_BRANCH_TENSOR_FILE_HASH_MISMATCH")
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        require(list(value.shape) == list(reference["shape"]), "SST_BRANCH_TENSOR_SHAPE_MISMATCH")
        require(str(value.dtype) == str(reference["dtype"]), "SST_BRANCH_TENSOR_DTYPE_MISMATCH")
        require(
            hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
            == reference["raw_tensor_sha256"],
            "SST_BRANCH_TENSOR_RAW_HASH_MISMATCH",
        )
        return np.asarray(value)

    for index, row in enumerate(observation["forwards"]):
        for key in ("logits", "margins", "predictions", "group_membership", "group_q10_margin"):
            result[f"forward/{index}/{key}"] = load(row[key])
    for key in (
        "double_failure_slack",
        "effective_support",
        "necessity",
        "pair_backup",
        "single_failure_slack",
        "support_allocation",
        "support_concentration",
    ):
        result[key] = load(observation[key])
    return result


def _encode_effects(
    store: TensorStore,
    *,
    seed_id: str,
    horizon: int,
    observations: dict[str, tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    identity_fields = (
        "probe_contract_id",
        "probe_contract_sha256",
        "component_registry_id",
        "component_registry_sha256",
        "component_ids",
        "pair_ids",
    )
    identity = {
        field: observations["full_step"][1][field]
        for field in identity_fields
    }
    require(
        all(
            all(observation[field] == identity[field] for field in identity_fields)
            for _root, observation in observations.values()
        ),
        "SST_BRANCH_PROBE_IDENTITY_ALIGNMENT_FAILURE",
    )
    arrays = {
        branch: _load_observation_arrays(root, observation)
        for branch, (root, observation) in observations.items()
    }
    keys = tuple(sorted(arrays["full_step"]))
    require(all(tuple(sorted(value)) == keys for value in arrays.values()), "SST_BRANCH_PROBE_OUTPUT_SET_MISMATCH")
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for key in keys:
        full = arrays["full_step"][key]
        skip = arrays["skip_step"][key]
        parameter = arrays["parameter_only"][key]
        optimizer = arrays["optimizer_state_only"][key]
        require(full.shape == skip.shape == parameter.shape == optimizer.shape, f"SST_BRANCH_EFFECT_SHAPE_MISMATCH:{key}")
        if np.issubdtype(full.dtype, np.floating):
            effect_values = {
                "Phi_full": full - skip,
                "Phi_P": parameter - skip,
                "Phi_O": optimizer - skip,
                "Phi_PxO": full - parameter - optimizer + skip,
            }
            numeric[key] = {
                name: store.put(
                    value,
                    representation=f"{seed_id}:h{horizon}:{name}:{key}:complete_numeric_effect",
                )
                for name, value in effect_values.items()
            }
        else:
            categorical[key] = {
                branch: {
                    "equal_to_skip_count": int(np.count_nonzero(arrays[branch][key] == skip)),
                    "coordinate_count": int(skip.size),
                    "changed_mask": store.put(
                        arrays[branch][key] != skip,
                        representation=f"{seed_id}:h{horizon}:{branch}:{key}:complete_categorical_changed_mask",
                    ),
                }
                for branch in ("full_step", "parameter_only", "optimizer_state_only")
            }
    material = {
        "seed_id": seed_id,
        "horizon": horizon,
        "support_slot_identity": identity,
        "causal_difference_semantics": {
            "Phi_full": "D_U S_k = S_(k+1)^full - S_(k+1)^skip at horizon 1; for later horizons it is the propagated full-versus-skip effect under aligned subsequent opportunities.",
            "Phi_P": "parameter-only contribution relative to skip",
            "Phi_O": "optimizer-state-only contribution relative to skip",
            "Phi_PxO": "non-additive parameter-by-optimizer interaction",
            "trajectory_finite_difference_conflated": False,
        },
        "numeric_effects": numeric,
        "categorical_effects": categorical,
        "branch_probe_observation_ids": {
            branch: observation["probe_observation_id"]
            for branch, (_root, observation) in observations.items()
        },
    }
    return {**material, "effect_sha256": payload_sha256(material)}


def _probe_at_state(
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    *,
    branch_entry_root: Path,
    state: StateSnapshot,
    state_manifest: dict[str, Any],
) -> dict[str, Any]:
    runtime.restore(state)
    return ensure_probe_observation(
        runtime,
        store,
        entry_root=branch_entry_root,
        state_manifest=state_manifest,
    )


def execute_branch_seed(
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    *,
    formal_root: Path,
    branch_entry_root: Path,
    window: dict[str, Any],
    relative_h: int,
    pair_ids: list[str],
    branch_profile_sha256: str,
    main_protocol_sha256: str,
) -> dict[str, Any]:
    optimizer_step = int(window["start_optimizer_step"]) + relative_h
    seed_id = "branch-seed-" + payload_sha256(
        {
            "branch_profile_sha256": branch_profile_sha256,
            "entry_id": window["entry_id"],
            "optimizer_step": optimizer_step,
            "window_id": window["window_id"],
        }
    )[:32]
    seed_root = branch_entry_root / "branch-seeds" / seed_id
    receipt_path = seed_root / "branch_receipt.json"
    if receipt_path.exists():
        return _read_checked(receipt_path, "nanogpt-stepwise-four-branch-receipt-v1")

    source_record = _source_state_record(formal_root, window, optimizer_step)
    prestate = restorable_state_from_manifest(formal_root / str(window["entry_id"]), source_record["state"])
    batch = runtime.load_batch(optimizer_step)
    seed = runtime.derive_seed(branch_profile_sha256, str(window["entry_id"]), optimizer_step, 1)

    runtime.restore(prestate.clone())
    full_evidence = runtime.train_actual_step(batch, execute_optimizer=True, seed=seed)
    full_state = runtime.snapshot()
    runtime.restore(prestate.clone())
    skip_evidence = runtime.train_actual_step(batch, execute_optimizer=False, seed=seed)
    skip_state = runtime.snapshot()
    require(torch.equal(full_evidence.training_logits, skip_evidence.training_logits), "SST_BRANCH_SEED_LOGITS_NOT_MATCHED")
    require(_maps_exact(full_evidence.activation_outputs, skip_evidence.activation_outputs), "SST_BRANCH_SEED_ACTIVATIONS_NOT_MATCHED")
    require(_maps_exact(full_evidence.raw_gradients, skip_evidence.raw_gradients), "SST_BRANCH_SEED_RAW_GRADIENTS_NOT_MATCHED")
    require(_maps_exact(full_evidence.clipped_gradients, skip_evidence.clipped_gradients), "SST_BRANCH_SEED_CLIPPED_GRADIENTS_NOT_MATCHED")
    require(skip_state.commitment() == prestate.commitment(), "SST_BRANCH_SKIP_STATE_MUTATED")

    parameter_state = _state_from_parts(full_state.parameters, prestate.optimizer)
    optimizer_state = _state_from_parts(prestate.parameters, full_state.optimizer)
    branch_states = {
        "full_step": full_state.clone(),
        "skip_step": skip_state.clone(),
        "parameter_only": parameter_state,
        "optimizer_state_only": optimizer_state,
    }
    assert_snapshot_isolation(branch_states.values())
    seed_transition_id = "branch-transition-" + payload_sha256({"seed_id": seed_id, "optimizer_step": optimizer_step})[:32]
    seed_material = {
        "schema": "nanogpt-stepwise-branch-seed-v1",
        "status": "PASS",
        "seed_id": seed_id,
        "pair_ids": sorted(pair_ids),
        "entry_id": window["entry_id"],
        "window_id": window["window_id"],
        "relative_h": relative_h,
        "optimizer_step": optimizer_step,
        "immutable_prestate_id": source_record["state"]["state_id"],
        "immutable_prestate_sha256": prestate.commitment()["state_sha256"],
        "batch": _encode_batch(batch),
        "matched_forward_backward_clipping": True,
        "full_step": _encode_step(store, full_evidence, transition_id=seed_transition_id + ":full"),
        "skip_step": _encode_step(store, skip_evidence, transition_id=seed_transition_id + ":skip"),
        "parameter_only_composition": {
            "parameters_from": full_state.commitment()["state_sha256"],
            "optimizer_from": prestate.commitment()["state_sha256"],
            "native_training_occurrence_claimed": False,
        },
        "optimizer_state_only_composition": {
            "parameters_from": prestate.commitment()["state_sha256"],
            "optimizer_from": full_state.commitment()["state_sha256"],
            "native_training_occurrence_claimed": False,
        },
    }
    seed_result = _checked_result(seed_root / "seed_result.json", seed_material)

    remaining = int(window["end_optimizer_step"]) - optimizer_step
    legal_horizons = [value for value in HORIZONS if value <= remaining]
    require(bool(legal_horizons), f"SST_BRANCH_NO_LEGAL_HORIZON:{seed_id}")
    max_horizon = max(legal_horizons)
    horizon_results: list[dict[str, Any]] = []
    continuation_results: list[dict[str, Any]] = []

    for horizon in range(1, max_horizon + 1):
        if horizon in legal_horizons:
            observations: dict[str, tuple[Path, dict[str, Any]]] = {}
            state_rows: dict[str, Any] = {}
            for branch in BRANCHES:
                state_manifest = _encode_state(
                    store,
                    branch_states[branch],
                    entry_id=str(window["entry_id"]),
                    window_id=seed_id + ":" + branch,
                    optimizer_step=optimizer_step + horizon,
                    protocol_sha256=main_protocol_sha256,
                )
                state_result = _checked_result(
                    seed_root / "horizons" / f"h-{horizon:03d}" / f"{branch}-state.json",
                    {
                        "schema": "nanogpt-stepwise-branch-state-v1",
                        "status": "PASS",
                        "seed_id": seed_id,
                        "branch": branch,
                        "horizon": horizon,
                        "physical_optimizer_opportunity": optimizer_step + horizon,
                        "state": state_manifest,
                        "state_summary": runtime.state_summary(branch_states[branch]),
                    },
                )
                probe = _probe_at_state(
                    runtime,
                    store,
                    branch_entry_root=branch_entry_root,
                    state=branch_states[branch],
                    state_manifest=state_manifest,
                )
                state_rows[branch] = {
                    "state_id": state_manifest["state_id"],
                    "state_result_sha256": state_result["result_sha256"],
                    "probe_observation_id": probe["probe_observation_id"],
                    "probe_result_sha256": probe["result_sha256"],
                }
                observations[branch] = (branch_entry_root, probe)
            effects = _encode_effects(store, seed_id=seed_id, horizon=horizon, observations=observations)
            effect_result = _checked_result(
                seed_root / "horizons" / f"h-{horizon:03d}" / "effects.json",
                {"schema": "nanogpt-stepwise-branch-effects-v1", "status": "PASS", **effects},
            )
            horizon_results.append(
                {
                    "horizon": horizon,
                    "states": state_rows,
                    "effect_result_sha256": effect_result["result_sha256"],
                }
            )
        if horizon == max_horizon:
            break
        physical_step = optimizer_step + horizon
        continuation_batch = runtime.load_batch(physical_step)
        opportunity_seed = runtime.derive_seed(
            branch_profile_sha256,
            str(window["entry_id"]),
            physical_step,
            horizon + 1,
        )
        branch_steps: dict[str, Any] = {}
        for branch in BRANCHES:
            runtime.restore(branch_states[branch])
            pre_sha = branch_states[branch].commitment()["state_sha256"]
            evidence = runtime.train_actual_step(
                continuation_batch,
                execute_optimizer=True,
                seed=opportunity_seed,
            )
            branch_states[branch] = runtime.snapshot()
            branch_steps[branch] = {
                "from_state_sha256": pre_sha,
                "to_state_sha256": branch_states[branch].commitment()["state_sha256"],
                "step": _lightweight_step(evidence),
            }
        continuation_result = _checked_result(
            seed_root / "continuations" / f"step-{physical_step:05d}-to-{physical_step + 1:05d}.json",
            {
                "schema": "nanogpt-stepwise-branch-continuation-v1",
                "status": "PASS",
                "seed_id": seed_id,
                "relative_horizon_from": horizon,
                "physical_optimizer_step": physical_step,
                "training_opportunity_alignment": True,
                "same_batch_all_branches": _encode_batch(continuation_batch),
                "same_external_rng_opportunity_all_branches": opportunity_seed,
                "branches": branch_steps,
            },
        )
        continuation_results.append(
            {"physical_optimizer_step": physical_step, "result_sha256": continuation_result["result_sha256"]}
        )

    material = {
        "schema": "nanogpt-stepwise-four-branch-receipt-v1",
        "status": "PASS",
        "seed_id": seed_id,
        "pair_ids": sorted(pair_ids),
        "entry_id": window["entry_id"],
        "window_id": window["window_id"],
        "relative_h": relative_h,
        "optimizer_step": optimizer_step,
        "branch_profile_sha256": branch_profile_sha256,
        "main_protocol_sha256": main_protocol_sha256,
        "seed_result_sha256": seed_result["result_sha256"],
        "legal_horizons": legal_horizons,
        "horizon_results": horizon_results,
        "continuation_results": continuation_results,
        "branch_count": len(BRANCHES),
        "skip_consumed_no_extra_batch": True,
        "mutable_storage_isolation_checked": True,
    }
    return _checked_result(receipt_path, material)


def execute_all_branches(
    *,
    formal_root: Path,
    source_root: Path,
    source_archive_manifest_path: Path,
    trainer_root: Path,
    branch_output_root: Path,
    selection_path: Path,
    divergence_audit_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    main_protocol_path: Path,
    branch_profile_path: Path,
    budget_path: Path,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    audit = read_json(divergence_audit_path)
    archive = read_json(source_archive_manifest_path)
    budget = read_json(budget_path)
    windows = {str(row["window_id"]): row for row in selection["windows"]}
    bundles = {str(row["entry_id"]): str(row["gfg_bundle_id"]) for row in archive["support_bundles"]}
    requests: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in audit["results"]:
        for window_key in ("decline_window_id", "stable_window_id"):
            window = windows[str(row[window_key])]
            for relative_h in row["causal_seed_relative_h"]:
                key = (str(window["entry_id"]), str(window["window_id"]), int(relative_h))
                request = requests.setdefault(key, {"window": window, "relative_h": int(relative_h), "pair_ids": []})
                request["pair_ids"].append(str(row["pair_id"]))
    require(len(requests) <= int(budget["key_step_branch_seed_ceiling"]), "SST_BRANCH_SEED_BUDGET_EXCEEDED")
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    branch_profile_sha = file_sha256(branch_profile_path)
    main_protocol_sha = file_sha256(main_protocol_path)
    completed: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (entry_id, _window_id, _h), request in sorted(requests.items()):
        grouped.setdefault(entry_id, []).append(request)
    ordinal = 0
    for entry_id, entry_requests in sorted(grouped.items()):
        entry_root = branch_output_root / entry_id
        store = TensorStore(entry_root / "tensor-objects")
        runtime = StepwiseTrainingRuntime(source_root / bundles[entry_id], trainer_root, registry, probe_contract)
        try:
            for request in entry_requests:
                ordinal += 1
                result = execute_branch_seed(
                    runtime,
                    store,
                    formal_root=formal_root,
                    branch_entry_root=entry_root,
                    window=request["window"],
                    relative_h=request["relative_h"],
                    pair_ids=sorted(set(request["pair_ids"])),
                    branch_profile_sha256=branch_profile_sha,
                    main_protocol_sha256=main_protocol_sha,
                )
                completed.append({"seed_id": result["seed_id"], "result_sha256": result["result_sha256"]})
                print({"event": "SST_BRANCH_SEED_COMPLETE", "ordinal": ordinal, "seed_count": len(requests), "seed_id": result["seed_id"]}, flush=True)
        finally:
            runtime.close()
    material = {
        "schema": "nanogpt-stepwise-all-branches-receipt-v1",
        "status": "PASS",
        "formal_root": str(formal_root.resolve()),
        "branch_output_root": str(branch_output_root.resolve()),
        "selection_sha256": selection["selection_sha256"],
        "divergence_audit_sha256": audit["audit_sha256"],
        "branch_profile_sha256": branch_profile_sha,
        "main_protocol_sha256": main_protocol_sha,
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "unique_seed_count": len(requests),
        "seed_ceiling": budget["key_step_branch_seed_ceiling"],
        "completed": completed,
    }
    return _checked_result(branch_output_root / "all_branches_receipt.json", material)
