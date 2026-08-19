from __future__ import annotations

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
from experiments.gfg_nanogpt_support_transition_v1.runtime import (
    StateSnapshot,
    assert_snapshot_isolation,
)

from .branches import (
    _load_observation_arrays,
    _maps_exact,
    _probe_at_state,
    _source_state_record,
    _state_from_parts,
)
from .contracts import ComponentRegistry, ProbeContract
from .execution import (
    _checked_result,
    _encode_batch,
    _encode_state,
    _encode_step,
)
from .runtime import StepEvidence, StepwiseTrainingRuntime
from .storage import TensorStore, restorable_state_from_manifest


RECIPROCAL_BRANCHES = (
    "skip",
    "native_full",
    "native_parameter_only",
    "native_optimizer_only",
    "donor_parameter_delta",
    "donor_optimizer_innovation",
    "donor_joint_update",
)


def _clone_optimizer(
    values: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            key: value.detach().contiguous().cpu().clone()
            for key, value in child.items()
        }
        for name, child in values.items()
    }


def _optimizer_innovation(
    post: Mapping[str, Mapping[str, torch.Tensor]],
    pre: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    betas: tuple[float, float],
) -> dict[str, dict[str, torch.Tensor]]:
    require(set(post) == set(pre), "SST_RECIPROCAL_OPTIMIZER_PARAMETER_SET_MISMATCH")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name in sorted(pre):
        require(set(post[name]) == set(pre[name]), f"SST_RECIPROCAL_OPTIMIZER_KEY_SET_MISMATCH:{name}")
        result[name] = {
            "step": post[name]["step"] - pre[name]["step"],
            "exp_avg": post[name]["exp_avg"] - betas[0] * pre[name]["exp_avg"],
            "exp_avg_sq": post[name]["exp_avg_sq"] - betas[1] * pre[name]["exp_avg_sq"],
        }
    return result


def _transplant_state(
    recipient: StateSnapshot,
    *,
    parameter_delta: Mapping[str, torch.Tensor] | None = None,
    optimizer_innovation: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    optimizer_betas: tuple[float, float] | None = None,
) -> StateSnapshot:
    parameters = {
        name: value.detach().contiguous().cpu().clone()
        for name, value in recipient.parameters.items()
    }
    optimizer = _clone_optimizer(recipient.optimizer)
    if parameter_delta is not None:
        require(set(parameter_delta) == set(parameters), "SST_RECIPROCAL_PARAMETER_DELTA_SET_MISMATCH")
        for name in sorted(parameters):
            require(parameter_delta[name].shape == parameters[name].shape, f"SST_RECIPROCAL_PARAMETER_DELTA_SHAPE_MISMATCH:{name}")
            parameters[name].add_(parameter_delta[name].to(parameters[name].dtype))
    if optimizer_innovation is not None:
        require(optimizer_betas is not None, "SST_RECIPROCAL_OPTIMIZER_BETAS_MISSING")
        require(set(optimizer_innovation) == set(optimizer), "SST_RECIPROCAL_OPTIMIZER_DELTA_SET_MISMATCH")
        for name in sorted(optimizer):
            require(set(optimizer_innovation[name]) == set(optimizer[name]), f"SST_RECIPROCAL_OPTIMIZER_DELTA_KEY_SET_MISMATCH:{name}")
            for key in sorted(optimizer[name]):
                require(optimizer_innovation[name][key].shape == optimizer[name][key].shape, f"SST_RECIPROCAL_OPTIMIZER_DELTA_SHAPE_MISMATCH:{name}.{key}")
                if key == "step":
                    optimizer[name][key].add_(optimizer_innovation[name][key].to(optimizer[name][key].dtype))
                else:
                    beta = optimizer_betas[0] if key == "exp_avg" else optimizer_betas[1]
                    optimizer[name][key].mul_(beta).add_(optimizer_innovation[name][key].to(optimizer[name][key].dtype))
                require(bool(torch.isfinite(optimizer[name][key]).all()), f"SST_RECIPROCAL_OPTIMIZER_DELTA_NONFINITE:{name}.{key}")
            require(bool((optimizer[name]["exp_avg_sq"] >= 0).all()), f"SST_RECIPROCAL_EXP_AVG_SQ_NEGATIVE:{name}")
    return StateSnapshot(parameters, optimizer)


def _window(formal_root: Path, endpoint: dict[str, Any]) -> dict[str, Any]:
    receipt = read_json(
        formal_root
        / str(endpoint["entry_id"])
        / "windows"
        / str(endpoint["window_id"])
        / "window_receipt.json"
    )
    require(receipt["schema"] == "nanogpt-stepwise-window-receipt-v1", "SST_RECIPROCAL_WINDOW_SCHEMA_INVALID")
    require(receipt["status"] == "PASS", "SST_RECIPROCAL_WINDOW_NOT_PASS")
    window = receipt["window"]
    for key in ("entry_id", "window_id", "source_bundle_id"):
        require(str(window[key]) == str(endpoint[key]), f"SST_RECIPROCAL_ENDPOINT_WINDOW_MISMATCH:{key}")
    step = int(endpoint["optimizer_step"])
    require(int(window["start_optimizer_step"]) <= step < int(window["end_optimizer_step"]), "SST_RECIPROCAL_STEP_OUTSIDE_WINDOW")
    return window


def _seed_execution(
    runtime: StepwiseTrainingRuntime,
    *,
    formal_root: Path,
    endpoint: dict[str, Any],
    window: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    step = int(endpoint["optimizer_step"])
    source_record = _source_state_record(formal_root, window, step)
    prestate = restorable_state_from_manifest(formal_root / str(endpoint["entry_id"]), source_record["state"])
    batch = runtime.load_batch(step)
    seed = runtime.derive_seed(protocol_sha256, str(endpoint["entry_id"]), step, 1)
    runtime.restore(prestate.clone())
    full_evidence = runtime.train_actual_step(batch, execute_optimizer=True, seed=seed)
    full_state = runtime.snapshot()
    runtime.restore(prestate.clone())
    skip_evidence = runtime.train_actual_step(batch, execute_optimizer=False, seed=seed)
    skip_state = runtime.snapshot()
    require(_maps_exact(full_evidence.activation_outputs, skip_evidence.activation_outputs), "SST_RECIPROCAL_SEED_ACTIVATIONS_NOT_MATCHED")
    require(_maps_exact(full_evidence.raw_gradients, skip_evidence.raw_gradients), "SST_RECIPROCAL_SEED_RAW_GRADIENTS_NOT_MATCHED")
    require(_maps_exact(full_evidence.clipped_gradients, skip_evidence.clipped_gradients), "SST_RECIPROCAL_SEED_CLIPPED_GRADIENTS_NOT_MATCHED")
    require(torch.equal(full_evidence.training_logits, skip_evidence.training_logits), "SST_RECIPROCAL_SEED_LOGITS_NOT_MATCHED")
    require(skip_state.commitment() == prestate.commitment(), "SST_RECIPROCAL_SKIP_STATE_MUTATED")
    return {
        "endpoint": endpoint,
        "window": window,
        "source_record": source_record,
        "prestate": prestate,
        "full_state": full_state,
        "full_evidence": full_evidence,
        "skip_evidence": skip_evidence,
        "batch": batch,
        "seed": seed,
        "parameter_delta": {name: full_state.parameters[name] - prestate.parameters[name] for name in prestate.parameters},
        "optimizer_betas": tuple(float(value) for value in full_evidence.optimizer_config["betas"]),
        "optimizer_innovation": _optimizer_innovation(
            full_state.optimizer,
            prestate.optimizer,
            betas=tuple(float(value) for value in full_evidence.optimizer_config["betas"]),
        ),
    }


def _branch_states(recipient: dict[str, Any], donor: dict[str, Any]) -> dict[str, StateSnapshot]:
    prestate: StateSnapshot = recipient["prestate"]
    full_state: StateSnapshot = recipient["full_state"]
    states = {
        "skip": prestate.clone(),
        "native_full": full_state.clone(),
        "native_parameter_only": _state_from_parts(full_state.parameters, prestate.optimizer),
        "native_optimizer_only": _state_from_parts(prestate.parameters, full_state.optimizer),
        "donor_parameter_delta": _transplant_state(prestate, parameter_delta=donor["parameter_delta"]),
        "donor_optimizer_innovation": _transplant_state(
            prestate,
            optimizer_innovation=donor["optimizer_innovation"],
            optimizer_betas=donor["optimizer_betas"],
        ),
        "donor_joint_update": _transplant_state(
            prestate,
            parameter_delta=donor["parameter_delta"],
            optimizer_innovation=donor["optimizer_innovation"],
            optimizer_betas=donor["optimizer_betas"],
        ),
    }
    require(tuple(states) == RECIPROCAL_BRANCHES, "SST_RECIPROCAL_BRANCH_ORDER_INVALID")
    assert_snapshot_isolation(states.values())
    return states


def _numeric_responses(observation_root: Path, observation: dict[str, Any]) -> dict[str, np.ndarray]:
    values = _load_observation_arrays(observation_root, observation)
    return {key: value.astype(np.float64, copy=False) for key, value in values.items() if np.issubdtype(value.dtype, np.floating)}


def _effect_rows(
    store: TensorStore,
    *,
    recipient_label: str,
    horizon: int,
    observation_root: Path,
    observations: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    arrays = {branch: _numeric_responses(observation_root, observation) for branch, observation in observations.items()}
    keys = tuple(sorted(arrays["skip"]))
    require(all(tuple(sorted(value)) == keys for value in arrays.values()), "SST_RECIPROCAL_PROBE_OUTPUT_SET_MISMATCH")
    effects: dict[str, Any] = {}
    for key in keys:
        skip = arrays["skip"][key]
        effects[key] = {
            branch: store.put(
                arrays[branch][key] - skip,
                representation=f"reciprocal:{recipient_label}:h{horizon}:{branch}:{key}:branch_minus_skip_response",
            )
            for branch in RECIPROCAL_BRANCHES
            if branch != "skip"
        }
    return effects


def _execute_recipient(
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    *,
    output_root: Path,
    recipient: dict[str, Any],
    donor: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    main_protocol_sha256: str,
) -> dict[str, Any]:
    label = str(recipient["endpoint"]["label"])
    donor_label = str(donor["endpoint"]["label"])
    root = output_root / f"recipient-{label}"
    states = _branch_states(recipient, donor)
    endpoint = recipient["endpoint"]
    step = int(endpoint["optimizer_step"])
    seed_id = "reciprocal-seed-" + payload_sha256({"protocol_sha256": protocol_sha256, "recipient": endpoint, "donor": donor["endpoint"]})[:32]
    seed_material = {
        "schema": "nanogpt-reciprocal-seed-v1",
        "status": "PASS",
        "seed_id": seed_id,
        "recipient": endpoint,
        "donor": donor["endpoint"],
        "recipient_prestate_id": recipient["source_record"]["state"]["state_id"],
        "recipient_prestate_sha256": recipient["prestate"].commitment()["state_sha256"],
        "donor_prestate_id": donor["source_record"]["state"]["state_id"],
        "donor_prestate_sha256": donor["prestate"].commitment()["state_sha256"],
        "recipient_batch": _encode_batch(recipient["batch"]),
        "donor_batch": _encode_batch(donor["batch"]),
        "recipient_native_full_step": _encode_step(store, recipient["full_evidence"], transition_id=f"{seed_id}:recipient-native-full"),
        "recipient_native_skip_step": _encode_step(store, recipient["skip_evidence"], transition_id=f"{seed_id}:recipient-native-skip"),
        "donor_native_full_step": _encode_step(store, donor["full_evidence"], transition_id=f"{seed_id}:donor-native-full"),
        "named_identity_alignment": sorted(recipient["prestate"].parameters),
        "transplant_semantics": protocol["transplant_semantics"],
        "absolute_donor_poststate_used": False,
    }
    seed_result = _checked_result(root / "seed_result.json", seed_material)
    horizons = tuple(int(value) for value in protocol["horizons"])
    max_horizon = max(horizons)
    horizon_rows: list[dict[str, Any]] = []
    continuation_rows: list[dict[str, Any]] = []
    for horizon in range(1, max_horizon + 1):
        if horizon in horizons:
            observations: dict[str, dict[str, Any]] = {}
            state_rows: dict[str, Any] = {}
            for branch in RECIPROCAL_BRANCHES:
                manifest = _encode_state(
                    store,
                    states[branch],
                    entry_id=str(endpoint["entry_id"]),
                    window_id=f"{seed_id}:{branch}",
                    optimizer_step=step + horizon,
                    protocol_sha256=main_protocol_sha256,
                )
                state_result = _checked_result(
                    root / "horizons" / f"h-{horizon:03d}" / f"{branch}-state.json",
                    {
                        "schema": "nanogpt-reciprocal-branch-state-v1",
                        "status": "PASS",
                        "seed_id": seed_id,
                        "recipient_label": label,
                        "donor_label": donor_label,
                        "branch": branch,
                        "horizon": horizon,
                        "physical_optimizer_opportunity": step + horizon,
                        "state": manifest,
                        "state_summary": runtime.state_summary(states[branch]),
                    },
                )
                observation = _probe_at_state(runtime, store, branch_entry_root=root, state=states[branch], state_manifest=manifest)
                observations[branch] = observation
                state_rows[branch] = {
                    "state_id": manifest["state_id"],
                    "state_result_sha256": state_result["result_sha256"],
                    "probe_observation_id": observation["probe_observation_id"],
                    "probe_result_sha256": observation["result_sha256"],
                    "capability_accuracy": observation["capability_accuracy"],
                }
            effects = _effect_rows(
                store,
                recipient_label=label,
                horizon=horizon,
                observation_root=root,
                observations=observations,
            )
            effect_result = _checked_result(
                root / "horizons" / f"h-{horizon:03d}" / "effects.json",
                {
                    "schema": "nanogpt-reciprocal-branch-effects-v1",
                    "status": "PASS",
                    "seed_id": seed_id,
                    "recipient_label": label,
                    "donor_label": donor_label,
                    "horizon": horizon,
                    "response_semantics": "branch_minus_recipient_skip",
                    "numeric_effects": effects,
                },
            )
            horizon_rows.append({"horizon": horizon, "states": state_rows, "effect_result_sha256": effect_result["result_sha256"]})
        if horizon == max_horizon:
            break
        physical_step = step + horizon
        batch = runtime.load_batch(physical_step)
        opportunity_seed = runtime.derive_seed(protocol_sha256, str(endpoint["entry_id"]), physical_step, horizon + 1)
        branch_steps: dict[str, Any] = {}
        for branch in RECIPROCAL_BRANCHES:
            runtime.restore(states[branch])
            pre_sha = states[branch].commitment()["state_sha256"]
            evidence = runtime.train_actual_step(batch, execute_optimizer=True, seed=opportunity_seed)
            states[branch] = runtime.snapshot()
            branch_steps[branch] = {
                "from_state_sha256": pre_sha,
                "to_state_sha256": states[branch].commitment()["state_sha256"],
                "loss": evidence.loss,
                "total_gradient_norm": evidence.total_gradient_norm,
                "execute_optimizer": evidence.execute_optimizer,
                "rng_before": evidence.rng_before,
                "rng_after": evidence.rng_after,
            }
        continuation = _checked_result(
            root / "continuations" / f"step-{physical_step:05d}-to-{physical_step + 1:05d}.json",
            {
                "schema": "nanogpt-reciprocal-continuation-v1",
                "status": "PASS",
                "seed_id": seed_id,
                "recipient_label": label,
                "physical_optimizer_step": physical_step,
                "same_batch_all_branches": _encode_batch(batch),
                "same_external_rng_opportunity_all_branches": opportunity_seed,
                "branches": branch_steps,
            },
        )
        continuation_rows.append({"physical_optimizer_step": physical_step, "result_sha256": continuation["result_sha256"]})
    return _checked_result(
        root / "reciprocal_receipt.json",
        {
            "schema": "nanogpt-reciprocal-recipient-receipt-v1",
            "status": "PASS",
            "seed_id": seed_id,
            "recipient": endpoint,
            "donor": donor["endpoint"],
            "seed_result_sha256": seed_result["result_sha256"],
            "branches": list(RECIPROCAL_BRANCHES),
            "horizons": list(horizons),
            "horizon_results": horizon_rows,
            "continuation_results": continuation_rows,
            "future_information_used": False,
        },
    )


def execute_reciprocal_pair(
    *,
    formal_root: Path,
    source_root: Path,
    trainer_root: Path,
    output_root: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    main_protocol_path: Path,
    reciprocal_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(reciprocal_protocol_path)
    require(protocol["schema"] == "nanogpt-reciprocal-matched-pair-protocol-v2", "SST_RECIPROCAL_PROTOCOL_SCHEMA_INVALID")
    require(tuple(protocol["branches"]) == RECIPROCAL_BRANCHES, "SST_RECIPROCAL_PROTOCOL_BRANCHES_INVALID")
    endpoints = protocol["endpoints"]
    require(len(endpoints) == 2 and {row["label"] for row in endpoints} == {"A", "B"}, "SST_RECIPROCAL_ENDPOINTS_INVALID")
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    protocol_sha = file_sha256(reciprocal_protocol_path)
    main_protocol_sha = file_sha256(main_protocol_path)
    executions: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        window = _window(formal_root, endpoint)
        runtime = StepwiseTrainingRuntime(source_root / str(endpoint["source_bundle_id"]), trainer_root, registry, probe_contract)
        try:
            executions[str(endpoint["label"])] = _seed_execution(
                runtime,
                formal_root=formal_root,
                endpoint=endpoint,
                window=window,
                protocol_sha256=protocol_sha,
            )
        finally:
            runtime.close()
    completed: list[dict[str, Any]] = []
    for label, donor_label in (("A", "B"), ("B", "A")):
        endpoint = executions[label]["endpoint"]
        store = TensorStore(output_root / f"recipient-{label}" / "tensor-objects")
        runtime = StepwiseTrainingRuntime(source_root / str(endpoint["source_bundle_id"]), trainer_root, registry, probe_contract)
        try:
            receipt = _execute_recipient(
                runtime,
                store,
                output_root=output_root,
                recipient=executions[label],
                donor=executions[donor_label],
                protocol=protocol,
                protocol_sha256=protocol_sha,
                main_protocol_sha256=main_protocol_sha,
            )
            completed.append({"recipient_label": label, "receipt_sha256": receipt["result_sha256"]})
            print({"event": "SST_RECIPROCAL_RECIPIENT_COMPLETE", "recipient": label}, flush=True)
        finally:
            runtime.close()
    return _checked_result(
        output_root / "reciprocal_pair_receipt.json",
        {
            "schema": "nanogpt-reciprocal-pair-receipt-v1",
            "status": "PASS",
            "formal_root": str(formal_root.resolve()),
            "source_root": str(source_root.resolve()),
            "trainer_root": str(trainer_root.resolve()),
            "output_root": str(output_root.resolve()),
            "reciprocal_protocol_sha256": protocol_sha,
            "main_protocol_sha256": main_protocol_sha,
            "component_registry_sha256": registry.source_sha256,
            "probe_contract_sha256": probe_contract.source_sha256,
            "completed": completed,
            "recipient_count": 2,
            "branch_count_per_recipient": len(RECIPROCAL_BRANCHES),
            "horizons": protocol["horizons"],
            "future_information_used": False,
        },
    )
