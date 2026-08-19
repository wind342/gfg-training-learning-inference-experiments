from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
)
from experiments.gfg_nanogpt_support_transition_v1.runtime import assert_snapshot_isolation

from .branches import _load_observation_arrays, _probe_at_state
from .contracts import ComponentRegistry, ProbeContract
from .execution import _checked_result, _encode_state, _encode_step
from .reciprocal import _seed_execution, _transplant_state, _window
from .runtime import StepwiseTrainingRuntime
from .storage import TensorStore


LOCAL_RESPONSE_BRANCHES = ("baseline", "plus_epsilon", "minus_epsilon")
LOCAL_RESPONSE_TRANSPORT_BRANCHES = (
    "baseline",
    "plus_epsilon",
    "minus_epsilon",
    "plus_full",
)


def _scaled_delta(delta: dict[str, Any], scale: float) -> dict[str, Any]:
    return {
        name: value.detach().contiguous().cpu().mul(scale)
        for name, value in sorted(delta.items())
    }


def _categorical_transition(
    store: TensorStore,
    *,
    baseline: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    full: np.ndarray | None,
    prefix: str,
) -> dict[str, Any]:
    plus_changed = np.not_equal(plus, baseline)
    minus_changed = np.not_equal(minus, baseline)
    result = {
        "baseline": store.put(baseline, representation=f"{prefix}:categorical_baseline"),
        "plus": store.put(plus, representation=f"{prefix}:categorical_plus_epsilon"),
        "minus": store.put(minus, representation=f"{prefix}:categorical_minus_epsilon"),
        "plus_changed_mask": store.put(plus_changed, representation=f"{prefix}:plus_changed_mask"),
        "minus_changed_mask": store.put(minus_changed, representation=f"{prefix}:minus_changed_mask"),
        "plus_changed_count": int(np.count_nonzero(plus_changed)),
        "minus_changed_count": int(np.count_nonzero(minus_changed)),
    }
    if full is not None:
        full_changed = np.not_equal(full, baseline)
        result.update(
            {
                "full": store.put(full, representation=f"{prefix}:categorical_plus_full"),
                "full_changed_mask": store.put(
                    full_changed,
                    representation=f"{prefix}:full_changed_mask",
                ),
                "full_changed_count": int(np.count_nonzero(full_changed)),
            }
        )
    return result


def _execute_receiver(
    *,
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    output_root: Path,
    receiver: dict[str, Any],
    donor: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    main_protocol_sha256: str,
) -> dict[str, Any]:
    label = str(receiver["endpoint"]["label"])
    root = output_root / f"receiver-{label}"
    epsilon = float(protocol["epsilon"])
    receiver_state_kind = str(protocol.get("receiver_state_kind", "skip"))
    require(
        receiver_state_kind in {"skip", "native_full"},
        "SST_LOCAL_RESPONSE_RECEIVER_STATE_KIND_INVALID",
    )
    prestate = (
        receiver["prestate"]
        if receiver_state_kind == "skip"
        else receiver["full_state"]
    )
    donor_delta = donor["parameter_delta"]
    states = {
        "baseline": prestate.clone(),
        "plus_epsilon": _transplant_state(
            prestate,
            parameter_delta=_scaled_delta(donor_delta, epsilon),
        ),
        "minus_epsilon": _transplant_state(
            prestate,
            parameter_delta=_scaled_delta(donor_delta, -epsilon),
        ),
    }
    branches = tuple(str(value) for value in protocol["branches"])
    if branches == LOCAL_RESPONSE_TRANSPORT_BRANCHES:
        states["plus_full"] = _transplant_state(
            prestate,
            parameter_delta=_scaled_delta(donor_delta, 1.0),
        )
    require(tuple(states) == branches, "SST_LOCAL_RESPONSE_BRANCH_ORDER_INVALID")
    assert_snapshot_isolation(states.values())
    endpoint = receiver["endpoint"]
    step = int(endpoint["optimizer_step"])
    experiment_id = "local-response-" + payload_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "receiver": endpoint,
            "donor": donor["endpoint"],
        }
    )[:32]
    seed = _checked_result(
        root / "local_response_seed.json",
        {
            "schema": "nanogpt-local-response-seed-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver": endpoint,
            "donor": donor["endpoint"],
            "donor_update_source_object_id": protocol["donor_update"]["source_object_id"],
            "receiver_prestate_id": receiver["source_record"]["state"]["state_id"],
            "receiver_prestate_sha256": prestate.commitment()["state_sha256"],
            "receiver_state_kind": receiver_state_kind,
            "receiver_source_object_id": protocol.get(
                "receiver_source_object_ids", {}
            ).get(label),
            "donor_prestate_id": donor["source_record"]["state"]["state_id"],
            "donor_prestate_sha256": donor["prestate"].commitment()["state_sha256"],
            "donor_native_full_step": _encode_step(
                store,
                donor["full_evidence"],
                transition_id=f"{experiment_id}:donor-native-full",
            ),
            "epsilon": epsilon,
            "adam_state_transplanted": False,
            "future_information_used": False,
        },
    )
    observations: dict[str, dict[str, Any]] = {}
    states_out: dict[str, Any] = {}
    scales = {
        "baseline": 0.0,
        "plus_epsilon": epsilon,
        "minus_epsilon": -epsilon,
        "plus_full": 1.0,
    }
    for branch in branches:
        manifest = _encode_state(
            store,
            states[branch],
            entry_id=str(endpoint["entry_id"]),
            window_id=f"{experiment_id}:{branch}",
            optimizer_step=step + 1,
            protocol_sha256=main_protocol_sha256,
        )
        state_result = _checked_result(
            root / "h-001" / f"{branch}-state.json",
            {
                "schema": "nanogpt-local-response-state-v1",
                "status": "PASS",
                "experiment_id": experiment_id,
                "receiver_label": label,
                "branch": branch,
                "horizon": 1,
                "scale": scales[branch],
                "receiver_state_kind": receiver_state_kind,
                "state": manifest,
                "state_summary": runtime.state_summary(states[branch]),
            },
        )
        observation = _probe_at_state(
            runtime,
            store,
            branch_entry_root=root,
            state=states[branch],
            state_manifest=manifest,
        )
        observations[branch] = observation
        states_out[branch] = {
            "state_id": manifest["state_id"],
            "state_sha256": manifest["commitment"]["state_sha256"],
            "state_result_sha256": state_result["result_sha256"],
            "probe_observation_id": observation["probe_observation_id"],
            "probe_result_sha256": observation["result_sha256"],
            "capability_accuracy": observation["capability_accuracy"],
        }

    arrays = {
        branch: _load_observation_arrays(root, observation)
        for branch, observation in observations.items()
    }
    keys = tuple(sorted(arrays["baseline"]))
    require(
        all(tuple(sorted(value)) == keys for value in arrays.values()),
        "SST_LOCAL_RESPONSE_PROBE_OUTPUT_SET_MISMATCH",
    )
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for key in keys:
        baseline = arrays["baseline"][key]
        plus = arrays["plus_epsilon"][key]
        minus = arrays["minus_epsilon"][key]
        full = arrays["plus_full"][key] if "plus_full" in arrays else None
        require(baseline.shape == plus.shape == minus.shape, "SST_LOCAL_RESPONSE_SHAPE_MISMATCH")
        prefix = f"local-response:{label}:h1:{key}"
        if np.issubdtype(baseline.dtype, np.floating):
            base64 = baseline.astype(np.float64, copy=False)
            plus64 = plus.astype(np.float64, copy=False)
            minus64 = minus.astype(np.float64, copy=False)
            j_value = (plus64 - minus64) / (2.0 * epsilon)
            k_value = (plus64 + minus64 - 2.0 * base64) / (epsilon * epsilon)
            numeric[key] = {
                "baseline": store.put(base64, representation=f"{prefix}:numeric_baseline_float64"),
                "plus": store.put(plus64, representation=f"{prefix}:numeric_plus_float64"),
                "minus": store.put(minus64, representation=f"{prefix}:numeric_minus_float64"),
                "j_first_order": store.put(j_value, representation=f"{prefix}:central_first_order_response"),
                "k_curvature": store.put(k_value, representation=f"{prefix}:central_second_order_curvature"),
            }
            if full is not None:
                full64 = full.astype(np.float64, copy=False)
                numeric[key].update(
                    {
                        "full": store.put(
                            full64,
                            representation=f"{prefix}:numeric_plus_full_float64",
                        ),
                        "full_delta": store.put(
                            full64 - base64,
                            representation=f"{prefix}:exact_plus_full_response",
                        ),
                    }
                )
        else:
            categorical[key] = _categorical_transition(
                store,
                baseline=baseline,
                plus=plus,
                minus=minus,
                full=full,
                prefix=prefix,
            )
    response = _checked_result(
        root / "local_response_jk.json",
        {
            "schema": "nanogpt-local-response-jk-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver_label": label,
            "donor_label": str(donor["endpoint"]["label"]),
            "horizon": 1,
            "epsilon": epsilon,
            "branches": list(branches),
            "receiver_state_kind": receiver_state_kind,
            "numeric_responses": numeric,
            "categorical_transitions": categorical,
            "categorical_values_subtracted": False,
            "future_information_used": False,
        },
    )
    return _checked_result(
        root / "local_response_receipt.json",
        {
            "schema": "nanogpt-local-response-receiver-receipt-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver": endpoint,
            "donor": donor["endpoint"],
            "seed_result_sha256": seed["result_sha256"],
            "states": states_out,
            "response_result_sha256": response["result_sha256"],
            "future_information_used": False,
        },
    )


def execute_local_response_jk(
    *,
    formal_root: Path,
    source_root: Path,
    trainer_root: Path,
    output_root: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    main_protocol_path: Path,
    local_response_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(local_response_protocol_path)
    require(protocol["schema"] == "nanogpt-local-response-jk-protocol-v1", "SST_LOCAL_RESPONSE_PROTOCOL_SCHEMA_INVALID")
    branches = tuple(str(value) for value in protocol["branches"])
    require(
        branches in {LOCAL_RESPONSE_BRANCHES, LOCAL_RESPONSE_TRANSPORT_BRANCHES},
        "SST_LOCAL_RESPONSE_PROTOCOL_BRANCHES_INVALID",
    )
    require(float(protocol["epsilon"]) == 0.125, "SST_LOCAL_RESPONSE_EPSILON_INVALID")
    receivers = protocol["receivers"]
    receiver_labels = {str(row["label"]) for row in receivers}
    require(len(receivers) == 2 and receiver_labels == {"A", "B"}, "SST_LOCAL_RESPONSE_RECEIVERS_INVALID")
    donor_label = str(protocol["donor_update"]["label"])
    require(donor_label in receiver_labels, "SST_LOCAL_RESPONSE_DONOR_INVALID")
    require(
        str(protocol.get("receiver_state_kind", "skip"))
        in {"skip", "native_full"},
        "SST_LOCAL_RESPONSE_PROTOCOL_RECEIVER_STATE_KIND_INVALID",
    )
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    protocol_sha = file_sha256(local_response_protocol_path)
    main_protocol_sha = file_sha256(main_protocol_path)
    executions: dict[str, dict[str, Any]] = {}
    for endpoint in receivers:
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
    donor = executions[donor_label]
    for label in ("A", "B"):
        endpoint = executions[label]["endpoint"]
        store = TensorStore(output_root / f"receiver-{label}" / "tensor-objects")
        runtime = StepwiseTrainingRuntime(source_root / str(endpoint["source_bundle_id"]), trainer_root, registry, probe_contract)
        try:
            receipt = _execute_receiver(
                runtime=runtime,
                store=store,
                output_root=output_root,
                receiver=executions[label],
                donor=donor,
                protocol=protocol,
                protocol_sha256=protocol_sha,
                main_protocol_sha256=main_protocol_sha,
            )
            completed.append({"receiver_label": label, "receipt_sha256": receipt["result_sha256"]})
            print({"event": "SST_LOCAL_RESPONSE_RECEIVER_COMPLETE", "receiver": label}, flush=True)
        finally:
            runtime.close()
    return _checked_result(
        output_root / "local_response_pair_receipt.json",
        {
            "schema": "nanogpt-local-response-pair-receipt-v1",
            "status": "PASS",
            "formal_root": str(formal_root.resolve()),
            "source_root": str(source_root.resolve()),
            "trainer_root": str(trainer_root.resolve()),
            "output_root": str(output_root.resolve()),
            "local_response_protocol_sha256": protocol_sha,
            "main_protocol_sha256": main_protocol_sha,
            "component_registry_sha256": registry.source_sha256,
            "probe_contract_sha256": probe_contract.source_sha256,
            "completed": completed,
            "receiver_count": 2,
            "branch_count_per_receiver": len(branches),
            "epsilon": protocol["epsilon"],
            "horizon": 1,
            "receiver_state_kind": str(
                protocol.get("receiver_state_kind", "skip")
            ),
            "future_information_used": False,
        },
    )
