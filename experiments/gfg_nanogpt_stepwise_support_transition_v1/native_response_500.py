from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    read_json,
    require,
)
from experiments.gfg_nanogpt_support_transition_v1.runtime import (
    StateSnapshot,
    assert_snapshot_isolation,
)

from .branches import _load_observation_arrays, _probe_at_state, _source_state_record
from .contracts import ComponentRegistry, ProbeContract
from .execution import _checked_result, _encode_state, _read_checked
from .local_response import _categorical_transition, _scaled_delta
from .p2_response import _load_exact_update, _optimizer_exact
from .reciprocal import _transplant_state
from .runtime import StepwiseTrainingRuntime
from .storage import TensorStore, restorable_state_from_manifest


NATIVE_RESPONSE_500_SCHEMA = "nanogpt-native-direction-response-500-protocol-v1"
BRANCHES = ("baseline", "native_minus_0.125", "native_plus_0.125")


def _load_receiver(formal_root: Path, endpoint: dict[str, Any]) -> dict[str, Any]:
    receipt = read_json(
        formal_root
        / str(endpoint["entry_id"])
        / "windows"
        / str(endpoint["window_id"])
        / "window_receipt.json"
    )
    require(receipt["schema"] == "nanogpt-stepwise-window-receipt-v1", "NATIVE_RESPONSE_500_WINDOW_SCHEMA_INVALID")
    require(receipt["status"] == "PASS", "NATIVE_RESPONSE_500_WINDOW_NOT_PASS")
    window = receipt["window"]
    for key in ("entry_id", "window_id", "source_bundle_id"):
        require(str(window[key]) == str(endpoint[key]), f"NATIVE_RESPONSE_500_WINDOW_IDENTITY_MISMATCH:{key}")
    step = int(endpoint["optimizer_step"])
    require(
        int(window["capture_start_optimizer_step"]) <= step < int(window["capture_end_optimizer_step"]),
        "NATIVE_RESPONSE_500_STEP_OUTSIDE_CAPTURED_WINDOW",
    )
    entry_root = formal_root / str(endpoint["entry_id"])
    state_path = entry_root / str(endpoint["prestate"]["state_record_path"])
    require(file_sha256(state_path) == endpoint["prestate"]["state_record_file_sha256"], "NATIVE_RESPONSE_500_PRESTATE_FILE_HASH_MISMATCH")
    state_record = _source_state_record(formal_root, window, step)
    require(state_record["result_sha256"] == endpoint["prestate"]["state_record_result_sha256"], "NATIVE_RESPONSE_500_PRESTATE_RESULT_HASH_MISMATCH")
    require(state_record["state"]["state_id"] == endpoint["prestate"]["state_id"], "NATIVE_RESPONSE_500_PRESTATE_ID_MISMATCH")
    require(state_record["state"]["commitment"]["state_sha256"] == endpoint["prestate"]["state_sha256"], "NATIVE_RESPONSE_500_PRESTATE_COMMITMENT_MISMATCH")
    prestate = restorable_state_from_manifest(entry_root, state_record["state"])

    transition_path = entry_root / str(endpoint["native_update"]["transition_record_path"])
    require(file_sha256(transition_path) == endpoint["native_update"]["transition_record_file_sha256"], "NATIVE_RESPONSE_500_TRANSITION_FILE_HASH_MISMATCH")
    transition = _read_checked(transition_path, "nanogpt-stepwise-transition-v1")
    require(transition["result_sha256"] == endpoint["native_update"]["transition_record_result_sha256"], "NATIVE_RESPONSE_500_TRANSITION_RESULT_HASH_MISMATCH")
    require(transition["transition_id"] == endpoint["native_update"]["transition_id"], "NATIVE_RESPONSE_500_TRANSITION_ID_MISMATCH")
    require(bool(transition["step"]["execute_optimizer"]), "NATIVE_RESPONSE_500_UPDATE_NOT_EXECUTED")
    update_ref = transition["step"]["parameter_update"]
    expected_update = endpoint["native_update"]
    require(update_ref["raw_tensor_sha256"] == expected_update["raw_tensor_sha256"], "NATIVE_RESPONSE_500_UPDATE_RAW_HASH_MISMATCH")
    require(update_ref["file_sha256"] == expected_update["file_sha256"], "NATIVE_RESPONSE_500_UPDATE_FILE_HASH_MISMATCH")
    require(int(np.prod(update_ref["shape"])) == int(expected_update["element_count"]), "NATIVE_RESPONSE_500_UPDATE_COUNT_MISMATCH")
    update = _load_exact_update(entry_root, transition)
    require(set(update) == set(prestate.parameters), "NATIVE_RESPONSE_500_UPDATE_PARAMETER_SET_MISMATCH")
    return {
        "endpoint": endpoint,
        "window": window,
        "entry_root": entry_root,
        "state_record": state_record,
        "prestate": prestate,
        "transition": transition,
        "update_reference": update_ref,
        "update": update,
    }


def _analysis_states(receiver: dict[str, Any], epsilon: float) -> dict[str, StateSnapshot]:
    prestate: StateSnapshot = receiver["prestate"]
    update = receiver["update"]
    states = {
        "baseline": prestate.clone(),
        "native_minus_0.125": _transplant_state(prestate, parameter_delta=_scaled_delta(update, -epsilon)),
        "native_plus_0.125": _transplant_state(prestate, parameter_delta=_scaled_delta(update, epsilon)),
    }
    assert_snapshot_isolation(states.values())
    require(tuple(states) == BRANCHES, "NATIVE_RESPONSE_500_BRANCH_ORDER_INVALID")
    require(all(_optimizer_exact(prestate, state) for state in states.values()), "NATIVE_RESPONSE_500_OPTIMIZER_MUTATED")
    return states


def _derive_native_response(
    *,
    store: TensorStore,
    output_root: Path,
    sample_id: str,
    observations: Mapping[str, dict[str, Any]],
    epsilon: float,
) -> dict[str, Any]:
    baseline = _load_observation_arrays(output_root, observations["baseline"])
    minus = _load_observation_arrays(output_root, observations["native_minus_0.125"])
    plus = _load_observation_arrays(output_root, observations["native_plus_0.125"])
    keys = tuple(sorted(baseline))
    require(tuple(sorted(minus)) == keys and tuple(sorted(plus)) == keys, "NATIVE_RESPONSE_500_OUTPUT_SET_MISMATCH")
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for key in keys:
        base = baseline[key]
        neg = minus[key]
        pos = plus[key]
        require(base.shape == neg.shape == pos.shape, f"NATIVE_RESPONSE_500_SHAPE_MISMATCH:{sample_id}:{key}")
        prefix = f"native-response-500:{sample_id}:{key}"
        if np.issubdtype(base.dtype, np.floating):
            base64 = base.astype(np.float64, copy=False)
            neg64 = neg.astype(np.float64, copy=False)
            pos64 = pos.astype(np.float64, copy=False)
            numeric[key] = {
                "baseline": store.put(base64, representation=f"{prefix}:baseline_float64"),
                "minus": store.put(neg64, representation=f"{prefix}:minus_float64"),
                "plus": store.put(pos64, representation=f"{prefix}:plus_float64"),
                "j_native": store.put(
                    (pos64 - neg64) / (2.0 * epsilon),
                    representation=f"{prefix}:central_first_order_native_direction_response",
                ),
                "k_native": store.put(
                    (pos64 + neg64 - 2.0 * base64) / (epsilon * epsilon),
                    representation=f"{prefix}:central_second_order_native_direction_curvature",
                ),
            }
        else:
            categorical[key] = _categorical_transition(
                store,
                baseline=base,
                plus=pos,
                minus=neg,
                full=None,
                prefix=prefix,
            )
    return _checked_result(
        output_root / "responses" / f"{sample_id}.json",
        {
            "schema": "nanogpt-native-direction-response-v1",
            "status": "PASS",
            "sample_id": sample_id,
            "epsilon": epsilon,
            "numeric_responses": numeric,
            "categorical_transitions": categorical,
            "categorical_values_subtracted": False,
            "future_information_used": False,
        },
    )


def _existing_sample_receipt(output_root: Path, sample_id: str, protocol_sha: str) -> dict[str, Any] | None:
    path = output_root / "samples" / sample_id / "sample_receipt.json"
    if not path.exists():
        return None
    receipt = _read_checked(path, "nanogpt-native-direction-response-sample-receipt-v1")
    require(receipt["status"] == "PASS", f"NATIVE_RESPONSE_500_EXISTING_SAMPLE_NOT_PASS:{sample_id}")
    require(receipt["protocol_sha256"] == protocol_sha, f"NATIVE_RESPONSE_500_EXISTING_SAMPLE_PROTOCOL_DRIFT:{sample_id}")
    response = _read_checked(output_root / "responses" / f"{sample_id}.json", "nanogpt-native-direction-response-v1")
    require(response["result_sha256"] == receipt["response_result_sha256"], f"NATIVE_RESPONSE_500_EXISTING_RESPONSE_DRIFT:{sample_id}")
    for branch, row in receipt["branches"].items():
        state = _read_checked(output_root / "samples" / sample_id / "states" / f"{branch}.json", "nanogpt-native-direction-analysis-state-v1")
        require(state["result_sha256"] == row["state_result_sha256"], f"NATIVE_RESPONSE_500_EXISTING_STATE_DRIFT:{sample_id}:{branch}")
        probe = _read_checked(
            output_root / "probe-observations" / receipt["probe_contract_id"] / f"{row['state_id']}.json",
            "nanogpt-stepwise-probe-observation-v1",
        )
        require(probe["result_sha256"] == row["probe_result_sha256"], f"NATIVE_RESPONSE_500_EXISTING_PROBE_DRIFT:{sample_id}:{branch}")
    return receipt


def _execute_receiver(
    *,
    receiver: dict[str, Any],
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    output_root: Path,
    protocol: dict[str, Any],
    protocol_sha: str,
) -> dict[str, Any]:
    endpoint = receiver["endpoint"]
    sample_id = str(endpoint["sample_id"])
    existing = _existing_sample_receipt(output_root, sample_id, protocol_sha)
    if existing is not None:
        print({"event": "NATIVE_RESPONSE_500_SAMPLE_REUSED", "sample_id": sample_id}, flush=True)
        return existing

    observations: dict[str, dict[str, Any]] = {}
    state_rows: dict[str, Any] = {}
    states = _analysis_states(receiver, float(protocol["epsilon"]))
    for branch, state in states.items():
        manifest = _encode_state(
            store,
            state,
            entry_id=str(endpoint["entry_id"]),
            window_id=f"{protocol['protocol_id']}:{sample_id}:{branch}",
            optimizer_step=int(endpoint["optimizer_step"]),
            protocol_sha256=protocol_sha,
        )
        state_result = _checked_result(
            output_root / "samples" / sample_id / "states" / f"{branch}.json",
            {
                "schema": "nanogpt-native-direction-analysis-state-v1",
                "status": "PASS",
                "sample_id": sample_id,
                "branch": branch,
                "physical_optimizer_step": int(endpoint["optimizer_step"]),
                "analysis_state_not_native_training_state": branch != "baseline",
                "optimizer_identical_to_receiver_prestate": True,
                "state": manifest,
                "state_summary": runtime.state_summary(state),
            },
        )
        observation = _probe_at_state(
            runtime,
            store,
            branch_entry_root=output_root,
            state=state,
            state_manifest=manifest,
        )
        require(int(observation["actual_forward_count"]) == int(protocol["expected_probe_forward_count_per_state"]), "NATIVE_RESPONSE_500_PROBE_FORWARD_COUNT_INVALID")
        require(bool(observation["baseline_byte_exact"]), "NATIVE_RESPONSE_500_PROBE_BASELINE_NOT_EXACT")
        observations[branch] = observation
        state_rows[branch] = {
            "state_id": manifest["state_id"],
            "state_sha256": manifest["commitment"]["state_sha256"],
            "state_result_sha256": state_result["result_sha256"],
            "probe_observation_id": observation["probe_observation_id"],
            "probe_result_sha256": observation["result_sha256"],
        }

    response = _derive_native_response(
        store=store,
        output_root=output_root,
        sample_id=sample_id,
        observations=observations,
        epsilon=float(protocol["epsilon"]),
    )
    receipt = _checked_result(
        output_root / "samples" / sample_id / "sample_receipt.json",
        {
            "schema": "nanogpt-native-direction-response-sample-receipt-v1",
            "status": "PASS",
            "protocol_sha256": protocol_sha,
            "sample_id": sample_id,
            "entry_id": endpoint["entry_id"],
            "run_id": endpoint["run_id"],
            "window_id": endpoint["window_id"],
            "optimizer_step": endpoint["optimizer_step"],
            "source_prestate_id": endpoint["prestate"]["state_id"],
            "source_transition_id": endpoint["native_update"]["transition_id"],
            "source_native_update_raw_tensor_sha256": endpoint["native_update"]["raw_tensor_sha256"],
            "probe_contract_id": runtime.probe_contract.probe_contract_id,
            "branches": state_rows,
            "response_result_sha256": response["result_sha256"],
            "state_count": 3,
            "probe_forward_count": 36,
            "backward_pass_count": 0,
            "optimizer_step_count": 0,
            "training_continuation_count": 0,
            "native_target_content_opened": False,
            "future_information_used": False,
        },
    )
    print({"event": "NATIVE_RESPONSE_500_SAMPLE_COMPLETE", "sample_id": sample_id}, flush=True)
    return receipt


def execute_native_response_500(
    *,
    formal_root: Path,
    source_root: Path,
    trainer_root: Path,
    output_root: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    main_protocol_path: Path,
    response_protocol_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    protocol = read_json(response_protocol_path)
    require(protocol["schema"] == NATIVE_RESPONSE_500_SCHEMA, "NATIVE_RESPONSE_500_PROTOCOL_SCHEMA_INVALID")
    require(protocol["status"] == "FROZEN_BEFORE_RESPONSE_EXECUTION", "NATIVE_RESPONSE_500_PROTOCOL_NOT_FROZEN")
    require(float(protocol["epsilon"]) == 0.125, "NATIVE_RESPONSE_500_EPSILON_INVALID")
    endpoints = list(protocol["receivers"])
    require(len(endpoints) == 500, "NATIVE_RESPONSE_500_ENDPOINT_COUNT_INVALID")
    require(len({row["sample_id"] for row in endpoints}) == 500, "NATIVE_RESPONSE_500_ENDPOINT_ID_NOT_UNIQUE")
    selected = endpoints if limit is None else endpoints[:limit]
    require(bool(selected), "NATIVE_RESPONSE_500_EMPTY_EXECUTION")
    if limit is None:
        require(len(selected) == 500, "NATIVE_RESPONSE_500_FULL_EXECUTION_COUNT_INVALID")

    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    protocol_sha = file_sha256(response_protocol_path)
    output_root.mkdir(parents=True, exist_ok=True)
    store = TensorStore(output_root / "tensor-objects")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for endpoint in selected:
        grouped[str(endpoint["source_bundle_id"])].append(endpoint)

    receipts: list[dict[str, Any]] = []
    for source_bundle_id in sorted(grouped):
        runtime = StepwiseTrainingRuntime(
            source_root / source_bundle_id,
            trainer_root,
            registry,
            probe_contract,
        )
        try:
            for endpoint in sorted(grouped[source_bundle_id], key=lambda row: row["sample_id"]):
                receiver = _load_receiver(formal_root, endpoint)
                receipt = _execute_receiver(
                    receiver=receiver,
                    runtime=runtime,
                    store=store,
                    output_root=output_root,
                    protocol=protocol,
                    protocol_sha=protocol_sha,
                )
                receipts.append(
                    {
                        "sample_id": receipt["sample_id"],
                        "result_sha256": receipt["result_sha256"],
                    }
                )
                del receiver
        finally:
            runtime.close()

    receipts.sort(key=lambda row: row["sample_id"])
    completed = len(receipts)
    run_receipt = _checked_result(
        output_root / ("native_response_500_run_receipt.json" if limit is None else f"native_response_smoke_{limit}_run_receipt.json"),
        {
            "schema": "nanogpt-native-direction-response-run-receipt-v1",
            "status": "PASS" if limit is None else "SMOKE_PASS",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "main_protocol_sha256": file_sha256(main_protocol_path),
            "component_registry_sha256": registry.source_sha256,
            "probe_contract_sha256": probe_contract.source_sha256,
            "sample_receipts": receipts,
            "receiver_count": completed,
            "state_count": completed * 3,
            "probe_forward_count": completed * 36,
            "backward_pass_count": 0,
            "optimizer_step_count": 0,
            "training_continuation_count": 0,
            "native_target_content_opened": False,
            "full_500_execution": limit is None,
        },
    )
    if limit is not None:
        return run_receipt

    files = sorted(path for path in output_root.rglob("*.json") if path.name != "PRE_TARGET_RESPONSE_500_SEAL.json")
    return _checked_result(
        output_root / "PRE_TARGET_RESPONSE_500_SEAL.json",
        {
            "schema": "nanogpt-native-direction-response-500-pretarget-seal-v1",
            "status": "SEALED_BEFORE_NATIVE_TARGET_ACCESS",
            "protocol_sha256": protocol_sha,
            "run_receipt_result_sha256": run_receipt["result_sha256"],
            "sealed_json_files": [
                {"relative_path": path.relative_to(output_root).as_posix(), "file_sha256": file_sha256(path)}
                for path in files
            ],
            "sealed_json_file_count": len(files),
            "native_target_content_opened": False,
            "future_information_used": False,
        },
    )


__all__ = ["execute_native_response_500"]
