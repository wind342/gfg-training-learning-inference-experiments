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

from .branches import _load_observation_arrays, _probe_at_state, _source_state_record
from .contracts import ComponentRegistry, ProbeContract
from .execution import _checked_result, _encode_state, _read_checked
from .local_response import _categorical_transition, _scaled_delta
from .reciprocal import _transplant_state, _window
from .runtime import StepwiseTrainingRuntime
from .storage import TensorStore, restorable_state_from_manifest


P2_SCHEMA = "nanogpt-p2-reciprocal-local-response-protocol-v1"
P2_LABELS = ("P2a", "P2b")


def _optimizer_exact(left: StateSnapshot, right: StateSnapshot) -> bool:
    return set(left.optimizer) == set(right.optimizer) and all(
        set(left.optimizer[name]) == set(right.optimizer[name])
        and all(
            torch.equal(left.optimizer[name][key], right.optimizer[name][key])
            for key in left.optimizer[name]
        )
        for name in left.optimizer
    )


def _load_exact_update(entry_root: Path, transition: dict[str, Any]) -> dict[str, torch.Tensor]:
    reference = transition["step"]["parameter_update"]
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "P2_UPDATE_LOCATOR_INVALID")
    path = entry_root / locator
    require(file_sha256(path) == reference["file_sha256"], "P2_UPDATE_FILE_HASH_MISMATCH")
    packed = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(packed.shape) == list(reference["shape"]), "P2_UPDATE_PACKED_SHAPE_MISMATCH")
    require(str(packed.dtype) == str(reference["dtype"]), "P2_UPDATE_PACKED_DTYPE_MISMATCH")
    require(
        hashlib.sha256(np.ascontiguousarray(packed).tobytes(order="C")).hexdigest()
        == reference["raw_tensor_sha256"],
        "P2_UPDATE_PACKED_RAW_HASH_MISMATCH",
    )
    result: dict[str, torch.Tensor] = {}
    for row in reference["layout"]:
        offset = int(row["offset"])
        count = int(row["element_count"])
        value = np.asarray(packed[offset : offset + count]).astype(np.dtype(row["dtype"]), copy=True)
        value = value.reshape(tuple(int(child) for child in row["shape"]))
        tensor = torch.from_numpy(value)
        require(tensor_sha256(tensor) == row["raw_tensor_sha256"], f"P2_UPDATE_CHILD_HASH_MISMATCH:{row['name']}")
        result[str(row["name"])] = tensor
    require(tuple(sorted(result)) == tuple(reference["canonical_name_order"]), "P2_UPDATE_NAME_ORDER_MISMATCH")
    return result


def _load_inputs(
    *,
    formal_root: Path,
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for endpoint in protocol["receivers"]:
        label = str(endpoint["label"])
        require(label in P2_LABELS, f"P2_RECEIVER_LABEL_INVALID:{label}")
        window = _window(formal_root, endpoint)
        step = int(endpoint["optimizer_step"])
        state_path = (
            formal_root
            / str(endpoint["entry_id"])
            / "windows"
            / str(endpoint["window_id"])
            / "states"
            / f"step-{step:05d}.json"
        )
        require(file_sha256(state_path) == endpoint["prestate"]["state_record_file_sha256"], f"P2_PRESTATE_FILE_HASH_MISMATCH:{label}")
        state_record = _source_state_record(formal_root, window, step)
        require(state_record["result_sha256"] == endpoint["prestate"]["state_record_result_sha256"], f"P2_PRESTATE_RESULT_HASH_MISMATCH:{label}")
        require(state_record["state"]["state_id"] == endpoint["prestate"]["state_id"], f"P2_PRESTATE_ID_MISMATCH:{label}")
        require(state_record["state"]["commitment"]["state_sha256"] == endpoint["prestate"]["state_sha256"], f"P2_PRESTATE_COMMITMENT_MISMATCH:{label}")
        entry_root = formal_root / str(endpoint["entry_id"])
        prestate = restorable_state_from_manifest(entry_root, state_record["state"])
        transition_path = (
            entry_root
            / "windows"
            / str(endpoint["window_id"])
            / "transitions"
            / f"step-{step:05d}-to-{step + 1:05d}.json"
        )
        transition = _read_checked(transition_path, "nanogpt-stepwise-transition-v1")
        require(bool(transition["step"]["execute_optimizer"]), f"P2_NATIVE_UPDATE_NOT_EXECUTED:{label}")
        reference = transition["step"]["parameter_update"]
        expected = endpoint["native_update"]
        require(reference["raw_tensor_sha256"] == expected["raw_tensor_sha256"], f"P2_UPDATE_RAW_ID_MISMATCH:{label}")
        require(reference["file_sha256"] == expected["file_sha256"], f"P2_UPDATE_FILE_ID_MISMATCH:{label}")
        require(int(np.prod(reference["shape"])) == int(expected["element_count"]), f"P2_UPDATE_ELEMENT_COUNT_MISMATCH:{label}")
        update = _load_exact_update(entry_root, transition)
        require(set(update) == set(prestate.parameters), f"P2_UPDATE_PARAMETER_SET_MISMATCH:{label}")
        loaded[label] = {
            "endpoint": endpoint,
            "window": window,
            "state_record": state_record,
            "prestate": prestate,
            "transition": transition,
            "transition_path": transition_path,
            "update_reference": reference,
            "update": update,
        }
    require(tuple(sorted(loaded)) == tuple(sorted(P2_LABELS)), "P2_RECEIVER_SET_INVALID")
    return loaded


def _branch_states(
    receiver: dict[str, Any],
    inputs: Mapping[str, dict[str, Any]],
    epsilon: float,
) -> dict[str, StateSnapshot]:
    prestate: StateSnapshot = receiver["prestate"]
    states: dict[str, StateSnapshot] = {"baseline": prestate.clone()}
    for donor_label in P2_LABELS:
        delta = inputs[donor_label]["update"]
        states[f"update_{donor_label}_minus_0.125"] = _transplant_state(
            prestate,
            parameter_delta=_scaled_delta(delta, -epsilon),
        )
        states[f"update_{donor_label}_plus_0.125"] = _transplant_state(
            prestate,
            parameter_delta=_scaled_delta(delta, epsilon),
        )
    assert_snapshot_isolation(states.values())
    require(all(_optimizer_exact(prestate, state) for state in states.values()), "P2_SIGNED_BRANCH_OPTIMIZER_MUTATED")
    return states


def _derive_response(
    *,
    store: TensorStore,
    output_root: Path,
    receiver_label: str,
    donor_label: str,
    observations: Mapping[str, dict[str, Any]],
    epsilon: float,
) -> dict[str, Any]:
    baseline = _load_observation_arrays(output_root, observations["baseline"])
    minus = _load_observation_arrays(output_root, observations[f"update_{donor_label}_minus_0.125"])
    plus = _load_observation_arrays(output_root, observations[f"update_{donor_label}_plus_0.125"])
    keys = tuple(sorted(baseline))
    require(tuple(sorted(minus)) == keys and tuple(sorted(plus)) == keys, "P2_RESPONSE_OUTPUT_SET_MISMATCH")
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for key in keys:
        base = baseline[key]
        neg = minus[key]
        pos = plus[key]
        require(base.shape == neg.shape == pos.shape, f"P2_RESPONSE_SHAPE_MISMATCH:{key}")
        prefix = f"p2:{receiver_label}:{donor_label}:{key}"
        if np.issubdtype(base.dtype, np.floating):
            base64 = base.astype(np.float64, copy=False)
            neg64 = neg.astype(np.float64, copy=False)
            pos64 = pos.astype(np.float64, copy=False)
            numeric[key] = {
                "baseline": store.put(base64, representation=f"{prefix}:baseline_float64"),
                "minus": store.put(neg64, representation=f"{prefix}:minus_float64"),
                "plus": store.put(pos64, representation=f"{prefix}:plus_float64"),
                "j_first_order": store.put(
                    (pos64 - neg64) / (2.0 * epsilon),
                    representation=f"{prefix}:central_first_order_response",
                ),
                "k_curvature": store.put(
                    (pos64 + neg64 - 2.0 * base64) / (epsilon * epsilon),
                    representation=f"{prefix}:central_second_order_curvature",
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
        output_root / "responses" / f"receiver-{receiver_label}-donor-{donor_label}.json",
        {
            "schema": "nanogpt-p2-local-response-jk-v1",
            "status": "PASS",
            "receiver_label": receiver_label,
            "donor_label": donor_label,
            "epsilon": epsilon,
            "numeric_responses": numeric,
            "categorical_transitions": categorical,
            "categorical_values_subtracted": False,
            "future_information_used": False,
        },
    )


def execute_p2_response(
    *,
    formal_root: Path,
    source_root: Path,
    trainer_root: Path,
    output_root: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    main_protocol_path: Path,
    p2_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(p2_protocol_path)
    require(protocol["schema"] == P2_SCHEMA, "P2_PROTOCOL_SCHEMA_INVALID")
    require(protocol["status"] == "FROZEN_BEFORE_RESPONSE_EXECUTION_AND_BEFORE_NATIVE_TARGET_ACCESS", "P2_PROTOCOL_NOT_FROZEN")
    require(tuple(str(row["label"]) for row in protocol["receivers"]) == P2_LABELS, "P2_PROTOCOL_RECEIVER_ORDER_INVALID")
    epsilon = float(protocol["epsilon"])
    require(epsilon == 0.125, "P2_EPSILON_INVALID")
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    protocol_sha = file_sha256(p2_protocol_path)
    main_protocol_sha = file_sha256(main_protocol_path)
    inputs = _load_inputs(formal_root=formal_root, protocol=protocol)
    store = TensorStore(output_root / "tensor-objects")

    input_receipt = _checked_result(
        output_root / "verified_inputs.json",
        {
            "schema": "nanogpt-p2-verified-inputs-v1",
            "status": "PASS",
            "protocol_sha256": protocol_sha,
            "inputs": {
                label: {
                    "entry_id": inputs[label]["endpoint"]["entry_id"],
                    "optimizer_step": inputs[label]["endpoint"]["optimizer_step"],
                    "prestate_id": inputs[label]["state_record"]["state"]["state_id"],
                    "prestate_sha256": inputs[label]["prestate"].commitment()["state_sha256"],
                    "transition_id": inputs[label]["transition"]["transition_id"],
                    "transition_result_sha256": inputs[label]["transition"]["result_sha256"],
                    "native_update_raw_tensor_sha256": inputs[label]["update_reference"]["raw_tensor_sha256"],
                    "native_update_file_sha256": inputs[label]["update_reference"]["file_sha256"],
                }
                for label in P2_LABELS
            },
            "native_target_content_opened": False,
        },
    )

    receiver_receipts: list[dict[str, Any]] = []
    response_receipts: list[dict[str, Any]] = []
    for receiver_label in P2_LABELS:
        receiver = inputs[receiver_label]
        states = _branch_states(receiver, inputs, epsilon)
        expected_branches = tuple(protocol["branches_per_receiver"])
        require(tuple(states) == expected_branches, f"P2_BRANCH_ORDER_INVALID:{receiver_label}")
        runtime = StepwiseTrainingRuntime(
            source_root / str(receiver["endpoint"]["source_bundle_id"]),
            trainer_root,
            registry,
            probe_contract,
        )
        observations: dict[str, dict[str, Any]] = {}
        state_rows: dict[str, Any] = {}
        try:
            for branch, state in states.items():
                manifest = _encode_state(
                    store,
                    state,
                    entry_id=str(receiver["endpoint"]["entry_id"]),
                    window_id=f"{protocol['protocol_id']}:{receiver_label}:{branch}",
                    optimizer_step=int(receiver["endpoint"]["optimizer_step"]),
                    protocol_sha256=protocol_sha,
                )
                state_result = _checked_result(
                    output_root / "receivers" / receiver_label / "states" / f"{branch}.json",
                    {
                        "schema": "nanogpt-p2-analysis-state-v1",
                        "status": "PASS",
                        "receiver_label": receiver_label,
                        "branch": branch,
                        "physical_optimizer_step": int(receiver["endpoint"]["optimizer_step"]),
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
                require(int(observation["actual_forward_count"]) == int(protocol["expected_probe_forward_count_per_state"]), "P2_PROBE_FORWARD_COUNT_INVALID")
                require(bool(observation["baseline_byte_exact"]), "P2_PROBE_BASELINE_NOT_BYTE_EXACT")
                observations[branch] = observation
                state_rows[branch] = {
                    "state_id": manifest["state_id"],
                    "state_sha256": manifest["commitment"]["state_sha256"],
                    "state_result_sha256": state_result["result_sha256"],
                    "probe_observation_id": observation["probe_observation_id"],
                    "probe_result_sha256": observation["result_sha256"],
                }
                print({"event": "P2_PROBE_COMPLETE", "receiver": receiver_label, "branch": branch}, flush=True)
        finally:
            runtime.close()

        receiver_receipt = _checked_result(
            output_root / "receivers" / receiver_label / "receiver_receipt.json",
            {
                "schema": "nanogpt-p2-receiver-receipt-v1",
                "status": "PASS",
                "receiver_label": receiver_label,
                "prestate_id": receiver["state_record"]["state"]["state_id"],
                "branches": state_rows,
                "branch_count": len(state_rows),
                "probe_forward_count": sum(int(row["actual_forward_count"]) for row in observations.values()),
                "backward_pass_count": 0,
                "optimizer_step_count": 0,
                "future_information_used": False,
            },
        )
        receiver_receipts.append({"receiver_label": receiver_label, "result_sha256": receiver_receipt["result_sha256"]})
        for donor_label in P2_LABELS:
            response = _derive_response(
                store=store,
                output_root=output_root,
                receiver_label=receiver_label,
                donor_label=donor_label,
                observations=observations,
                epsilon=epsilon,
            )
            response_receipts.append(
                {
                    "receiver_label": receiver_label,
                    "donor_label": donor_label,
                    "result_sha256": response["result_sha256"],
                }
            )

    pair = _checked_result(
        output_root / "p2_response_pair_receipt.json",
        {
            "schema": "nanogpt-p2-response-pair-receipt-v1",
            "status": "PASS",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "main_protocol_sha256": main_protocol_sha,
            "component_registry_sha256": registry.source_sha256,
            "probe_contract_sha256": probe_contract.source_sha256,
            "verified_inputs_result_sha256": input_receipt["result_sha256"],
            "receiver_receipts": receiver_receipts,
            "response_receipts": response_receipts,
            "receiver_count": 2,
            "state_count": 10,
            "probe_forward_count": 120,
            "backward_pass_count": 0,
            "optimizer_step_count": 0,
            "training_continuation_count": 0,
            "native_target_content_opened": False,
        },
    )
    files = sorted(
        path for path in output_root.rglob("*.json")
        if path.name != "PRE_TARGET_RESPONSE_SEAL.json"
    )
    seal = _checked_result(
        output_root / "PRE_TARGET_RESPONSE_SEAL.json",
        {
            "schema": "nanogpt-p2-pre-target-response-seal-v1",
            "status": "SEALED_BEFORE_NATIVE_TARGET_ACCESS",
            "protocol_sha256": protocol_sha,
            "pair_receipt_result_sha256": pair["result_sha256"],
            "sealed_json_files": [
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "file_sha256": file_sha256(path),
                }
                for path in files
            ],
            "sealed_json_file_count": len(files),
            "native_target_content_opened": False,
            "future_information_used": False,
        },
    )
    return seal


__all__ = ["execute_p2_response"]
