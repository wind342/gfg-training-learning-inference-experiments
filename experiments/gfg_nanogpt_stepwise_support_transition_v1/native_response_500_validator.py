from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    read_json,
    require,
)

from .branches import _load_observation_arrays
from .execution import _checked_result, _read_checked
from .native_response_500 import (
    BRANCHES,
    NATIVE_RESPONSE_500_SCHEMA,
    _analysis_states,
    _load_receiver,
)
from .p2_response import _optimizer_exact
from .storage import restorable_state_from_manifest


def _load_ref(root: Path, reference: dict[str, Any]) -> np.ndarray:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "NATIVE_RESPONSE_500_VALIDATION_LOCATOR_INVALID")
    path = root / locator
    require(file_sha256(path) == reference["file_sha256"], "NATIVE_RESPONSE_500_VALIDATION_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "NATIVE_RESPONSE_500_VALIDATION_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "NATIVE_RESPONSE_500_VALIDATION_DTYPE_MISMATCH")
    require(
        hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
        == reference["raw_tensor_sha256"],
        "NATIVE_RESPONSE_500_VALIDATION_RAW_HASH_MISMATCH",
    )
    return np.asarray(value)


def _state_exact(left, right) -> bool:
    return (
        set(left.parameters) == set(right.parameters)
        and all(torch.equal(left.parameters[name], right.parameters[name]) for name in left.parameters)
        and _optimizer_exact(left, right)
    )


def validate_native_response_500(
    *,
    formal_root: Path,
    output_root: Path,
    response_protocol_path: Path,
    validation_path: Path,
    expected_count: int = 500,
) -> dict[str, Any]:
    protocol = read_json(response_protocol_path)
    require(protocol["schema"] == NATIVE_RESPONSE_500_SCHEMA, "NATIVE_RESPONSE_500_VALIDATION_PROTOCOL_SCHEMA_INVALID")
    require(expected_count > 0 and expected_count <= 500, "NATIVE_RESPONSE_500_VALIDATION_COUNT_INVALID")
    endpoints = list(protocol["receivers"])[:expected_count]
    protocol_sha = file_sha256(response_protocol_path)
    receipt_name = "native_response_500_run_receipt.json" if expected_count == 500 else f"native_response_smoke_{expected_count}_run_receipt.json"
    run_receipt = _read_checked(output_root / receipt_name, "nanogpt-native-direction-response-run-receipt-v1")
    require(run_receipt["protocol_sha256"] == protocol_sha, "NATIVE_RESPONSE_500_VALIDATION_PROTOCOL_HASH_MISMATCH")
    require(int(run_receipt["receiver_count"]) == expected_count, "NATIVE_RESPONSE_500_VALIDATION_RECEIVER_COUNT_MISMATCH")
    if expected_count == 500:
        seal = _read_checked(output_root / "PRE_TARGET_RESPONSE_500_SEAL.json", "nanogpt-native-direction-response-500-pretarget-seal-v1")
        require(seal["status"] == "SEALED_BEFORE_NATIVE_TARGET_ACCESS", "NATIVE_RESPONSE_500_VALIDATION_SEAL_INVALID")
        require(seal["protocol_sha256"] == protocol_sha, "NATIVE_RESPONSE_500_VALIDATION_SEAL_PROTOCOL_MISMATCH")
        for row in seal["sealed_json_files"]:
            path = output_root / str(row["relative_path"])
            require(file_sha256(path) == row["file_sha256"], f"NATIVE_RESPONSE_500_VALIDATION_SEALED_FILE_DRIFT:{row['relative_path']}")
        seal_result_sha = seal["result_sha256"]
    else:
        seal_result_sha = None

    epsilon = float(protocol["epsilon"])
    checked_states = 0
    checked_probes = 0
    checked_numeric = 0
    checked_categorical = 0
    rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        sample_id = str(endpoint["sample_id"])
        receiver = _load_receiver(formal_root, endpoint)
        expected_states = _analysis_states(receiver, epsilon)
        receipt = _read_checked(
            output_root / "samples" / sample_id / "sample_receipt.json",
            "nanogpt-native-direction-response-sample-receipt-v1",
        )
        require(receipt["protocol_sha256"] == protocol_sha, f"NATIVE_RESPONSE_500_VALIDATION_SAMPLE_PROTOCOL_MISMATCH:{sample_id}")
        observations: dict[str, dict[str, Any]] = {}
        for branch in BRANCHES:
            state_record = _read_checked(
                output_root / "samples" / sample_id / "states" / f"{branch}.json",
                "nanogpt-native-direction-analysis-state-v1",
            )
            actual_state = restorable_state_from_manifest(output_root, state_record["state"])
            require(_state_exact(actual_state, expected_states[branch]), f"NATIVE_RESPONSE_500_VALIDATION_STATE_NOT_EXACT:{sample_id}:{branch}")
            require(_optimizer_exact(actual_state, receiver["prestate"]), f"NATIVE_RESPONSE_500_VALIDATION_OPTIMIZER_CHANGED:{sample_id}:{branch}")
            observation = _read_checked(
                output_root / "probe-observations" / receipt["probe_contract_id"] / f"{state_record['state']['state_id']}.json",
                "nanogpt-stepwise-probe-observation-v1",
            )
            require(observation["observed_state_id"] == state_record["state"]["state_id"], f"NATIVE_RESPONSE_500_VALIDATION_PROBE_STATE_MISMATCH:{sample_id}:{branch}")
            require(int(observation["actual_forward_count"]) == 12, f"NATIVE_RESPONSE_500_VALIDATION_FORWARD_COUNT_INVALID:{sample_id}:{branch}")
            require(bool(observation["baseline_byte_exact"]), f"NATIVE_RESPONSE_500_VALIDATION_BASELINE_FLAG_FALSE:{sample_id}:{branch}")
            observations[branch] = observation
            checked_states += 1
            checked_probes += 1

        baseline = _load_observation_arrays(output_root, observations["baseline"])
        minus = _load_observation_arrays(output_root, observations["native_minus_0.125"])
        plus = _load_observation_arrays(output_root, observations["native_plus_0.125"])
        response = _read_checked(output_root / "responses" / f"{sample_id}.json", "nanogpt-native-direction-response-v1")
        require(response["sample_id"] == sample_id, f"NATIVE_RESPONSE_500_VALIDATION_RESPONSE_ID_MISMATCH:{sample_id}")
        for key in sorted(baseline):
            base = baseline[key]
            neg = minus[key]
            pos = plus[key]
            if np.issubdtype(base.dtype, np.floating):
                base64 = base.astype(np.float64, copy=False)
                neg64 = neg.astype(np.float64, copy=False)
                pos64 = pos.astype(np.float64, copy=False)
                expected = {
                    "baseline": base64,
                    "minus": neg64,
                    "plus": pos64,
                    "j_native": (pos64 - neg64) / (2.0 * epsilon),
                    "k_native": (pos64 + neg64 - 2.0 * base64) / (epsilon * epsilon),
                }
                for name, value in expected.items():
                    actual = _load_ref(output_root, response["numeric_responses"][key][name])
                    require(np.array_equal(actual, value, equal_nan=True), f"NATIVE_RESPONSE_500_VALIDATION_RESPONSE_MISMATCH:{sample_id}:{key}:{name}")
                    checked_numeric += 1
            else:
                expected_masks = {
                    "plus_changed_mask": np.not_equal(pos, base),
                    "minus_changed_mask": np.not_equal(neg, base),
                }
                for name, value in expected_masks.items():
                    actual = _load_ref(output_root, response["categorical_transitions"][key][name])
                    require(np.array_equal(actual, value), f"NATIVE_RESPONSE_500_VALIDATION_CATEGORICAL_MISMATCH:{sample_id}:{key}:{name}")
                    checked_categorical += 1
        rows.append({"sample_id": sample_id, "sample_receipt_result_sha256": receipt["result_sha256"], "response_result_sha256": response["result_sha256"]})
        if len(rows) % 25 == 0:
            print({"event": "NATIVE_RESPONSE_500_VALIDATION_PROGRESS", "checked": len(rows)}, flush=True)

    require(int(run_receipt["state_count"]) == checked_states == expected_count * 3, "NATIVE_RESPONSE_500_VALIDATION_STATE_TOTAL_INVALID")
    require(int(run_receipt["probe_forward_count"]) == checked_probes * 12 == expected_count * 36, "NATIVE_RESPONSE_500_VALIDATION_FORWARD_TOTAL_INVALID")
    require(run_receipt["backward_pass_count"] == run_receipt["optimizer_step_count"] == run_receipt["training_continuation_count"] == 0, "NATIVE_RESPONSE_500_VALIDATION_FORBIDDEN_EXECUTION")
    require(run_receipt["native_target_content_opened"] is False, "NATIVE_RESPONSE_500_VALIDATION_TARGET_OPENED")
    return _checked_result(
        validation_path,
        {
            "schema": "nanogpt-native-direction-response-500-pretarget-validation-v1",
            "status": "PASS" if expected_count == 500 else "SMOKE_PASS",
            "protocol_sha256": protocol_sha,
            "run_receipt_result_sha256": run_receipt["result_sha256"],
            "seal_result_sha256": seal_result_sha,
            "checked_receiver_count": len(rows),
            "checked_state_count": checked_states,
            "checked_probe_count": checked_probes,
            "checked_forward_count": checked_probes * 12,
            "checked_numeric_response_array_count": checked_numeric,
            "checked_categorical_response_array_count": checked_categorical,
            "response_rows": rows,
            "state_and_optimizer_reconstruction_exact": True,
            "response_recomputation_exact": True,
            "sealed_file_hashes_exact": expected_count == 500,
            "native_target_content_opened": False,
            "future_information_used": False,
        },
    )


__all__ = ["validate_native_response_500"]
