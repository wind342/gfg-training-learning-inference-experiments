from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
)

from .branches import _load_observation_arrays
from .execution import _checked_result, _read_checked
from .p2_response import P2_LABELS, P2_SCHEMA, _branch_states, _load_inputs, _optimizer_exact
from .storage import restorable_state_from_manifest


def _load_ref(root: Path, reference: dict[str, Any]) -> np.ndarray:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "P2_VALIDATION_TENSOR_LOCATOR_INVALID")
    path = root / locator
    require(file_sha256(path) == reference["file_sha256"], "P2_VALIDATION_TENSOR_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "P2_VALIDATION_TENSOR_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "P2_VALIDATION_TENSOR_DTYPE_MISMATCH")
    require(
        hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
        == reference["raw_tensor_sha256"],
        "P2_VALIDATION_TENSOR_RAW_HASH_MISMATCH",
    )
    return np.asarray(value)


def _state_exact(left, right) -> bool:
    return (
        set(left.parameters) == set(right.parameters)
        and all(torch.equal(left.parameters[name], right.parameters[name]) for name in left.parameters)
        and _optimizer_exact(left, right)
    )


def _validate_baseline_repeat(output_root: Path, observation: dict[str, Any]) -> None:
    require(len(observation["forwards"]) == 12, "P2_VALIDATION_FORWARD_COUNT_INVALID")
    require(observation["forwards"][0]["gate_components"] == [], "P2_VALIDATION_BASELINE0_GATE_INVALID")
    require(observation["forwards"][1]["gate_components"] == [], "P2_VALIDATION_BASELINE1_GATE_INVALID")
    for key in ("logits", "margins", "predictions", "group_membership", "group_q10_margin"):
        left = observation["forwards"][0][key]
        right = observation["forwards"][1][key]
        require(left["raw_tensor_sha256"] == right["raw_tensor_sha256"], f"P2_VALIDATION_BASELINE_REPEAT_MISMATCH:{key}")
        _load_ref(output_root, left)
        _load_ref(output_root, right)


def validate_p2_response(
    *,
    formal_root: Path,
    output_root: Path,
    p2_protocol_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    protocol = read_json(p2_protocol_path)
    require(protocol["schema"] == P2_SCHEMA, "P2_VALIDATION_PROTOCOL_SCHEMA_INVALID")
    protocol_sha = file_sha256(p2_protocol_path)
    inputs = _load_inputs(formal_root=formal_root, protocol=protocol)
    seal = _read_checked(output_root / "PRE_TARGET_RESPONSE_SEAL.json", "nanogpt-p2-pre-target-response-seal-v1")
    require(seal["status"] == "SEALED_BEFORE_NATIVE_TARGET_ACCESS", "P2_VALIDATION_SEAL_STATUS_INVALID")
    require(seal["protocol_sha256"] == protocol_sha, "P2_VALIDATION_SEAL_PROTOCOL_MISMATCH")
    for row in seal["sealed_json_files"]:
        path = output_root / str(row["relative_path"])
        require(file_sha256(path) == row["file_sha256"], f"P2_VALIDATION_SEALED_FILE_DRIFT:{row['relative_path']}")

    checked_states = 0
    checked_probes = 0
    checked_numeric_arrays = 0
    checked_categorical_arrays = 0
    response_rows: list[dict[str, Any]] = []
    epsilon = float(protocol["epsilon"])
    for receiver_label in P2_LABELS:
        expected_states = _branch_states(inputs[receiver_label], inputs, epsilon)
        observations: dict[str, dict[str, Any]] = {}
        for branch, expected in expected_states.items():
            state_record = _read_checked(
                output_root / "receivers" / receiver_label / "states" / f"{branch}.json",
                "nanogpt-p2-analysis-state-v1",
            )
            actual = restorable_state_from_manifest(output_root, state_record["state"])
            require(_state_exact(actual, expected), f"P2_VALIDATION_STATE_NOT_EXACT:{receiver_label}:{branch}")
            require(_optimizer_exact(actual, inputs[receiver_label]["prestate"]), f"P2_VALIDATION_OPTIMIZER_CHANGED:{receiver_label}:{branch}")
            observation_path = (
                output_root
                / "probe-observations"
                / "CSRG-4C-v1"
                / f"{state_record['state']['state_id']}.json"
            )
            observation = _read_checked(observation_path, "nanogpt-stepwise-probe-observation-v1")
            require(observation["observed_state_id"] == state_record["state"]["state_id"], "P2_VALIDATION_PROBE_STATE_MISMATCH")
            require(bool(observation["baseline_byte_exact"]), "P2_VALIDATION_PROBE_BASELINE_FLAG_FALSE")
            _validate_baseline_repeat(output_root, observation)
            observations[branch] = observation
            checked_states += 1
            checked_probes += 1

        baseline = _load_observation_arrays(output_root, observations["baseline"])
        for donor_label in P2_LABELS:
            minus = _load_observation_arrays(output_root, observations[f"update_{donor_label}_minus_0.125"])
            plus = _load_observation_arrays(output_root, observations[f"update_{donor_label}_plus_0.125"])
            response = _read_checked(
                output_root / "responses" / f"receiver-{receiver_label}-donor-{donor_label}.json",
                "nanogpt-p2-local-response-jk-v1",
            )
            require(response["receiver_label"] == receiver_label and response["donor_label"] == donor_label, "P2_VALIDATION_RESPONSE_IDENTITY_MISMATCH")
            for key in sorted(baseline):
                base = baseline[key]
                neg = minus[key]
                pos = plus[key]
                if np.issubdtype(base.dtype, np.floating):
                    row = response["numeric_responses"][key]
                    base64 = base.astype(np.float64, copy=False)
                    neg64 = neg.astype(np.float64, copy=False)
                    pos64 = pos.astype(np.float64, copy=False)
                    expected = {
                        "baseline": base64,
                        "minus": neg64,
                        "plus": pos64,
                        "j_first_order": (pos64 - neg64) / (2.0 * epsilon),
                        "k_curvature": (pos64 + neg64 - 2.0 * base64) / (epsilon * epsilon),
                    }
                    for name, value in expected.items():
                        actual = _load_ref(output_root, row[name])
                        require(np.array_equal(actual, value, equal_nan=True), f"P2_VALIDATION_RESPONSE_RECOMPUTE_MISMATCH:{receiver_label}:{donor_label}:{key}:{name}")
                        checked_numeric_arrays += 1
                else:
                    row = response["categorical_transitions"][key]
                    expected_masks = {
                        "plus_changed_mask": np.not_equal(pos, base),
                        "minus_changed_mask": np.not_equal(neg, base),
                    }
                    for name, value in expected_masks.items():
                        actual = _load_ref(output_root, row[name])
                        require(np.array_equal(actual, value), f"P2_VALIDATION_CATEGORICAL_RECOMPUTE_MISMATCH:{receiver_label}:{donor_label}:{key}:{name}")
                        checked_categorical_arrays += 1
            response_rows.append(
                {
                    "receiver_label": receiver_label,
                    "donor_label": donor_label,
                    "response_result_sha256": response["result_sha256"],
                }
            )

    pair = _read_checked(output_root / "p2_response_pair_receipt.json", "nanogpt-p2-response-pair-receipt-v1")
    require(pair["state_count"] == checked_states == 10, "P2_VALIDATION_TOTAL_STATE_COUNT_INVALID")
    require(pair["probe_forward_count"] == checked_probes * 12 == 120, "P2_VALIDATION_TOTAL_FORWARD_COUNT_INVALID")
    require(pair["backward_pass_count"] == pair["optimizer_step_count"] == pair["training_continuation_count"] == 0, "P2_VALIDATION_FORBIDDEN_EXECUTION_RECORDED")
    require(pair["native_target_content_opened"] is False, "P2_VALIDATION_TARGET_OPENED_PRESEAL")
    return _checked_result(
        validation_path,
        {
            "schema": "nanogpt-p2-response-pre-target-validation-v1",
            "status": "PASS",
            "protocol_sha256": protocol_sha,
            "seal_result_sha256": seal["result_sha256"],
            "pair_receipt_result_sha256": pair["result_sha256"],
            "checked_state_count": checked_states,
            "checked_probe_count": checked_probes,
            "checked_forward_count": checked_probes * 12,
            "checked_numeric_response_array_count": checked_numeric_arrays,
            "checked_categorical_response_array_count": checked_categorical_arrays,
            "response_rows": response_rows,
            "state_and_optimizer_reconstruction_exact": True,
            "response_recomputation_exact": True,
            "sealed_file_hashes_exact": True,
            "native_target_content_opened": False,
            "future_information_used": False,
        },
    )


__all__ = ["validate_p2_response"]
