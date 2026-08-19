from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)

from .contracts import ComponentRegistry, ProbeContract
from .phase import (
    CATEGORICAL_PROBE_FIELDS,
    NUMERIC_PROBE_FIELDS,
    _load_ref,
    load_support_state,
    verified_phase_protocol_sha256,
)


def _verify_ref(
    entry_root: Path,
    reference: dict[str, Any],
    expected: np.ndarray,
    *,
    temporal_role: str,
    operation: str,
) -> None:
    actual = _load_ref(entry_root, reference)
    require(reference["temporal_role"] == temporal_role, "SST_PHASE_VALIDATOR_TEMPORAL_ROLE_MISMATCH")
    require(reference["operation"] == operation, "SST_PHASE_VALIDATOR_OPERATION_MISMATCH")
    require(actual.dtype == expected.dtype, "SST_PHASE_VALIDATOR_DTYPE_MISMATCH")
    require(actual.shape == expected.shape, "SST_PHASE_VALIDATOR_SHAPE_MISMATCH")
    require(np.array_equal(actual, expected, equal_nan=True), "SST_PHASE_VALIDATOR_VALUE_MISMATCH")


def _difference(left: np.ndarray, right: np.ndarray, divisor: int = 1) -> np.ndarray:
    return (right.astype(np.float64) - left.astype(np.float64)) / float(divisor)


def validate_phase_window(
    *,
    formal_root: Path,
    window: dict[str, Any],
    registry: ComponentRegistry,
    probe_contract: ProbeContract,
    phase_protocol_sha256: str,
) -> dict[str, Any]:
    entry_root = formal_root / str(window["entry_id"])
    window_id = str(window["window_id"])
    scientific_start = int(window["scientific_start_optimizer_step"])
    scientific_end = int(window["scientific_end_optimizer_step"])
    capture_start = int(window["capture_start_optimizer_step"])
    phase_root = entry_root / "derived" / "support-phase-finite-difference-v1" / window_id
    receipt = read_json(phase_root / "phase_window_receipt.json")
    receipt_material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    require(payload_sha256(receipt_material) == receipt["receipt_sha256"], "SST_PHASE_VALIDATOR_RECEIPT_HASH_MISMATCH")
    require(receipt["phase_protocol_sha256"] == phase_protocol_sha256, "SST_PHASE_VALIDATOR_PROTOCOL_DRIFT")
    require(receipt["phase_state_count"] == scientific_end - scientific_start + 1, "SST_PHASE_VALIDATOR_STATE_COUNT_MISMATCH")

    states = {
        step: load_support_state(
            entry_root=entry_root,
            window_id=window_id,
            optimizer_step=step,
            registry=registry,
            probe_contract=probe_contract,
        )
        for step in range(capture_start, scientific_end + 1)
    }
    slot_hashes = {state.identity["slot_identity_sha256"] for state in states.values()}
    require(len(slot_hashes) == 1, "SST_PHASE_VALIDATOR_SLOT_IDENTITY_DRIFT")
    verified_numeric_arrays = 0
    verified_categorical_arrays = 0
    target_only_arrays = 0
    input_arrays = 0
    for step in range(scientific_start, scientific_end + 1):
        record = read_json(phase_root / "states" / f"step-{step:05d}.json")
        record_material = {key: value for key, value in record.items() if key != "phase_state_sha256"}
        require(payload_sha256(record_material) == record["phase_state_sha256"], "SST_PHASE_VALIDATOR_RECORD_HASH_MISMATCH")
        require(record["phase_protocol_sha256"] == phase_protocol_sha256, "SST_PHASE_VALIDATOR_RECORD_PROTOCOL_DRIFT")
        require(record["support_slot_identity"]["slot_identity_sha256"] in slot_hashes, "SST_PHASE_VALIDATOR_SLOT_IDENTITY_MISMATCH")
        require(record["continuous_derivative_claimed"] is False, "SST_PHASE_VALIDATOR_CONTINUOUS_DERIVATIVE_MISCLAIM")
        current = states[step]
        for scale in (1, 2, 5, 10):
            predecessor = states[step - scale]
            for name in NUMERIC_PROBE_FIELDS:
                expected = _difference(predecessor.numeric[name], current.numeric[name], scale)
                _verify_ref(
                    entry_root,
                    record["left_rates"][str(scale)][name],
                    expected,
                    temporal_role="input_available_at_cut",
                    operation="finite_difference_left",
                )
                verified_numeric_arrays += 1
                input_arrays += 1
            for name in CATEGORICAL_PROBE_FIELDS:
                expected_change = np.not_equal(predecessor.categorical[name], current.categorical[name])
                _verify_ref(
                    entry_root,
                    record["left_prediction_change_masks"][str(scale)][name],
                    expected_change,
                    temporal_role="input_available_at_cut",
                    operation="identity_aligned_categorical_change",
                )
                verified_categorical_arrays += 1
                input_arrays += 1
        previous = states[step - 1]
        previous_previous = states[step - 2]
        v_now = {name: _difference(previous.numeric[name], current.numeric[name]) for name in NUMERIC_PROBE_FIELDS}
        for name in NUMERIC_PROBE_FIELDS:
            v_previous = _difference(previous_previous.numeric[name], previous.numeric[name])
            expected_acceleration = v_now[name] - v_previous
            _verify_ref(
                entry_root,
                record["left_acceleration"][name],
                expected_acceleration,
                temporal_role="input_available_at_cut",
                operation="finite_difference_left_acceleration",
            )
            verified_numeric_arrays += 1
            input_arrays += 1
        require(
            record["categorical_acceleration_disposition"]["disposition_type"]
            == "CATEGORICAL_ACCELERATION_NOT_NUMERICALLY_DEFINED",
            "SST_PHASE_VALIDATOR_CATEGORICAL_ACCELERATION_DISPOSITION_MISSING",
        )
        if step < scientific_end:
            successor = states[step + 1]
            for name in NUMERIC_PROBE_FIELDS:
                expected_right = _difference(current.numeric[name], successor.numeric[name])
                _verify_ref(
                    entry_root,
                    record["right_rate_target_only"][name],
                    expected_right,
                    temporal_role="target_only_after_cut",
                    operation="finite_difference_right",
                )
                _verify_ref(
                    entry_root,
                    record["law_break_target_only"][name],
                    expected_right - v_now[name],
                    temporal_role="target_only_after_cut",
                    operation="left_right_discrete_rate_difference",
                )
                verified_numeric_arrays += 2
                target_only_arrays += 2
            for name in CATEGORICAL_PROBE_FIELDS:
                left_change = np.not_equal(previous.categorical[name], current.categorical[name])
                right_change = np.not_equal(current.categorical[name], successor.categorical[name])
                _verify_ref(
                    entry_root,
                    record["right_prediction_change_target_only"][name],
                    right_change,
                    temporal_role="target_only_after_cut",
                    operation="identity_aligned_categorical_change",
                )
                _verify_ref(
                    entry_root,
                    record["categorical_law_break_target_only"][name],
                    np.not_equal(left_change, right_change),
                    temporal_role="target_only_after_cut",
                    operation="categorical_transition_change",
                )
                verified_categorical_arrays += 2
                target_only_arrays += 2
        else:
            require(record["right_rate_target_only"] is None, "SST_PHASE_VALIDATOR_TERMINAL_RIGHT_RATE_PRESENT")
            require(record["law_break_target_only"] is None, "SST_PHASE_VALIDATOR_TERMINAL_LAW_BREAK_PRESENT")
            require(
                record["right_difference_disposition"]["disposition_type"]
                == "FINITE_DIFFERENCE_RIGHT_OUTSIDE_FROZEN_WINDOW",
                "SST_PHASE_VALIDATOR_TERMINAL_DISPOSITION_MISSING",
            )
        partition = record["future_leakage_partition"]
        require("right_rate_target_only" not in partition["input_available_at_cut"], "SST_PHASE_VALIDATOR_FUTURE_LEAKAGE")
        require("law_break_target_only" not in partition["input_available_at_cut"], "SST_PHASE_VALIDATOR_FUTURE_LEAKAGE")
    return {
        "entry_id": window["entry_id"],
        "window_id": window_id,
        "receipt_sha256": receipt["receipt_sha256"],
        "verified_numeric_array_count": verified_numeric_arrays,
        "verified_categorical_array_count": verified_categorical_arrays,
        "input_available_array_count": input_arrays,
        "target_only_array_count": target_only_arrays,
        "future_leakage_audit": "PASS",
        "identity_alignment_audit": "PASS",
        "finite_difference_replay": "PASS",
    }


def validate_phase_evidence(
    *,
    formal_root: Path,
    selection_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    phase_protocol_path: Path,
    output_path: Path,
    parent_selection_path: Path | None = None,
    entry_id: str | None = None,
    max_windows: int | None = None,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    phase_protocol_sha = verified_phase_protocol_sha256(
        selection=selection,
        phase_protocol_path=phase_protocol_path,
        parent_selection_path=parent_selection_path,
    )
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    windows = list(selection["windows"])
    if entry_id is not None:
        windows = [window for window in windows if str(window["entry_id"]) == entry_id]
        require(bool(windows), f"SST_PHASE_VALIDATOR_ENTRY_WINDOWS_EMPTY:{entry_id}")
    if max_windows is not None:
        windows = windows[:max_windows]
    results = [
        validate_phase_window(
            formal_root=formal_root,
            window=window,
            registry=registry,
            probe_contract=probe_contract,
            phase_protocol_sha256=phase_protocol_sha,
        )
        for window in windows
    ]
    material = {
        "schema": "nanogpt-support-phase-validation-v1",
        "status": "PASS",
        "formal_root": str(formal_root.resolve()),
        "selection_sha256": selection["selection_sha256"],
        "phase_protocol_sha256": phase_protocol_sha,
        "validated_window_count": len(results),
        "validated_numeric_array_count": sum(row["verified_numeric_array_count"] for row in results),
        "validated_categorical_array_count": sum(row["verified_categorical_array_count"] for row in results),
        "future_leakage_audit": "PASS",
        "identity_alignment_audit": "PASS",
        "finite_difference_replay": "PASS",
        "window_results": results,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
