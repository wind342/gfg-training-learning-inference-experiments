from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
from .storage import TensorStore


NUMERIC_PROBE_FIELDS = (
    "component_target_group_response",
    "forward_logits",
    "forward_margins",
    "necessity",
    "pair_backup",
    "single_failure_slack",
    "double_failure_slack",
    "support_edge_strength",
    "support_allocation",
    "support_concentration",
    "effective_support",
)
CATEGORICAL_PROBE_FIELDS = ("forward_predictions",)


def verified_phase_protocol_sha256(
    *,
    selection: dict[str, Any],
    phase_protocol_path: Path,
    parent_selection_path: Path | None = None,
) -> str:
    """Bind a full or independently replayed selection to the phase protocol.

    Independent replay selections were frozen as exact subsets before results
    existed.  Their content hash binds the subset and the parent selection hash,
    while the parent selection binds the finite-difference protocol.  Verify the
    complete chain instead of mutating the already-used replay selection.
    """

    selection_material = {
        key: value for key, value in selection.items() if key != "selection_sha256"
    }
    require(
        payload_sha256(selection_material) == selection["selection_sha256"],
        "SST_PHASE_SELECTION_HASH_INVALID",
    )
    protocol_sha = file_sha256(phase_protocol_path)
    declared = selection.get("finite_difference_protocol_sha256")
    if declared is not None:
        require(declared == protocol_sha, "SST_PHASE_SELECTION_PROTOCOL_DRIFT")
        return protocol_sha

    require(
        selection.get("schema")
        == "nanogpt-stepwise-independent-replay-window-selection-v1",
        "SST_PHASE_SELECTION_PROTOCOL_BINDING_MISSING",
    )
    require(parent_selection_path is not None, "SST_PHASE_PARENT_SELECTION_REQUIRED")
    parent = read_json(parent_selection_path)
    parent_material = {
        key: value for key, value in parent.items() if key != "selection_sha256"
    }
    require(
        payload_sha256(parent_material) == parent["selection_sha256"],
        "SST_PHASE_PARENT_SELECTION_HASH_INVALID",
    )
    require(
        selection["parent_selection_sha256"] == parent["selection_sha256"],
        "SST_PHASE_PARENT_SELECTION_DRIFT",
    )
    require(
        parent["finite_difference_protocol_sha256"] == protocol_sha,
        "SST_PHASE_PARENT_SELECTION_PROTOCOL_DRIFT",
    )
    parent_windows = {str(row["window_id"]): row for row in parent["windows"]}
    for window in selection["windows"]:
        window_id = str(window["window_id"])
        require(window_id in parent_windows, f"SST_PHASE_REPLAY_WINDOW_UNKNOWN:{window_id}")
        require(
            window == parent_windows[window_id],
            f"SST_PHASE_REPLAY_WINDOW_DRIFT:{window_id}",
        )
    return protocol_sha


@dataclass(frozen=True)
class SupportState:
    optimizer_step: int
    state_id: str
    probe_observation_id: str
    identity: dict[str, Any]
    numeric: dict[str, np.ndarray]
    categorical: dict[str, np.ndarray]
    source_refs: dict[str, Any]


def _load_ref(entry_root: Path, reference: dict[str, Any]) -> np.ndarray:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "SST_PHASE_TENSOR_LOCATOR_INVALID")
    path = entry_root / locator
    require(path.is_file(), f"SST_PHASE_TENSOR_MISSING:{locator}")
    require(file_sha256(path) == reference["file_sha256"], f"SST_PHASE_TENSOR_FILE_HASH_MISMATCH:{locator}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    materialized = np.asarray(array)
    require(list(materialized.shape) == list(reference["shape"]), f"SST_PHASE_TENSOR_SHAPE_MISMATCH:{locator}")
    require(str(materialized.dtype) == str(reference["dtype"]), f"SST_PHASE_TENSOR_DTYPE_MISMATCH:{locator}")
    raw_sha = hashlib.sha256(np.ascontiguousarray(materialized).tobytes(order="C")).hexdigest()
    require(raw_sha == reference["raw_tensor_sha256"], f"SST_PHASE_TENSOR_RAW_HASH_MISMATCH:{locator}")
    return materialized


def _state_path(entry_root: Path, window_id: str, step: int) -> Path:
    return entry_root / "windows" / window_id / "states" / f"step-{step:05d}.json"


def _slot_identity(
    *,
    probe_contract: ProbeContract,
    registry: ComponentRegistry,
    gate_plan: tuple[tuple[str, ...], ...],
    target_group_identity: str,
    numeric: dict[str, np.ndarray],
    categorical: dict[str, np.ndarray],
) -> dict[str, Any]:
    slots = {
        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in sorted({**numeric, **categorical}.items())
    }
    material = {
        "component_registry_id": registry.registry_id,
        "component_registry_sha256": registry.source_sha256,
        "component_ids": list(registry.component_ids),
        "probe_contract_id": probe_contract.probe_contract_id,
        "probe_contract_sha256": probe_contract.source_sha256,
        "gate_plan": [list(value) for value in gate_plan],
        "target_group_identity": target_group_identity,
        "result_slots": slots,
    }
    return {**material, "slot_identity_sha256": payload_sha256(material)}


def load_support_state(
    *,
    entry_root: Path,
    window_id: str,
    optimizer_step: int,
    registry: ComponentRegistry,
    probe_contract: ProbeContract,
) -> SupportState:
    state_path = _state_path(entry_root, window_id, optimizer_step)
    state = read_json(state_path)
    require(state["schema"] == "nanogpt-stepwise-state-v1", "SST_PHASE_STATE_SCHEMA_INVALID")
    require(int(state["optimizer_step"]) == optimizer_step, "SST_PHASE_STATE_STEP_MISMATCH")
    state_id = str(state["state"]["state_id"])
    probe_path = entry_root / "probe-observations" / probe_contract.probe_contract_id / f"{state_id}.json"
    probe = read_json(probe_path)
    require(probe["schema"] == "nanogpt-stepwise-probe-observation-v1", "SST_PHASE_PROBE_SCHEMA_INVALID")
    require(probe["observed_state_id"] == state_id, "SST_PHASE_PROBE_STATE_ID_MISMATCH")
    require(probe["probe_contract_id"] == probe_contract.probe_contract_id, "SST_PHASE_PROBE_CONTRACT_MISMATCH")
    require(probe["probe_contract_sha256"] == probe_contract.source_sha256, "SST_PHASE_PROBE_CONTRACT_HASH_MISMATCH")
    require(probe["component_registry_id"] == registry.registry_id, "SST_PHASE_COMPONENT_REGISTRY_MISMATCH")
    require(probe["component_registry_sha256"] == registry.source_sha256, "SST_PHASE_COMPONENT_REGISTRY_HASH_MISMATCH")
    require(tuple(probe["component_ids"]) == registry.component_ids, "SST_PHASE_COMPONENT_ID_ALIGNMENT_FAILURE")

    expected_gates = ((),) * probe_contract.baseline_repetitions + probe_contract.gate_sets
    actual_gates = tuple(tuple(str(value) for value in row["gate_components"]) for row in probe["forwards"])
    require(actual_gates == expected_gates, "SST_PHASE_GATE_PLAN_ALIGNMENT_FAILURE")
    pair_ids = tuple(tuple(str(value) for value in pair) for pair in probe["pair_ids"])
    require(pair_ids == probe_contract.pair_gates(), "SST_PHASE_PAIR_ID_ALIGNMENT_FAILURE")

    group_membership = [_load_ref(entry_root, row["group_membership"]) for row in probe["forwards"]]
    group_hashes = [hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest() for value in group_membership]
    require(len(set(group_hashes)) == 1, "SST_PHASE_TARGET_GROUP_IDENTITY_INCONSISTENT_WITHIN_STATE")
    target_group_identity = group_hashes[0]

    forward_group_q10 = np.stack([_load_ref(entry_root, row["group_q10_margin"]) for row in probe["forwards"]])
    forward_logits = np.stack([_load_ref(entry_root, row["logits"]) for row in probe["forwards"]])
    forward_margins = np.stack([_load_ref(entry_root, row["margins"]) for row in probe["forwards"]])
    forward_predictions = np.stack([_load_ref(entry_root, row["predictions"]) for row in probe["forwards"]])
    necessity = _load_ref(entry_root, probe["necessity"])
    pair_backup = _load_ref(entry_root, probe["pair_backup"])
    require(necessity.shape == (len(registry.component_ids), probe_contract.target_group_count), "SST_PHASE_NECESSITY_SHAPE_INVALID")
    require(pair_backup.shape == (len(probe_contract.pair_gates()), probe_contract.target_group_count), "SST_PHASE_PAIR_BACKUP_SHAPE_INVALID")
    numeric = {
        "component_target_group_response": forward_group_q10,
        "forward_logits": forward_logits,
        "forward_margins": forward_margins,
        "necessity": necessity,
        "pair_backup": pair_backup,
        "single_failure_slack": _load_ref(entry_root, probe["single_failure_slack"]),
        "double_failure_slack": _load_ref(entry_root, probe["double_failure_slack"]),
        "support_edge_strength": np.concatenate([necessity, pair_backup], axis=0),
        "support_allocation": _load_ref(entry_root, probe["support_allocation"]),
        "support_concentration": _load_ref(entry_root, probe["support_concentration"]),
        "effective_support": _load_ref(entry_root, probe["effective_support"]),
    }
    categorical = {"forward_predictions": forward_predictions}
    require(tuple(numeric) == NUMERIC_PROBE_FIELDS, "SST_PHASE_NUMERIC_FIELD_SET_DRIFT")
    require(tuple(categorical) == CATEGORICAL_PROBE_FIELDS, "SST_PHASE_CATEGORICAL_FIELD_SET_DRIFT")
    identity = _slot_identity(
        probe_contract=probe_contract,
        registry=registry,
        gate_plan=actual_gates,
        target_group_identity=target_group_identity,
        numeric=numeric,
        categorical=categorical,
    )
    return SupportState(
        optimizer_step=optimizer_step,
        state_id=state_id,
        probe_observation_id=str(probe["probe_observation_id"]),
        identity=identity,
        numeric=numeric,
        categorical=categorical,
        source_refs={
            "state_result_sha256": state["result_sha256"],
            "state_id": state_id,
            "probe_observation_id": probe["probe_observation_id"],
            "probe_result_sha256": probe["result_sha256"],
        },
    )


def _require_aligned(states: list[SupportState]) -> None:
    require(bool(states), "SST_PHASE_ALIGNMENT_SET_EMPTY")
    expected = states[0].identity["slot_identity_sha256"]
    for state in states[1:]:
        require(state.identity["slot_identity_sha256"] == expected, f"SST_PHASE_SLOT_IDENTITY_MISMATCH:{state.optimizer_step}")


def _put_fields(
    store: TensorStore,
    fields: dict[str, np.ndarray],
    *,
    representation_prefix: str,
    temporal_role: str,
    operation: str,
    source_steps: list[int],
) -> dict[str, Any]:
    return {
        name: store.put(
            value,
            representation=f"{representation_prefix}:{name}",
            extra={
                "operation": operation,
                "source_optimizer_steps": source_steps,
                "temporal_role": temporal_role,
            },
        )
        for name, value in sorted(fields.items())
    }


def _numeric_difference(left: SupportState, right: SupportState, divisor: int = 1) -> dict[str, np.ndarray]:
    _require_aligned([left, right])
    return {
        name: (right.numeric[name].astype(np.float64) - left.numeric[name].astype(np.float64)) / float(divisor)
        for name in NUMERIC_PROBE_FIELDS
    }


def _categorical_change(left: SupportState, right: SupportState) -> dict[str, np.ndarray]:
    _require_aligned([left, right])
    return {name: np.not_equal(left.categorical[name], right.categorical[name]) for name in CATEGORICAL_PROBE_FIELDS}


def derive_phase_window(
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
    output_root = entry_root / "derived" / "support-phase-finite-difference-v1" / window_id
    receipt_path = output_root / "phase_window_receipt.json"
    if receipt_path.exists():
        existing = read_json(receipt_path)
        require(existing["phase_protocol_sha256"] == phase_protocol_sha256, "SST_PHASE_RECEIPT_PROTOCOL_DRIFT")
        return existing
    store = TensorStore(entry_root / "tensor-objects")
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
    _require_aligned(list(states.values()))
    records: list[dict[str, Any]] = []
    lookbacks = tuple(int(value) for value in window["lookback_scales"])
    require(lookbacks == (1, 2, 5, 10), "SST_PHASE_WINDOW_LOOKBACK_DRIFT")
    for step in range(scientific_start, scientific_end + 1):
        current = states[step]
        prefix = f"phase:{window_id}:step:{step}"
        left_rates: dict[str, Any] = {}
        left_prediction_change: dict[str, Any] = {}
        for scale in lookbacks:
            predecessor = states[step - scale]
            left_rates[str(scale)] = _put_fields(
                store,
                _numeric_difference(predecessor, current, scale),
                representation_prefix=f"{prefix}:V_minus:m={scale}",
                temporal_role="input_available_at_cut",
                operation="finite_difference_left",
                source_steps=[step - scale, step],
            )
            left_prediction_change[str(scale)] = _put_fields(
                store,
                _categorical_change(predecessor, current),
                representation_prefix=f"{prefix}:categorical_left_change:m={scale}",
                temporal_role="input_available_at_cut",
                operation="identity_aligned_categorical_change",
                source_steps=[step - scale, step],
            )
        previous = states[step - 1]
        previous_previous = states[step - 2]
        v_now = _numeric_difference(previous, current)
        v_previous = _numeric_difference(previous_previous, previous)
        acceleration = {name: v_now[name] - v_previous[name] for name in NUMERIC_PROBE_FIELDS}
        a_minus = _put_fields(
            store,
            acceleration,
            representation_prefix=f"{prefix}:A_minus:m=1",
            temporal_role="input_available_at_cut",
            operation="finite_difference_left_acceleration",
            source_steps=[step - 2, step - 1, step],
        )
        if step < scientific_end:
            successor = states[step + 1]
            v_plus_values = _numeric_difference(current, successor)
            v_plus = _put_fields(
                store,
                v_plus_values,
                representation_prefix=f"{prefix}:V_plus",
                temporal_role="target_only_after_cut",
                operation="finite_difference_right",
                source_steps=[step, step + 1],
            )
            j_values = {name: v_plus_values[name] - v_now[name] for name in NUMERIC_PROBE_FIELDS}
            law_break = _put_fields(
                store,
                j_values,
                representation_prefix=f"{prefix}:J",
                temporal_role="target_only_after_cut",
                operation="left_right_discrete_rate_difference",
                source_steps=[step - 1, step, step + 1],
            )
            right_prediction_change = _put_fields(
                store,
                _categorical_change(current, successor),
                representation_prefix=f"{prefix}:categorical_right_change",
                temporal_role="target_only_after_cut",
                operation="identity_aligned_categorical_change",
                source_steps=[step, step + 1],
            )
            left_change = _categorical_change(previous, current)
            right_change = _categorical_change(current, successor)
            categorical_law_break = _put_fields(
                store,
                {name: np.not_equal(left_change[name], right_change[name]) for name in CATEGORICAL_PROBE_FIELDS},
                representation_prefix=f"{prefix}:categorical_J",
                temporal_role="target_only_after_cut",
                operation="categorical_transition_change",
                source_steps=[step - 1, step, step + 1],
            )
            right_disposition = None
        else:
            v_plus = None
            law_break = None
            right_prediction_change = None
            categorical_law_break = None
            right_disposition = {
                "disposition_type": "FINITE_DIFFERENCE_RIGHT_OUTSIDE_FROZEN_WINDOW",
                "scope_end_optimizer_step": scientific_end,
            }
        material = {
            "schema": "nanogpt-support-phase-state-v1",
            "status": "PASS",
            "entry_id": window["entry_id"],
            "window_id": window_id,
            "optimizer_step": step,
            "phase_protocol_sha256": phase_protocol_sha256,
            "support_slot_identity": current.identity,
            "source_support_states": {
                str(source_step): states[source_step].source_refs
                for source_step in range(step - max(lookbacks), min(step + 1, scientific_end) + 1)
            },
            "left_rates": left_rates,
            "left_prediction_change_masks": left_prediction_change,
            "left_acceleration": a_minus,
            "categorical_acceleration_disposition": {
                "disposition_type": "CATEGORICAL_ACCELERATION_NOT_NUMERICALLY_DEFINED"
            },
            "right_rate_target_only": v_plus,
            "law_break_target_only": law_break,
            "right_prediction_change_target_only": right_prediction_change,
            "categorical_law_break_target_only": categorical_law_break,
            "right_difference_disposition": right_disposition,
            "future_leakage_partition": {
                "input_available_at_cut": ["support_state", "left_rates", "left_acceleration", "left_prediction_change_masks"],
                "target_only_after_cut": ["right_rate_target_only", "law_break_target_only", "right_prediction_change_target_only", "categorical_law_break_target_only"],
            },
            "continuous_derivative_claimed": False,
        }
        result = {**material, "phase_state_sha256": payload_sha256(material)}
        record_path = output_root / "states" / f"step-{step:05d}.json"
        write_json(record_path, result)
        records.append({"optimizer_step": step, "phase_state_sha256": result["phase_state_sha256"]})
    receipt_material = {
        "schema": "nanogpt-support-phase-window-receipt-v1",
        "status": "PASS",
        "entry_id": window["entry_id"],
        "window_id": window_id,
        "phase_protocol_sha256": phase_protocol_sha256,
        "scientific_interval": [scientific_start, scientific_end],
        "captured_prehistory_interval": [capture_start, scientific_start - 1],
        "lookback_scales": list(lookbacks),
        "phase_state_count": len(records),
        "records": records,
        "finite_difference_not_derivative": True,
    }
    receipt = {**receipt_material, "receipt_sha256": payload_sha256(receipt_material)}
    write_json(receipt_path, receipt)
    return receipt


def derive_phase_evidence(
    *,
    formal_root: Path,
    selection_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    phase_protocol_path: Path,
    parent_selection_path: Path | None = None,
    entry_id: str | None = None,
    max_windows: int | None = None,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    protocol_sha = verified_phase_protocol_sha256(
        selection=selection,
        phase_protocol_path=phase_protocol_path,
        parent_selection_path=parent_selection_path,
    )
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    windows = list(selection["windows"])
    if entry_id is not None:
        windows = [window for window in windows if str(window["entry_id"]) == entry_id]
        require(bool(windows), f"SST_PHASE_ENTRY_WINDOWS_EMPTY:{entry_id}")
    if max_windows is not None:
        windows = windows[:max_windows]
    receipts: list[dict[str, Any]] = []
    for ordinal, window in enumerate(windows, start=1):
        receipt = derive_phase_window(
            formal_root=formal_root,
            window=window,
            registry=registry,
            probe_contract=probe_contract,
            phase_protocol_sha256=protocol_sha,
        )
        receipts.append({"entry_id": window["entry_id"], "window_id": window["window_id"], "receipt_sha256": receipt["receipt_sha256"]})
        print({"event": "SST_PHASE_WINDOW_COMPLETE", "ordinal": ordinal, "window_count": len(windows), "window_id": window["window_id"]}, flush=True)
    material = {
        "schema": "nanogpt-support-phase-evidence-index-v1",
        "status": "PASS",
        "formal_root": str(formal_root.resolve()),
        "selection_sha256": selection["selection_sha256"],
        "phase_protocol_sha256": protocol_sha,
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "window_receipts": receipts,
    }
    result = {**material, "index_sha256": payload_sha256(material)}
    write_json(formal_root / "SUPPORT_PHASE_EVIDENCE_INDEX.json", result)
    return result
