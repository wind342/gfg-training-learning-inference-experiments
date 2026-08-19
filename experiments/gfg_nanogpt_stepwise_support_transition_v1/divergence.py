from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
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

from .contracts import ComponentRegistry
from .phase import NUMERIC_PROBE_FIELDS


STATE_CHANNELS = (
    "raw_probe_response",
    "support_concentration",
    "pair_backup",
    "adam_state",
    "parameter_adam_alignment",
    "phase_left_rate_m1",
    "phase_left_rate_multiscale",
    "phase_left_acceleration",
    "visible_capability",
)
TRANSITION_CHANNELS = (
    "parameter_update_structure",
    "optimizer_transition",
)
PREDICTOR_CHANNELS = tuple(value for value in (*STATE_CHANNELS, *TRANSITION_CHANNELS) if value != "visible_capability")


@dataclass(frozen=True)
class NumericFeature:
    values: np.ndarray
    defined: np.ndarray


def _load_ref(entry_root: Path, reference: dict[str, Any]) -> np.ndarray:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "SST_DIVERGENCE_TENSOR_LOCATOR_INVALID")
    path = entry_root / locator
    require(file_sha256(path) == reference["file_sha256"], "SST_DIVERGENCE_TENSOR_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "SST_DIVERGENCE_TENSOR_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "SST_DIVERGENCE_TENSOR_DTYPE_MISMATCH")
    require(
        hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
        == reference["raw_tensor_sha256"],
        "SST_DIVERGENCE_TENSOR_RAW_HASH_MISMATCH",
    )
    return np.asarray(value)


def _feature(values: np.ndarray | list[float]) -> NumericFeature:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return NumericFeature(array, np.isfinite(array))


def normalized_l2(left: NumericFeature, right: NumericFeature) -> dict[str, Any]:
    require(left.values.shape == right.values.shape, "SST_DIVERGENCE_FEATURE_SHAPE_MISMATCH")
    defined = left.defined & right.defined
    if not np.any(defined):
        return {
            "distance": None,
            "disposition": "DISTANCE_UNDEFINED_NO_SHARED_NUMERIC_COORDINATE",
            "shared_coordinate_count": 0,
        }
    a = left.values[defined]
    b = right.values[defined]
    numerator = float(np.linalg.norm(a - b))
    denominator = max((float(np.linalg.norm(a)) + float(np.linalg.norm(b))) / 2.0, 1e-30)
    return {
        "distance": numerator / denominator,
        "disposition": None,
        "shared_coordinate_count": int(np.count_nonzero(defined)),
    }


def _state_record(formal_root: Path, entry_id: str, window_id: str, step: int) -> tuple[Path, dict[str, Any]]:
    entry_root = formal_root / entry_id
    path = entry_root / "windows" / window_id / "states" / f"step-{step:05d}.json"
    value = read_json(path)
    require(value["schema"] == "nanogpt-stepwise-state-v1", "SST_DIVERGENCE_STATE_SCHEMA_INVALID")
    return entry_root, value


def _probe_record(entry_root: Path, state: dict[str, Any], probe_contract_id: str) -> dict[str, Any]:
    state_id = str(state["state"]["state_id"])
    path = entry_root / "probe-observations" / probe_contract_id / f"{state_id}.json"
    value = read_json(path)
    require(value["schema"] == "nanogpt-stepwise-probe-observation-v1", "SST_DIVERGENCE_PROBE_SCHEMA_INVALID")
    require(value["observed_state_id"] == state_id, "SST_DIVERGENCE_PROBE_STATE_MISMATCH")
    return value


def _transition_record(formal_root: Path, entry_id: str, window_id: str, step: int) -> tuple[Path, dict[str, Any]]:
    entry_root = formal_root / entry_id
    path = entry_root / "windows" / window_id / "transitions" / f"step-{step:05d}-to-{step + 1:05d}.json"
    value = read_json(path)
    require(value["schema"] == "nanogpt-stepwise-transition-v1", "SST_DIVERGENCE_TRANSITION_SCHEMA_INVALID")
    return entry_root, value


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64).reshape(-1)
    right64 = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left64)) * float(np.linalg.norm(right64))
    return float(np.dot(left64, right64) / denominator) if denominator > 0.0 else math.nan


def _parameter_adam_alignment(
    entry_root: Path,
    state_manifest: dict[str, Any],
    registry: ComponentRegistry,
) -> NumericFeature:
    parameter_ref = state_manifest["parameters"]
    exp_avg_ref = state_manifest["optimizer_exp_avg"]
    exp_avg_sq_ref = state_manifest["optimizer_exp_avg_sq"]
    parameters = _load_ref(entry_root, parameter_ref).reshape(-1)
    exp_avg = _load_ref(entry_root, exp_avg_ref).reshape(-1)
    exp_avg_sq = _load_ref(entry_root, exp_avg_sq_ref).reshape(-1)
    require(parameters.shape == exp_avg.shape == exp_avg_sq.shape, "SST_ALIGNMENT_PACKED_SHAPE_MISMATCH")
    preconditioned = exp_avg / (np.sqrt(exp_avg_sq) + 1e-8)
    values = [_cosine(parameters, preconditioned)]
    layout = parameter_ref["layout"]
    for component in registry.components:
        indices: list[np.ndarray] = []
        for row in layout:
            if str(row["name"]).startswith(component.parameter_prefix):
                start = int(row["offset"])
                indices.append(np.arange(start, start + int(row["element_count"]), dtype=np.int64))
        require(bool(indices), f"SST_ALIGNMENT_COMPONENT_EMPTY:{component.component_id}")
        selected = np.concatenate(indices)
        values.append(_cosine(parameters[selected], preconditioned[selected]))
    return _feature(values)


def state_features(
    formal_root: Path,
    *,
    entry_id: str,
    window_id: str,
    optimizer_step: int,
    probe_contract_id: str,
    registry: ComponentRegistry,
) -> dict[str, NumericFeature]:
    entry_root, state = _state_record(formal_root, entry_id, window_id, optimizer_step)
    probe = _probe_record(entry_root, state, probe_contract_id)
    raw: list[np.ndarray] = []
    for row in probe["forwards"]:
        for key in ("logits", "margins", "group_q10_margin"):
            raw.append(_load_ref(entry_root, row[key]).reshape(-1))
    baseline = probe["forwards"][0]
    visible = np.concatenate(
        [
            np.asarray([probe["capability_accuracy"]], dtype=np.float64),
            _load_ref(entry_root, baseline["margins"]).reshape(-1),
            _load_ref(entry_root, baseline["group_q10_margin"]).reshape(-1),
        ]
    )
    manifest = state["state"]
    adam_state = np.concatenate(
        [
            _load_ref(entry_root, manifest["optimizer_exp_avg"]).reshape(-1),
            np.sqrt(_load_ref(entry_root, manifest["optimizer_exp_avg_sq"]).reshape(-1)),
            _load_ref(entry_root, manifest["optimizer_steps"]).reshape(-1),
        ]
    )
    phase_path = (
        entry_root
        / "derived"
        / "support-phase-finite-difference-v1"
        / window_id
        / "states"
        / f"step-{optimizer_step:05d}.json"
    )
    phase = read_json(phase_path)

    def phase_fields(group: dict[str, Any]) -> np.ndarray:
        return np.concatenate([_load_ref(entry_root, group[name]).reshape(-1) for name in NUMERIC_PROBE_FIELDS])

    phase_left_m1 = phase_fields(phase["left_rates"]["1"])
    phase_left_multiscale = np.concatenate(
        [phase_fields(phase["left_rates"][str(scale)]) for scale in (1, 2, 5, 10)]
    )
    phase_left_acceleration = phase_fields(phase["left_acceleration"])
    return {
        "raw_probe_response": _feature(np.concatenate(raw)),
        "support_concentration": _feature(_load_ref(entry_root, probe["support_concentration"])),
        "pair_backup": _feature(_load_ref(entry_root, probe["pair_backup"])),
        "adam_state": _feature(adam_state),
        "parameter_adam_alignment": _parameter_adam_alignment(entry_root, manifest, registry),
        "phase_left_rate_m1": _feature(phase_left_m1),
        "phase_left_rate_multiscale": _feature(phase_left_multiscale),
        "phase_left_acceleration": _feature(phase_left_acceleration),
        "visible_capability": _feature(visible),
    }


def transition_features(
    formal_root: Path,
    *,
    entry_id: str,
    window_id: str,
    optimizer_step: int,
) -> dict[str, NumericFeature]:
    entry_root, transition = _transition_record(formal_root, entry_id, window_id, optimizer_step)
    step = transition["step"]
    require(step["execute_optimizer"], "SST_DIVERGENCE_MAIN_TRANSITION_NOT_FULL_STEP")
    optimizer = step["optimizer_deltas"]
    return {
        "parameter_update_structure": _feature(_load_ref(entry_root, step["parameter_update"])),
        "optimizer_transition": _feature(
            np.concatenate(
                [
                    _load_ref(entry_root, optimizer["exp_avg"]).reshape(-1),
                    _load_ref(entry_root, optimizer["exp_avg_sq"]).reshape(-1),
                    _load_ref(entry_root, optimizer["adam_step"]).reshape(-1),
                ]
            )
        ),
    }


def _distance_sequences(
    formal_root: Path,
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    probe_contract_id: str,
    registry: ComponentRegistry,
) -> dict[str, list[dict[str, Any]]]:
    result = {channel: [] for channel in (*STATE_CHANNELS, *TRANSITION_CHANNELS)}
    for h in range(101):
        left_state = state_features(
            formal_root,
            entry_id=str(left["entry_id"]),
            window_id=str(left["window_id"]),
            optimizer_step=int(left["start_optimizer_step"]) + h,
            probe_contract_id=probe_contract_id,
            registry=registry,
        )
        right_state = state_features(
            formal_root,
            entry_id=str(right["entry_id"]),
            window_id=str(right["window_id"]),
            optimizer_step=int(right["start_optimizer_step"]) + h,
            probe_contract_id=probe_contract_id,
            registry=registry,
        )
        for channel in STATE_CHANNELS:
            result[channel].append(normalized_l2(left_state[channel], right_state[channel]))
        if h < 100:
            left_transition = transition_features(
                formal_root,
                entry_id=str(left["entry_id"]),
                window_id=str(left["window_id"]),
                optimizer_step=int(left["start_optimizer_step"]) + h,
            )
            right_transition = transition_features(
                formal_root,
                entry_id=str(right["entry_id"]),
                window_id=str(right["window_id"]),
                optimizer_step=int(right["start_optimizer_step"]) + h,
            )
            for channel in TRANSITION_CHANNELS:
                result[channel].append(normalized_l2(left_transition[channel], right_transition[channel]))
    return result


def build_divergence_distance_cache(
    *,
    formal_root: Path,
    selection_path: Path,
    pairing_path: Path,
    component_registry_path: Path,
    probe_contract_id: str,
    audit_protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    pairing = read_json(pairing_path)
    registry = ComponentRegistry.load(component_registry_path)
    windows = {str(row["window_id"]): row for row in selection["windows"]}
    pair_rows: list[dict[str, Any]] = []
    stable_ids = sorted({str(row["stable_window_id"]) for row in pairing["pairings"]})

    for index, row in enumerate(pairing["pairings"], start=1):
        decline = windows[str(row["decline_window_id"])]
        stable = windows[str(row["stable_window_id"])]
        distances = _distance_sequences(
            formal_root,
            left=decline,
            right=stable,
            probe_contract_id=probe_contract_id,
            registry=registry,
        )
        pair_rows.append(
            {
                "pair_id": row["pair_id"],
                "decline_entry_id": decline["entry_id"],
                "decline_window_id": decline["window_id"],
                "stable_entry_id": stable["entry_id"],
                "stable_window_id": stable["window_id"],
                "distances": distances,
            }
        )
        print({"event": "SST_PAIR_DISTANCE_COMPLETE", "ordinal": index, "pair_count": len(pairing["pairings"]), "pair_id": row["pair_id"]}, flush=True)

    stable_rows: list[dict[str, Any]] = []
    for index, window_id in enumerate(stable_ids, start=1):
        window = windows[window_id]
        distances: dict[str, list[dict[str, Any]]] = {
            channel: [] for channel in (*STATE_CHANNELS, *TRANSITION_CHANNELS)
        }
        state_zero = state_features(
            formal_root,
            entry_id=str(window["entry_id"]),
            window_id=window_id,
            optimizer_step=int(window["start_optimizer_step"]),
            probe_contract_id=probe_contract_id,
            registry=registry,
        )
        transition_zero = transition_features(
            formal_root,
            entry_id=str(window["entry_id"]),
            window_id=window_id,
            optimizer_step=int(window["start_optimizer_step"]),
        )
        for h in range(101):
            current = state_features(
                formal_root,
                entry_id=str(window["entry_id"]),
                window_id=window_id,
                optimizer_step=int(window["start_optimizer_step"]) + h,
                probe_contract_id=probe_contract_id,
                registry=registry,
            )
            for channel in STATE_CHANNELS:
                distances[channel].append(normalized_l2(current[channel], state_zero[channel]))
            if h < 100:
                current_transition = transition_features(
                    formal_root,
                    entry_id=str(window["entry_id"]),
                    window_id=window_id,
                    optimizer_step=int(window["start_optimizer_step"]) + h,
                )
                for channel in TRANSITION_CHANNELS:
                    distances[channel].append(normalized_l2(current_transition[channel], transition_zero[channel]))
        stable_rows.append({"entry_id": window["entry_id"], "window_id": window_id, "distances_from_h0": distances})
        print({"event": "SST_STABLE_ENVELOPE_DISTANCE_COMPLETE", "ordinal": index, "stable_window_count": len(stable_ids), "window_id": window_id}, flush=True)

    material = {
        "schema": "nanogpt-stepwise-divergence-distance-cache-v1",
        "status": "PASS",
        "formal_root": str(formal_root.resolve()),
        "selection_sha256": selection["selection_sha256"],
        "pairing_sha256": pairing["pairing_sha256"],
        "component_registry_id": registry.registry_id,
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_id": probe_contract_id,
        "audit_protocol_sha256": file_sha256(audit_protocol_path),
        "state_channels": list(STATE_CHANNELS),
        "transition_channels": list(TRANSITION_CHANNELS),
        "pair_rows": pair_rows,
        "stable_rows": stable_rows,
    }
    result = {**material, "distance_cache_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result


def adjudicate_divergence(
    *,
    distance_cache_path: Path,
    audit_protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    cache = read_json(distance_cache_path)
    require(cache["audit_protocol_sha256"] == file_sha256(audit_protocol_path), "SST_DIVERGENCE_PROTOCOL_DRIFT")
    results: list[dict[str, Any]] = []
    for pair in cache["pair_rows"]:
        excluded = {str(pair["decline_entry_id"]), str(pair["stable_entry_id"])}
        development = [row for row in cache["stable_rows"] if str(row["entry_id"]) not in excluded]
        require(bool(development), f"SST_DIVERGENCE_LOO_POPULATION_EMPTY:{pair['pair_id']}")
        channels: dict[str, Any] = {}
        detected: list[tuple[int, str]] = []
        fallback: list[tuple[float, int, str]] = []
        for channel in (*STATE_CHANNELS, *TRANSITION_CHANNELS):
            sequence = pair["distances"][channel]
            initial = sequence[0]["distance"]
            require(initial is not None, f"SST_DIVERGENCE_PAIR_INITIAL_UNDEFINED:{pair['pair_id']}:{channel}")
            thresholds: list[float] = []
            excess: list[float] = []
            ratios: list[float] = []
            for h, row in enumerate(sequence):
                population = [
                    float(candidate["distances_from_h0"][channel][h]["distance"])
                    for candidate in development
                    if candidate["distances_from_h0"][channel][h]["distance"] is not None
                ]
                require(bool(population), f"SST_DIVERGENCE_ENVELOPE_EMPTY:{pair['pair_id']}:{channel}:{h}")
                threshold = max(float(np.quantile(np.asarray(population, dtype=np.float64), 0.99)), 1e-12)
                value = max(0.0, float(row["distance"]) - float(initial)) if row["distance"] is not None else 0.0
                thresholds.append(threshold)
                excess.append(value)
                ratios.append(value / threshold)
            earliest = None
            for h in range(0, min(len(sequence), 100) - 2):
                if all(excess[index] > thresholds[index] for index in (h, h + 1, h + 2)):
                    earliest = h
                    break
            channels[channel] = {
                "initial_distance": initial,
                "thresholds": thresholds,
                "pair_excess": excess,
                "ratios": ratios,
                "earliest_persistent_crossing_h": earliest,
            }
            if channel in PREDICTOR_CHANNELS:
                if earliest is not None:
                    detected.append((earliest, channel))
                best_h = max(range(min(len(ratios), 100)), key=lambda value: (ratios[value], -value))
                fallback.append((ratios[best_h], best_h, channel))
        if detected:
            key_h, key_channel = min(detected, key=lambda value: (value[0], value[1]))
            key_status = "DETECTED_PERSISTENT_PREDICTOR_DIVERGENCE"
        else:
            _ratio, key_h, key_channel = max(fallback, key=lambda value: (value[0], -value[1], value[2]))
            key_status = "NO_THRESHOLD_CROSSING_FALLBACK"
        legal_h = sorted({value for offset in (-1, 0, 1) if 0 <= (value := key_h + offset) <= 99})
        results.append(
            {
                "pair_id": pair["pair_id"],
                "decline_entry_id": pair["decline_entry_id"],
                "decline_window_id": pair["decline_window_id"],
                "stable_entry_id": pair["stable_entry_id"],
                "stable_window_id": pair["stable_window_id"],
                "leave_out_entry_ids": sorted(excluded),
                "development_stable_window_count": len(development),
                "channels": channels,
                "key_step_relative_h": key_h,
                "key_step_channel": key_channel,
                "key_step_status": key_status,
                "causal_seed_relative_h": legal_h,
                "visible_capability_excluded_from_key_selection": True,
            }
        )
    material = {
        "schema": "nanogpt-stepwise-earliest-divergence-audit-v1",
        "status": "PASS",
        "distance_cache_sha256": cache["distance_cache_sha256"],
        "audit_protocol_sha256": file_sha256(audit_protocol_path),
        "results": results,
    }
    result = {**material, "audit_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
