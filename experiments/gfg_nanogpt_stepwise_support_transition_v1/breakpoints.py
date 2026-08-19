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

from .phase import NUMERIC_PROBE_FIELDS, _load_ref


SIGNALS = (
    "parameter_update_rms",
    "optimizer_transition_rms",
    "left_evolution_rate_change_rms",
    "support_state_change_rms",
    "candidate_transition_law_switch_rms",
    "margin_change_rms",
    "prediction_change_fraction",
    "capability_change_abs",
)


def _rms(values: list[np.ndarray]) -> float:
    total = 0.0
    count = 0
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        defined = np.isfinite(array)
        total += float(np.sum(array[defined] * array[defined]))
        count += int(np.count_nonzero(defined))
    return float(np.sqrt(total / count)) if count else 0.0


def _phase_group(entry_root: Path, references: dict[str, Any]) -> list[np.ndarray]:
    return [_load_ref(entry_root, references[name]) for name in NUMERIC_PROBE_FIELDS]


def _probe_accuracy(entry_root: Path, window_id: str, step: int, probe_contract_id: str) -> float:
    state = read_json(entry_root / "windows" / window_id / "states" / f"step-{step:05d}.json")
    probe = read_json(entry_root / "probe-observations" / probe_contract_id / f"{state['state']['state_id']}.json")
    return float(probe["capability_accuracy"])


def _transition_signals(entry_root: Path, window_id: str, step: int) -> tuple[float, float]:
    transition = read_json(
        entry_root / "windows" / window_id / "transitions" / f"step-{step:05d}-to-{step + 1:05d}.json"
    )
    parameter = _load_ref(entry_root, transition["step"]["parameter_update"])
    optimizer = transition["step"]["optimizer_deltas"]
    optimizer_arrays = [
        _load_ref(entry_root, optimizer[key])
        for key in ("exp_avg", "exp_avg_sq", "adam_step")
    ]
    return _rms([parameter]), _rms(optimizer_arrays)


def _window_sequence(
    *,
    formal_root: Path,
    window: dict[str, Any],
    probe_contract_id: str,
) -> dict[str, list[float]]:
    entry_root = formal_root / str(window["entry_id"])
    window_id = str(window["window_id"])
    start = int(window["scientific_start_optimizer_step"])
    end = int(window["scientific_end_optimizer_step"])
    require(end - start == 100, "SST_BREAKPOINT_SCIENTIFIC_WINDOW_LENGTH_INVALID")
    result = {name: [] for name in SIGNALS}
    for step in range(start, end):
        phase = read_json(
            entry_root
            / "derived"
            / "support-phase-finite-difference-v1"
            / window_id
            / "states"
            / f"step-{step:05d}.json"
        )
        parameter_rms, optimizer_rms = _transition_signals(entry_root, window_id, step)
        result["parameter_update_rms"].append(parameter_rms)
        result["optimizer_transition_rms"].append(optimizer_rms)
        result["left_evolution_rate_change_rms"].append(_rms(_phase_group(entry_root, phase["left_acceleration"])))
        result["support_state_change_rms"].append(_rms(_phase_group(entry_root, phase["right_rate_target_only"])))
        result["candidate_transition_law_switch_rms"].append(_rms(_phase_group(entry_root, phase["law_break_target_only"])))
        result["margin_change_rms"].append(
            _rms([_load_ref(entry_root, phase["right_rate_target_only"]["forward_margins"])])
        )
        categorical = _load_ref(entry_root, phase["right_prediction_change_target_only"]["forward_predictions"])
        result["prediction_change_fraction"].append(float(np.mean(categorical.astype(np.float64))))
        current_accuracy = _probe_accuracy(entry_root, window_id, step, probe_contract_id)
        next_accuracy = _probe_accuracy(entry_root, window_id, step + 1, probe_contract_id)
        result["capability_change_abs"].append(abs(next_accuracy - current_accuracy))
    return result


def build_breakpoint_cache(
    *,
    formal_root: Path,
    selection_path: Path,
    probe_contract_id: str,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    rows: list[dict[str, Any]] = []
    for ordinal, window in enumerate(selection["windows"], start=1):
        rows.append(
            {
                "entry_id": window["entry_id"],
                "window_id": window["window_id"],
                "categories": window["categories"],
                "signals": _window_sequence(
                    formal_root=formal_root,
                    window=window,
                    probe_contract_id=probe_contract_id,
                ),
            }
        )
        print({"event": "SST_BREAKPOINT_WINDOW_CACHE_COMPLETE", "ordinal": ordinal, "window_count": len(selection["windows"]), "window_id": window["window_id"]}, flush=True)
    material = {
        "schema": "nanogpt-support-breakpoint-cache-v1",
        "status": "PASS",
        "selection_sha256": selection["selection_sha256"],
        "protocol_sha256": file_sha256(protocol_path),
        "probe_contract_id": probe_contract_id,
        "signals": list(SIGNALS),
        "windows": rows,
        "future_target_used_for_window_selection": False,
    }
    result = {**material, "cache_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result


def _earliest_persistent(values: list[float], thresholds: list[float], length: int) -> int | None:
    for start in range(0, len(values) - length + 1):
        if all(values[index] > thresholds[index] for index in range(start, start + length)):
            return start
    return None


def adjudicate_breakpoints(
    *,
    cache_path: Path,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    cache = read_json(cache_path)
    protocol = read_json(protocol_path)
    require(cache["protocol_sha256"] == file_sha256(protocol_path), "SST_BREAKPOINT_PROTOCOL_DRIFT")
    quantile = float(protocol["stable_envelope_quantile"])
    persistence = int(protocol["persistent_crossing_steps"])
    stable_rows = [row for row in cache["windows"] if "matched_stable" in row["categories"]]
    require(bool(stable_rows), "SST_BREAKPOINT_STABLE_POPULATION_EMPTY")
    results: list[dict[str, Any]] = []
    for row in cache["windows"]:
        if not ({"pre_decline", "recovery"} & set(row["categories"])):
            continue
        development = [value for value in stable_rows if value["entry_id"] != row["entry_id"]]
        require(bool(development), f"SST_BREAKPOINT_LOO_POPULATION_EMPTY:{row['window_id']}")
        crossings: dict[str, int | None] = {}
        thresholds: dict[str, list[float]] = {}
        for signal in SIGNALS:
            signal_thresholds = [
                max(
                    float(np.quantile(np.asarray([candidate["signals"][signal][h] for candidate in development]), quantile)),
                    1e-15,
                )
                for h in range(100)
            ]
            thresholds[signal] = signal_thresholds
            crossings[signal] = _earliest_persistent(row["signals"][signal], signal_thresholds, persistence)
        update_crossing_values = [crossings["parameter_update_rms"], crossings["optimizer_transition_rms"]]
        update_crossing = min(value for value in update_crossing_values if value is not None) if any(value is not None for value in update_crossing_values) else None
        ordered = {
            "optimizer_or_update_change": update_crossing,
            "left_evolution_rate_change": crossings["left_evolution_rate_change_rms"],
            "support_state_change": crossings["support_state_change_rms"],
            "candidate_mode_switch": crossings["candidate_transition_law_switch_rms"],
            "margin_change": crossings["margin_change_rms"],
            "capability_decline_or_recovery": min(
                value
                for value in (crossings["prediction_change_fraction"], crossings["capability_change_abs"])
                if value is not None
            ) if any(value is not None for value in (crossings["prediction_change_fraction"], crossings["capability_change_abs"])) else None,
        }
        structural = ordered["support_state_change"] is not None
        law = ordered["candidate_mode_switch"] is not None
        readout = ordered["capability_decline_or_recovery"] is not None
        if sum((structural, law, readout)) > 1:
            classification = "combined_mechanism"
        elif structural:
            classification = "support_state_jump"
        elif law:
            classification = "transition_law_switch"
        elif readout:
            classification = "capability_readout_crossing"
        else:
            classification = "no_persistent_breakpoint_detected"
        results.append(
            {
                "entry_id": row["entry_id"],
                "window_id": row["window_id"],
                "categories": row["categories"],
                "development_stable_window_count": len(development),
                "classification": classification,
                "earliest_order_relative_h": ordered,
                "signal_crossings_relative_h": crossings,
                "thresholds": thresholds,
                "candidate_mode_switch_is_observed_mode": False,
                "right_difference_used_only_for_adjudication": True,
            }
        )
    material = {
        "schema": "nanogpt-support-breakpoint-adjudication-v1",
        "status": "PASS",
        "cache_sha256": cache["cache_sha256"],
        "protocol_sha256": file_sha256(protocol_path),
        "event_window_count": len(results),
        "results": results,
    }
    result = {**material, "adjudication_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result
