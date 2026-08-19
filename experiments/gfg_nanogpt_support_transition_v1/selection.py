from __future__ import annotations

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


LEGACY_MATCH_FEATURES = (
    "validation_accuracy",
    "margin_q10",
    "margin_q50",
    "margin_std",
    "correct_probability_mean",
    "cyclic_spectral_top_fraction",
    "cyclic_spectral_entropy",
    "ln_f_parameter_rms",
    "ln_f_parameter_std",
    "ln_f_moment1_rms",
    "ln_f_moment2_rms",
    "ln_f_preconditioned_rms",
    "all_moment1_rms",
    "all_moment2_rms",
    "all_preconditioned_rms",
    "all_parameter_displacement_rms",
    "gradient_shock_impulse_300",
    "gradient_clip_fraction_300",
    "gradient_q90_300",
)


def _row_map(value: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["optimizer_step"]): row for row in value["rows"]}


def _inside_interval(step: int, interval: dict[str, Any]) -> bool:
    end = int(interval["recovery"] or interval["end"])
    return int(interval["start"]) <= step <= end


def _nearest_boundary_distance(step: int, intervals: list[dict[str, Any]]) -> int:
    boundaries = [
        int(value)
        for interval in intervals
        for value in (interval["start"], interval.get("recovery"))
        if value is not None
    ]
    return min(abs(step - value) for value in boundaries)


def _numeric_features(
    legacy_row: dict[str, Any],
    csrg_row: dict[str, Any],
    csrg_names: tuple[str, ...],
) -> dict[str, float]:
    source = legacy_row["features"]
    result: dict[str, float] = {}
    for name in LEGACY_MATCH_FEATURES:
        value = source.get(name)
        require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"CST_ANCHOR_LEGACY_FEATURE_INVALID:{name}",
        )
        result["legacy:" + name] = float(value)
    for name in csrg_names:
        value = csrg_row["features"].get(name)
        require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"CST_ANCHOR_CSRG_FEATURE_INVALID:{name}",
        )
        result["csrg:" + name] = float(value)
    return result


def freeze_anchor_selection(
    *,
    stability_feature_cache: Path,
    csrg_feature_index: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze four outcome-defined anchor categories for every historical run.

    Historical outcomes define the categories because all thirteen runs are
    already development evidence.  They never enter the matching distance or
    any later transition-model feature.  No new causal-branch result is read.
    """

    legacy = read_json(stability_feature_cache)
    csrg = read_json(csrg_feature_index)
    contract_sha = file_sha256(contract_path)
    csrg_names = tuple(sorted(csrg["feature_definitions"]["all_derived_feature_names"]))
    require(len(csrg_names) == 34, "CST_CSRG_MATCH_FEATURE_COUNT_INVALID")
    csrg_by_entry = {row["entry_id"]: row for row in csrg["entries"]}
    require(len(legacy["runs"]) == len(csrg_by_entry) == 13, "CST_ENTRY_COUNT_INVALID")

    merged: dict[str, dict[int, dict[str, Any]]] = {}
    all_vectors: list[dict[str, float]] = []
    for run in legacy["runs"]:
        entry_id = str(run["entry_id"])
        legacy_rows = _row_map(run)
        csrg_rows = _row_map(csrg_by_entry[entry_id])
        require(set(legacy_rows) == set(csrg_rows) == set(range(100, 10001, 100)), "CST_GRID_MISMATCH")
        merged[entry_id] = {}
        for step in sorted(legacy_rows):
            values = _numeric_features(legacy_rows[step], csrg_rows[step], csrg_names)
            merged[entry_id][step] = {
                "legacy": legacy_rows[step],
                "csrg": csrg_rows[step],
                "values": values,
            }
            if step >= int(run["formation_transition_step"]):
                all_vectors.append(values)

    feature_names = tuple(sorted(all_vectors[0]))
    matrix = np.asarray([[row[name] for name in feature_names] for row in all_vectors], dtype=np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    admitted = scales > 1e-12
    require(int(admitted.sum()) >= 20, "CST_MATCH_FEATURES_DEGENERATE")
    admitted_names = tuple(name for name, keep in zip(feature_names, admitted) if keep)
    means = means[admitted]
    scales = scales[admitted]

    def standardized(row: dict[str, float]) -> np.ndarray:
        return (np.asarray([row[name] for name in admitted_names], dtype=np.float64) - means) / scales

    anchors: list[dict[str, Any]] = []
    for run in legacy["runs"]:
        entry_id = str(run["entry_id"])
        formation = int(run["formation_transition_step"])
        intervals = [
            value
            for value in run["stability_intervals"]
            if int(value["start"]) >= formation + 100 and value.get("recovery") is not None
        ]
        require(bool(intervals), f"CST_NO_COMPLETE_POSTFORMATION_DECLINE:{entry_id}")
        severe = min(intervals, key=lambda row: (float(row["minimum"]), int(row["start"])))
        pre_step = int(severe["start"]) - 100
        recovery_step = int(severe["recovery"])
        require(pre_step >= formation, f"CST_PREDECLINE_BEFORE_FORMATION:{entry_id}")
        require(pre_step in merged[entry_id] and recovery_step in merged[entry_id], "CST_ANCHOR_GRID_MISSING")

        stable_candidates = []
        for step, row in merged[entry_id].items():
            accuracy = float(row["legacy"]["features"]["validation_accuracy"])
            if step < formation or accuracy < 0.90:
                continue
            if any(_inside_interval(step, interval) for interval in run["stability_intervals"]):
                continue
            future_onsets = [int(value["start"]) for value in run["stability_intervals"] if int(value["start"]) > step]
            stable_forward = min(future_onsets) - step if future_onsets else 10000 - step
            if stable_forward < 500:
                continue
            stable_candidates.append((step, _nearest_boundary_distance(step, run["stability_intervals"]), stable_forward))
        require(bool(stable_candidates), f"CST_NO_SUSTAINED_STABLE_ANCHOR:{entry_id}")
        stable_step, stable_distance, stable_forward = max(
            stable_candidates,
            key=lambda row: (row[1], row[2], -row[0]),
        )

        pre_vector = standardized(merged[entry_id][pre_step]["values"])
        control_candidates: list[tuple[float, int]] = []
        for step, row in merged[entry_id].items():
            if step < formation or step in {stable_step, pre_step, recovery_step} or step >= 10000:
                continue
            current_accuracy = float(row["legacy"]["features"]["validation_accuracy"])
            next_accuracy = float(merged[entry_id][step + 100]["legacy"]["features"]["validation_accuracy"])
            if current_accuracy < 0.90 or next_accuracy < 0.90:
                continue
            if any(_inside_interval(step, interval) or int(interval["start"]) == step + 100 for interval in run["stability_intervals"]):
                continue
            distance = float(np.sqrt(np.mean((standardized(row["values"]) - pre_vector) ** 2)))
            control_candidates.append((distance, step))
        require(bool(control_candidates), f"CST_NO_MATCHED_CONTROL:{entry_id}")
        control_distance, control_step = min(control_candidates, key=lambda row: (row[0], row[1]))

        source_rows = {
            "post_formation_sustained_stable": (stable_step, {"boundary_distance": stable_distance, "forward_stable_steps": stable_forward}),
            "immediate_pre_severe_decline": (pre_step, {"decline_start": int(severe["start"]), "decline_minimum": float(severe["minimum"])}),
            "matched_non_declining_control": (control_step, {"standardized_rms_distance": control_distance, "matched_to_step": pre_step}),
            "recovery_after_same_severe_decline": (recovery_step, {"decline_start": int(severe["start"]), "decline_minimum": float(severe["minimum"])}),
        }
        for category, (step, rationale) in source_rows.items():
            selected = merged[entry_id][step]
            anchors.append(
                {
                    "anchor_category": category,
                    "entry_id": entry_id,
                    "evaluation_occurrence_id": selected["legacy"]["evaluation_occurrence_id"],
                    "optimizer_step": step,
                    "rationale": rationale,
                    "source_csrg_occurrence_ids": selected["csrg"]["analysis_occurrence_ids"],
                    "source_support_object_ids": selected["csrg"]["support_object_ids"],
                }
            )

    require(len(anchors) == 52, "CST_ANCHOR_COUNT_INVALID")
    require(len({(row["entry_id"], row["anchor_category"]) for row in anchors}) == 52, "CST_ANCHOR_DUPLICATE")
    material = {
        "anchors": sorted(anchors, key=lambda row: (row["entry_id"], row["anchor_category"])),
        "category_outcomes_used_only_for_selection": True,
        "contract_sha256": contract_sha,
        "csrg_feature_index_sha256": file_sha256(csrg_feature_index),
        "distance_feature_names": list(admitted_names),
        "distance_feature_standardization": "global post-formation mean and population standard deviation computed before causal execution",
        "entry_id_used_in_distance": False,
        "future_outcome_admitted_as_transition_feature": False,
        "new_branch_results_read": False,
        "optimizer_step_used_in_distance": False,
        "schema": "nanogpt-support-transition-anchor-selection-v1",
        "stability_feature_cache_sha256": file_sha256(stability_feature_cache),
        "status": "FROZEN_BEFORE_GPU_BRANCH_EXECUTION",
    }
    result = {**material, "selection_sha256": payload_sha256(material)}
    write_json(output_path, result)
    return result


def independent_replay_steps(contract_sha256: str, entry_ids: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry_id in sorted(entry_ids):
        ranked = sorted(
            range(100, 10001, 100),
            key=lambda step: hashlib.sha256(
                f"{contract_sha256}\0{entry_id}\0{step}".encode("utf-8")
            ).hexdigest(),
        )
        result[entry_id] = ranked[0]
    return result
