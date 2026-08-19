from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import CONFIRMATION_RUNS, DEVELOPMENT_RUNS


METHODS = (
    "f1_f3_f5_outcome",
    "all_competitor_gaps",
    "all_competitor_geometry",
    "all_competitor_combined",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    truth = np.asarray([row["truth_final_correct"] for row in rows], dtype=bool)
    prediction = np.asarray([row["predicted_final_correct"][method] for row in rows], dtype=bool)
    boundary = np.asarray([row["truth_boundary"] for row in rows], dtype=object)
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    tn = int(np.sum(~prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "count": len(rows),
        "correct_count": int(np.sum(prediction == truth)),
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "final_correct_precision": precision,
        "final_correct_recall": tpr,
        "final_correct_f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "transition_recall": {
            name: float(np.mean(prediction[boundary == name] == truth[boundary == name])) if np.any(boundary == name) else None
            for name in ("MAINTAIN_CORRECT", "CORRECT_TO_WRONG", "WRONG_TO_CORRECT", "MAINTAIN_WRONG")
        },
    }


def _equal(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict):
        return set(left) == set(right) and all(_equal(left[key], right[key], tolerance) for key in left)
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, bool)) and isinstance(right, (int, bool)):
        return left == right
    return abs(float(left) - float(right)) <= tolerance


def check(report_root: Path) -> dict[str, Any]:
    results = _read_json(report_root / "OUTCOME_RESULTS.json")
    repairs = _read_json(report_root / "OUTCOME_REPAIRS.json")
    decision = _read_json(report_root / "DECISION.json")
    feature = _read_json(report_root / "FEATURE_MANIFEST.json")
    rows = _read_rows(report_root / "OUTCOME_PREDICTION_LEDGER.jsonl.gz")
    sources = _read_rows(report_root / "SOURCE_SECTION_LEDGER.jsonl.gz")
    with np.load(report_root / "COMPETITOR_COORDINATES.npz", allow_pickle=False) as payload:
        shapes = {
            "record_ids": tuple(payload["record_ids"].shape),
            "gaps": tuple(payload["all_competitor_gaps"].shape),
            "geometry": tuple(payload["all_competitor_geometry"].shape),
        }
        finite = bool(
            np.all(np.isfinite(payload["all_competitor_gaps"]))
            and np.all(np.isfinite(payload["all_competitor_geometry"]))
        )
    run_sets = {
        "development": set(DEVELOPMENT_RUNS),
        "confirmation": set(CONFIRMATION_RUNS),
        "all_runs": set(DEVELOPMENT_RUNS) | set(CONFIRMATION_RUNS),
    }
    metric_checks: dict[str, bool] = {}
    repair_checks: dict[str, bool] = {}
    for split_name, run_ids in run_sets.items():
        split = [row for row in rows if row["entry_id"] in run_ids]
        subsets = {
            "overall": split,
            "severe_conflict": [row for row in split if row["severe_conflict"]],
            "group_level_remainder_311": [row for row in split if row["group_level_remainder_311"]],
        }
        for method in METHODS:
            for subset_name, subset in subsets.items():
                recomputed = _metrics(subset, method)
                metric_checks[f"{split_name}:{method}:{subset_name}"] = _equal(
                    recomputed, results["metrics"][split_name][method][subset_name]
                )
                baseline = np.asarray(
                    [row["predicted_final_correct"]["f1_f3_f5_outcome"] for row in subset], dtype=bool
                )
                prediction = np.asarray([row["predicted_final_correct"][method] for row in subset], dtype=bool)
                truth = np.asarray([row["truth_final_correct"] for row in subset], dtype=bool)
                fixed = int(np.sum((baseline != truth) & (prediction == truth)))
                broken = int(np.sum((baseline == truth) & (prediction != truth)))
                repair_checks[f"{split_name}:{method}:{subset_name}"] = repairs["repairs"][split_name][method][subset_name] == {
                    "fixed_baseline_errors": fixed,
                    "newly_broken_baseline_answers": broken,
                    "net_repairs": fixed - broken,
                }
    same_run = sum(row["entry_id"] in set(row["neighbor_entry_ids"]) for row in rows)
    confirmation_isolated = all(
        set(row["neighbor_entry_ids"]).issubset(set(DEVELOPMENT_RUNS))
        for row in rows if row["entry_id"] in set(CONFIRMATION_RUNS)
    )
    checks = {
        "record_count_exact": len(rows) == 15264,
        "remainder_count_exact": sum(row["group_level_remainder_311"] for row in rows) == decision["remainder_count"] == 311,
        "source_section_count_exact": len(sources) == 72,
        "all_sources_use_alpha_zero": all(abs(float(row["alpha_value_used"])) <= 1e-12 for row in sources),
        "coordinate_shapes_exact": shapes == {"record_ids": (15264,), "gaps": (15264, 23), "geometry": (15264, 115)},
        "coordinates_finite": finite,
        "raw_class_identity_not_numeric": not feature["raw_class_identity_used_as_numeric_coordinate"],
        "post_outcome_inputs_empty": feature["post_outcome_input_fields"] == [],
        "curve_metrics_not_used": not decision["curve_metrics_used_for_decision"],
        "no_same_run_neighbors": same_run == 0,
        "confirmation_history_development_only": confirmation_isolated,
        "all_metrics_recomputed": all(metric_checks.values()),
        "all_repairs_recomputed": all(repair_checks.values()),
    }
    result = {
        "schema": "nanogpt-competitor-handoff-outcome-independent-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "coordinate_shapes": {key: list(value) for key, value in shapes.items()},
        "same_run_neighbor_count": same_run,
        "metric_checks": metric_checks,
        "repair_checks": repair_checks,
    }
    (report_root / "INDEPENDENT_CHECK.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"INDEPENDENT_CHECK_FAILED:{checks}")
    return result
