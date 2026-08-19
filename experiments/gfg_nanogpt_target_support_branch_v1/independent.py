from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import (
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
)


METHODS = ("f1_f3_f5", "target_support", "competitor_boundary", "target_support_competitor")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    truth = np.asarray([row["true_curve"] for row in rows], dtype=np.float64)
    prediction = np.asarray([row["predicted_curves"][method] for row in rows], dtype=np.float64)
    start = np.asarray([row["margin0"] for row in rows], dtype=np.float64)
    truth_endpoint = truth[:, -1]
    prediction_endpoint = prediction[:, -1]
    truth_start_correct = start > 0.0
    truth_end_correct = start + truth_endpoint > 0.0
    prediction_end_correct = start + prediction_endpoint > 0.0
    unchanged = truth_start_correct == truth_end_correct
    wrong_to_correct = (~truth_start_correct) & truth_end_correct
    return {
        "count": len(rows),
        "curve_rmse": float(np.sqrt(np.mean(np.square(prediction - truth)))),
        "endpoint_direction_accuracy": float(np.mean(np.sign(prediction_endpoint) == np.sign(truth_endpoint))),
        "boundary_accuracy": float(np.mean(prediction_end_correct == truth_end_correct)),
        "unchanged_false_crossing_rate": float(np.mean(prediction_end_correct[unchanged] != truth_start_correct[unchanged])) if np.any(unchanged) else None,
        "wrong_to_correct_recall": float(np.mean(prediction_end_correct[wrong_to_correct])) if np.any(wrong_to_correct) else None,
    }


def _same(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
        return int(left) == int(right)
    return abs(float(left) - float(right)) <= tolerance


def check(report_root: Path) -> dict[str, Any]:
    metrics = _read_json(report_root / "TARGET_SUPPORT_RESPONSE_RESULTS.json")["metrics"]
    decision = _read_json(report_root / "DECISION.json")
    feature = _read_json(report_root / "FEATURE_MANIFEST.json")
    source_rows = _read_rows(report_root / "SOURCE_SECTION_LEDGER.jsonl.gz")
    prediction_rows = _read_rows(report_root / "PREDICTION_LEDGER.jsonl.gz")
    pair_rows = _read_rows(report_root / "TARGET_PAIR_DIAGNOSTIC_LEDGER.jsonl.gz")
    with np.load(report_root / "TARGET_COORDINATES.npz", allow_pickle=False) as payload:
        coordinate_shapes = {
            "record_ids": tuple(payload["record_ids"].shape),
            "support": tuple(payload["target_support"].shape),
            "boundary": tuple(payload["competitor_boundary"].shape),
        }
        finite_coordinates = bool(
            np.all(np.isfinite(payload["target_support"]))
            and np.all(np.isfinite(payload["competitor_boundary"]))
        )
    run_sets = {
        "development": set(DEVELOPMENT_RUNS),
        "confirmation": set(CONFIRMATION_RUNS),
        "all_runs": set(DEVELOPMENT_RUNS) | set(CONFIRMATION_RUNS),
    }
    metric_checks: dict[str, bool] = {}
    for split_name, run_ids in run_sets.items():
        split = [row for row in prediction_rows if row["entry_id"] in run_ids]
        masks = {
            "overall": split,
            "severe_conflict": [row for row in split if row["severe_conflict"]],
            "group_level_remainder_311": [row for row in split if row["group_level_remainder_311"]],
        }
        for method in METHODS:
            for subset_name, rows in masks.items():
                recomputed = _metrics(rows, method)
                reported = metrics[split_name][method][subset_name]
                metric_checks[f"{split_name}:{method}:{subset_name}"] = all(
                    _same(recomputed[key], reported[key]) for key in recomputed
                )
    same_run_neighbor_count = sum(
        row["entry_id"] in set(row["neighbor_entry_ids"]) for row in prediction_rows
    )
    confirmation_history = set(DEVELOPMENT_RUNS)
    confirmation_isolated = all(
        set(row["neighbor_entry_ids"]).issubset(confirmation_history)
        for row in prediction_rows
        if row["entry_id"] in set(CONFIRMATION_RUNS)
    )
    remainder_prediction = sum(bool(row["group_level_remainder_311"]) for row in prediction_rows)
    remainder_pairs = sum(bool(row["group_level_remainder_311"]) for row in pair_rows)
    checks = {
        "source_section_count_exact": len(source_rows) == 72,
        "all_sources_use_only_alpha_zero": all(
            row["alpha_index_used"] == 1 and abs(float(row["alpha_value_used"])) <= 1e-12 for row in source_rows
        ),
        "future_response_fields_empty": feature["future_response_fields_used"] == [],
        "coordinate_shapes_exact": coordinate_shapes == {
            "record_ids": (15264,), "support": (15264, 19), "boundary": (15264, 40)
        },
        "coordinates_finite": finite_coordinates,
        "prediction_count_exact": len(prediction_rows) == 15264,
        "pair_count_exact": len(pair_rows) == 14896,
        "remainder_count_exact": remainder_prediction == remainder_pairs == decision["remainder_count"] == 311,
        "no_same_run_neighbors": same_run_neighbor_count == 0,
        "confirmation_history_development_only": confirmation_isolated,
        "all_reported_metrics_recomputed": all(metric_checks.values()),
        "post_update_not_executable_input": not decision["post_update_evidence_used_as_executable_input"],
    }
    result = {
        "schema": "nanogpt-target-support-branch-independent-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "coordinate_shapes": {key: list(value) for key, value in coordinate_shapes.items()},
        "metric_checks": metric_checks,
        "same_run_neighbor_count": same_run_neighbor_count,
    }
    (report_root / "INDEPENDENT_CHECK.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"INDEPENDENT_CHECK_FAILED:{checks}")
    return result
