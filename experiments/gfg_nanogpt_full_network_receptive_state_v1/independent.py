from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_competitor_handoff_outcome_v1.independent import _equal, _metrics
from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import CONFIRMATION_RUNS, DEVELOPMENT_RUNS

from .analysis import BASELINE, COMPONENTS, K, METHODS


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check(report_root: Path, require_central_direction_gate: bool = True) -> dict[str, Any]:
    result = _read_json(report_root / "OUTCOME_RESULTS.json")
    repairs = _read_json(report_root / "OUTCOME_REPAIRS.json")
    decision = _read_json(report_root / "DECISION.json")
    feature = _read_json(report_root / "FEATURE_MANIFEST.json")
    context = _read_json(report_root / "EXPERIMENT_CONTEXT.json")
    audits = _read_json(report_root / "SECTION_JVP_AUDIT.json")["sections"]
    ledger = _rows(report_root / "OUTCOME_PREDICTION_LEDGER.jsonl.gz")
    entries = np.asarray([row["entry_id"] for row in ledger], dtype=object)
    with np.load(report_root / "RECEPTIVE_COORDINATES.npz", allow_pickle=False) as payload:
        record_ids = np.asarray(payload["record_ids"]).astype(str)
        total = np.asarray(payload["total_gap_jvp"], dtype=np.float64)
        component = np.asarray(payload["component_gap_jvp"], dtype=np.float64)
    with np.load(report_root / "RETRIEVAL_NEIGHBORS.npz", allow_pickle=False) as payload:
        neighbors = {method: np.asarray(payload[method], dtype=np.int64) for method in METHODS}
    run_sets = {
        "development": set(DEVELOPMENT_RUNS),
        "confirmation": set(CONFIRMATION_RUNS),
        "all_runs": set(DEVELOPMENT_RUNS) | set(CONFIRMATION_RUNS),
    }
    metric_checks: dict[str, bool] = {}
    repair_checks: dict[str, bool] = {}
    for split_name, run_ids in run_sets.items():
        split = [row for row in ledger if row["entry_id"] in run_ids]
        subsets = {
            "overall": split,
            "severe_conflict": [row for row in split if row["severe_conflict"]],
            "group_level_remainder_311": [row for row in split if row["group_level_remainder_311"]],
        }
        for method in METHODS:
            for subset_name, subset in subsets.items():
                metric_checks[f"{split_name}:{method}:{subset_name}"] = _equal(
                    _metrics(subset, method), result["metrics"][split_name][method][subset_name]
                )
                truth = np.asarray([row["truth_final_correct"] for row in subset], dtype=bool)
                baseline = np.asarray([row["predicted_final_correct"][BASELINE] for row in subset], dtype=bool)
                prediction = np.asarray([row["predicted_final_correct"][method] for row in subset], dtype=bool)
                fixed = int(np.sum((baseline != truth) & (prediction == truth)))
                broken = int(np.sum((baseline == truth) & (prediction != truth)))
                repair_checks[f"{split_name}:{method}:{subset_name}"] = repairs["repairs"][split_name][method][subset_name] == {
                    "fixed_baseline_errors": fixed,
                    "newly_broken_baseline_answers": broken,
                    "net_repairs": fixed - broken,
                }
    same_run = {
        method: int(np.sum(entries[:, None] == entries[values]))
        for method, values in neighbors.items()
    }
    confirmation_mask = np.isin(entries, np.asarray(CONFIRMATION_RUNS, dtype=object))
    confirmation_outside = {
        method: int(np.sum(~np.isin(entries[values[confirmation_mask]], np.asarray(DEVELOPMENT_RUNS, dtype=object))))
        for method, values in neighbors.items()
    }
    checks = {
        "record_count_exact": len(ledger) == 15264,
        "coordinate_identity_order_exact": np.array_equal(record_ids, np.asarray([row["record_id"] for row in ledger])),
        "coordinate_shapes_exact": total.shape == (15264, 23) and component.shape == (15264, 23, len(COMPONENTS)),
        "coordinates_finite": bool(np.all(np.isfinite(total)) and np.all(np.isfinite(component))),
        "section_count_exact": len(audits) == 72,
        "coordinate_alpha_zero_only": all(row["alpha_values_used_for_coordinate"] == [0.0] for row in audits),
        "central_difference_validation_only": all(row["alpha_values_used_for_validation_only"] == [-0.125, 0.125] for row in audits),
        "alpha_zero_forward_gate": all(float(row["alpha_zero_forward_max_abs"]) <= 5e-5 for row in audits),
        "central_direction_values_valid": all(
            np.isfinite(float(row["jvp_central_difference_correlation"])) for row in audits
        ),
        "central_direction_gate": (
            all(float(row["jvp_central_difference_correlation"]) >= 0.98 for row in audits)
            if require_central_direction_gate
            else True
        ),
        "component_partition_exact": all(set(row["parameter_partition"].values()) == set(COMPONENTS) for row in audits),
        "neighbor_shape_exact": all(value.shape == (15264, K) for value in neighbors.values()),
        "no_same_run_neighbors": all(value == 0 for value in same_run.values()),
        "confirmation_history_development_only": all(value == 0 for value in confirmation_outside.values()),
        "diagnostic_status_disclosed": bool(feature["diagnostic_only"]) and not bool(feature["formal_predictor_input"]),
        "post_hoc_context_disclosed": context["evidence_status"] == "POST_HOC_MECHANISM_DIAGNOSTIC_ONLY",
        "curve_metrics_not_used": not decision["curve_metrics_used_for_decision"],
        "all_metrics_recomputed": all(metric_checks.values()),
        "all_repairs_recomputed": all(repair_checks.values()),
    }
    output = {
        "schema": "nanogpt-full-network-receptive-state-independent-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "same_run_neighbor_edge_count": same_run,
        "confirmation_neighbor_edge_outside_development": confirmation_outside,
        "metric_checks": metric_checks,
        "repair_checks": repair_checks,
    }
    (report_root / "INDEPENDENT_CHECK.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    if output["status"] != "PASS":
        raise RuntimeError(f"INDEPENDENT_CHECK_FAILED:{checks}")
    return output
