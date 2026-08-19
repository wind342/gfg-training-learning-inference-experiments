from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import SPACES, TASKS, binary_metrics, file_sha256, response_metrics


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def close(left: float | None, right: float | None, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def check(report_root: Path) -> dict[str, Any]:
    freeze = read_json(report_root / "EXPERIMENT_FREEZE.json")
    results = read_json(report_root / "RESULTS.json")
    audit = read_json(report_root / "AVAILABILITY_AUDIT.json")
    feature_manifest = read_json(report_root / "FEATURE_MANIFEST.json")
    ledger = read_rows(report_root / "LEFT_HISTORY_LEDGER.jsonl.gz")
    record_index = read_rows(report_root / "RECORD_INDEX.jsonl.gz")
    source_objects = read_rows(report_root / "SOURCE_OBJECT_LEDGER.jsonl.gz")
    checks: list[str] = []

    for source, expected in freeze["source_hashes"].items():
        if file_sha256(Path(source)) != expected:
            raise RuntimeError(f"SOURCE_HASH_MISMATCH:{source}")
        checks.append(f"source:{Path(source).name}")
    if audit["status"] != "PASS" or audit["prohibited_current_response_inputs_used"]:
        raise RuntimeError("AVAILABILITY_OR_LEAKAGE_AUDIT_INVALID")
    checks.append("availability_and_leakage_audit")
    prohibited = ("run_id", "entry_id", "optimizer_step", "section_id", "record_id", "phase", "alpha", "future", "response")
    violations = [
        name
        for names in feature_manifest["feature_names"].values()
        for name in names
        if any(token in name.lower() for token in prohibited)
    ]
    if violations:
        raise RuntimeError(f"PROHIBITED_FEATURE_NAME:{violations[0]}")
    checks.append("feature_name_leakage_scan")
    if len(ledger) != 15264 or len(record_index) != 15264 or len(source_objects) != 72:
        raise RuntimeError("LEDGER_CARDINALITY_MISMATCH")
    checks.append("ledger_cardinality")

    index_entries = np.asarray([row["entry_id"] for row in record_index], dtype=object)
    for row in ledger:
        if len(row["neighbor_source_record_indices_X4"]) != 64:
            raise RuntimeError(f"NEIGHBOR_CARDINALITY_INVALID:{row['record_id']}")
        entry = row["entry_id"]
        if any(index_entries[int(value)] == entry for value in row["neighbor_source_record_indices_X4"]):
            raise RuntimeError(f"HELD_OUT_RUN_USED_AS_NEIGHBOR:{row['record_id']}")
    checks.append("complete_leave_one_run_out_neighbor_isolation")

    reported_tasks = results["task_results"]
    for task in TASKS:
        valid_rows = [row for row in ledger if row["labels"][task] is not None]
        labels = np.asarray([bool(row["labels"][task]) for row in valid_rows], dtype=bool)
        for space in SPACES:
            scores = np.asarray([float(row["risks"][space][task]) for row in valid_rows], dtype=np.float64)
            thresholds = (
                np.asarray([float(row["main_thresholds"][space]) for row in valid_rows], dtype=np.float64)
                if task == "competitor_switch"
                else np.full(len(valid_rows), 0.5, dtype=np.float64)
            )
            rebuilt = binary_metrics(labels, scores, thresholds)
            reported = reported_tasks[task][space]
            for field in ("roc_auc", "pr_auc", "brier", "prevalence"):
                if not close(rebuilt[field], reported[field]):
                    raise RuntimeError(f"METRIC_RECOMPUTATION_MISMATCH:{task}:{space}:{field}")
            if task == "competitor_switch":
                for field in ("threshold_recall", "threshold_fpr", "threshold_precision"):
                    if not close(rebuilt[field], reported[field]):
                        raise RuntimeError(f"THRESHOLD_RECOMPUTATION_MISMATCH:{space}:{field}")
    checks.append("all_task_auc_ap_brier_recomputed")

    main_rows = [row for row in ledger if row["labels"]["competitor_switch"] is not None]
    main_labels = np.asarray([bool(row["labels"]["competitor_switch"]) for row in main_rows], dtype=bool)
    for baseline_name in ("gap_only", "past_switch_count_only", "prevalence_only"):
        scores = np.asarray([float(row["baseline_risks"][baseline_name]) for row in main_rows], dtype=np.float64)
        thresholds = np.asarray(
            [float(row["baseline_thresholds"][baseline_name]) for row in main_rows], dtype=np.float64
        )
        rebuilt = binary_metrics(main_labels, scores, thresholds)
        reported = reported_tasks["competitor_switch"][baseline_name]
        for field in (
            "roc_auc",
            "pr_auc",
            "brier",
            "prevalence",
            "threshold_recall",
            "threshold_fpr",
            "threshold_precision",
        ):
            if not close(rebuilt[field], reported[field]):
                raise RuntimeError(f"BASELINE_RECOMPUTATION_MISMATCH:{baseline_name}:{field}")
    checks.append("frozen_scalar_baselines_recomputed")

    curves = np.asarray([row["true_response_displacement"] for row in ledger], dtype=np.float64)
    margin0 = np.asarray([row["margin0"] for row in ledger], dtype=np.float64)
    severe = np.asarray([bool(row["labels"]["severe_conflict"]) for row in ledger], dtype=bool)
    prediction_fields = {
        "ordinary_X4": "ordinary_X4_response_prediction",
        "oracle_same_competitor_switch_branch_X4": "oracle_same_switch_branch_response_prediction",
        "executable_routed_X4": "executable_routed_response_prediction",
    }
    for result_name, ledger_name in prediction_fields.items():
        prediction = np.asarray([row[ledger_name] for row in ledger], dtype=np.float64)
        rebuilt = response_metrics(curves, prediction, margin0, severe)
        reported = results["response_prediction"][result_name]
        for subset in ("overall", "severe_conflict"):
            for field in ("curve_rmse", "endpoint_direction_accuracy", "boundary_accuracy"):
                if not close(rebuilt[subset][field], reported[subset][field]):
                    raise RuntimeError(f"RESPONSE_RECOMPUTATION_MISMATCH:{result_name}:{subset}:{field}")
    checks.append("response_metrics_recomputed")
    if not all(row["target_only_objects_present_and_excluded"] for row in source_objects):
        raise RuntimeError("TARGET_ONLY_PARTITION_AUDIT_INCOMPLETE")
    checks.append("target_only_partition_excluded")

    result = {
        "schema": "nanogpt-native-prebranch-independent-recomputation-v1",
        "status": "PASS",
        "check_count": len(checks),
        "checks": checks,
        "recomputed_record_count": len(ledger),
        "recomputed_run_count": len(set(index_entries.tolist())),
        "future_leakage_detected": False,
        "test_run_neighbor_leakage_detected": False,
    }
    write_json(report_root / "INDEPENDENT_RECOMPUTATION.json", result)
    return result


__all__ = ["check"]
