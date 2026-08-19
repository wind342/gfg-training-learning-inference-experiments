from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
    FACTOR_RECORDS,
    binary_metrics,
    file_sha256,
    response_metrics,
)


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
    checks: list[str] = []
    freeze = read_json(report_root / "EXPERIMENT_FREEZE.json")
    audit = read_json(report_root / "AVAILABILITY_AUDIT.json")
    selection = read_json(report_root / "CANDIDATE_SELECTION.json")
    results = read_json(report_root / "RESULTS.json")
    ledger = read_rows(report_root / "CONFIRMATION_LEDGER.jsonl.gz")
    sources = read_rows(report_root / "SOURCE_OBJECT_LEDGER.jsonl.gz")
    require = lambda condition, message: None if condition else (_ for _ in ()).throw(RuntimeError(message))

    require(tuple(freeze["development_runs"]) == DEVELOPMENT_RUNS, "DEVELOPMENT_SPLIT_DRIFT")
    require(tuple(freeze["confirmation_runs"]) == CONFIRMATION_RUNS, "CONFIRMATION_SPLIT_DRIFT")
    require(selection["confirmation_results_seen_during_selection"] is False, "SELECTION_LEAKAGE_FLAG")
    checks.append("frozen_split_and_selection_boundary")
    require(audit["status"] == "PASS", "AVAILABILITY_NOT_PASS")
    require(not audit["post_response_inputs_used"], "POST_RESPONSE_INPUT_USED")
    require(not audit["run_or_step_identity_used_as_feature"], "IDENTITY_FEATURE_USED")
    require(not audit["current_alpha_positive_probe_used_as_feature"], "CURRENT_PROBE_USED")
    checks.append("input_time_boundary")

    for row in sources:
        for path_key, digest_key in (
            ("transition_path", "transition_sha256"),
            ("state_path", "state_sha256"),
            ("response_section_path", "response_section_sha256"),
            ("evaluation_input_path", "evaluation_input_file_sha256"),
            ("batch_input_path", "batch_input_file_sha256"),
            ("batch_target_path", "batch_target_file_sha256"),
        ):
            require(file_sha256(Path(row[path_key])) == row[digest_key], f"SOURCE_HASH_MISMATCH:{path_key}")
    checks.append("all_source_file_hashes")

    with gzip.open(FACTOR_RECORDS, "rt", encoding="utf-8") as handle:
        record_entry = {row["record_id"]: row["entry_id"] for row in map(json.loads, handle)}
    require(len(ledger) == 5088, f"CONFIRMATION_LEDGER_CARDINALITY:{len(ledger)}")
    for row in ledger:
        require(row["entry_id"] in CONFIRMATION_RUNS, "NON_CONFIRMATION_TARGET")
        require(row["neighbor_entries_are_development_only"], "NEIGHBOR_BOUNDARY_FLAG")
        for field in ("X3_neighbor_record_ids", "X3_plus_q_neighbor_record_ids"):
            require(len(row[field]) == 64, f"NEIGHBOR_COUNT:{field}")
            require(all(record_entry[value] in DEVELOPMENT_RUNS for value in row[field]), f"NEIGHBOR_RUN_LEAKAGE:{field}")
    checks.append("confirmation_neighbor_run_isolation")

    labels = np.asarray([row["true_severe_conflict"] for row in ledger], dtype=bool)
    reported_binary = results["confirmation_branch_risk"]
    for name, field in (("X3", "X3_risk"), ("X3_plus_q", "X3_plus_q_risk")):
        scores = np.asarray([row[field] for row in ledger], dtype=np.float64)
        rebuilt = binary_metrics(labels, scores, 0.5)
        for metric in ("roc_auc", "pr_auc", "brier", "prevalence"):
            require(close(rebuilt[metric], reported_binary[name][metric]), f"BINARY_METRIC_MISMATCH:{name}:{metric}")
    checks.append("confirmation_branch_metrics_recomputed")

    curves = np.asarray([row["true_response_curve"] for row in ledger], dtype=np.float64)
    margin0 = np.asarray([row["margin0"] for row in ledger], dtype=np.float64)
    fields = {
        "X3": "X3_response_prediction",
        "X3_plus_q": "X3_plus_q_response_prediction",
        "oracle_same_true_branch_X3_plus_q": "oracle_response_prediction",
    }
    for name, field in fields.items():
        predictions = np.asarray([row[field] for row in ledger], dtype=np.float64)
        rebuilt = response_metrics(curves, predictions, margin0, labels)
        reported = results["confirmation_response"][name]
        for subset in ("overall", "severe_conflict"):
            for metric in ("curve_rmse", "endpoint_direction_accuracy", "boundary_accuracy"):
                require(close(rebuilt[subset][metric], reported[subset][metric]), f"RESPONSE_METRIC_MISMATCH:{name}:{subset}:{metric}")
    checks.append("confirmation_response_metrics_recomputed")

    require(all(row["selected_q_name"] == selection["selected_coordinate"] for row in ledger), "SELECTED_Q_DRIFT")
    require(all(row["selected_q_value"] is not None for row in ledger), "SELECTED_Q_MISSING")
    checks.append("selected_coordinate_identity_and_coverage")
    result = {
        "schema": "nanogpt-local-branch-coordinate-independent-recomputation-v1",
        "status": "PASS",
        "check_count": len(checks),
        "checks": checks,
        "confirmation_record_count": len(ledger),
        "source_section_count": len(sources),
        "future_leakage_detected": False,
        "confirmation_run_neighbor_leakage_detected": False,
    }
    write_json(report_root / "INDEPENDENT_RECOMPUTATION.json", result)
    return result


__all__ = ["check"]
