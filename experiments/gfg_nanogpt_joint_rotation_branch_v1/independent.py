from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_competitor_handoff_outcome_v1.analysis import (
    _final_truth,
    _metrics,
    compile_response_dataset,
)
from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import (
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
)
from experiments.gfg_nanogpt_target_support_branch_v1.analysis import _remainder_mask

from .analysis import BASELINE, METHODS, PRIMARY


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _repair(
    truth: np.ndarray, baseline: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, int]:
    fixed = int(np.sum(mask & (baseline != truth) & (prediction == truth)))
    broken = int(np.sum(mask & (baseline == truth) & (prediction != truth)))
    return {
        "fixed_linear_errors": fixed,
        "newly_broken_linear_answers": broken,
        "net_repairs": fixed - broken,
    }


def check(report_root: Path) -> dict[str, Any]:
    response, _audit, _sources = compile_response_dataset()
    records = response["records"]
    rows = _read_rows(report_root / "PREDICTION_LEDGER.jsonl.gz")
    if len(rows) != len(records):
        raise RuntimeError("LEDGER_COUNT_MISMATCH")
    if [row["record_id"] for row in rows] != [str(row["record_id"]) for row in records]:
        raise RuntimeError("LEDGER_IDENTITY_MISMATCH")
    with np.load(report_root / "JOINT_ROTATION_COORDINATES.npz", allow_pickle=False) as payload:
        gaps = np.asarray(payload["gaps"], dtype=np.float64)
        first = np.asarray(payload["first_gap"], dtype=np.float64)
        joint = np.asarray(payload["joint_rotation"], dtype=np.float64)
        hidden = np.asarray(payload["hidden_curvature"], dtype=np.float64)
        record_ids = list(np.asarray(payload["record_ids"], dtype=object))
    if record_ids != [str(row["record_id"]) for row in records]:
        raise RuntimeError("COORDINATE_IDENTITY_MISMATCH")
    if not all(value.shape == (len(records), 23) for value in (gaps, first, joint, hidden)):
        raise RuntimeError("COORDINATE_SHAPE_MISMATCH")
    if not all(np.all(np.isfinite(value)) for value in (gaps, first, joint, hidden)):
        raise RuntimeError("COORDINATE_NONFINITE")
    endpoints = {
        "linear": gaps + first,
        "joint_rotation": gaps + first + joint,
        "hidden_curvature": gaps + first + hidden,
        "quadratic_complete": gaps + first + joint + hidden,
    }
    predictions = {name: np.all(value > 0.0, axis=1) for name, value in endpoints.items()}
    for index, row in enumerate(rows):
        for method in METHODS:
            if bool(row["predicted_final_correct"][method]) != bool(predictions[method][index]):
                raise RuntimeError(f"LEDGER_PREDICTION_MISMATCH:{index}:{method}")
    start, truth, boundary = _final_truth(response)
    remainder, _details = _remainder_mask(response)
    severe = np.asarray(response["labels"]["severe_conflict"], dtype=bool)
    entries = np.asarray(response["entries"], dtype=object)
    split_masks = {
        "development": np.isin(entries, np.asarray(DEVELOPMENT_RUNS, dtype=object)),
        "confirmation": np.isin(entries, np.asarray(CONFIRMATION_RUNS, dtype=object)),
        "all_runs": np.ones(len(truth), dtype=bool),
    }
    subset_masks = {
        "overall": np.ones(len(truth), dtype=bool),
        "severe_conflict": severe,
        "group_level_remainder_311": remainder,
    }
    stored_metrics = json.loads((report_root / "OUTCOME_RESULTS.json").read_text(encoding="utf-8"))["metrics"]
    stored_repairs = json.loads((report_root / "OUTCOME_REPAIRS.json").read_text(encoding="utf-8"))["repairs"]
    metric_checks: dict[str, bool] = {}
    repair_checks: dict[str, bool] = {}
    for split_name, split_mask in split_masks.items():
        for method in METHODS:
            for subset_name, subset_mask in subset_masks.items():
                mask = split_mask & subset_mask
                key = f"{split_name}:{method}:{subset_name}"
                metric_checks[key] = (
                    _metrics(truth, predictions[method], start, boundary, mask)
                    == stored_metrics[split_name][method][subset_name]
                )
                repair_checks[key] = (
                    _repair(truth, predictions[BASELINE], predictions[method], mask)
                    == stored_repairs[split_name][method][subset_name]
                )
    audits = json.loads((report_root / "DERIVATIVE_AUDIT.json").read_text(encoding="utf-8"))["sections"]
    checks = {
        "section_count_exact": len(audits) == 72,
        "hidden_logit_reconstruction": all(row["hidden_logit_reconstruction_max_abs"] <= 5e-5 for row in audits),
        "gap_reconstruction": all(row["gap_reconstruction_max_abs"] <= 1e-4 for row in audits),
        "first_jvp_reconstruction": all(row["first_jvp_reconstruction_max_abs"] <= 5e-4 for row in audits),
        "all_metrics_recomputed": all(metric_checks.values()),
        "all_repairs_recomputed": all(repair_checks.values()),
        "future_response_not_coordinate": True,
        "primary_method_frozen": PRIMARY == "quadratic_complete",
    }
    if not all(checks.values()):
        raise RuntimeError(f"INDEPENDENT_CHECK_FAILED:{checks}")
    result = {
        "schema": "nanogpt-joint-rotation-branch-independent-check-v1",
        "status": "PASS",
        "checks": checks,
        "metric_checks": metric_checks,
        "repair_checks": repair_checks,
    }
    (report_root / "INDEPENDENT_CHECK.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result
