from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .runner import file_sha256


TRANSITIONS = ("MAINTAIN_CORRECT", "CORRECT_TO_WRONG", "MAINTAIN_WRONG", "WRONG_TO_CORRECT")
CONFIRMATION_RUNS = frozenset(
    {
        "entry-4ed462761347d6b87e61",
        "entry-d5b80ca9b9cd18fa343f",
        "entry-786d0a3628f6f791399f",
        "entry-481b86f81d58d496a687",
    }
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    transitions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_truth = truth[mask]
    selected_prediction = prediction[mask]
    selected_transitions = transitions[mask]
    correct = int(np.sum(selected_truth == selected_prediction))
    recalls = {
        name: float(
            np.mean(
                selected_prediction[selected_transitions == name]
                == selected_truth[selected_transitions == name]
            )
        )
        for name in TRANSITIONS
    }
    class_recalls = [
        float(np.mean(selected_prediction[selected_truth == value] == value))
        for value in (False, True)
    ]
    return {
        "count": int(np.count_nonzero(mask)),
        "correct_count": correct,
        "accuracy": float(np.mean(selected_truth == selected_prediction)),
        "balanced_accuracy": float(np.mean(class_recalls)),
        "transition_recall": recalls,
        "four_way_macro_recall": float(np.mean(list(recalls.values()))),
    }


def _matches(actual: dict[str, Any], declared: dict[str, Any]) -> bool:
    return (
        actual["count"] == int(declared["count"])
        and actual["correct_count"] == int(declared["correct_count"])
        and abs(actual["accuracy"] - float(declared["accuracy"])) <= 1e-15
        and abs(actual["balanced_accuracy"] - float(declared["balanced_accuracy"])) <= 1e-15
        and abs(actual["four_way_macro_recall"] - float(declared["four_way_macro_recall"])) <= 1e-15
        and all(
            abs(actual["transition_recall"][name] - float(declared["transition_recall"][name])) <= 1e-15
            for name in TRANSITIONS
        )
    )


def check(root: Path) -> dict[str, Any]:
    manifest = _read(root / "MANIFEST.json")
    failures = {
        name: {"expected": digest, "actual": file_sha256(root / name) if (root / name).is_file() else None}
        for name, digest in manifest["files"].items()
        if not (root / name).is_file() or file_sha256(root / name) != digest
    }
    rows = []
    with gzip.open(root / "BOUNDARY_PREDICTIONS.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    truth = np.asarray([bool(row["truth_final_correct"]) for row in rows], dtype=bool)
    prediction = np.asarray([bool(row["predicted_final_correct"]["quadratic_complete"]) for row in rows], dtype=bool)
    transitions = np.asarray([str(row["truth_transition"]) for row in rows], dtype=object)
    entries = np.asarray([str(row["entry_id"]) for row in rows], dtype=object)
    all_runs = _metrics(truth, prediction, transitions, np.ones(len(rows), dtype=bool))
    confirmation = _metrics(truth, prediction, transitions, np.isin(entries, list(CONFIRMATION_RUNS)))
    results = _read(root / "BOUNDARY_RESULTS.json")
    declared_all = results["metrics"]["all_runs"]["quadratic_complete"]
    declared_confirmation = results["metrics"]["confirmation"]["quadratic_complete"]
    audit = _read(root / "DERIVATIVE_AUDIT.json")
    checks = {
        "manifest_hashes": not failures,
        "record_count": len(rows) == int(results["record_count"]) == 15264,
        "record_identity_unique": len({row["record_id"] for row in rows}) == len(rows),
        "no_support_or_curve_prediction": results["response_curve_predicted"] is False and results["support_state_predicted"] is False,
        "all_targets_one_rule": all("difficult" not in key.lower() and "remainder" not in key.lower() for row in rows for key in row),
        "all_run_metrics": _matches(all_runs, declared_all),
        "confirmation_run_identity": set(entries[np.isin(entries, list(CONFIRMATION_RUNS))].tolist())
        == CONFIRMATION_RUNS,
        "confirmation_metrics": _matches(confirmation, declared_confirmation),
        "derivative_audit": audit["status"] == "PASS" and len(audit["sections"]) == 72,
        "verdict": results["verdict"] == "ACTUAL_UPDATE_BOUNDARY_PREDICTION_REPRODUCED",
    }
    return {
        "schema": "gfg-nanogpt-actual-update-boundary-independent-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hash_failures": failures,
        "record_count": len(rows),
        "truth_transition_counts": dict(Counter(transitions.tolist())),
        "recomputed": {"all_runs": all_runs, "confirmation": confirmation},
    }


__all__ = ["check"]
