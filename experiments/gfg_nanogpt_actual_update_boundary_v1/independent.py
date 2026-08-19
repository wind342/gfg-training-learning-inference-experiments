from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .runner import file_sha256


TRANSITIONS = ("MAINTAIN_CORRECT", "CORRECT_TO_WRONG", "MAINTAIN_WRONG", "WRONG_TO_CORRECT")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    correct = int(np.sum(truth == prediction))
    accuracy = float(np.mean(truth == prediction))
    recalls = {
        name: float(np.mean(prediction[transitions == name] == truth[transitions == name]))
        for name in TRANSITIONS
    }
    macro = float(np.mean(list(recalls.values())))
    results = _read(root / "BOUNDARY_RESULTS.json")
    declared = results["metrics"]["all_runs"]["quadratic_complete"]
    audit = _read(root / "DERIVATIVE_AUDIT.json")
    checks = {
        "manifest_hashes": not failures,
        "record_count": len(rows) == int(results["record_count"]) == 15264,
        "record_identity_unique": len({row["record_id"] for row in rows}) == len(rows),
        "no_support_or_curve_prediction": results["response_curve_predicted"] is False and results["support_state_predicted"] is False,
        "all_targets_one_rule": all("difficult" not in key.lower() and "remainder" not in key.lower() for row in rows for key in row),
        "accuracy": correct == int(declared["correct_count"]) and abs(accuracy - float(declared["accuracy"])) <= 1e-15,
        "transition_recalls": all(abs(recalls[name] - float(declared["transition_recall"][name])) <= 1e-15 for name in TRANSITIONS),
        "four_way_macro_recall": abs(macro - float(declared["four_way_macro_recall"])) <= 1e-15,
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
        "recomputed": {
            "all_runs_correct_count": correct,
            "all_runs_accuracy": accuracy,
            "all_runs_transition_recall": recalls,
            "all_runs_four_way_macro_recall": macro,
        },
    }


__all__ = ["check"]

