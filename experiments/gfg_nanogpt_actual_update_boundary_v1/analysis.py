from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_competitor_handoff_outcome_v1.analysis import (
    _final_truth,
    _metrics,
    compile_competitor_coordinates,
    compile_response_dataset,
)
from experiments.gfg_nanogpt_full_network_receptive_state_v1.analysis import DEFAULT_TRAINER_ROOT
from experiments.gfg_nanogpt_joint_rotation_branch_v1.analysis import compile_coordinates
from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import (
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
)


METHODS = ("linear", "joint_rotation", "hidden_curvature", "quadratic_complete")
PRIMARY = "quadratic_complete"
TRANSITIONS = ("MAINTAIN_CORRECT", "CORRECT_TO_WRONG", "MAINTAIN_WRONG", "WRONG_TO_CORRECT")


def _macro_recall(metric: dict[str, Any]) -> float:
    values = [value for value in metric["transition_recall"].values() if value is not None]
    return float(np.mean(values))


def _transition_confusion(
    start: np.ndarray, truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    actual = np.where(
        start,
        np.where(truth, "MAINTAIN_CORRECT", "CORRECT_TO_WRONG"),
        np.where(truth, "WRONG_TO_CORRECT", "MAINTAIN_WRONG"),
    )[mask]
    predicted = np.where(
        start,
        np.where(prediction, "MAINTAIN_CORRECT", "CORRECT_TO_WRONG"),
        np.where(prediction, "WRONG_TO_CORRECT", "MAINTAIN_WRONG"),
    )[mask]
    index = {name: position for position, name in enumerate(TRANSITIONS)}
    matrix = np.zeros((len(TRANSITIONS), len(TRANSITIONS)), dtype=np.int64)
    for left, right in zip(actual.tolist(), predicted.tolist()):
        matrix[index[str(left)], index[str(right)]] += 1
    return {
        "labels": list(TRANSITIONS),
        "matrix": matrix.tolist(),
        "actual_counts": dict(Counter(str(value) for value in actual.tolist())),
        "predicted_counts": dict(Counter(str(value) for value in predicted.tolist())),
    }


def _repair(truth: np.ndarray, baseline: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    fixed = int(np.sum(mask & (baseline != truth) & (prediction == truth)))
    broken = int(np.sum(mask & (baseline == truth) & (prediction != truth)))
    return {"fixed_linear_errors": fixed, "newly_broken_linear_answers": broken, "net_repairs": fixed - broken}


def run_analysis(trainer_root: Path = DEFAULT_TRAINER_ROOT) -> dict[str, Any]:
    response, response_audit, response_sources = compile_response_dataset()
    competitor = compile_competitor_coordinates(response)
    coordinates = compile_coordinates(response, competitor, trainer_root.resolve())
    gaps = np.asarray(competitor["gaps"], dtype=np.float64)
    first = np.asarray(coordinates["first_gap"], dtype=np.float64)
    joint = np.asarray(coordinates["joint_rotation"], dtype=np.float64)
    hidden = np.asarray(coordinates["hidden_curvature"], dtype=np.float64)
    endpoints = {
        "linear": gaps + first,
        "joint_rotation": gaps + first + joint,
        "hidden_curvature": gaps + first + hidden,
        "quadratic_complete": gaps + first + joint + hidden,
    }
    predictions = {name: np.all(values > 0.0, axis=1) for name, values in endpoints.items()}
    start, truth, boundary = _final_truth(response)
    entries = np.asarray(response["entries"], dtype=object)
    split_masks = {
        "development": np.isin(entries, np.asarray(DEVELOPMENT_RUNS, dtype=object)),
        "confirmation": np.isin(entries, np.asarray(CONFIRMATION_RUNS, dtype=object)),
        "all_runs": np.ones(len(truth), dtype=bool),
    }
    metrics: dict[str, Any] = {}
    repairs: dict[str, Any] = {}
    for split, mask in split_masks.items():
        metrics[split] = {}
        repairs[split] = {}
        for method in METHODS:
            metric = _metrics(truth, predictions[method], start, boundary, mask)
            metric["four_way_macro_recall"] = _macro_recall(metric)
            metric["four_way_confusion"] = _transition_confusion(start, truth, predictions[method], mask)
            metrics[split][method] = metric
            repairs[split][method] = _repair(truth, predictions["linear"], predictions[method], mask)

    per_run = {}
    for run in sorted(set(entries.tolist())):
        mask = entries == run
        metric = _metrics(truth, predictions[PRIMARY], start, boundary, mask)
        metric["four_way_macro_recall"] = _macro_recall(metric)
        metric["four_way_confusion"] = _transition_confusion(start, truth, predictions[PRIMARY], mask)
        per_run[str(run)] = metric

    ledger = []
    for index, record in enumerate(response["records"]):
        ledger.append({
            "row_index": index,
            "record_id": str(record["record_id"]),
            "entry_id": str(record["entry_id"]),
            "section_id": str(record["section_id"]),
            "evaluation_unit_id": str(record["evaluation_unit_id"]),
            "target_group": int(record["target_group"]),
            "start_correct": bool(start[index]),
            "truth_final_correct": bool(truth[index]),
            "truth_transition": str(boundary[index]),
            "predicted_final_correct": {name: bool(predictions[name][index]) for name in METHODS},
            "predicted_transition": {
                name: (
                    "MAINTAIN_CORRECT" if start[index] and predictions[name][index]
                    else "CORRECT_TO_WRONG" if start[index]
                    else "WRONG_TO_CORRECT" if predictions[name][index]
                    else "MAINTAIN_WRONG"
                )
                for name in METHODS
            },
            "minimum_predicted_endpoint_gap": {
                name: float(np.min(endpoints[name][index])) for name in METHODS
            },
        })

    confirmation = metrics["confirmation"][PRIMARY]
    verdict = (
        "ACTUAL_UPDATE_BOUNDARY_PREDICTION_REPRODUCED"
        if confirmation["accuracy"] >= 0.90
        and confirmation["four_way_macro_recall"] >= 0.90
        else "ACTUAL_UPDATE_BOUNDARY_PREDICTION_NOT_REPRODUCED"
    )
    return {
        "schema": "gfg-nanogpt-actual-update-boundary-results-v1",
        "status": "PASS",
        "verdict": verdict,
        "evidence_status": "REEXECUTION_OF_ESTABLISHED_ALGORITHM_NOT_FRESH_CONFIRMATION",
        "primary_method": PRIMARY,
        "prediction_target": "IMMEDIATE_TARGET_LEVEL_POST_UPDATE_BOUNDARY",
        "response_curve_predicted": False,
        "support_state_predicted": False,
        "record_count": len(truth),
        "run_count": len(set(entries.tolist())),
        "metrics": metrics,
        "repairs": repairs,
        "per_run": per_run,
        "ledger": ledger,
        "coordinates": coordinates,
        "response_audit": response_audit,
        "response_sources": response_sources,
    }


__all__ = ["METHODS", "PRIMARY", "run_analysis"]
