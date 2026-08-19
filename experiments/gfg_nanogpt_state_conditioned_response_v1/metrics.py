from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np

from .model import POSITIVE_ALPHAS, boundary_class, normalized_shape, response_type, stable_seed


BOUNDARY_CLASSES = (
    "MAINTAIN_CORRECT",
    "CORRECT_TO_WRONG",
    "MAINTAIN_WRONG",
    "WRONG_TO_CORRECT",
)
RESPONSE_TYPES = (
    "NEAR_LINEAR",
    "SATURATING",
    "ACCELERATING",
    "TURNBACK",
    "SIGN_REVERSAL",
    "OTHER",
)


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _classification_metrics(truth: list[str], predicted: list[str], labels: tuple[str, ...]) -> dict[str, Any]:
    matrix = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(truth, predicted, strict=True):
        if actual not in matrix:
            matrix[actual] = {name: 0 for name in labels}
        if guess not in matrix[actual]:
            matrix[actual][guess] = 0
        matrix[actual][guess] += 1
    per_class: dict[str, Any] = {}
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(matrix.get(actual, {}).get(label, 0) for actual in labels if actual != label)
        fn = sum(matrix.get(label, {}).get(guess, 0) for guess in labels if guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": int(sum(matrix.get(label, {}).values())),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def evaluate_curves(
    truth: np.ndarray,
    prediction: np.ndarray,
    margin0: np.ndarray,
    normalization_scale: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    abs_error = np.abs(error)
    squared = np.square(error)
    scale = np.asarray(normalization_scale, dtype=np.float64)
    if scale.ndim == 1:
        scale = np.broadcast_to(scale, truth.shape)
    normalized_error = error / np.maximum(scale, 1e-12)
    true_shape = normalized_shape(truth)
    predicted_shape = normalized_shape(prediction)
    shape_error = predicted_shape - true_shape
    truth_boundary = [boundary_class(float(value), curve) for value, curve in zip(margin0, truth, strict=True)]
    predicted_boundary = [boundary_class(float(value), curve) for value, curve in zip(margin0, prediction, strict=True)]
    truth_types = [response_type(curve) for curve in truth]
    predicted_types = [response_type(curve) for curve in prediction]
    unchanged = np.asarray([value in {"MAINTAIN_CORRECT", "MAINTAIN_WRONG"} for value in truth_boundary], dtype=bool)
    false_cross = np.asarray(
        [value in {"CORRECT_TO_WRONG", "WRONG_TO_CORRECT"} for value in predicted_boundary],
        dtype=bool,
    )
    endpoint_truth = truth[:, -1]
    endpoint_prediction = prediction[:, -1]
    direction_truth = np.sign(endpoint_truth)
    direction_prediction = np.sign(endpoint_prediction)
    flattened_true_shape = true_shape.ravel()
    flattened_predicted_shape = predicted_shape.ravel()
    correlation = 0.0
    if np.std(flattened_true_shape) > 0 and np.std(flattened_predicted_shape) > 0:
        correlation = float(np.corrcoef(flattened_true_shape, flattened_predicted_shape)[0, 1])
    return {
        "count": int(len(truth)),
        "displacement": {
            "mae": float(np.mean(abs_error)),
            "rmse": float(np.sqrt(np.mean(squared))),
            "nrmse": float(np.sqrt(np.mean(np.square(normalized_error)))),
            "per_alpha": {
                str(alpha): {
                    "mae": float(np.mean(abs_error[:, index])),
                    "rmse": float(np.sqrt(np.mean(squared[:, index]))),
                }
                for index, alpha in enumerate(POSITIVE_ALPHAS.tolist())
            },
        },
        "normalized_shape": {
            "rmse": float(np.sqrt(np.mean(np.square(shape_error)))),
            "correlation": correlation,
        },
        "endpoint": {
            "mae": float(np.mean(np.abs(endpoint_prediction - endpoint_truth))),
            "rmse": float(np.sqrt(np.mean(np.square(endpoint_prediction - endpoint_truth)))),
            "direction_accuracy": float(np.mean(direction_truth == direction_prediction)),
        },
        "response_type": _classification_metrics(truth_types, predicted_types, RESPONSE_TYPES),
        "boundary": _classification_metrics(truth_boundary, predicted_boundary, BOUNDARY_CLASSES),
        "unchanged_target": {
            "count": int(np.sum(unchanged)),
            "false_crossing_count": int(np.sum(false_cross & unchanged)),
            "false_crossing_rate": float(np.mean(false_cross[unchanged])) if np.any(unchanged) else 0.0,
        },
    }


def evaluate_by_run(
    truth: np.ndarray,
    prediction: np.ndarray,
    margin0: np.ndarray,
    normalization_scale: np.ndarray,
    entries: np.ndarray,
) -> dict[str, Any]:
    overall = evaluate_curves(truth, prediction, margin0, normalization_scale)
    per_run = {}
    for entry in sorted(set(entries.tolist())):
        mask = entries == entry
        per_run[entry] = evaluate_curves(
            truth[mask],
            prediction[mask],
            margin0[mask],
            normalization_scale[mask],
        )
    return {"overall": overall, "per_run": per_run}


def clustered_bootstrap_improvement(
    per_run_candidate: dict[str, Any],
    per_run_baseline: dict[str, Any],
    metric_path: tuple[str, ...],
    *,
    label: str,
    resamples: int = 2000,
) -> dict[str, Any]:
    entries = sorted(per_run_candidate)

    def get(payload: dict[str, Any], path: tuple[str, ...]) -> float:
        value: Any = payload
        for item in path:
            value = value[item]
        return float(value)

    improvements = np.asarray(
        [get(per_run_baseline[entry], metric_path) - get(per_run_candidate[entry], metric_path) for entry in entries],
        dtype=np.float64,
    )
    generator = np.random.default_rng(stable_seed(f"bootstrap:{label}"))
    draws = generator.integers(0, len(entries), size=(resamples, len(entries)))
    values = np.mean(improvements[draws], axis=1)
    return {
        "cluster_unit": "entry_id",
        "run_count": len(entries),
        "resamples": resamples,
        "mean_improvement": float(np.mean(improvements)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "per_run_improvement": {entry: float(value) for entry, value in zip(entries, improvements, strict=True)},
    }


def label_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


__all__ = [
    "BOUNDARY_CLASSES",
    "RESPONSE_TYPES",
    "clustered_bootstrap_improvement",
    "evaluate_by_run",
    "evaluate_curves",
    "label_counts",
]
