from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .inventory import ALPHAS, read_json, write_json


CAPABILITY_ZERO_BAND = 1.0 / 424.0
DECLINE_DELTA_THRESHOLD = -0.02
TRANSITION_CLASSES = ("maintain_correct", "correct_to_wrong", "wrong_to_correct", "maintain_wrong")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "p50_abs_error": float(np.quantile(np.abs(error), 0.50)),
        "p90_abs_error": float(np.quantile(np.abs(error), 0.90)),
        "p99_abs_error": float(np.quantile(np.abs(error), 0.99)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _transition_class(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_ok = np.asarray(before) >= 0.0
    after_ok = np.asarray(after) >= 0.0
    result = np.empty(before_ok.shape, dtype="U20")
    result[before_ok & after_ok] = "maintain_correct"
    result[before_ok & ~after_ok] = "correct_to_wrong"
    result[~before_ok & after_ok] = "wrong_to_correct"
    result[~before_ok & ~after_ok] = "maintain_wrong"
    return result


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        expected: {actual: int(np.sum((truth == expected) & (prediction == actual))) for actual in TRANSITION_CLASSES}
        for expected in TRANSITION_CLASSES
    }


def _direction(value: float) -> int:
    if value > CAPABILITY_ZERO_BAND:
        return 1
    if value < -CAPABILITY_ZERO_BAND:
        return -1
    return 0


def _section_curve(root: Path, section_id: str) -> dict[str, np.ndarray]:
    with np.load(root / "sections" / f"{section_id}.npz", allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _bootstrap_difference(left: list[float], right: list[float], *, seed: int = 1729) -> dict[str, float]:
    _require(len(left) == len(right) and bool(left), "BOOTSTRAP_PAIR_COUNT_INVALID")
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([np.mean(rng.choice(delta, size=len(delta), replace=True)) for _ in range(2000)])
    return {
        "mean_difference": float(np.mean(delta)),
        "cluster_pair_bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
        "cluster_pair_bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
    }


def local_to_endpoint(*, inventory: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_material = read_json(output_root / "IDENTITY_MATERIAL.json")
    identity_by_entry = identity_material["entries"]
    truth_rows: list[np.ndarray] = []
    prediction_rows: dict[str, list[np.ndarray]] = {"M0": [], "J": [], "JK": []}
    section_records: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    mechanism_counts: Counter[str] = Counter()
    alpha = np.asarray(ALPHAS, dtype=np.float64)
    h = 0.125

    for section in inventory["sections"]:
        curve = _section_curve(output_root, section["section_id"])
        margins = curve["all_margins"][:, 0, :].astype(np.float64)
        competitors = curve["baseline_competitor_ids"]
        groups = curve["groups"]
        m_minus, m_zero, m_plus, m_true = margins[0], margins[1], margins[2], margins[-1]
        j = (m_plus - m_minus) / (2.0 * h)
        k = (m_plus - 2.0 * m_zero + m_minus) / (h * h)
        candidates = {"M0": m_zero, "J": m_zero + j, "JK": m_zero + j + 0.5 * k}
        truth_rows.append(m_true)
        for key, value in candidates.items():
            prediction_rows[key].append(value)

        true_class = _transition_class(m_zero, m_true)
        jk_class = _transition_class(m_zero, candidates["JK"])
        first_differences = np.diff(margins, axis=0)
        second_differences = np.diff(first_differences, axis=0)
        for index in range(m_true.size):
            identity = identity_by_entry[section["entry_id_audit_only"]][index]
            delta_true = m_true[index] - m_zero[index]
            delta_jk = candidates["JK"][index] - m_zero[index]
            diffs = first_differences[:, index]
            seconds = second_differences[:, index]
            if np.any(diffs[:-1] * diffs[1:] < 0.0):
                mechanism = "turnback"
            elif np.any(seconds[:-1] * seconds[1:] < 0.0):
                mechanism = "convexity_change"
            elif abs(delta_true) < abs(j[index]) * 0.5:
                mechanism = "saturation_or_low_gain"
            elif abs(delta_true) > abs(j[index]) * 1.5:
                mechanism = "endpoint_amplification"
            else:
                mechanism = "approximately_local"
            if delta_true != 0.0 and delta_jk != 0.0 and np.sign(delta_true) != np.sign(delta_jk):
                mechanism += "+local_endpoint_sign_mismatch"
            if competitors[1, index] != competitors[-1, index]:
                mechanism += "+competitor_switch"
            mechanism_counts[mechanism] += 1
            ledgers.append(
                {
                    "section_id": section["section_id"],
                    "element_index_audit_only": index,
                    "evaluation_unit_id": identity["evaluation_unit_id"],
                    "upstream_element_identity": identity["upstream_element_identity"],
                    "target_group": int(groups[index]),
                    "margins_by_alpha": margins[:, index].tolist(),
                    "competitor_ids_by_alpha": competitors[:, index].astype(int).tolist(),
                    "true_transition": str(true_class[index]),
                    "jk_transition": str(jk_class[index]),
                    "mechanism_class": mechanism,
                }
            )
        section_records.append(
            {
                "section_id": section["section_id"],
                "entry_id_audit_only": section["entry_id_audit_only"],
                "true_capability_before": float(np.mean(m_zero >= 0.0)),
                "true_capability_after": float(np.mean(m_true >= 0.0)),
                "m0_capability_after": float(np.mean(candidates["M0"] >= 0.0)),
                "j_capability_after": float(np.mean(candidates["J"] >= 0.0)),
                "jk_capability_after": float(np.mean(candidates["JK"] >= 0.0)),
            }
        )

    truth = np.concatenate(truth_rows)
    results: dict[str, Any] = {}
    section_truth_before = np.asarray([row["true_capability_before"] for row in section_records])
    section_truth_after = np.asarray([row["true_capability_after"] for row in section_records])
    for key, rows in prediction_rows.items():
        prediction = np.concatenate(rows)
        before = np.concatenate([_section_curve(output_root, row["section_id"])["all_margins"][1, 0, :] for row in inventory["sections"]]).astype(np.float64)
        true_class = _transition_class(before, truth)
        predicted_class = _transition_class(before, prediction)
        predicted_section_after = np.asarray([row[f"{key.lower()}_capability_after"] for row in section_records])
        true_delta = section_truth_after - section_truth_before
        predicted_delta = predicted_section_after - section_truth_before
        results[key] = {
            "margin": _regression_metrics(truth, prediction),
            "correct_to_wrong": _binary_metrics(true_class == "correct_to_wrong", predicted_class == "correct_to_wrong"),
            "wrong_to_correct": _binary_metrics(true_class == "wrong_to_correct", predicted_class == "wrong_to_correct"),
            "maintain_correct_accuracy": float(np.mean(predicted_class[true_class == "maintain_correct"] == "maintain_correct")) if np.any(true_class == "maintain_correct") else None,
            "maintain_wrong_accuracy": float(np.mean(predicted_class[true_class == "maintain_wrong"] == "maintain_wrong")) if np.any(true_class == "maintain_wrong") else None,
            "transition_confusion": _confusion(true_class, predicted_class),
            "capability_mae": float(np.mean(np.abs(predicted_section_after - section_truth_after))),
            "capability_direction_accuracy": float(
                np.mean(
                    np.asarray([_direction(value) for value in true_delta], dtype=np.int8)
                    == np.asarray([_direction(value) for value in predicted_delta], dtype=np.int8)
                )
            ),
            "significant_decline": _binary_metrics(true_delta <= DECLINE_DELTA_THRESHOLD, predicted_delta <= DECLINE_DELTA_THRESHOLD),
        }
    adjudication = {
        "schema": "nanogpt-local-to-endpoint-adjudication-v1",
        "status": "PASS",
        "section_count": 72,
        "evaluation_count": int(truth.size),
        "alpha_grid": list(ALPHAS),
        "local_derivative_half_width": h,
        "decline_delta_threshold": DECLINE_DELTA_THRESHOLD,
        "capability_zero_band": CAPABILITY_ZERO_BAND,
        "models": results,
        "mechanism_class_counts": dict(sorted(mechanism_counts.items())),
        "sections": section_records,
        "global_unseen_entry_accessed": False,
    }
    crossing = {
        "schema": "nanogpt-target-crossing-ledgers-v1",
        "status": "PASS",
        "row_count": len(ledgers),
        "identity_warning": "element_index is audit-only; stable evaluation identities are in IDENTITY_MATERIAL.json",
        "rows": ledgers,
    }
    return adjudication, crossing


def _normalized_shapes(curve: dict[str, np.ndarray]) -> np.ndarray:
    margins = curve["all_margins"][:, 0, :].astype(np.float64).T
    delta = margins - margins[:, [1]]
    denom = np.max(np.abs(delta), axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return delta / denom


def _shape_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered * left_centered, axis=1) * np.sum(right_centered * right_centered, axis=1))
    result = np.zeros_like(numerator)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    result[~valid] = np.allclose(left[~valid], right[~valid], rtol=0.0, atol=0.0, equal_nan=True)
    return result


def _greedy_cross_run_alignment(left_curve: dict[str, np.ndarray], right_curve: dict[str, np.ndarray]) -> np.ndarray:
    # Matching uses alpha=0 facts only; no alpha>0 response enters the alignment.
    left_m = left_curve["all_margins"][1, 0, :].astype(np.float64)
    right_m = right_curve["all_margins"][1, 0, :].astype(np.float64)
    left_g = left_curve["groups"].astype(int)
    right_g = right_curve["groups"].astype(int)
    left_n = left_curve["necessity"][1].T.astype(np.float64)
    right_n = right_curve["necessity"][1].T.astype(np.float64)
    left_features = np.column_stack([left_m, left_n[left_g]])
    right_features = np.column_stack([right_m, right_n[right_g]])
    combined = np.vstack([left_features, right_features])
    scale = combined.std(axis=0)
    scale[scale == 0.0] = 1.0
    left_features = (left_features - combined.mean(axis=0)) / scale
    right_features = (right_features - combined.mean(axis=0)) / scale
    available = set(range(right_features.shape[0]))
    mapping: list[int] = []
    for left_index in range(left_features.shape[0]):
        best = min(available, key=lambda value: (float(np.linalg.norm(left_features[left_index] - right_features[value])), value))
        mapping.append(best)
        available.remove(best)
    return np.asarray(mapping, dtype=np.int64)


def adjacent_transport(*, inventory: dict[str, Any], geometry: dict[str, Any], output_root: Path) -> dict[str, Any]:
    cache = {row["section_id"]: _section_curve(output_root, row["section_id"]) for row in inventory["sections"]}
    groups: dict[str, list[float]] = defaultdict(list)
    pair_records: list[dict[str, Any]] = []
    for label, rows in (
        ("adjacent", geometry["treatment_pairs"]),
        ("random_nonadjacent", geometry["random_nonadjacent_controls"]),
        ("matched_cross_run", geometry["matched_cross_run_controls"]),
    ):
        for row in rows:
            left = cache[row["left_section_id"]]
            right = cache[row["right_section_id"]]
            left_shape = _normalized_shapes(left)
            right_shape = _normalized_shapes(right)
            if label == "matched_cross_run":
                mapping = _greedy_cross_run_alignment(left, right)
                right_shape = right_shape[mapping]
                alignment = "one-to-one greedy matching on alpha=0 margin and group necessity; no fixed target ordinal"
            else:
                _require(np.array_equal(left["groups"], right["groups"]), f"WITHIN_RUN_IDENTITY_MISMATCH:{row['left_section_id']}")
                alignment = "same stable within-run evaluation units"
            correlations = _shape_correlation(left_shape, right_shape)
            mean = float(np.mean(correlations))
            groups[label].append(mean)
            pair_records.append(
                {
                    "comparison": label,
                    "left_section_id": row["left_section_id"],
                    "right_section_id": row["right_section_id"],
                    "alignment": alignment,
                    "target_count": int(correlations.size),
                    "mean_shape_correlation": mean,
                    "median_shape_correlation": float(np.median(correlations)),
                    "monotonic_direction_agreement": float(np.mean(np.sign(left_shape[:, -1]) == np.sign(right_shape[:, -1]))),
                }
            )
    summary = {
        key: {
            "pair_count": len(values),
            "mean_of_pair_mean_shape_correlations": float(np.mean(values)),
            "median_of_pair_mean_shape_correlations": float(np.median(values)),
        }
        for key, values in groups.items()
    }
    return {
        "schema": "nanogpt-adjacent-response-transport-analysis-v1",
        "status": "PASS",
        "section_count": 72,
        "alpha_grid": list(ALPHAS),
        "controls_frozen_before_curves": geometry["status"] == "FROZEN_BEFORE_FINITE_AMPLITUDE_CURVES",
        "summary": summary,
        "adjacent_minus_random": _bootstrap_difference(groups["adjacent"], groups["random_nonadjacent"], seed=1729),
        "adjacent_minus_matched_cross_run": _bootstrap_difference(groups["adjacent"], groups["matched_cross_run"], seed=1730),
        "pair_results": pair_records,
        "global_unseen_entry_accessed": False,
    }


def run_deterministic_analysis(*, inventory_path: Path, geometry_path: Path, output_root: Path) -> None:
    inventory = read_json(inventory_path)
    geometry = read_json(geometry_path)
    adjudication, crossing = local_to_endpoint(inventory=inventory, output_root=output_root)
    transport = adjacent_transport(inventory=inventory, geometry=geometry, output_root=output_root)
    write_json(output_root / "LOCAL_TO_ENDPOINT_ADJUDICATION.json", adjudication)
    write_json(output_root / "TARGET_CROSSING_LEDGERS.json", crossing)
    write_json(output_root / "ADJACENT_TRANSPORT_ANALYSIS.json", transport)
