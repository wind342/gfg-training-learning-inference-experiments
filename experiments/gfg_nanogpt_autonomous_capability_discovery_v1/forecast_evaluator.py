from __future__ import annotations

import math
from typing import Any

from .candidate_validator import (
    normalize_component_curve_row,
    normalize_forecast_curve_row,
)
from .common import payload_sha256
from .evaluation_capture import capability_transition


STABILITY_THRESHOLD = 0.90
STABILITY_EVENT_PRECISION_MIN = 0.50
STABILITY_EVENT_RECALL_MIN = 0.50
STABILITY_EVENT_F1_MIN = 0.50
STABILITY_DEFICIT_RMSE_MAX = 0.15
STABILITY_RECOVERY_ERROR_MAX_STEPS = 200


def post_formation_stability(
    metrics: list[dict[str, Any]], transition_step: int | None
) -> dict[str, Any]:
    if transition_step is None:
        return {
            "classification": None,
            "degradation_steps": [],
            "first_recovery_step": None,
        }
    post_transition = [
        row for row in metrics if row["step"] > transition_step
    ]
    degradation_steps = [
        row["step"]
        for row in post_transition
        if row["validation_accuracy"] < STABILITY_THRESHOLD
    ]
    if not degradation_steps:
        classification = "STABLE"
        first_recovery_step = None
    else:
        first_degradation = degradation_steps[0]
        recovery_rows = [
            row
            for row in post_transition
            if row["step"] > first_degradation
            and row["validation_accuracy"] >= STABILITY_THRESHOLD
        ]
        first_recovery_step = (
            recovery_rows[0]["step"] if recovery_rows else None
        )
        classification = (
            "TRANSIENT_DEGRADATION_RECOVERY"
            if first_recovery_step is not None
            else "PERSISTENT_DEGRADATION"
        )
    return {
        "classification": classification,
        "degradation_steps": degradation_steps,
        "first_recovery_step": first_recovery_step,
    }


def _predicted_interval_coverage(
    intervals: list[dict[str, Any]], actual_steps: list[int]
) -> dict[str, Any]:
    covered = [
        step
        for step in actual_steps
        if any(
            row["step_low"] <= step <= row["step_high"]
            for row in intervals
        )
    ]
    return {
        "actual_degradation_step_count": len(actual_steps),
        "covered_degradation_step_count": len(covered),
        "all_actual_degradation_steps_covered": (
            len(covered) == len(actual_steps)
        ),
    }


def _binary_event_metrics(
    *, actual_steps: set[int], predicted_steps: set[int]
) -> dict[str, Any]:
    true_positive = len(actual_steps & predicted_steps)
    false_positive = len(predicted_steps - actual_steps)
    false_negative = len(actual_steps - predicted_steps)
    true_negative_is_not_reported = True
    if predicted_steps:
        precision = true_positive / len(predicted_steps)
    else:
        precision = 1.0 if not actual_steps else 0.0
    if actual_steps:
        recall = true_positive / len(actual_steps)
    else:
        recall = 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "actual_event_count": len(actual_steps),
        "f1": f1,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "precision": precision,
        "predicted_event_count": len(predicted_steps),
        "recall": recall,
        "true_negative_is_not_reported": true_negative_is_not_reported,
        "true_positive": true_positive,
    }


def _first_recovery_step(
    rows: list[dict[str, Any]], *, value_name: str
) -> int | None:
    first_degradation: int | None = None
    for row in rows:
        value = float(row[value_name])
        if first_degradation is None and value < STABILITY_THRESHOLD:
            first_degradation = int(row["step"])
        elif first_degradation is not None and value >= STABILITY_THRESHOLD:
            return int(row["step"])
    return None


def evaluate_forecast(
    *,
    sealed_forecast: dict[str, Any],
    prefix_metrics: list[dict[str, Any]],
    future_metrics: list[dict[str, Any]],
    candidate_seal_sha256: str,
) -> dict[str, Any]:
    forecast = sealed_forecast["forecast"]
    combined = sorted(
        [*prefix_metrics, *future_metrics],
        key=lambda row: row["step"],
    )
    actual_transition = capability_transition(combined)
    will_transition = actual_transition is not None
    normalized_curve = [
        normalize_forecast_curve_row(row)
        for row in forecast["predicted_validation_curve"]
    ]
    predicted_curve = {
        step: accuracy
        for point in normalized_curve
        if point is not None
        for step, accuracy in (point,)
    }
    predicted_formation_curve = {
        step: value
        for row in forecast["predicted_formation_curve"]
        for point in (normalize_component_curve_row(row, "capability"),)
        if point is not None
        for step, value in (point,)
    }
    predicted_stability_curve = {
        step: value
        for row in forecast["predicted_stability_degradation_curve"]
        for point in (normalize_component_curve_row(row, "degradation"),)
        if point is not None
        for step, value in (point,)
    }
    actual_future = [
        row
        for row in combined
        if row["step"] > sealed_forecast["prediction_cut_step"]
    ]
    common = [
        row for row in actual_future if row["step"] in predicted_curve
    ]
    squared = [
        (
            predicted_curve[row["step"]]
            - row["validation_accuracy"]
        )
        ** 2
        for row in common
    ]
    normalized_rmse = (
        math.sqrt(sum(squared) / len(squared)) if squared else None
    )
    stability = post_formation_stability(combined, actual_transition)
    predicted_stability = forecast["post_formation_stability"]
    predicted_instability_intervals = forecast[
        "predicted_instability_intervals"
    ]
    stability_interval_coverage = _predicted_interval_coverage(
        predicted_instability_intervals,
        stability["degradation_steps"],
    )
    post_transition_common = [
        row
        for row in common
        if actual_transition is not None and row["step"] > actual_transition
    ]
    actual_degradation_steps = {
        int(row["step"])
        for row in post_transition_common
        if row["validation_accuracy"] < STABILITY_THRESHOLD
    }
    predicted_degradation_steps = {
        int(row["step"])
        for row in post_transition_common
        if predicted_curve[row["step"]] < STABILITY_THRESHOLD
        and predicted_stability_curve.get(row["step"], 0.0) > 0.0
    }
    event_metrics = _binary_event_metrics(
        actual_steps=actual_degradation_steps,
        predicted_steps=predicted_degradation_steps,
    )
    stability_squared = [
        (
            max(0.0, STABILITY_THRESHOLD - predicted_curve[row["step"]])
            - max(0.0, STABILITY_THRESHOLD - row["validation_accuracy"])
        )
        ** 2
        for row in post_transition_common
    ]
    stability_deficit_rmse = (
        math.sqrt(sum(stability_squared) / len(stability_squared))
        if stability_squared
        else None
    )
    predicted_post_transition_rows = [
        {
            "step": int(row["step"]),
            "predicted_validation_accuracy": predicted_curve[row["step"]],
        }
        for row in post_transition_common
    ]
    predicted_first_recovery = _first_recovery_step(
        predicted_post_transition_rows,
        value_name="predicted_validation_accuracy",
    )
    actual_first_recovery = stability["first_recovery_step"]
    if stability["classification"] == "TRANSIENT_DEGRADATION_RECOVERY":
        recovery_timing_error = (
            abs(predicted_first_recovery - actual_first_recovery)
            if predicted_first_recovery is not None
            and actual_first_recovery is not None
            else None
        )
        recovery_timing_gate = (
            recovery_timing_error is not None
            and recovery_timing_error <= STABILITY_RECOVERY_ERROR_MAX_STEPS
        )
    else:
        recovery_timing_error = None
        recovery_timing_gate = True
    low_200 = forecast["transition_step_low_200"]
    high_200 = forecast["transition_step_high_200"]
    low_500 = forecast["transition_step_low_500"]
    high_500 = forecast["transition_step_high_500"]
    interval_200_contains_actual = (
        actual_transition is not None
        and isinstance(low_200, int)
        and isinstance(high_200, int)
        and low_200 <= actual_transition <= high_200
    )
    interval_500_contains_actual = (
        actual_transition is not None
        and isinstance(low_500, int)
        and isinstance(high_500, int)
        and low_500 <= actual_transition <= high_500
    )
    shared_gates = {
        "candidate_seal_unchanged": (
            sealed_forecast["candidate_seal_sha256"]
            == candidate_seal_sha256
        ),
        "curve_complete_on_future_grid": (
            len(common) == len(actual_future)
        ),
        "future_gfg_reads_zero": (
            sealed_forecast["future_gfg_reads"] == 0
        ),
        "normalized_rmse_at_most_0_2": (
            normalized_rmse is not None and normalized_rmse <= 0.2
        ),
    }
    formation_gates = {
        "transition_classification_correct": (
            forecast["will_transition"] == will_transition
        ),
        "transition_interval_500_contains_actual": (
            interval_500_contains_actual
        )
        if will_transition
        else low_500 is None and high_500 is None,
        "transition_interval_200_width_valid": (
            isinstance(low_200, int)
            and isinstance(high_200, int)
            and high_200 - low_200 <= 200
        )
        if will_transition
        else True,
        "transition_interval_500_width_valid": (
            isinstance(low_500, int)
            and isinstance(high_500, int)
            and high_500 - low_500 <= 500
        )
        if will_transition
        else True,
    }
    stability_gates = {
        "stability_classification_correct": (
            predicted_stability == stability["classification"]
        ),
        "stability_curve_complete_on_post_transition_grid": (
            actual_transition is not None
            and bool(post_transition_common)
            and all(
                row["step"] in predicted_formation_curve
                and row["step"] in predicted_stability_curve
                for row in post_transition_common
            )
        ),
        "stability_component_accounts_for_post_formation_deficits": (
            bool(post_transition_common)
            and all(
                predicted_curve[row["step"]] >= STABILITY_THRESHOLD
                or (
                    predicted_formation_curve.get(row["step"], 0.0)
                    >= STABILITY_THRESHOLD
                    and predicted_stability_curve.get(row["step"], 0.0) > 0.0
                )
                for row in post_transition_common
            )
        ),
        "stability_degradation_event_precision_at_least_0_5": (
            event_metrics["precision"] >= STABILITY_EVENT_PRECISION_MIN
        ),
        "stability_degradation_event_recall_at_least_0_5": (
            event_metrics["recall"] >= STABILITY_EVENT_RECALL_MIN
        ),
        "stability_degradation_event_f1_at_least_0_5": (
            event_metrics["f1"] >= STABILITY_EVENT_F1_MIN
        ),
        "stability_deficit_rmse_at_most_0_15": (
            stability_deficit_rmse is not None
            and stability_deficit_rmse <= STABILITY_DEFICIT_RMSE_MAX
        ),
        "transient_first_recovery_error_at_most_200_steps": (
            recovery_timing_gate
        ),
    }
    gates = {**shared_gates, **formation_gates, **stability_gates}
    material = {
        "actual_transition_step": actual_transition,
        "curve_point_count": len(common),
        "gates": gates,
        "high_precision_diagnostics": {
            "transition_interval_200_contains_actual": (
                interval_200_contains_actual
                if will_transition
                else low_200 is None and high_200 is None
            ),
            "transition_step_high_200": high_200,
            "transition_step_low_200": low_200,
        },
        "stability_validation": {
            "actual": stability,
            "candidate_classification": predicted_stability,
            "classification_correct": stability_gates[
                "stability_classification_correct"
            ],
            "degradation_event_metrics": event_metrics,
            "post_formation_deficit_rmse": stability_deficit_rmse,
            "predicted_first_recovery_step": predicted_first_recovery,
            "recovery_timing_absolute_error_steps": recovery_timing_error,
            "predicted_instability_intervals": (
                predicted_instability_intervals
            ),
            "interval_coverage": stability_interval_coverage,
            "primary_pass_gate": True,
            "gates": stability_gates,
            "status": (
                "STABILITY_FORECAST_PASS"
                if all(stability_gates.values())
                else "STABILITY_FORECAST_NOT_ESTABLISHED"
            ),
            "threshold": STABILITY_THRESHOLD,
        },
        "formation_validation": {
            "gates": formation_gates,
            "status": (
                "FORMATION_FORECAST_PASS"
                if all(formation_gates.values())
                else "FORMATION_FORECAST_NOT_ESTABLISHED"
            ),
        },
        "primary_interval": {
            "transition_interval_500_contains_actual": (
                interval_500_contains_actual
                if will_transition
                else low_500 is None and high_500 is None
            ),
            "transition_step_high_500": high_500,
            "transition_step_low_500": low_500,
        },
        "normalized_rmse": normalized_rmse,
        "schema": "unseen-training-dual-dynamics-forecast-validation-v2",
        "unified_curve_validation": {
            "gates": shared_gates,
            "status": (
                "UNIFIED_CURVE_PASS"
                if all(shared_gates.values())
                else "UNIFIED_CURVE_NOT_ESTABLISHED"
            ),
        },
        "status": (
            "FORECAST_VALIDATION_PASS"
            if all(gates.values())
            else "FORECAST_VALIDATION_NOT_ESTABLISHED"
        ),
    }
    material["validation_sha256"] = payload_sha256(material)
    return material
