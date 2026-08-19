from __future__ import annotations

from typing import Any

from .candidate_validator import (
    intervention_direction,
    intervention_shift_interval,
)
from .common import payload_sha256
from .evaluation_capture import capability_transition
from .forecast_evaluator import post_formation_stability


STABILITY_EFFECT_EPSILON = 0.0001


def _stability_deficit(
    metrics: list[dict[str, Any]], transition_step: int | None
) -> float | None:
    if transition_step is None:
        return None
    post_transition = [
        row for row in metrics if row["step"] > transition_step
    ]
    if not post_transition:
        return None
    return sum(
        max(0.0, 0.90 - row["validation_accuracy"])
        for row in post_transition
    ) / len(post_transition)


def evaluate_causal_intervention(
    *,
    prefix_metrics: list[dict[str, Any]],
    baseline_metrics: list[dict[str, Any]],
    intervention_metrics: list[dict[str, Any]],
    intervention_spec: dict[str, Any],
    fork_audit: dict[str, Any],
    intervention_receipt: dict[str, Any],
) -> dict[str, Any]:
    baseline_transition = capability_transition(
        sorted(
            [*prefix_metrics, *baseline_metrics],
            key=lambda row: row["step"],
        )
    )
    intervention_transition = capability_transition(
        sorted(
            [*prefix_metrics, *intervention_metrics],
            key=lambda row: row["step"],
        )
    )
    direction = intervention_direction(intervention_spec)
    if direction is None:
        raise ValueError("INTERVENTION_DIRECTION_INVALID")
    if baseline_transition is None or intervention_transition is None:
        magnitude = None
    elif direction == "DELAY":
        magnitude = intervention_transition - baseline_transition
    else:
        magnitude = baseline_transition - intervention_transition
    shift_interval = intervention_shift_interval(intervention_spec)
    if shift_interval is None:
        raise ValueError("INTERVENTION_SHIFT_INTERVAL_MISSING")
    low, high = shift_interval
    baseline_combined = sorted(
        [*prefix_metrics, *baseline_metrics], key=lambda row: row["step"]
    )
    intervention_combined = sorted(
        [*prefix_metrics, *intervention_metrics],
        key=lambda row: row["step"],
    )
    baseline_stability = post_formation_stability(
        baseline_combined, baseline_transition
    )
    intervention_stability = post_formation_stability(
        intervention_combined, intervention_transition
    )
    baseline_deficit = _stability_deficit(
        baseline_combined, baseline_transition
    )
    intervention_deficit = _stability_deficit(
        intervention_combined, intervention_transition
    )
    if baseline_deficit is None or intervention_deficit is None:
        actual_stability_effect = None
    elif intervention_deficit < baseline_deficit - STABILITY_EFFECT_EPSILON:
        actual_stability_effect = "IMPROVE"
    elif intervention_deficit > baseline_deficit + STABILITY_EFFECT_EPSILON:
        actual_stability_effect = "WORSEN"
    else:
        actual_stability_effect = "NO_CHANGE"
    shared_gates = {
        "branches_share_exact_checkpoint": fork_audit["identical"],
        "intervention_actually_executed": (
            intervention_receipt["event_count"] > 0
        ),
    }
    formation_gates = {
        "direction_correct": magnitude is not None and magnitude > 0,
        "minimum_shift_600": magnitude is not None and magnitude >= 600,
        "predicted_shift_interval_contains_actual": (
            magnitude is not None and low <= magnitude <= high
        ),
        "both_branches_reached_transition": (
            baseline_transition is not None
            and intervention_transition is not None
        ),
    }
    stability_gates = {
        "stability_effect_direction_correct": (
            actual_stability_effect
            == intervention_spec["predicted_stability_effect"]
        )
    }
    gates = {**shared_gates, **formation_gates, **stability_gates}
    material = {
        "actual_shift_optimizer_steps": magnitude,
        "baseline_transition_step": baseline_transition,
        "direction": direction,
        "gates": gates,
        "formation_validation": {
            "gates": formation_gates,
            "status": (
                "FORMATION_CAUSAL_PASS"
                if all(formation_gates.values())
                else "FORMATION_CAUSAL_NOT_ESTABLISHED"
            ),
        },
        "intervention_transition_step": intervention_transition,
        "predicted_shift_high": high,
        "predicted_shift_low": low,
        "stability_validation": {
            "actual_effect": actual_stability_effect,
            "baseline": baseline_stability,
            "baseline_mean_deficit_below_0_90": baseline_deficit,
            "candidate_predicted_effect": intervention_spec[
                "predicted_stability_effect"
            ],
            "effect_direction_correct": (
                stability_gates["stability_effect_direction_correct"]
            ),
            "intervention": intervention_stability,
            "intervention_mean_deficit_below_0_90": (
                intervention_deficit
            ),
            "effect_epsilon": STABILITY_EFFECT_EPSILON,
            "gates": stability_gates,
            "primary_pass_gate": True,
            "status": (
                "STABILITY_CAUSAL_PASS"
                if all(stability_gates.values())
                else "STABILITY_CAUSAL_NOT_ESTABLISHED"
            ),
        },
        "schema": "causal-training-dual-dynamics-validation-v2",
        "shared_validation": {
            "gates": shared_gates,
            "status": (
                "CAUSAL_SHARED_EXECUTION_PASS"
                if all(shared_gates.values())
                else "CAUSAL_SHARED_EXECUTION_NOT_ESTABLISHED"
            ),
        },
        "status": (
            "CAUSAL_INTERVENTION_PASS"
            if all(gates.values())
            else "CAUSAL_INTERVENTION_NOT_ESTABLISHED"
        ),
    }
    material["validation_sha256"] = payload_sha256(material)
    return material
