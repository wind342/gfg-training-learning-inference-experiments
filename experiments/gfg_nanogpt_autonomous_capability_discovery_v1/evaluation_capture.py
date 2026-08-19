from __future__ import annotations

from typing import Any

from .nanogpt_adapter import detect_transition


def capability_transition(
    metrics: list[dict[str, Any]],
) -> int | None:
    return detect_transition(
        metrics,
        train_threshold=0.99,
        pre_transition_validation_max=0.30,
        validation_threshold=0.90,
        sustained_points=3,
    )


def is_prediction_cut(
    current: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> bool:
    if int(current["step"]) < 500:
        return False
    if len(metrics) < 3:
        return False
    if not all(
        row["train_accuracy"] >= 0.99 for row in metrics[-3:]
    ):
        return False
    if not 0.20 <= current["validation_accuracy"] <= 0.75:
        return False
    return capability_transition(metrics) is None
