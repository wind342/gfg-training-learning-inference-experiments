from __future__ import annotations

from typing import Any

from experiments.gfg_resnet_cifar_training_learning_generalization_v1.gfg import (
    CompactGFG,
)


def add_update_event(
    graph: CompactGFG,
    event_index: int,
    epoch: int,
    batch_ids: list[int],
    timestep_sha256: str,
    noise_sha256: str,
    training_loss: float,
    analysis: dict[str, Any],
) -> None:
    receiving = graph.source(
        "parameter_adamw_receiving_state",
        {"sha256": analysis["pre_state_sha256"], "epoch": epoch},
    )
    batch = graph.source(
        "identified_cifar10_training_batch", {"sample_ids": batch_ids}
    )
    diffusion_occurrences = graph.source(
        "identified_diffusion_occurrences",
        {"timestep_sha256": timestep_sha256, "noise_sha256": noise_sha256},
    )
    update = graph.outcome(
        "formed_actual_parameter_update",
        {"sha256": analysis["delta_state_sha256"], "loss": training_loss},
    )
    update_occurrence = graph.occurrence(
        "actual_adamw_diffusion_training_update",
        {"event_index": event_index, "epoch": epoch},
    )
    graph.fact(
        receiving,
        "adamw_epsilon_prediction_training_step",
        update_occurrence,
        update,
        "receiving_state",
    )
    graph.fact(
        batch,
        "adamw_epsilon_prediction_training_step",
        update_occurrence,
        update,
        "training_source",
    )
    graph.fact(
        diffusion_occurrences,
        "adamw_epsilon_prediction_training_step",
        update_occurrence,
        update,
        "noise_and_timestep_occurrences",
    )
    response = graph.outcome(
        "finite_amplitude_residual_boundary_response",
        {
            "target_ids": analysis["selected_target_ids"],
            "alpha_grid": analysis["alpha_grid"],
        },
    )
    response_occurrence = graph.occurrence(
        "finite_amplitude_realized_update_replay",
        {"event_index": event_index, "epoch": epoch},
    )
    graph.fact(
        receiving,
        "apply_realized_update_path",
        response_occurrence,
        response,
        "receiving_state",
    )
    update_source = graph.source(
        "generated_actual_update",
        {"outcome_id": update, "sha256": analysis["delta_state_sha256"]},
    )
    graph.fact(
        update_source,
        "apply_realized_update_path",
        response_occurrence,
        response,
        "actual_training_action",
    )
    support = graph.outcome(
        "distributed_unet_support_reallocation",
        {"target_ids": analysis["selected_target_ids"], "component_count": 4},
    )
    support_occurrence = graph.occurrence(
        "unet_component_coalition_gating",
        {"event_index": event_index, "coalition_count_per_state": 16},
    )
    response_source = graph.source(
        "generated_functional_response", {"outcome_id": response}
    )
    graph.fact(
        response_source,
        "support_coalition_adjudication",
        support_occurrence,
        support,
        "functional_response",
    )


__all__ = ["CompactGFG", "add_update_event"]
