from __future__ import annotations

from typing import Any

from experiments.gfg_resnet_cifar_training_learning_generalization_v1.gfg import CompactGFG


def add_update_event(
    graph: CompactGFG,
    event_index: int,
    epoch: int,
    batch_ids: list[int],
    training_loss: float,
    analysis: dict[str, Any],
) -> None:
    parameter_state = graph.source(
        "parameter_adamw_receiving_state",
        {"sha256": analysis["pre_state_sha256"], "epoch": epoch},
    )
    batch = graph.source("identified_cifar100_training_batch", {"sample_ids": batch_ids})
    update = graph.outcome(
        "formed_actual_adamw_update",
        {"sha256": analysis["delta_state_sha256"], "loss": training_loss},
    )
    occurrence = graph.occurrence(
        "actual_moe_adamw_training_update",
        {"event_index": event_index, "epoch": epoch},
    )
    graph.fact(parameter_state, "adamw_training_step", occurrence, update, "receiving_state")
    graph.fact(batch, "adamw_training_step", occurrence, update, "training_source")
    update_source = graph.source(
        "generated_actual_update",
        {"outcome_id": update, "sha256": analysis["delta_state_sha256"]},
    )
    response = graph.outcome(
        "finite_amplitude_target_response",
        {"target_ids": analysis["selected_target_ids"], "alpha_grid": analysis["alpha_grid"]},
    )
    replay = graph.occurrence(
        "finite_amplitude_update_replay",
        {"event_index": event_index, "epoch": epoch},
    )
    graph.fact(parameter_state, "apply_realized_update_path", replay, response, "receiving_state")
    graph.fact(update_source, "apply_realized_update_path", replay, response, "update_action")
    response_source = graph.source("generated_target_response", {"outcome_id": response})
    support = graph.outcome(
        "distributed_expert_support_reallocation",
        {"target_ids": analysis["selected_target_ids"], "component_count": 4},
    )
    coalition = graph.occurrence(
        "identified_expert_coalition_gating",
        {"event_index": event_index, "coalition_count_per_state": 16},
    )
    graph.fact(response_source, "expert_support_adjudication", coalition, support, "functional_response")
    route = graph.outcome("top2_query_conditioned_routing", analysis["routing"])
    routing_occurrence = graph.occurrence(
        "router_audit", {"event_index": event_index, "epoch": epoch}
    )
    graph.fact(parameter_state, "route_identified_tokens", routing_occurrence, route, "receiving_state")


def add_inference_event(
    graph: CompactGFG,
    seed: int,
    trained_state_sha256: str,
    inference: dict[str, Any],
) -> None:
    learned = graph.source("trained_moe_parameter_version", {"sha256": trained_state_sha256})
    query = graph.source("identified_cifar100_query_set", {"seed": seed})
    result = graph.outcome(
        "frozen_query_conditioned_inference",
        {"tests": inference["tests"], "pass": inference["pass"]},
    )
    occurrence = graph.occurrence("frozen_moe_inference", {"seed": seed})
    graph.fact(learned, "project_learned_expert_support", occurrence, result, "learned_state")
    graph.fact(query, "project_learned_expert_support", occurrence, result, "current_query")
    for component, row in inference["rollbacks"].items():
        rollback = graph.occurrence("prelearning_component_rollback", {"seed": seed, "component": component})
        outcome = graph.outcome(
            "rollback_inference_change",
            {"component": component, "logit_max_abs_change": row["logit_max_abs_change"]},
        )
        graph.fact(learned, "rollback_component_version", rollback, outcome, "trained_version")
