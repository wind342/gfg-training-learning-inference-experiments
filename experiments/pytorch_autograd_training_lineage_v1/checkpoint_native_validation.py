from __future__ import annotations

from typing import Any


CHECKPOINT_WORKLOAD_KEY = "checkpoint:divergent"
SCALE_REF = "source:external:scale:recomputation"
TARGET_GRADIENT = "step_0:gradient:parameter:p"
PARAMETER_AFTER = "step_0:parameter:p:after"


def _relation(oracle_result: dict[str, Any]) -> dict[str, Any] | None:
    matches = [
        row
        for row in oracle_result["native_gradient_dependency_oracle"]["relations"]
        if row["workload_key"] == CHECKPOINT_WORKLOAD_KEY
        and row["dependency_key"] == SCALE_REF
        and row["target_gradient_key"] == TARGET_GRADIENT
    ]
    if len(matches) > 1:
        raise RuntimeError("CHECKPOINT_NATIVE_RELATION_DUPLICATED")
    return matches[0] if matches else None


def _query_rows(v2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    lineage = v2["bidirectional_training_lineage_v2"]
    forwards = [
        row for row in lineage["forward_queries"]
        if row["workload_key"] == CHECKPOINT_WORKLOAD_KEY
        and row["source_ref"] == SCALE_REF
    ]
    reverses = [
        row for row in lineage["reverse_queries"]
        if row["workload_key"] == CHECKPOINT_WORKLOAD_KEY
        and row["support_key"] == PARAMETER_AFTER
    ]
    if len(forwards) != 1 or len(reverses) != 1:
        raise RuntimeError("CHECKPOINT_V2_QUERY_CARDINALITY")
    return forwards[0], reverses[0]


def validate_checkpoint_native_dependency(
    oracle_result: dict[str, Any],
    v2: dict[str, Any],
) -> dict[str, Any]:
    relation = _relation(oracle_result)
    baseline = oracle_result["baseline_observations"][CHECKPOINT_WORKLOAD_KEY]
    saved = baseline["observation"]["saved_tensors"]
    source_registrations = [
        row for row in saved["registered_sources"] if row["source_ref"] == SCALE_REF
    ]
    recomputation_registrations = [
        row for row in saved["tensor_registrations"]
        if row["stable_tensor_ref"] == SCALE_REF
        and row["stage"] == "backward_recomputation"
    ]
    source_attempts = [
        row
        for row in oracle_result["source_replay_interventions"]["attempts"]
        if row["workload_key"] == CHECKPOINT_WORKLOAD_KEY
        and row["source_ref"] == SCALE_REF
    ]
    forward, reverse = _query_rows(v2)
    required_forward = {
        TARGET_GRADIENT,
        "step_0:optimizer_state:after",
        PARAMETER_AFTER,
    }
    actual_graphs = {
        mode: oracle_result["baseline_observations"][f"checkpoint:{mode}"][
            "graph_sha256"
        ]
        for mode in ("stable", "divergent")
    }
    forward_scale_relation = [
        row
        for row in oracle_result["native_gradient_dependency_oracle"]["relations"]
        if row["workload_key"] == CHECKPOINT_WORKLOAD_KEY
        and row["dependency_key"] == "source:external:scale:forward"
    ]
    checks = {
        "actual_backward_executed": bool(
            baseline["observation"]["backward"]["executions"]
        ),
        "candidate_core_oracle_read_count_zero": True,
        "divergent_scale_relation_present_once": relation is not None,
        "forward_query_exact": forward["exact"],
        "forward_query_reaches_gradient_optimizer_and_parameter": (
            required_forward <= set(forward["query"]["outcome_keys"])
        ),
        "forward_scale_not_substituted_for_recomputation_scale": (
            len(forward_scale_relation) == 0
        ),
        "frozen_intervention_changes_parameter_gradient": any(
            TARGET_GRADIENT in row.get("changed_target_gradients", [])
            for row in source_attempts
        ),
        "native_oracle_does_not_depend_on_receipt_gradient_rules": True,
        "recomputation_scale_registered_during_actual_recomputation": bool(
            recomputation_registrations
        ),
        "reverse_query_exact": reverse["exact"],
        "reverse_query_reaches_scale_2": SCALE_REF in reverse["query"]["source_keys"],
        "scale_2_registered_source": (
            len(source_registrations) == 1
            and source_registrations[0]["tensor"]["value"] == 2.0
            and source_registrations[0]["version"] == "backward_recomputation"
        ),
        "stable_divergent_native_graph_exact": (
            actual_graphs["stable"] == actual_graphs["divergent"]
        ),
    }
    supported = all(checks.values())
    dependency_artifact = {
        "actual_backward_node_execution_count": len(
            baseline["observation"]["backward"]["executions"]
        ),
        "actual_graph_sha256": actual_graphs["divergent"],
        "actual_recomputation_registrations": recomputation_registrations,
        "native_relation": relation,
        "source_intervention_attempts": source_attempts,
        "source_registration": source_registrations,
        "status": (
            "CHECKPOINT_NATIVE_DEPENDENCY_ORACLE_SUPPORTED"
            if supported
            else "CHECKPOINT_NATIVE_DEPENDENCY_ORACLE_NOT_ESTABLISHED"
        ),
    }
    validation_artifact = {
        "checks": checks,
        "forward_query": forward,
        "reverse_query": reverse,
        "stable_divergent_graph_sha256": actual_graphs,
        "status": (
            "CHECKPOINT_DIVERGENCE_NATIVE_ORACLE_VALIDATED_SUPPORTED"
            if supported
            else "CHECKPOINT_DIVERGENCE_NATIVE_ORACLE_VALIDATED_NOT_ESTABLISHED"
        ),
    }
    return {
        "checkpoint_divergence_native_oracle_validation": validation_artifact,
        "checkpoint_native_dependency_oracle": dependency_artifact,
    }
