from __future__ import annotations

from typing import Any

from ..common import content_id
from ..common import ExperimentError


def semantic_occurrence_key(
    *, actor_id: str, operation: str, semantic_slot: int
) -> str:
    return content_id(
        "semocc1_",
        {
            "actor_id": actor_id,
            "operation": operation,
            "semantic_slot": semantic_slot,
        },
    )


def core_content_occurrence_id(semantic_key: str) -> str:
    """Experimental stand-in for the frozen Core content-addressed join identity."""

    return content_id(
        "coreocc1_",
        {
            "semantic_occurrence_key": semantic_key,
            "identity_semantics": "content_addressed_without_runtime_run_membership",
        },
    )


def concrete_occurrence_instance_id(
    *, execution_run_id: str, semantic_key: str, runtime_ordinal: int
) -> str:
    return content_id(
        "concocc1_",
        {
            "execution_run_id": execution_run_id,
            "semantic_occurrence_key": semantic_key,
            "runtime_ordinal": runtime_ordinal,
        },
    )


def make_occurrence(
    *,
    execution_run_id: str,
    actor_id: str,
    sequence_index: int,
    operation: str,
    semantic_slot: int,
    scope_id: str,
) -> dict[str, Any]:
    semantic_key = semantic_occurrence_key(
        actor_id=actor_id,
        operation=operation,
        semantic_slot=semantic_slot,
    )
    return {
        "execution_run_id": execution_run_id,
        "semantic_occurrence_key": semantic_key,
        "concrete_occurrence_instance_id": concrete_occurrence_instance_id(
            execution_run_id=execution_run_id,
            semantic_key=semantic_key,
            runtime_ordinal=sequence_index,
        ),
        "core_content_occurrence_id": core_content_occurrence_id(semantic_key),
        "actor_id": actor_id,
        "sequence_index": sequence_index,
        "operation": operation,
        "scope_id": scope_id,
    }


def make_fact(
    *,
    occurrence: dict[str, Any],
    fact_slot: int,
    value: Any,
    support_id: str | None = None,
) -> dict[str, Any]:
    semantic_projection = {
        "semantic_occurrence_key": occurrence["semantic_occurrence_key"],
        "fact_slot": fact_slot,
        "value": value,
        "coordinates": {
            "u": {"kind": "controlled_input", "fact_slot": fact_slot},
            "tau": {"operation": occurrence["operation"]},
            "omega_bar": {
                "core_content_occurrence_id": occurrence[
                    "core_content_occurrence_id"
                ]
            },
            "z": {"value": value},
            "rho": "produces",
        },
    }
    support_id = support_id or content_id(
        "support1_",
        {
            "semantic_occurrence_key": occurrence["semantic_occurrence_key"],
            "fact_slot": fact_slot,
            "value": value,
        },
    )
    core_fact_id = content_id(
        "corefact1_",
        {
            "core_content_occurrence_id": occurrence["core_content_occurrence_id"],
            "fact_slot": fact_slot,
            "support_id": support_id,
        },
    )
    return {
        "fact_id": content_id(
            "fact1_",
            {
                "concrete_occurrence_instance_id": occurrence[
                    "concrete_occurrence_instance_id"
                ],
                "fact_slot": fact_slot,
                "support_id": support_id,
            },
        ),
        "semantic_fact_id": content_id("semfact1_", semantic_projection),
        "core_fact_id": core_fact_id,
        "occurrence_id": occurrence["concrete_occurrence_instance_id"],
        "support_id": support_id,
        "fact_slot": fact_slot,
        "semantic_projection": semantic_projection,
    }


def compare_runs(
    *,
    left_output: Any,
    right_output: Any,
    left_facts: list[dict[str, Any]],
    right_facts: list[dict[str, Any]],
    left_relation_hash: str,
    right_relation_hash: str,
) -> dict[str, Any]:
    left_semantic = sorted(row["semantic_fact_id"] for row in left_facts)
    right_semantic = sorted(row["semantic_fact_id"] for row in right_facts)
    left_core = sorted(row["core_fact_id"] for row in left_facts)
    right_core = sorted(row["core_fact_id"] for row in right_facts)
    left_strict = sorted(row["fact_id"] for row in left_facts)
    right_strict = sorted(row["fact_id"] for row in right_facts)
    return {
        "ordinary_output_equal": left_output == right_output,
        "semantic_fact_projection_equal": left_semantic == right_semantic,
        "core_fact_id_equal": left_core == right_core,
        "concrete_run_scoped_gamma_equal": left_strict == right_strict,
        "exact_gamma_equality_status": (
            "ESTABLISHED"
            if left_strict == right_strict
            else "EXACT_GAMMA_EQUALITY_NOT_ESTABLISHED"
        ),
        "relation_graph_different": left_relation_hash != right_relation_hash,
    }


def validate_identity_record(occurrence: dict[str, Any]) -> None:
    if (
        occurrence["concrete_occurrence_instance_id"]
        == occurrence["semantic_occurrence_key"]
    ):
        raise ExperimentError("SEMANTIC_OCCURRENCE_AS_CONCRETE_INSTANCE")
    expected = concrete_occurrence_instance_id(
        execution_run_id=occurrence["execution_run_id"],
        semantic_key=occurrence["semantic_occurrence_key"],
        runtime_ordinal=occurrence["sequence_index"],
    )
    if occurrence["concrete_occurrence_instance_id"] != expected:
        raise ExperimentError("CONCRETE_OCCURRENCE_IDENTITY_INVALID")


def validate_frozen_fact_coordinates(fact: dict[str, Any]) -> None:
    coordinates = fact["semantic_projection"]["coordinates"]
    if set(coordinates) != {"u", "tau", "omega_bar", "z", "rho"}:
        if "logical_clock" in coordinates or "vector_clock" in coordinates:
            raise ExperimentError("VECTOR_CLOCK_AS_SIXTH_COORDINATE")
        raise ExperimentError("FROZEN_FIVE_COORDINATE_BOUNDARY_VIOLATED")
