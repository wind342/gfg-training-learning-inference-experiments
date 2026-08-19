from __future__ import annotations

from typing import Any

from ..src.semantic_evidence_validator import validate_primitive_store
from .common import RuntimeScenarioBuilder


def build() -> RuntimeScenarioBuilder:
    builder = RuntimeScenarioBuilder(label="primitive-semantic-validation-v1")
    occurrence_a, facts_a = builder.add_occurrence(
        actor_id="actor-A",
        sequence_index=0,
        operation="write-and-send",
        semantic_slot=0,
        scope_id="semantic-validation",
    )
    occurrence_a2, facts_a2 = builder.add_occurrence(
        actor_id="actor-A",
        sequence_index=1,
        operation="next-operation",
        semantic_slot=1,
        scope_id="semantic-validation",
    )
    occurrence_b, facts_b = builder.add_occurrence(
        actor_id="actor-B",
        sequence_index=0,
        operation="receive-and-read",
        semantic_slot=2,
        scope_id="semantic-validation",
    )
    occurrence_c, facts_c = builder.add_occurrence(
        actor_id="actor-C",
        sequence_index=0,
        operation="conflicting-write",
        semantic_slot=3,
        scope_id="semantic-validation",
    )
    a = occurrence_a["concrete_occurrence_instance_id"]
    a2 = occurrence_a2["concrete_occurrence_instance_id"]
    b = occurrence_b["concrete_occurrence_instance_id"]
    builder.add_program_order(a, a2)
    builder.add_message(
        a,
        b,
        channel_id="semantic-validation-queue",
        payload={"message": "payload"},
    )
    builder.add_synchronization([a2], b, generation=1)
    builder.add_generated_origin(facts_a[1], facts_b[0])
    builder.add_reads_from(
        facts_a2[2],
        facts_b[2],
        resource_id="semantic-versioned-resource",
        version_id="version-1",
    )
    builder.add_conflict(
        facts_a[0],
        facts_c[0],
        resource_id="semantic-conflict-resource",
        version_id="version-current",
    )
    return builder


def run() -> dict[str, Any]:
    builder = build()
    validated = validate_primitive_store(
        builder.primitive_store(), builder.runtime_receipts()
    )
    return {
        "status": validated["status"],
        "execution_run_id": builder.run_id,
        "primitive_relation_count": validated["primitive_relation_count"],
        "relation_type_counts": validated["relation_type_counts"],
        "all_six_relation_types_exercised": all(
            count > 0 for count in validated["relation_type_counts"].values()
        ),
    }
