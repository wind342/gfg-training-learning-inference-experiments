from __future__ import annotations

from typing import Any

from ..src.semantic_evidence_validator import validate_primitive_store
from .common import RuntimeScenarioBuilder


def build() -> RuntimeScenarioBuilder:
    builder = RuntimeScenarioBuilder(label="reads-from-version-v1")
    _, producer_facts = builder.add_occurrence(
        actor_id="writer",
        sequence_index=0,
        operation="write-version",
        semantic_slot=0,
        scope_id="reads-from-version",
    )
    _, consumer_facts = builder.add_occurrence(
        actor_id="reader",
        sequence_index=0,
        operation="read-version",
        semantic_slot=1,
        scope_id="reads-from-version",
    )
    builder.add_reads_from(
        producer_facts[1],
        consumer_facts[2],
        resource_id="account-record",
        version_id="account-version-0007",
    )
    return builder


def run() -> dict[str, Any]:
    builder = build()
    validated = validate_primitive_store(
        builder.primitive_store(), builder.runtime_receipts()
    )
    receipt = builder.reads_from_receipts[0]
    return {
        "status": validated["status"],
        "reads_from_relation_count": validated["relation_type_counts"][
            "reads_from"
        ],
        "resource_id": receipt["resource_id"],
        "observed_version_id": receipt["version_id"],
        "source_fact_id": receipt["source_fact_id"],
        "target_fact_id": receipt["target_fact_id"],
        "resource_identity_alone_was_not_used": True,
    }
