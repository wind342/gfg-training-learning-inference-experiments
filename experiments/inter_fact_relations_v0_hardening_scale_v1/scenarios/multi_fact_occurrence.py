from __future__ import annotations

from typing import Any

from ..src.capture_auditor import audit_capture
from ..src.indexed_candidate_resolver import IndexedCandidateResolver
from ..src.selective_lifting import (
    compare_lifting_strategies,
    selected_fact_pairs,
)
from ..src.semantic_evidence_validator import validate_primitive_store
from .common import RuntimeScenarioBuilder


def run() -> dict[str, Any]:
    builder = RuntimeScenarioBuilder(label="multi-fact-occurrence-v1")
    occurrence_a, facts_a = builder.add_occurrence(
        actor_id="producer",
        sequence_index=0,
        operation="emit-three-facts",
        semantic_slot=0,
        scope_id="multi-fact",
        fact_count=3,
    )
    occurrence_b, facts_b = builder.add_occurrence(
        actor_id="consumer",
        sequence_index=0,
        operation="consume-selected-fact",
        semantic_slot=1,
        scope_id="multi-fact",
        fact_count=1,
    )
    selected_source = facts_a[1]
    target = facts_b[0]
    dependency = builder.add_generated_origin(selected_source, target)
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    capture = audit_capture(builder.capture_contract(), receipts, validated)
    resolver = IndexedCandidateResolver(
        execution_run_id=builder.run_id,
        primitive_store=validated,
        capture_audit=capture,
        lifting_rules=compare_lifting_strategies(),
    )
    source_occurrence_id = occurrence_a["concrete_occurrence_instance_id"]
    target_occurrence_id = occurrence_b["concrete_occurrence_instance_id"]
    happens_before = resolver.happens_before(
        source_occurrence_id, target_occurrence_id
    )
    occurrence_order_pairs = selected_fact_pairs(
        relation_type="happens_before",
        source_occurrence_id=source_occurrence_id,
        target_occurrence_id=target_occurrence_id,
        occurrence_to_facts=resolver.occurrence_to_facts_map,
    )
    dependency_pairs = selected_fact_pairs(
        relation_type="generated_origin_dependency",
        source_occurrence_id=source_occurrence_id,
        target_occurrence_id=target_occurrence_id,
        occurrence_to_facts=resolver.occurrence_to_facts_map,
        evidence_selected_pair=(
            dependency["source_id"],
            dependency["target_id"],
        ),
    )
    false_dependency_pairs = [
        [fact["fact_id"], target["fact_id"]]
        for fact in (facts_a[0], facts_a[2])
        if (fact["fact_id"], target["fact_id"]) in dependency_pairs
    ]
    return {
        "status": "PASS" if not false_dependency_pairs else "FAIL",
        "occurrence_happens_before": happens_before,
        "source_fact_count": len(facts_a),
        "occurrence_order_lift_pair_count": len(occurrence_order_pairs),
        "fact_specific_dependency_pairs": [
            list(pair) for pair in dependency_pairs
        ],
        "false_dependency_pairs": false_dependency_pairs,
        "recommended_lifting": "RELATION_TYPE_SPECIFIC_LIFTING",
        "strategy_comparison": compare_lifting_strategies(),
    }
