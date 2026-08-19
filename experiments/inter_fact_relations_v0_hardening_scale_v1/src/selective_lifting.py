from __future__ import annotations

from typing import Any

from ..common import ExperimentError


ALL_FACTS_OF_OCCURRENCE = "ALL_FACTS_OF_OCCURRENCE"
EVIDENCE_SELECTED_FACTS = "EVIDENCE_SELECTED_FACTS"
RELATION_TYPE_SPECIFIC_LIFTING = "RELATION_TYPE_SPECIFIC_LIFTING"


def lifting_policy(relation_type: str) -> dict[str, Any]:
    if relation_type == "happens_before":
        return {
            "strategy": ALL_FACTS_OF_OCCURRENCE,
            "meaning": "order_of_producing_occurrences_only",
            "materialization": "query_only",
        }
    if relation_type in {
        "generated_origin_dependency",
        "reads_from",
        "conflicts_with",
    }:
        return {
            "strategy": EVIDENCE_SELECTED_FACTS,
            "meaning": "fact_specific_relation",
            "materialization": "primitive_exact_endpoints",
        }
    if relation_type == "concurrent_with":
        return {
            "strategy": RELATION_TYPE_SPECIFIC_LIFTING,
            "meaning": "concurrent_producing_occurrences",
            "materialization": "query_only_with_complete_capture",
        }
    raise ExperimentError("LIFTING_RELATION_TYPE_UNKNOWN")


def selected_fact_pairs(
    *,
    relation_type: str,
    source_occurrence_id: str,
    target_occurrence_id: str,
    occurrence_to_facts: dict[str, list[str]],
    evidence_selected_pair: tuple[str, str] | None = None,
    capture_complete: bool = False,
) -> list[tuple[str, str]]:
    policy = lifting_policy(relation_type)
    if policy["strategy"] == EVIDENCE_SELECTED_FACTS:
        if evidence_selected_pair is None:
            raise ExperimentError("FACT_SPECIFIC_EVIDENCE_ENDPOINTS_REQUIRED")
        return [evidence_selected_pair]
    if policy["strategy"] == RELATION_TYPE_SPECIFIC_LIFTING and not capture_complete:
        return []
    return [
        (left, right)
        for left in sorted(occurrence_to_facts[source_occurrence_id])
        for right in sorted(occurrence_to_facts[target_occurrence_id])
    ]


def compare_lifting_strategies() -> dict[str, Any]:
    return {
        "ALL_FACTS_OF_OCCURRENCE": {
            "appropriate_for": ["happens_before_as_occurrence_order"],
            "inappropriate_for": [
                "generated_origin_dependency",
                "reads_from",
                "conflicts_with",
            ],
        },
        "EVIDENCE_SELECTED_FACTS": {
            "appropriate_for": [
                "generated_origin_dependency",
                "reads_from",
                "conflicts_with",
            ],
            "inappropriate_for": [],
        },
        "RELATION_TYPE_SPECIFIC_LIFTING": {
            "appropriate_for": [
                "mixed_endpoint_models",
                "concurrent_with_under_complete_capture",
            ],
            "inappropriate_for": ["untyped_global_cartesian_lifting"],
        },
        "recommended": "RELATION_TYPE_SPECIFIC_LIFTING",
    }
