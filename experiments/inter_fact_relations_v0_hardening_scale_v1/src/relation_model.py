from __future__ import annotations

from typing import Any

from ..common import ExperimentError, content_id


PRIMITIVE_RELATION_TYPES = frozenset(
    {
        "program_order",
        "generated_origin_dependency",
        "message_send_receive",
        "synchronizes_with",
        "reads_from",
        "conflicts_with",
    }
)
CAUSAL_PRIMITIVE_TYPES = frozenset(
    PRIMITIVE_RELATION_TYPES - {"conflicts_with"}
)
OCCURRENCE_PRIMITIVE_TYPES = frozenset(
    {"program_order", "message_send_receive", "synchronizes_with"}
)
FACT_PRIMITIVE_TYPES = frozenset(
    {"generated_origin_dependency", "reads_from", "conflicts_with"}
)
DERIVED_RELATION_TYPES = frozenset({"happens_before", "concurrent_with"})
ESTABLISHMENT_SOURCES = frozenset(
    {
        "generator_established",
        "wrapper_established",
        "inferred",
        "independent_reference",
    }
)
SYMMETRIC_RELATION_TYPES = frozenset({"conflicts_with", "concurrent_with"})


def make_evidence(
    *,
    evidence_kind: str,
    establishment_source: str,
    authority_id: str,
    execution_run_id: str,
    receipt_ref: str,
    occurrence_ids: list[str],
    fact_ids: list[str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    if establishment_source not in {
        "generator_established",
        "wrapper_established",
    }:
        raise ExperimentError("EVIDENCE_ESTABLISHMENT_SOURCE_INVALID")
    material = {
        "evidence_kind": evidence_kind,
        "establishment_source": establishment_source,
        "authority_id": authority_id,
        "execution_run_id": execution_run_id,
        "receipt_ref": receipt_ref,
        "occurrence_ids": sorted(occurrence_ids),
        "fact_ids": sorted(fact_ids),
        "payload": payload,
        "schema_version": "inter-fact-evidence-hardening-v1",
    }
    return {"evidence_id": content_id("ifev1_", material), **material}


def make_relation(
    *,
    endpoint_level: str,
    relation_type: str,
    source_id: str,
    target_id: str,
    establishment_source: str,
    authority_id: str,
    execution_run_id: str,
    evidence_refs: list[str] | None = None,
    rule_id: str | None = None,
    input_relation_refs: list[str] | None = None,
) -> dict[str, Any]:
    if endpoint_level not in {"occurrence", "fact"}:
        raise ExperimentError("RELATION_ENDPOINT_LEVEL_INVALID")
    if relation_type not in PRIMITIVE_RELATION_TYPES | DERIVED_RELATION_TYPES:
        raise ExperimentError("RELATION_TYPE_INVALID")
    if establishment_source not in ESTABLISHMENT_SOURCES:
        raise ExperimentError("RELATION_ESTABLISHMENT_SOURCE_INVALID")
    if establishment_source == "independent_reference":
        raise ExperimentError("INDEPENDENT_REFERENCE_IN_CANDIDATE_GRAPH")
    if source_id == target_id:
        raise ExperimentError("RELATION_SELF_EDGE_INVALID")
    if relation_type in SYMMETRIC_RELATION_TYPES and target_id < source_id:
        source_id, target_id = target_id, source_id
    evidence_refs = list(evidence_refs or [])
    input_relation_refs = list(input_relation_refs or [])
    if len(evidence_refs) != len(set(evidence_refs)):
        raise ExperimentError("DUPLICATE_EVIDENCE_REFERENCE")
    if len(input_relation_refs) != len(set(input_relation_refs)):
        raise ExperimentError("DUPLICATE_INPUT_RELATION_REFERENCE")
    material = {
        "endpoint_level": endpoint_level,
        "relation_type": relation_type,
        "source_id": source_id,
        "target_id": target_id,
        "establishment_source": establishment_source,
        "authority_id": authority_id,
        "execution_run_id": execution_run_id,
        "evidence_refs": sorted(evidence_refs),
        "rule_id": rule_id,
        "input_relation_refs": input_relation_refs,
        "schema_version": "inter-fact-relation-hardening-v1",
    }
    return {"relation_id": content_id("ifr1_", material), **material}


def relation_semantic_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["endpoint_level"],
        row["relation_type"],
        row["source_id"],
        row["target_id"],
    )
