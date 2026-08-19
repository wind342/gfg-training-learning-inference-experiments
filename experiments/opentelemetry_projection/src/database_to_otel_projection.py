from __future__ import annotations

from collections import defaultdict

from .canonical_otel import (
    SCHEMA_VERSION,
    canonicalize_trace,
    occurrence_span_key,
    root_span_key,
    trace_key,
)
from .database_projection import DatabaseDomainProjection, DatabaseOccurrence
from .projection_errors import ProjectionError


def _key(occurrence: DatabaseOccurrence) -> str:
    return occurrence_span_key(
        occurrence_index=occurrence.occurrence_index,
        occurrence_type=occurrence.occurrence_type,
        stable_instance_key=occurrence.stable_instance_key,
    )


def project_database_to_otel(
    database: DatabaseDomainProjection,
) -> dict:
    """Independently extract the OTel substructure from the database view."""

    occurrence_by_id = {
        occurrence.generation_occurrence_id: occurrence
        for occurrence in database.occurrences
    }
    if len(occurrence_by_id) != len(database.occurrences):
        raise ProjectionError("UNKNOWN_OCCURRENCE", "DUPLICATE_DATABASE_OCCURRENCE")
    bindings_by_occurrence = defaultdict(list)
    producer_candidates: dict[str, set[str]] = defaultdict(set)
    for binding in database.bindings:
        if binding.occurrence_id not in occurrence_by_id:
            raise ProjectionError("UNKNOWN_OCCURRENCE", binding.occurrence_id)
        bindings_by_occurrence[binding.occurrence_id].append(binding)
        if binding.outcome_kind == "support":
            producer_candidates[binding.outcome_id].add(binding.occurrence_id)
    producers = {}
    for support_id, candidates in producer_candidates.items():
        if len(candidates) != 1:
            raise ProjectionError("LINK_EDGE_MISMATCH", support_id)
        producers[support_id] = next(iter(candidates))
    prior_support_by_origin = {
        bridge.generated_origin_id: bridge.prior_support_id
        for bridge in database.generated_bridges
    }
    root_key = root_span_key(database.run_id)
    spans = [
        {
            "span_semantic_key": root_key,
            "name": "query.execute",
            "parent_semantic_key": None,
            "linked_semantic_keys": [],
            "status": "OK",
            "attributes": {
                "logical.order": 0,
                "span.kind": "query_root",
                "execution.kind": "deterministic_relational_query",
                "execution.run_id": database.run_id,
            },
            "events": [
                {
                    "name": "query.execution",
                    "attributes": {"execution.run_id": database.run_id},
                }
            ],
        }
    ]
    for occurrence in sorted(
        database.occurrences, key=lambda item: item.occurrence_index
    ):
        bindings = bindings_by_occurrence.get(occurrence.generation_occurrence_id, [])
        if not bindings:
            raise ProjectionError("MISSING_SPAN", occurrence.generation_occurrence_id)
        outcomes = {binding.outcome_kind for binding in bindings}
        if len(outcomes) != 1:
            raise ProjectionError(
                "OCCURRENCE_MERGED", occurrence.generation_occurrence_id
            )
        outcome_kind = next(iter(outcomes))
        operator_type = occurrence.transform_operator_type
        if occurrence.occurrence_type != f"relational_{operator_type}_execution":
            raise ProjectionError(
                "OPERATION_TYPE_MISMATCH", occurrence.generation_occurrence_id
            )
        if occurrence.transform_stage != occurrence.occurrence_stage:
            raise ProjectionError(
                "ATTRIBUTE_MISMATCH", occurrence.generation_occurrence_id
            )
        links = []
        for binding in bindings:
            if binding.origin_kind != "generated_origin":
                continue
            support_id = prior_support_by_origin.get(binding.origin_id)
            producer_id = producers.get(support_id or "")
            if producer_id is None:
                raise ProjectionError("LINK_EDGE_MISMATCH", binding.origin_id)
            links.append(_key(occurrence_by_id[producer_id]))
        spans.append(
            {
                "span_semantic_key": _key(occurrence),
                "name": f"operator.{operator_type}",
                "parent_semantic_key": root_key,
                "linked_semantic_keys": links,
                "status": "OK",
                "attributes": {
                    "logical.order": occurrence.occurrence_index + 1,
                    "span.kind": "occurrence",
                    "operation.type": operator_type,
                    "operation.stage": occurrence.occurrence_stage,
                    "occurrence.type": occurrence.occurrence_type,
                    "occurrence.stable_instance_key": occurrence.stable_instance_key,
                    "occurrence.index": occurrence.occurrence_index,
                    "outcome.kind": outcome_kind,
                    "transform.operator_type": operator_type,
                    "transform.stage": occurrence.transform_stage,
                    "occurrence.cardinality": 1,
                },
                "events": [
                    {
                        "name": "generation.occurrence",
                        "attributes": {
                            "occurrence.index": occurrence.occurrence_index,
                            "outcome.kind": outcome_kind,
                        },
                    }
                ],
            }
        )
    return canonicalize_trace(
        {
            "schema_version": SCHEMA_VERSION,
            "trace_semantic_key": trace_key(database.run_id),
            "spans": spans,
        }
    )
