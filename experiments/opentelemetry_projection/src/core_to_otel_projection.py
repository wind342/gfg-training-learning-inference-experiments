from __future__ import annotations

from collections import defaultdict
from typing import Any

from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot

from .canonical_otel import (
    SCHEMA_VERSION,
    canonicalize_trace,
    occurrence_span_key,
    root_span_key,
    trace_key,
)
from .projection_errors import ProjectionError
from .snapshot_access import require_validated_snapshot


def _run_id(occurrences: list[dict[str, Any]]) -> str:
    run_ids = {
        occurrence.get("occurrence_payload", {}).get("run_id")
        for occurrence in occurrences
    }
    if None in run_ids or len(run_ids) != 1:
        raise ProjectionError("SELECTED_CONTEXT_MISMATCH", "RUN_ID")
    return str(next(iter(run_ids)))


def _semantic_key(occurrence: dict[str, Any]) -> str:
    return occurrence_span_key(
        occurrence_index=occurrence["occurrence_index"],
        occurrence_type=occurrence["occurrence_type"],
        stable_instance_key=occurrence["stable_instance_key"],
    )


def project_core_to_otel(
    snapshot: ValidatedSnapshot,
    validation: SnapshotValidation,
) -> dict[str, Any]:
    """Project only validated Core facts into the frozen OTel shadow schema."""

    require_validated_snapshot(snapshot, validation)
    tables = snapshot.tables
    occurrences = list(tables.generation_occurrences)
    if not occurrences:
        raise ProjectionError("MISSING_SPAN", "NO_OCCURRENCES")
    indices = [row["occurrence_index"] for row in occurrences]
    if sorted(indices) != list(range(len(indices))):
        raise ProjectionError("OCCURRENCE_ORDER_INVALID")
    run_id = _run_id(occurrences)
    root_key = root_span_key(run_id)

    occurrence_by_id = {row["generation_occurrence_id"]: row for row in occurrences}
    bindings_by_occurrence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    support_producer_candidates: dict[str, set[str]] = defaultdict(set)
    for binding in tables.generation_bindings:
        occurrence_id = binding["generation_occurrence_id"]
        if occurrence_id not in occurrence_by_id:
            raise ProjectionError("UNKNOWN_OCCURRENCE", occurrence_id)
        bindings_by_occurrence[occurrence_id].append(binding)
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            support_producer_candidates[outcome["support_id"]].add(occurrence_id)
    support_producer: dict[str, str] = {}
    for support_id, candidates in support_producer_candidates.items():
        if len(candidates) != 1:
            raise ProjectionError("LINK_EDGE_MISMATCH", f"PRODUCER:{support_id}")
        support_producer[support_id] = next(iter(candidates))

    prior_support_by_generated_origin = {
        row["generated_origin_id"]: row["origin_payload"].get("prior_support_id")
        for row in tables.generated_origins
    }
    spans: list[dict[str, Any]] = [
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
                "execution.run_id": run_id,
            },
            "events": [
                {
                    "name": "query.execution",
                    "attributes": {"execution.run_id": run_id},
                }
            ],
        }
    ]

    for occurrence in sorted(occurrences, key=lambda row: row["occurrence_index"]):
        occurrence_id = occurrence["generation_occurrence_id"]
        bindings = bindings_by_occurrence.get(occurrence_id, [])
        if not bindings:
            raise ProjectionError("MISSING_SPAN", occurrence_id)
        outcome_kinds = {binding["outcome_reference"]["kind"] for binding in bindings}
        if len(outcome_kinds) != 1:
            raise ProjectionError("OCCURRENCE_MERGED", occurrence_id)
        outcome_kind = next(iter(outcome_kinds))
        transform = occurrence["transform_reference"]
        if not isinstance(transform, dict):
            raise ProjectionError("ATTRIBUTE_MISMATCH", occurrence_id)
        operator_type = transform.get("operator_type")
        transform_stage = transform.get("stage")
        stage = occurrence["occurrence_stage"]
        expected_type = f"relational_{operator_type}_execution"
        if occurrence["occurrence_type"] != expected_type:
            raise ProjectionError("OPERATION_TYPE_MISMATCH", occurrence_id)
        if transform_stage != stage:
            raise ProjectionError("ATTRIBUTE_MISMATCH", occurrence_id)

        linked_semantic_keys: list[str] = []
        for binding in bindings:
            origin = binding["origin_reference"]
            if origin["kind"] == "registered_source":
                continue
            generated_id = origin["generated_origin_id"]
            prior_support_id = prior_support_by_generated_origin.get(generated_id)
            if not isinstance(prior_support_id, str):
                raise ProjectionError("LINK_EDGE_MISMATCH", generated_id)
            producer_id = support_producer.get(prior_support_id)
            if producer_id is None:
                raise ProjectionError("LINK_EDGE_MISMATCH", prior_support_id)
            linked_semantic_keys.append(_semantic_key(occurrence_by_id[producer_id]))

        semantic_key = _semantic_key(occurrence)
        spans.append(
            {
                "span_semantic_key": semantic_key,
                "name": f"operator.{operator_type}",
                "parent_semantic_key": root_key,
                "linked_semantic_keys": linked_semantic_keys,
                "status": "OK",
                "attributes": {
                    "logical.order": occurrence["occurrence_index"] + 1,
                    "span.kind": "occurrence",
                    "operation.type": operator_type,
                    "operation.stage": stage,
                    "occurrence.type": occurrence["occurrence_type"],
                    "occurrence.stable_instance_key": occurrence["stable_instance_key"],
                    "occurrence.index": occurrence["occurrence_index"],
                    "outcome.kind": outcome_kind,
                    "transform.operator_type": operator_type,
                    "transform.stage": transform_stage,
                    "occurrence.cardinality": 1,
                },
                "events": [
                    {
                        "name": "generation.occurrence",
                        "attributes": {
                            "occurrence.index": occurrence["occurrence_index"],
                            "outcome.kind": outcome_kind,
                        },
                    }
                ],
            }
        )
    return canonicalize_trace(
        {
            "schema_version": SCHEMA_VERSION,
            "trace_semantic_key": trace_key(run_id),
            "spans": spans,
        }
    )
