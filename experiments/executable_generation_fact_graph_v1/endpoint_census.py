from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .canonical_graph import canonical_hash


def _fact_id(fact: dict[str, Any]) -> str:
    value = fact.get("fact_id") or fact.get("atomic_fact_id")
    if not isinstance(value, str) or not value:
        raise ValueError("NATIVE_FACT_ID_MISSING")
    return value


def _occurrence_id(fact: dict[str, Any]) -> str:
    coordinates = fact.get("coordinates")
    if coordinates is None:
        semantic = fact.get("semantic_projection")
        coordinates = (
            semantic.get("coordinates")
            if isinstance(semantic, dict)
            else fact
        )
    omega = coordinates.get("omega_bar", coordinates.get("omega"))
    if not isinstance(omega, dict):
        raise ValueError("NATIVE_FACT_OCCURRENCE_MISSING")
    value = (
        omega.get("concrete_occurrence_id")
        or omega.get("occurrence_id")
        or omega.get("generation_occurrence_id")
        or fact.get("occurrence_id")
        or omega.get("concrete_occurrence_instance_id")
        or omega.get("core_content_occurrence_id")
    )
    if not isinstance(value, str) or not value:
        raise ValueError("NATIVE_FACT_OCCURRENCE_ID_MISSING")
    return value


def _endpoint_kinds(relation: dict[str, Any]) -> tuple[str, str]:
    source = relation.get("source_endpoint_kind")
    target = relation.get("target_endpoint_kind")
    if source is None and target is None:
        level = relation.get("endpoint_level")
        source = level
        target = level
    if source not in {"fact", "occurrence"}:
        raise ValueError("RELATION_SOURCE_ENDPOINT_KIND_INVALID")
    if target not in {"fact", "occurrence"}:
        raise ValueError("RELATION_TARGET_ENDPOINT_KIND_INVALID")
    return source, target


def _mapping_ids(
    endpoint_kind: str,
    endpoint_id: str,
    *,
    facts_by_id: dict[str, list[str]],
    facts_by_occurrence: dict[str, list[str]],
) -> list[str]:
    if endpoint_kind == "fact":
        return facts_by_id.get(endpoint_id, [])
    return facts_by_occurrence.get(endpoint_id, [])


def _exclusive_mapping_class(
    source_count: int, target_count: int
) -> str:
    if source_count == 1 and target_count == 1:
        return "both_endpoints_unique_fact_mapping"
    if source_count == 0 and target_count == 0:
        return "both_endpoints_without_fact_mapping"
    if source_count == 0 or target_count == 0:
        return "one_endpoint_without_fact_mapping"
    if source_count > 1 and target_count > 1:
        return "both_endpoints_multiple_cartesian_ambiguity"
    return "one_endpoint_multiple_fact_mapping"


def census_primitive_relation_endpoints(
    atomic_fact_bundles: Iterable[dict[str, Any]],
    primitive_relation_sidecars: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Measure whether native primitive endpoints fit the v1 fact-only graph.

    This function audits; it never lifts, drops, fabricates, rewrites, or
    expands a relation.
    """

    facts_by_run_and_id: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    facts_by_run_and_occurrence: dict[
        str, dict[str, list[str]]
    ] = defaultdict(lambda: defaultdict(list))
    native_fact_count = 0
    for bundle in atomic_fact_bundles:
        run_id = bundle["execution_run_id"]
        for fact in bundle["facts"]:
            if fact.get("execution_run_id", run_id) != run_id:
                raise ValueError("ATOMIC_FACT_RUN_SCOPE_MISMATCH")
            fact_id = _fact_id(fact)
            occurrence_id = _occurrence_id(fact)
            facts_by_run_and_id[run_id][fact_id].append(fact_id)
            facts_by_run_and_occurrence[run_id][occurrence_id].append(fact_id)
            native_fact_count += 1

    relations: list[dict[str, Any]] = []
    for sidecar in primitive_relation_sidecars:
        run_id = sidecar["execution_run_id"]
        for relation in sidecar["relations"]:
            if relation["execution_run_id"] != run_id:
                raise ValueError("RELATION_RUN_SCOPE_MISMATCH")
            relations.append(relation)

    endpoint_type_counts: Counter[str] = Counter()
    relation_type_counts: Counter[str] = Counter()
    mapping_class_counts: Counter[str] = Counter()
    per_type: dict[str, Counter[str]] = defaultdict(Counter)
    unmappable: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for relation in sorted(relations, key=lambda row: row["relation_id"]):
        run_id = relation["execution_run_id"]
        source_kind, target_kind = _endpoint_kinds(relation)
        source_ids = _mapping_ids(
            source_kind,
            relation["source_id"],
            facts_by_id=facts_by_run_and_id[run_id],
            facts_by_occurrence=facts_by_run_and_occurrence[run_id],
        )
        target_ids = _mapping_ids(
            target_kind,
            relation["target_id"],
            facts_by_id=facts_by_run_and_id[run_id],
            facts_by_occurrence=facts_by_run_and_occurrence[run_id],
        )
        mapping_class = _exclusive_mapping_class(
            len(source_ids), len(target_ids)
        )
        endpoint_signature = f"{source_kind}->{target_kind}"
        relation_type = relation["relation_type"]
        endpoint_type_counts[endpoint_signature] += 1
        relation_type_counts[relation_type] += 1
        mapping_class_counts[mapping_class] += 1
        per_type[relation_type]["total"] += 1
        per_type[relation_type][mapping_class] += 1
        if len(source_ids) != 1 or len(target_ids) != 1:
            reason_codes = []
            if not source_ids:
                reason_codes.append("SOURCE_ENDPOINT_HAS_NO_FACT_MAPPING")
            elif len(source_ids) > 1:
                reason_codes.append(
                    "SOURCE_OCCURRENCE_HAS_MULTIPLE_FACT_MAPPINGS"
                )
            if not target_ids:
                reason_codes.append("TARGET_ENDPOINT_HAS_NO_FACT_MAPPING")
            elif len(target_ids) > 1:
                reason_codes.append(
                    "TARGET_OCCURRENCE_HAS_MULTIPLE_FACT_MAPPINGS"
                )
            row = {
                "execution_run_id": run_id,
                "relation_id": relation["relation_id"],
                "relation_type": relation_type,
                "source_endpoint_kind": source_kind,
                "source_endpoint_id": relation["source_id"],
                "source_fact_mapping_count": len(source_ids),
                "source_fact_ids": sorted(source_ids),
                "target_endpoint_kind": target_kind,
                "target_endpoint_id": relation["target_id"],
                "target_fact_mapping_count": len(target_ids),
                "target_fact_ids": sorted(target_ids),
                "mapping_class": mapping_class,
                "reason_codes": reason_codes,
            }
            unmappable.append(row)
            if len(source_ids) > 1 or len(target_ids) > 1:
                ambiguous.append(
                    {
                        **row,
                        "prohibited_cartesian_edge_count": (
                            len(source_ids) * len(target_ids)
                        ),
                    }
                )

    by_type = {}
    mapping_fields = (
        "both_endpoints_unique_fact_mapping",
        "one_endpoint_without_fact_mapping",
        "both_endpoints_without_fact_mapping",
        "one_endpoint_multiple_fact_mapping",
        "both_endpoints_multiple_cartesian_ambiguity",
    )
    for relation_type in sorted(relation_type_counts):
        counts = per_type[relation_type]
        by_type[relation_type] = {
            "total": counts["total"],
            **{field: counts[field] for field in mapping_fields},
            "unmappable_count": (
                counts["total"]
                - counts["both_endpoints_unique_fact_mapping"]
            ),
        }

    material = {
        "schema_version": "pure-fact-endpoint-census-v1",
        "vertex_definition": (
            "one graph vertex per GenerationBinding fact instance"
        ),
        "native_fact_count": native_fact_count,
        "primitive_relation_count": len(relations),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "endpoint_type_combination_counts": {
            key: endpoint_type_counts.get(key, 0)
            for key in (
                "fact->fact",
                "occurrence->occurrence",
                "fact->occurrence",
                "occurrence->fact",
            )
        },
        "mapping_class_counts": {
            key: mapping_class_counts.get(key, 0)
            for key in mapping_fields
        },
        "relation_type_mapping_counts": by_type,
        "legally_mappable_relation_count": (
            mapping_class_counts["both_endpoints_unique_fact_mapping"]
        ),
        "unmappable_primitive_relation_count": len(unmappable),
        "ambiguous_occurrence_relation_count": len(ambiguous),
        "unmappable_relations": unmappable,
        "ambiguous_occurrence_lifting": ambiguous,
        "prohibited_action_counts": {
            "discarded_relation_count": 0,
            "fabricated_fact_node_count": 0,
            "fabricated_relation_count": 0,
            "reattached_relation_count": 0,
            "cartesian_expanded_edge_count": 0,
        },
        "pure_fact_vertex_model_supported": not unmappable,
    }
    return {**material, "census_sha256": canonical_hash(material)}
