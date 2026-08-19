from __future__ import annotations

from collections import defaultdict
from typing import Any

from generation_relation_core.entities import generation_binding

from .canonical_graph import canonical_hash
from .graph_model import ValidatedGenerationFactGraph


def project_atomic_generation_state(
    validated_graph: ValidatedGenerationFactGraph,
) -> dict[str, Any]:
    facts = []
    for node in sorted(
        validated_graph.graph.nodes,
        key=lambda row: (
            row.execution_run_id,
            row.snapshot_id,
            row.generation_binding_id,
        ),
    ):
        binding = generation_binding(
            domain_scope_id=node.domain_scope_id,
            origin_reference=node.source_reference["reference"],
            generation_occurrence_id=node.occurrence_identity,
            outcome_reference=node.outcome_reference["reference"],
            relation_role=node.relation_role,
            evidence_ids=node.evidence_refs,
        )
        if binding["generation_binding_id"] != node.generation_binding_id:
            raise ValueError("GRAPH_ATOMIC_BINDING_ID_MISMATCH")
        facts.append(
            {
                "execution_run_id": node.execution_run_id,
                "snapshot_id": node.snapshot_id,
                "generation_binding_id": node.generation_binding_id,
                "domain_scope_id": node.domain_scope_id,
                "binding": binding,
                "coordinates": {
                    "u": node.source_reference,
                    "tau": node.realized_transformation,
                    "omega_bar": node.concrete_occurrence,
                    "z": node.outcome_reference,
                    "rho": node.relation_role,
                },
                "occurrence_identity": node.occurrence_identity,
                "outcome_identity": node.outcome_identity,
                "native_fact_identity": node.native_fact_identity,
            }
        )
    material = {
        "schema_version": "graph-atomic-state-projection-v1",
        "graph_id": validated_graph.graph_id,
        "fact_count": len(facts),
        "facts": facts,
    }
    return {**material, "projection_sha256": canonical_hash(material)}


def project_relation_store(
    validated_graph: ValidatedGenerationFactGraph,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in validated_graph.graph.edges:
        payload = edge.relation_payload
        if payload.get("projection_scope") != "relation_store":
            continue
        grouped[payload["source_relation_store_id"]].append(
            payload["native_relation"]
        )
    stores = [
        {
            "relation_store_id": store_id,
            "relations": sorted(rows, key=lambda row: row["relation_id"]),
        }
        for store_id, rows in sorted(grouped.items())
    ]
    material = {
        "schema_version": "graph-relation-store-projection-v1",
        "graph_id": validated_graph.graph_id,
        "store_count": len(stores),
        "relation_count": sum(len(row["relations"]) for row in stores),
        "stores": stores,
    }
    return {**material, "projection_sha256": canonical_hash(material)}


def project_signed_generation_algebra(
    validated_graph: ValidatedGenerationFactGraph,
    contract: dict[str, Any],
) -> dict[str, Any]:
    from .adapters.signed_algebra_adapter import project_graph_to_signed_algebra

    return project_graph_to_signed_algebra(validated_graph, contract)

