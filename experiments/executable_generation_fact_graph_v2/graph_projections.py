from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from generation_relation_core.entities import generation_binding

from .canonical_graph import canonical_hash
from .graph_model import ValidatedGenerationFactGraphV2


def project_gamma(
    validated_graph: ValidatedGenerationFactGraphV2,
) -> dict[str, Any]:
    facts = []
    for node in sorted(
        validated_graph.graph.fact_nodes,
        key=lambda row: (
            row.execution_run_id,
            row.snapshot_id,
            row.generation_binding_id,
        ),
    ):
        binding = generation_binding(
            domain_scope_id=node.domain_scope_id,
            origin_reference=node.u["reference"],
            generation_occurrence_id=node.generation_occurrence_id,
            outcome_reference=node.z["reference"],
            relation_role=node.rho,
            evidence_ids=node.evidence_refs,
        )
        if binding["generation_binding_id"] != node.generation_binding_id:
            raise ValueError("GAMMA_BINDING_ID_MISMATCH")
        facts.append(
            {
                "execution_run_id": node.execution_run_id,
                "snapshot_id": node.snapshot_id,
                "generation_binding_id": node.generation_binding_id,
                "domain_scope_id": node.domain_scope_id,
                "binding": binding,
                "coordinates": {
                    "u": node.u,
                    "tau": node.tau,
                    "omega_bar": node.omega_bar,
                    "z": node.z,
                    "rho": node.rho,
                },
                "generation_occurrence_id": (
                    node.generation_occurrence_id
                ),
                "concrete_occurrence_instance_id": (
                    node.concrete_occurrence_instance_id
                ),
                "outcome_identity": node.outcome_identity,
                "evidence_refs": node.evidence_refs,
                "native_fact_id": node.native_fact_id,
                "native_fact": node.native_fact,
            }
        )
    material = {
        "schema_version": "generation-fact-state-projection-v2",
        "graph_id": validated_graph.graph_id,
        "fact_count": len(facts),
        "facts": facts,
    }
    return {**material, "projection_sha256": canonical_hash(material)}


def project_occurrence_view(
    validated_graph: ValidatedGenerationFactGraphV2,
) -> dict[str, Any]:
    occurrences = [
        row.to_dict()
        for row in sorted(
            validated_graph.graph.occurrence_nodes,
            key=lambda item: item.graph_node_id,
        )
    ]
    relations = [
        row.to_dict()
        for row in sorted(
            validated_graph.graph.relation_edges,
            key=lambda item: item.graph_edge_id,
        )
        if row.source_node_kind == "generation_occurrence"
        and row.target_node_kind == "generation_occurrence"
    ]
    material = {
        "schema_version": "occurrence-execution-view-v2",
        "graph_id": validated_graph.graph_id,
        "occurrence_count": len(occurrences),
        "occurrences": occurrences,
        "relation_count": len(relations),
        "relations": relations,
    }
    return {**material, "projection_sha256": canonical_hash(material)}


def project_primitive_relation_sidecar(
    validated_graph: ValidatedGenerationFactGraphV2,
) -> dict[str, Any]:
    graph = validated_graph.graph
    envelopes = {
        row["relation_store_id"]: row
        for row in graph.metadata.relation_store_envelopes
    }
    by_store: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph.relation_edges:
        if edge.primitive_or_derived != "primitive":
            continue
        by_store[edge.source_relation_store_id].append(
            edge.native_relation
        )
    stores = []
    for store_id in sorted(envelopes):
        envelope = dict(envelopes[store_id])
        if envelope["relation_store_id"] != store_id:
            raise ValueError("RELATION_STORE_ENVELOPE_ID_MISMATCH")
        store = {
            **envelope,
            "relations": sorted(
                by_store.get(store_id, []),
                key=lambda row: row["relation_id"],
            ),
            "evidence": sorted(
                graph.metadata.relation_evidence_records,
                key=lambda row: row.get(
                    "evidence_id", canonical_hash(row)
                ),
            ),
        }
        stores.append(store)
    material = {
        "schema_version": "primitive-relation-sidecar-projection-v2",
        "graph_id": validated_graph.graph_id,
        "store_count": len(stores),
        "relation_count": sum(
            len(row["relations"]) for row in stores
        ),
        "stores": stores,
    }
    return {**material, "projection_sha256": canonical_hash(material)}


def project_fact_only_graph(
    validated_graph: ValidatedGenerationFactGraphV2,
) -> dict[str, Any]:
    retained = [
        row.to_dict()
        for row in validated_graph.graph.relation_edges
        if row.source_node_kind == "generation_fact"
        and row.target_node_kind == "generation_fact"
    ]
    omitted = [
        row.to_dict()
        for row in validated_graph.graph.relation_edges
        if row.source_node_kind != "generation_fact"
        or row.target_node_kind != "generation_fact"
    ]
    signature_counts = Counter(
        row["source_node_kind"] + "->" + row["target_node_kind"]
        for row in omitted
    )
    material = {
        "schema_version": "fact-only-graph-projection-v2",
        "graph_id": validated_graph.graph_id,
        "projection_kind": "fact-only projection",
        "complete_primitive_sidecar_recovery_claimed": False,
        "fact_nodes": [
            row.to_dict()
            for row in validated_graph.graph.fact_nodes
        ],
        "retained_primitive_relations": retained,
        "retained_relation_count": len(retained),
        "omitted_occurrence_level_relations": omitted,
        "omitted_relation_count": len(omitted),
        "omitted_endpoint_signature_counts": dict(
            sorted(signature_counts.items())
        ),
        "information_boundary": (
            "Occurrence-level primitive relations are intentionally absent "
            "from this lossy projection and remain authoritative only in "
            "the complete heterogeneous graph."
        ),
    }
    return {**material, "projection_sha256": canonical_hash(material)}


def project_signed_algebra(
    validated_graph: ValidatedGenerationFactGraphV2,
    contract: dict[str, Any],
) -> dict[str, Any]:
    from .adapters.signed_algebra_adapter import (
        project_graph_to_signed_algebra,
    )

    return project_graph_to_signed_algebra(validated_graph, contract)
