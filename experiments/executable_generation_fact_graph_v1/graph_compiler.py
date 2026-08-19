from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from generation_relation_core.snapshots import ValidatedSnapshot

from .canonical_graph import (
    canonical_graph_document,
    canonical_hash,
    content_id,
    implementation_hash,
)
from .graph_model import (
    ExecutableGenerationFactGraph,
    GraphFactNode,
    GraphMetadata,
    GraphRelationEdge,
)


def _entity_indexes(snapshot: ValidatedSnapshot) -> dict[str, dict[str, dict]]:
    tables = snapshot.tables
    return {
        "sources": {
            row["source_information_id"]: row
            for row in tables.source_information_records
        },
        "generated": {
            row["generated_origin_id"]: row for row in tables.generated_origins
        },
        "occurrences": {
            row["generation_occurrence_id"]: row
            for row in tables.generation_occurrences
        },
        "supports": {
            row["support_id"]: row for row in tables.perceptual_support_records
        },
        "dispositions": {
            row["disposition_id"]: row for row in tables.explicit_dispositions
        },
    }


def _origin_entity(binding: dict, indexes: dict[str, dict[str, dict]]) -> dict:
    reference = binding["origin_reference"]
    if reference["kind"] == "registered_source":
        return indexes["sources"][reference["source_information_id"]]
    return indexes["generated"][reference["generated_origin_id"]]


def _outcome_entity(binding: dict, indexes: dict[str, dict[str, dict]]) -> dict:
    reference = binding["outcome_reference"]
    if reference["kind"] == "support":
        return indexes["supports"][reference["support_id"]]
    return indexes["dispositions"][reference["disposition_id"]]


def _outcome_identity(binding: dict) -> str:
    reference = binding["outcome_reference"]
    return reference.get("support_id", reference.get("disposition_id"))


def _node(
    *,
    snapshot: ValidatedSnapshot,
    binding: dict,
    execution_run_id: str,
    native_identity: dict[str, Any] | None,
    indexes: dict[str, dict[str, dict]],
    schema_version: str,
) -> GraphFactNode:
    occurrence = indexes["occurrences"][binding["generation_occurrence_id"]]
    source_reference = {
        "reference": binding["origin_reference"],
        "entity": _origin_entity(binding, indexes),
    }
    outcome_reference = {
        "reference": binding["outcome_reference"],
        "entity": _outcome_entity(binding, indexes),
    }
    concrete_occurrence = {
        key: occurrence[key]
        for key in (
            "domain_scope_id",
            "generator_manifest_id",
            "occurrence_stage",
            "occurrence_type",
            "stable_instance_key",
            "occurrence_index",
            "occurrence_payload",
        )
    }
    content = {
        "u": source_reference,
        "tau": occurrence["transform_reference"],
        "omega_bar": concrete_occurrence,
        "z": outcome_reference,
        "rho": binding["relation_role"],
    }
    graph_node_id = content_id(
        "gfnode1_",
        {
            "graph_schema_version": schema_version,
            "execution_run_id": execution_run_id,
            "snapshot_id": snapshot.snapshot_id,
            "generation_binding_id": binding["generation_binding_id"],
        },
    )
    fact_content_hash = canonical_hash(content)
    instance = {
        "fact_content_hash": fact_content_hash,
        "execution_run_id": execution_run_id,
        "snapshot_id": snapshot.snapshot_id,
        "generation_binding_id": binding["generation_binding_id"],
        "occurrence_identity": binding["generation_occurrence_id"],
        "outcome_identity": _outcome_identity(binding),
    }
    return GraphFactNode(
        graph_node_id=graph_node_id,
        execution_run_id=execution_run_id,
        snapshot_id=snapshot.snapshot_id,
        generation_binding_id=binding["generation_binding_id"],
        domain_scope_id=binding["domain_scope_id"],
        source_reference=source_reference,
        realized_transformation=occurrence["transform_reference"],
        concrete_occurrence=concrete_occurrence,
        outcome_reference=outcome_reference,
        relation_role=binding["relation_role"],
        occurrence_identity=binding["generation_occurrence_id"],
        outcome_identity=_outcome_identity(binding),
        evidence_refs=sorted(binding["evidence_ids"]),
        fact_content_hash=fact_content_hash,
        node_instance_hash=canonical_hash(instance),
        native_fact_identity=native_identity,
    )


def _edge(
    *,
    execution_run_id: str,
    source_node_id: str,
    target_node_id: str,
    relation_type: str,
    relation_semantics: str,
    relation_payload: dict[str, Any],
    establishment_source: str,
    authority_id: str,
    evidence_refs: Iterable[str],
    rule_id: str | None,
    input_relation_refs: Iterable[str],
    primitive_or_derived: str,
    schema_version: str,
) -> GraphRelationEdge:
    if relation_semantics == "symmetric" and target_node_id < source_node_id:
        source_node_id, target_node_id = target_node_id, source_node_id
    evidence = sorted(evidence_refs)
    inputs = sorted(input_relation_refs)
    material = {
        "graph_schema_version": schema_version,
        "execution_run_id": execution_run_id,
        "source_graph_node_id": source_node_id,
        "target_graph_node_id": target_node_id,
        "relation_type": relation_type,
        "relation_semantics": relation_semantics,
        "relation_payload": relation_payload,
        "establishment_source": establishment_source,
        "authority_id": authority_id,
        "evidence_refs": evidence,
        "rule_id": rule_id,
        "input_relation_refs": inputs,
        "primitive_or_derived": primitive_or_derived,
    }
    instance_hash = canonical_hash(material)
    return GraphRelationEdge(
        graph_edge_id="gfedge1_" + instance_hash,
        execution_run_id=execution_run_id,
        source_graph_node_id=source_node_id,
        target_graph_node_id=target_node_id,
        relation_type=relation_type,
        relation_semantics=relation_semantics,
        relation_payload=relation_payload,
        establishment_source=establishment_source,
        authority_id=authority_id,
        evidence_refs=evidence,
        rule_id=rule_id,
        input_relation_refs=inputs,
        primitive_or_derived=primitive_or_derived,
        relation_instance_hash=instance_hash,
    )


def _generated_origin_edges(
    nodes: list[GraphFactNode],
    *,
    execution_run_id: str,
    schema_version: str,
) -> list[GraphRelationEdge]:
    by_support: dict[str, list[GraphFactNode]] = defaultdict(list)
    for node in nodes:
        reference = node.outcome_reference["reference"]
        if reference["kind"] == "support":
            by_support[reference["support_id"]].append(node)
    edges: list[GraphRelationEdge] = []
    for consumer in nodes:
        reference = consumer.source_reference["reference"]
        if reference["kind"] != "generated_origin":
            continue
        origin = consumer.source_reference["entity"]
        prior_support_id = origin["origin_payload"].get("prior_support_id")
        if not prior_support_id:
            raise ValueError("GENERATED_ORIGIN_PRIOR_SUPPORT_MISSING")
        producers = by_support.get(prior_support_id, [])
        if not producers:
            raise ValueError("GENERATED_ORIGIN_PRODUCER_BINDING_MISSING")
        for producer in sorted(producers, key=lambda row: row.graph_node_id):
            payload = {
                "projection_scope": "core_generated_origin",
                "generated_origin_id": origin["generated_origin_id"],
                "prior_support_id": prior_support_id,
                "producer_binding_id": producer.generation_binding_id,
                "consumer_binding_id": consumer.generation_binding_id,
            }
            edges.append(
                _edge(
                    execution_run_id=execution_run_id,
                    source_node_id=producer.graph_node_id,
                    target_node_id=consumer.graph_node_id,
                    relation_type="generated_origin_dependency",
                    relation_semantics="directed",
                    relation_payload=payload,
                    establishment_source="generator_established",
                    authority_id="validated-core-generated-origin-v1",
                    evidence_refs=consumer.evidence_refs,
                    rule_id="core-generated-origin-exact-support-v1",
                    input_relation_refs=[],
                    primitive_or_derived="primitive",
                    schema_version=schema_version,
                )
            )
    return edges


def _relation_edges(
    nodes: list[GraphFactNode],
    relation_store: dict[str, Any],
    lifting_contract: dict[str, Any],
    relation_registry: dict[str, Any],
    *,
    execution_run_id: str,
    schema_version: str,
) -> list[GraphRelationEdge]:
    binding_nodes = {row.generation_binding_id: row for row in nodes}
    occurrence_nodes: dict[str, list[GraphFactNode]] = defaultdict(list)
    for node in nodes:
        occurrence_nodes[node.occurrence_identity].append(node)
    for rows in occurrence_nodes.values():
        rows.sort(key=lambda row: row.generation_binding_id)

    edges: list[GraphRelationEdge] = []
    for relation in relation_store.get("relations", []):
        if relation["execution_run_id"] != execution_run_id:
            raise ValueError("EDGE_RUN_SCOPE_MISMATCH")
        relation_type = relation["relation_type"]
        registry = relation_registry["relations"].get(relation_type)
        if registry is None:
            raise ValueError("RELATION_TYPE_UNKNOWN")
        lifting = lifting_contract["rules"].get(relation_type)
        if lifting is None:
            raise ValueError("LIFTING_RULE_MISSING")
        endpoint_level = relation["endpoint_level"]
        if endpoint_level == "fact":
            source = binding_nodes.get(relation["source_id"])
            target = binding_nodes.get(relation["target_id"])
        elif endpoint_level == "occurrence":
            source_rows = occurrence_nodes.get(relation["source_id"], [])
            target_rows = occurrence_nodes.get(relation["target_id"], [])
            if not source_rows or not target_rows:
                raise ValueError(
                    "OCCURRENCE_ENDPOINT_WITHOUT_FACT_NODE:"
                    + relation["relation_id"]
                )
            if len(source_rows) != 1 or len(target_rows) != 1:
                raise ValueError(
                    "AMBIGUOUS_OCCURRENCE_TO_FACT_LIFTING:"
                    + relation["relation_id"]
                )
            source = source_rows[0]
            target = target_rows[0]
        else:
            raise ValueError("RELATION_ENDPOINT_LEVEL_INVALID")
        if source is None or target is None:
            raise ValueError("RELATION_ENDPOINT_MISSING")
        native_relation = relation.get("native_relation", relation)
        payload = {
            "projection_scope": "relation_store",
            "source_relation_store_id": relation_store["relation_store_id"],
            "source_relation_id": relation["relation_id"],
            "endpoint_level": endpoint_level,
            "native_source_id": native_relation["source_id"],
            "native_target_id": native_relation["target_id"],
            "native_relation": native_relation,
        }
        edges.append(
            _edge(
                execution_run_id=execution_run_id,
                source_node_id=source.graph_node_id,
                target_node_id=target.graph_node_id,
                relation_type=relation_type,
                relation_semantics=registry["semantics"],
                relation_payload=payload,
                establishment_source=relation["establishment_source"],
                authority_id=relation["authority_id"],
                evidence_refs=relation["evidence_refs"],
                rule_id=relation.get("rule_id") or lifting["rule_id"],
                input_relation_refs=relation.get("input_relation_refs", []),
                primitive_or_derived=(
                    "derived"
                    if relation.get("primitive_or_derived") == "derived"
                    or registry["kind"] == "derived"
                    else "primitive"
                ),
                schema_version=schema_version,
            )
        )
    return edges


def compile_generation_fact_graph(
    validated_snapshots: list[dict[str, Any]],
    validated_relation_store: dict[str, Any],
    capture_audit: dict[str, Any],
    graph_profile: dict[str, Any],
    relation_lifting_contract: dict[str, Any],
    *,
    relation_type_registry: dict[str, Any],
) -> ExecutableGenerationFactGraph:
    if not validated_snapshots:
        raise ValueError("VALIDATED_SNAPSHOT_REQUIRED")
    schema_version = graph_profile["graph_schema_version"]
    execution_ids = {row["execution_run_id"] for row in validated_snapshots}
    if len(execution_ids) != 1:
        raise ValueError("SNAPSHOT_RUN_SCOPE_MISMATCH")
    execution_run_id = next(iter(execution_ids))
    if validated_relation_store.get("execution_run_id", execution_run_id) != execution_run_id:
        raise ValueError("RELATION_STORE_RUN_SCOPE_MISMATCH")
    if capture_audit.get("execution_run_id", execution_run_id) != execution_run_id:
        raise ValueError("CAPTURE_AUDIT_RUN_SCOPE_MISMATCH")

    nodes: list[GraphFactNode] = []
    domains: set[str] = set()
    for snapshot_input in validated_snapshots:
        snapshot = snapshot_input["snapshot"]
        if not isinstance(snapshot, ValidatedSnapshot):
            raise TypeError("VALIDATED_SNAPSHOT_OBJECT_REQUIRED")
        snapshot_domains = {
            row["domain_scope_id"]
            for row in snapshot.tables.support_space_records
        }
        if len(snapshot_domains) != 1:
            raise ValueError("SNAPSHOT_DOMAIN_SCOPE_MISMATCH")
        domains.update(snapshot_domains)
        aliases = snapshot_input.get("native_binding_identities", {})
        indexes = _entity_indexes(snapshot)
        for binding in snapshot.tables.generation_bindings:
            domains.add(binding["domain_scope_id"])
            nodes.append(
                _node(
                    snapshot=snapshot,
                    binding=binding,
                    execution_run_id=execution_run_id,
                    native_identity=aliases.get(binding["generation_binding_id"]),
                    indexes=indexes,
                    schema_version=schema_version,
                )
            )
    nodes.sort(key=lambda row: row.graph_node_id)
    if len(domains) != 1:
        raise ValueError("GRAPH_DOMAIN_SCOPE_MISMATCH")
    if len({row.graph_node_id for row in nodes}) != len(nodes):
        raise ValueError("DUPLICATE_GRAPH_NODE_ID")

    edges = _generated_origin_edges(
        nodes,
        execution_run_id=execution_run_id,
        schema_version=schema_version,
    )
    edges.extend(
        _relation_edges(
            nodes,
            validated_relation_store,
            relation_lifting_contract,
            relation_type_registry,
            execution_run_id=execution_run_id,
            schema_version=schema_version,
        )
    )
    edges.sort(key=lambda row: row.graph_edge_id)
    if len({row.graph_edge_id for row in edges}) != len(edges):
        raise ValueError("DUPLICATE_GRAPH_EDGE_ID")

    primitive_count = sum(row.primitive_or_derived == "primitive" for row in edges)
    derived_count = len(edges) - primitive_count
    metadata = {
        "graph_schema_version": schema_version,
        "graph_id": "",
        "execution_run_id": execution_run_id,
        "domain_scope_id": next(iter(domains)),
        "source_snapshot_ids": sorted(
            row["snapshot"].snapshot_id for row in validated_snapshots
        ),
        "source_relation_store_ids": (
            [validated_relation_store["relation_store_id"]]
            if validated_relation_store.get("relation_store_id")
            else []
        ),
        "capture_audit_id": capture_audit["capture_audit_id"],
        "graph_profile_id": graph_profile["graph_profile_id"],
        "compiler_code_hash": implementation_hash("graph_compiler.py"),
        "validator_code_hash": implementation_hash("graph_validator.py"),
        "relation_contract_hash": canonical_hash(relation_lifting_contract),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "primitive_edge_count": primitive_count,
        "derived_edge_count": derived_count,
    }
    document = canonical_graph_document(
        {
            "metadata": metadata,
            "nodes": [row.to_dict() for row in nodes],
            "edges": [row.to_dict() for row in edges],
        }
    )
    return ExecutableGenerationFactGraph.from_dict(document)
