from __future__ import annotations

from collections import defaultdict
from typing import Any

from generation_relation_core.snapshots import ValidatedSnapshot

from .canonical_graph import (
    canonical_graph_document,
    canonical_hash,
    content_id,
    implementation_hash,
)
from .endpoint_registry import validate_occurrence_catalog
from .graph_model import (
    ExecutableGenerationFactGraphV2,
    FactNode,
    GraphMetadata,
    IncidenceEdge,
    OccurrenceNode,
    RelationEdge,
)


def _indexes(snapshot: ValidatedSnapshot) -> dict[str, dict[str, dict]]:
    tables = snapshot.tables
    return {
        "sources": {
            row["source_information_id"]: row
            for row in tables.source_information_records
        },
        "generated": {
            row["generated_origin_id"]: row
            for row in tables.generated_origins
        },
        "occurrences": {
            row["generation_occurrence_id"]: row
            for row in tables.generation_occurrences
        },
        "supports": {
            row["support_id"]: row
            for row in tables.perceptual_support_records
        },
        "dispositions": {
            row["disposition_id"]: row
            for row in tables.explicit_dispositions
        },
    }


def _origin(
    binding: dict[str, Any], indexes: dict[str, dict[str, dict]]
) -> dict[str, Any]:
    reference = binding["origin_reference"]
    entity = (
        indexes["sources"][reference["source_information_id"]]
        if reference["kind"] == "registered_source"
        else indexes["generated"][reference["generated_origin_id"]]
    )
    return {"reference": reference, "entity": entity}


def _outcome(
    binding: dict[str, Any], indexes: dict[str, dict[str, dict]]
) -> dict[str, Any]:
    reference = binding["outcome_reference"]
    entity = (
        indexes["supports"][reference["support_id"]]
        if reference["kind"] == "support"
        else indexes["dispositions"][reference["disposition_id"]]
    )
    return {"reference": reference, "entity": entity}


def _outcome_id(binding: dict[str, Any]) -> str:
    reference = binding["outcome_reference"]
    return reference.get("support_id", reference.get("disposition_id"))


def _required_occurrences(
    snapshot_inputs: list[dict[str, Any]],
    relation_store: dict[str, Any],
) -> set[str]:
    native_by_binding = {
        binding_id: identity.get("native_occurrence_id")
        for item in snapshot_inputs
        for binding_id, identity in item.get(
            "native_binding_identities", {}
        ).items()
    }
    required = set()
    for item in snapshot_inputs:
        snapshot = item["snapshot"]
        for binding in snapshot.tables.generation_bindings:
            required.add(
                native_by_binding.get(binding["generation_binding_id"])
                or binding["generation_occurrence_id"]
            )
    for relation in relation_store.get("relations", []):
        source_kind = relation.get(
            "source_endpoint_kind", relation.get("endpoint_level")
        )
        target_kind = relation.get(
            "target_endpoint_kind", relation.get("endpoint_level")
        )
        if source_kind == "occurrence":
            required.add(relation["source_id"])
        if target_kind == "occurrence":
            required.add(relation["target_id"])
    return required


def _fact_node(
    *,
    snapshot: ValidatedSnapshot,
    binding: dict[str, Any],
    indexes: dict[str, dict[str, dict]],
    execution_run_id: str,
    graph_schema_version: str,
    concrete_occurrence_id: str,
    native_identity: dict[str, Any] | None,
) -> FactNode:
    occurrence = indexes["occurrences"][
        binding["generation_occurrence_id"]
    ]
    u = _origin(binding, indexes)
    z = _outcome(binding, indexes)
    omega = {"generation_occurrence": occurrence}
    content = {
        "u": u,
        "tau": occurrence["transform_reference"],
        "omega_bar": omega,
        "z": z,
        "rho": binding["relation_role"],
    }
    node_id = content_id(
        "gff2_",
        {
            "graph_schema_version": graph_schema_version,
            "node_kind": "generation_fact",
            "execution_run_id": execution_run_id,
            "snapshot_id": snapshot.snapshot_id,
            "generation_binding_id": binding["generation_binding_id"],
        },
    )
    content_hash = canonical_hash(content)
    native_fact_id = (
        native_identity.get("native_fact_id")
        if native_identity is not None
        else None
    )
    native_fact = (
        native_identity.get("native_fact")
        if native_identity is not None
        else None
    )
    instance = {
        "fact_content_hash": content_hash,
        "execution_run_id": execution_run_id,
        "snapshot_id": snapshot.snapshot_id,
        "generation_binding_id": binding["generation_binding_id"],
        "concrete_occurrence_instance_id": concrete_occurrence_id,
        "outcome_identity": _outcome_id(binding),
        "native_fact_id": native_fact_id,
    }
    return FactNode(
        node_kind="generation_fact",
        graph_node_id=node_id,
        execution_run_id=execution_run_id,
        snapshot_id=snapshot.snapshot_id,
        generation_binding_id=binding["generation_binding_id"],
        domain_scope_id=binding["domain_scope_id"],
        u=u,
        tau=occurrence["transform_reference"],
        omega_bar=omega,
        z=z,
        rho=binding["relation_role"],
        generation_occurrence_id=binding["generation_occurrence_id"],
        concrete_occurrence_instance_id=concrete_occurrence_id,
        outcome_identity=_outcome_id(binding),
        evidence_refs=sorted(binding["evidence_ids"]),
        native_fact_id=native_fact_id,
        native_fact=native_fact,
        fact_content_hash=content_hash,
        fact_instance_hash=canonical_hash(instance),
    )


def _occurrence_node(
    row: dict[str, Any],
    *,
    graph_schema_version: str,
) -> OccurrenceNode:
    content = {
        key: row.get(key)
        for key in (
            "generation_occurrence_id",
            "occurrence_type",
            "occurrence_stage",
            "stable_instance_key",
            "occurrence_index",
            "transform_reference",
            "occurrence_payload",
            "generator_manifest_id",
            "evidence_refs",
            "catalog_authority",
        )
    }
    node_id = content_id(
        "gfo2_",
        {
            "graph_schema_version": graph_schema_version,
            "node_kind": "generation_occurrence",
            "execution_run_id": row["execution_run_id"],
            "concrete_occurrence_instance_id": row[
                "concrete_occurrence_instance_id"
            ],
        },
    )
    content_hash = canonical_hash(content)
    instance = {
        "occurrence_content_hash": content_hash,
        "execution_run_id": row["execution_run_id"],
        "concrete_occurrence_instance_id": row[
            "concrete_occurrence_instance_id"
        ],
    }
    return OccurrenceNode(
        node_kind="generation_occurrence",
        graph_node_id=node_id,
        execution_run_id=row["execution_run_id"],
        concrete_occurrence_instance_id=row[
            "concrete_occurrence_instance_id"
        ],
        generation_occurrence_id=row.get("generation_occurrence_id"),
        occurrence_type=row["occurrence_type"],
        occurrence_stage=row["occurrence_stage"],
        stable_instance_key=row["stable_instance_key"],
        occurrence_index=row.get("occurrence_index"),
        transform_reference=row.get("transform_reference"),
        occurrence_payload=row.get("occurrence_payload", {}),
        generator_manifest_id=row.get("generator_manifest_id"),
        evidence_refs=sorted(row.get("evidence_refs", [])),
        catalog_authority=row["catalog_authority"],
        occurrence_content_hash=content_hash,
        occurrence_instance_hash=canonical_hash(instance),
    )


def _incidence(
    occurrence: OccurrenceNode,
    fact: FactNode,
    *,
    graph_schema_version: str,
) -> IncidenceEdge:
    material = {
        "graph_schema_version": graph_schema_version,
        "edge_kind": "fact_occurrence_incidence",
        "relation_type": "realizes_fact",
        "execution_run_id": fact.execution_run_id,
        "source_occurrence_node_id": occurrence.graph_node_id,
        "target_fact_node_id": fact.graph_node_id,
        "source_concrete_occurrence_instance_id": (
            occurrence.concrete_occurrence_instance_id
        ),
        "target_generation_binding_id": fact.generation_binding_id,
        "derivation": "exact_from_generation_binding_occurrence",
    }
    instance_hash = canonical_hash(material)
    return IncidenceEdge(
        **{
            key: value
            for key, value in material.items()
            if key != "graph_schema_version"
        },
        graph_edge_id="gfi2_" + instance_hash,
        incidence_instance_hash=instance_hash,
    )


def _relation_edge(
    relation: dict[str, Any],
    *,
    source_node_id: str,
    source_node_kind: str,
    target_node_id: str,
    target_node_kind: str,
    store_id: str,
    graph_schema_version: str,
    registry: dict[str, Any],
) -> RelationEdge:
    relation_type = relation["relation_type"]
    spec = registry["relations"].get(relation_type)
    if spec is None:
        raise ValueError("RELATION_TYPE_UNKNOWN:" + relation_type)
    signature = f"{source_node_kind}->{target_node_kind}"
    if signature not in spec["allowed_endpoint_signatures"]:
        raise ValueError(
            "RELATION_ENDPOINT_SIGNATURE_INVALID:"
            + relation_type
            + ":"
            + signature
        )
    primitive_or_derived = relation.get(
        "primitive_or_derived", spec["kind"]
    )
    rule_id = relation.get("rule_id")
    inputs = sorted(relation.get("input_relation_refs", []))
    if primitive_or_derived == "derived" and (
        not rule_id or not inputs
    ):
        raise ValueError("DERIVED_RELATION_TRACEABILITY_MISSING")
    native_relation = relation.get("native_relation", relation)
    material = {
        "graph_schema_version": graph_schema_version,
        "execution_run_id": relation["execution_run_id"],
        "source_node_id": source_node_id,
        "source_node_kind": source_node_kind,
        "target_node_id": target_node_id,
        "target_node_kind": target_node_kind,
        "native_source_id": native_relation["source_id"],
        "native_target_id": native_relation["target_id"],
        "relation_type": relation_type,
        "relation_semantics": spec["semantics"],
        "relation_payload": relation.get("relation_payload", {}),
        "primitive_or_derived": primitive_or_derived,
        "establishment_source": relation["establishment_source"],
        "authority_id": relation["authority_id"],
        "evidence_refs": sorted(relation.get("evidence_refs", [])),
        "rule_id": rule_id,
        "input_relation_refs": inputs,
        "original_relation_id": relation["relation_id"],
        "source_relation_store_id": store_id,
        "native_relation": native_relation,
    }
    instance_hash = canonical_hash(material)
    return RelationEdge(
        edge_kind="execution_or_generation_relation",
        graph_edge_id="gfr2_" + instance_hash,
        relation_instance_hash=instance_hash,
        **{
            key: value
            for key, value in material.items()
            if key != "graph_schema_version"
        },
    )


def compile_executable_generation_fact_graph_v2(
    validated_snapshots: list[dict[str, Any]],
    validated_primitive_relation_store: dict[str, Any],
    occurrence_endpoint_catalog: dict[str, Any],
    capture_audit: dict[str, Any],
    graph_profile: dict[str, Any],
    relation_type_registry: dict[str, Any],
) -> ExecutableGenerationFactGraphV2:
    if not validated_snapshots:
        raise ValueError("VALIDATED_SNAPSHOT_REQUIRED")
    run_ids = {
        row["execution_run_id"] for row in validated_snapshots
    }
    run_ids.add(validated_primitive_relation_store["execution_run_id"])
    run_ids.add(occurrence_endpoint_catalog["execution_run_id"])
    run_ids.add(capture_audit["execution_run_id"])
    if len(run_ids) != 1:
        raise ValueError("GRAPH_RUN_SCOPE_MISMATCH")
    execution_run_id = next(iter(run_ids))
    schema_version = graph_profile["graph_schema_version"]

    required_occurrences = _required_occurrences(
        validated_snapshots, validated_primitive_relation_store
    )
    validate_occurrence_catalog(
        occurrence_endpoint_catalog,
        required_occurrence_ids=required_occurrences,
    )
    catalog_rows = {
        row["concrete_occurrence_instance_id"]: row
        for row in occurrence_endpoint_catalog["occurrences"]
    }
    core_to_concrete: dict[str, str] = {}
    for concrete_id, row in catalog_rows.items():
        core_id = row.get("generation_occurrence_id")
        if core_id:
            previous = core_to_concrete.setdefault(core_id, concrete_id)
            if previous != concrete_id:
                raise ValueError(
                    "CORE_OCCURRENCE_HAS_MULTIPLE_CATALOG_NODES"
                )

    fact_nodes: list[FactNode] = []
    domains: set[str] = set()
    for snapshot_input in validated_snapshots:
        snapshot = snapshot_input["snapshot"]
        if not isinstance(snapshot, ValidatedSnapshot):
            raise TypeError("VALIDATED_SNAPSHOT_OBJECT_REQUIRED")
        indexes = _indexes(snapshot)
        native_identities = snapshot_input.get(
            "native_binding_identities", {}
        )
        domains.update(
            row["domain_scope_id"]
            for row in snapshot.tables.support_space_records
        )
        for binding in snapshot.tables.generation_bindings:
            domains.add(binding["domain_scope_id"])
            native = native_identities.get(
                binding["generation_binding_id"]
            )
            concrete_id = (
                native.get("native_occurrence_id")
                or core_to_concrete.get(
                    binding["generation_occurrence_id"],
                    binding["generation_occurrence_id"],
                )
                if native is not None
                else core_to_concrete.get(
                    binding["generation_occurrence_id"],
                    binding["generation_occurrence_id"],
                )
            )
            if concrete_id not in catalog_rows:
                raise ValueError(
                    "FACT_OCCURRENCE_MISSING_FROM_CATALOG:"
                    + binding["generation_binding_id"]
                )
            catalog_core = catalog_rows[concrete_id].get(
                "generation_occurrence_id"
            )
            if catalog_core not in {
                None,
                binding["generation_occurrence_id"],
            }:
                raise ValueError("FACT_OCCURRENCE_ALIAS_MISMATCH")
            fact_nodes.append(
                _fact_node(
                    snapshot=snapshot,
                    binding=binding,
                    indexes=indexes,
                    execution_run_id=execution_run_id,
                    graph_schema_version=schema_version,
                    concrete_occurrence_id=concrete_id,
                    native_identity=native,
                )
            )
    if len(domains) != 1:
        raise ValueError("GRAPH_DOMAIN_SCOPE_MISMATCH")
    fact_nodes.sort(key=lambda row: row.graph_node_id)
    if len({row.graph_node_id for row in fact_nodes}) != len(fact_nodes):
        raise ValueError("DUPLICATE_FACT_NODE_ID")

    occurrence_nodes = sorted(
        (
            _occurrence_node(
                row, graph_schema_version=schema_version
            )
            for row in catalog_rows.values()
        ),
        key=lambda row: row.graph_node_id,
    )
    if len({row.graph_node_id for row in occurrence_nodes}) != len(
        occurrence_nodes
    ):
        raise ValueError("DUPLICATE_OCCURRENCE_NODE_ID")
    occurrence_by_concrete = {
        row.concrete_occurrence_instance_id: row
        for row in occurrence_nodes
    }
    incidence_edges = sorted(
        (
            _incidence(
                occurrence_by_concrete[
                    fact.concrete_occurrence_instance_id
                ],
                fact,
                graph_schema_version=schema_version,
            )
            for fact in fact_nodes
        ),
        key=lambda row: row.graph_edge_id,
    )

    facts_by_binding = {
        row.generation_binding_id: row for row in fact_nodes
    }
    facts_by_native = {
        row.native_fact_id: row
        for row in fact_nodes
        if row.native_fact_id is not None
    }
    occurrences_by_core = {
        row.generation_occurrence_id: row
        for row in occurrence_nodes
        if row.generation_occurrence_id is not None
    }
    relation_edges: list[RelationEdge] = []
    store_id = validated_primitive_relation_store["relation_store_id"]
    for relation in validated_primitive_relation_store.get(
        "relations", []
    ):
        if relation["execution_run_id"] != execution_run_id:
            raise ValueError("RELATION_RUN_SCOPE_MISMATCH")
        source_kind = relation.get(
            "source_endpoint_kind", relation.get("endpoint_level")
        )
        target_kind = relation.get(
            "target_endpoint_kind", relation.get("endpoint_level")
        )
        if source_kind == "fact":
            source = facts_by_native.get(
                relation["source_id"]
            ) or facts_by_binding.get(relation["source_id"])
        elif source_kind == "occurrence":
            source = occurrence_by_concrete.get(
                relation["source_id"]
            ) or occurrences_by_core.get(relation["source_id"])
        else:
            raise ValueError("RELATION_SOURCE_ENDPOINT_KIND_INVALID")
        if target_kind == "fact":
            target = facts_by_native.get(
                relation["target_id"]
            ) or facts_by_binding.get(relation["target_id"])
        elif target_kind == "occurrence":
            target = occurrence_by_concrete.get(
                relation["target_id"]
            ) or occurrences_by_core.get(relation["target_id"])
        else:
            raise ValueError("RELATION_TARGET_ENDPOINT_KIND_INVALID")
        if source is None:
            raise ValueError(
                "RELATION_SOURCE_ENDPOINT_MISSING:"
                + relation["relation_id"]
            )
        if target is None:
            raise ValueError(
                "RELATION_TARGET_ENDPOINT_MISSING:"
                + relation["relation_id"]
            )
        relation_edges.append(
            _relation_edge(
                relation,
                source_node_id=source.graph_node_id,
                source_node_kind=source.node_kind,
                target_node_id=target.graph_node_id,
                target_node_kind=target.node_kind,
                store_id=store_id,
                graph_schema_version=schema_version,
                registry=relation_type_registry,
            )
        )
    relation_edges.sort(key=lambda row: row.graph_edge_id)
    if len({row.graph_edge_id for row in relation_edges}) != len(
        relation_edges
    ):
        raise ValueError("DUPLICATE_RELATION_EDGE_ID")

    envelope = {
        key: value
        for key, value in validated_primitive_relation_store.items()
        if key not in {"relations", "evidence"}
    }
    metadata = {
        "graph_schema_version": schema_version,
        "graph_id": "",
        "execution_run_id": execution_run_id,
        "domain_scope_id": next(iter(domains)),
        "source_snapshot_ids": sorted(
            row["snapshot"].snapshot_id
            for row in validated_snapshots
        ),
        "source_relation_store_ids": [store_id],
        "relation_store_envelopes": [envelope],
        "relation_evidence_records": sorted(
            validated_primitive_relation_store.get("evidence", []),
            key=lambda row: row.get("evidence_id", canonical_hash(row)),
        ),
        "occurrence_catalog_id": occurrence_endpoint_catalog[
            "occurrence_catalog_id"
        ],
        "capture_audit_id": capture_audit["capture_audit_id"],
        "graph_profile_id": graph_profile["graph_profile_id"],
        "relation_registry_id": relation_type_registry["registry_id"],
        "compiler_code_hash": implementation_hash(
            "graph_compiler.py"
        ),
        "validator_code_hash": implementation_hash(
            "graph_validator.py"
        ),
        "fact_node_count": len(fact_nodes),
        "occurrence_node_count": len(occurrence_nodes),
        "incidence_edge_count": len(incidence_edges),
        "primitive_relation_edge_count": sum(
            row.primitive_or_derived == "primitive"
            for row in relation_edges
        ),
        "derived_relation_edge_count": sum(
            row.primitive_or_derived == "derived"
            for row in relation_edges
        ),
        "global_transitive_closure_materialized": False,
    }
    document = canonical_graph_document(
        {
            "metadata": metadata,
            "fact_nodes": [row.to_dict() for row in fact_nodes],
            "occurrence_nodes": [
                row.to_dict() for row in occurrence_nodes
            ],
            "incidence_edges": [
                row.to_dict() for row in incidence_edges
            ],
            "relation_edges": [
                row.to_dict() for row in relation_edges
            ],
        }
    )
    return ExecutableGenerationFactGraphV2.from_dict(document)
