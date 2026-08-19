from __future__ import annotations

from collections import defaultdict
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .canonical_graph import canonical_hash, graph_id, load_contract
from .graph_compiler import compile_generation_fact_graph
from .graph_model import (
    ExecutableGenerationFactGraph,
    GraphValidation,
    ValidatedGenerationFactGraph,
)


class GraphValidationError(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code + (f":{detail}" if detail else ""))


def _reject_cycle(edges: list[dict], relation_type: str) -> None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge["relation_type"] == relation_type:
            adjacency[edge["source_graph_node_id"]].add(
                edge["target_graph_node_id"]
            )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise GraphValidationError("DAG_RELATION_CYCLE", relation_type)
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, set()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(adjacency):
        visit(node)


def _compare_nodes(actual: list[dict], expected: list[dict]) -> None:
    expected_by_binding = {
        row["generation_binding_id"]: row for row in expected
    }
    actual_ids = [row["graph_node_id"] for row in actual]
    if len(actual_ids) != len(set(actual_ids)):
        raise GraphValidationError("DUPLICATE_GRAPH_NODE")
    actual_by_binding: dict[str, list[dict]] = defaultdict(list)
    for row in actual:
        actual_by_binding[row["generation_binding_id"]].append(row)
    missing = sorted(set(expected_by_binding) - set(actual_by_binding))
    if missing:
        raise GraphValidationError("NODE_BINDING_MISSING", missing[0])
    extra = sorted(set(actual_by_binding) - set(expected_by_binding))
    if extra:
        raise GraphValidationError("NODE_BINDING_UNKNOWN", extra[0])
    duplicates = sorted(
        key for key, rows in actual_by_binding.items() if len(rows) != 1
    )
    if duplicates:
        raise GraphValidationError("NODE_BINDING_DUPLICATED", duplicates[0])
    coordinate_fields = {
        "source_reference": "NODE_U_MISMATCH",
        "realized_transformation": "NODE_TAU_MISMATCH",
        "concrete_occurrence": "NODE_OMEGA_MISMATCH",
        "outcome_reference": "NODE_Z_MISMATCH",
        "relation_role": "NODE_RHO_MISMATCH",
    }
    for binding_id, expected_row in expected_by_binding.items():
        row = actual_by_binding[binding_id][0]
        for field, reason in coordinate_fields.items():
            if row[field] != expected_row[field]:
                raise GraphValidationError(reason, binding_id)
        if row["occurrence_identity"] != expected_row["occurrence_identity"]:
            raise GraphValidationError("NODE_OCCURRENCE_IDENTITY_MISMATCH", binding_id)
        if row["outcome_identity"] != expected_row["outcome_identity"]:
            raise GraphValidationError("NODE_OUTCOME_IDENTITY_MISMATCH", binding_id)
        if row["execution_run_id"] != expected_row["execution_run_id"]:
            raise GraphValidationError("NODE_RUN_IDENTITY_MISMATCH", binding_id)
        if row["snapshot_id"] != expected_row["snapshot_id"]:
            raise GraphValidationError("NODE_SNAPSHOT_IDENTITY_MISMATCH", binding_id)
        if row["evidence_refs"] != expected_row["evidence_refs"]:
            raise GraphValidationError("NODE_EVIDENCE_CLOSURE_MISMATCH", binding_id)
        if row["native_fact_identity"] != expected_row["native_fact_identity"]:
            raise GraphValidationError("NODE_NATIVE_IDENTITY_MISMATCH", binding_id)
        if row["fact_content_hash"] != expected_row["fact_content_hash"]:
            raise GraphValidationError("NODE_CONTENT_HASH_MISMATCH", binding_id)
        if row["node_instance_hash"] != expected_row["node_instance_hash"]:
            raise GraphValidationError("NODE_INSTANCE_HASH_MISMATCH", binding_id)
        if row["graph_node_id"] != expected_row["graph_node_id"]:
            raise GraphValidationError("NODE_GRAPH_ID_MISMATCH", binding_id)


def _compare_edges(actual: list[dict], expected: list[dict], node_ids: set[str]) -> None:
    ids = [row["graph_edge_id"] for row in actual]
    if len(ids) != len(set(ids)):
        raise GraphValidationError("DUPLICATE_GRAPH_EDGE")
    for row in actual:
        if row["source_graph_node_id"] not in node_ids:
            raise GraphValidationError("DANGLING_SOURCE_ENDPOINT", row["graph_edge_id"])
        if row["target_graph_node_id"] not in node_ids:
            raise GraphValidationError("DANGLING_TARGET_ENDPOINT", row["graph_edge_id"])
    expected_by_source = {
        row["relation_payload"].get(
            "source_relation_id",
            row["graph_edge_id"],
        ): row
        for row in expected
    }
    actual_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in actual:
        key = row["relation_payload"].get("source_relation_id", row["graph_edge_id"])
        actual_by_source[key].append(row)
    missing = sorted(set(expected_by_source) - set(actual_by_source))
    if missing:
        raise GraphValidationError("PRIMITIVE_EDGE_MISSING", missing[0])
    extra = sorted(set(actual_by_source) - set(expected_by_source))
    if extra:
        raise GraphValidationError("EDGE_UNKNOWN", extra[0])
    duplicates = sorted(
        key for key, rows in actual_by_source.items() if len(rows) != 1
    )
    if duplicates:
        raise GraphValidationError("PRIMITIVE_EDGE_DUPLICATED", duplicates[0])
    field_reasons = {
        "source_graph_node_id": "EDGE_SOURCE_ENDPOINT_MISMATCH",
        "target_graph_node_id": "EDGE_TARGET_ENDPOINT_MISMATCH",
        "relation_type": "EDGE_RELATION_TYPE_MISMATCH",
        "relation_semantics": "EDGE_RELATION_SEMANTICS_MISMATCH",
        "relation_payload": "EDGE_PAYLOAD_MISMATCH",
        "establishment_source": "EDGE_ESTABLISHMENT_SOURCE_MISMATCH",
        "authority_id": "EDGE_AUTHORITY_MISMATCH",
        "evidence_refs": "EDGE_EVIDENCE_MISMATCH",
        "rule_id": "EDGE_RULE_MISMATCH",
        "input_relation_refs": "EDGE_INPUT_RELATIONS_MISMATCH",
        "primitive_or_derived": "EDGE_PRIMITIVE_DERIVED_MISMATCH",
        "execution_run_id": "EDGE_RUN_SCOPE_MISMATCH",
    }
    for key, expected_row in expected_by_source.items():
        row = actual_by_source[key][0]
        for field, reason in field_reasons.items():
            if row[field] != expected_row[field]:
                raise GraphValidationError(reason, key)
        if row["relation_instance_hash"] != expected_row["relation_instance_hash"]:
            raise GraphValidationError("EDGE_INSTANCE_HASH_MISMATCH", key)
        if row["graph_edge_id"] != expected_row["graph_edge_id"]:
            raise GraphValidationError("EDGE_GRAPH_ID_MISMATCH", key)


def validate_generation_fact_graph(
    graph: ExecutableGenerationFactGraph,
    validated_snapshots: list[dict[str, Any]],
    validated_relation_store: dict[str, Any],
    capture_audit: dict[str, Any],
    contracts: dict[str, Any],
) -> ValidatedGenerationFactGraph:
    document = graph.to_dict()
    try:
        Draft202012Validator(contracts["graph_schema"]).validate(document)
    except ValidationError as exc:
        raise GraphValidationError("GRAPH_SCHEMA_INVALID", exc.json_path) from exc
    expected = compile_generation_fact_graph(
        validated_snapshots,
        validated_relation_store,
        capture_audit,
        contracts["graph_profile"],
        contracts["relation_lifting_contract"],
        relation_type_registry=contracts["relation_type_registry"],
    ).to_dict()
    _compare_nodes(document["nodes"], expected["nodes"])
    node_ids = {row["graph_node_id"] for row in document["nodes"]}
    _compare_edges(document["edges"], expected["edges"], node_ids)

    metadata_fields = set(expected["metadata"]) - {"graph_id"}
    for field in sorted(metadata_fields):
        if document["metadata"][field] != expected["metadata"][field]:
            raise GraphValidationError(
                "GRAPH_METADATA_MISMATCH", field
            )
    if document["metadata"]["graph_id"] != graph_id(document):
        raise GraphValidationError("CANONICAL_GRAPH_HASH_MISMATCH")
    if document["metadata"]["graph_id"] != expected["metadata"]["graph_id"]:
        raise GraphValidationError("CANONICAL_GRAPH_EXPECTED_HASH_MISMATCH")

    registry = contracts["relation_type_registry"]["relations"]
    for edge in document["edges"]:
        spec = registry.get(edge["relation_type"])
        if spec is None:
            raise GraphValidationError("RELATION_TYPE_UNKNOWN")
        if edge["relation_semantics"] != spec["semantics"]:
            raise GraphValidationError("EDGE_RELATION_SEMANTICS_MISMATCH")
        if edge["primitive_or_derived"] == "derived" and (
            not edge["rule_id"] or not edge["input_relation_refs"]
        ):
            raise GraphValidationError("DERIVED_EDGE_NOT_TRACEABLE")
    for relation_type, spec in registry.items():
        if spec["declared_dag"]:
            _reject_cycle(document["edges"], relation_type)

    if (
        capture_audit.get("concurrency_inference_allowed", False)
        and (
            capture_audit.get("status") != "CAPTURE_COMPLETE"
            or capture_audit.get("concurrency_scope")
            != "CONTROLLED_CAPTURE_SCOPE_ONLY"
        )
    ):
        raise GraphValidationError("CAPTURE_COMPLETENESS_GATE_INVALID")

    gates = {
        "graph_schema_valid": True,
        "graph_canonical_serialization_exact": True,
        "node_binding_coverage_exact": True,
        "node_content_exact": True,
        "node_instance_identity_preserved": True,
        "node_multiplicity_preserved": True,
        "edge_primitive_coverage_exact": True,
        "edge_endpoint_closure": True,
        "edge_endpoint_semantics_exact": True,
        "edge_relation_type_exact": True,
        "edge_evidence_closure": True,
        "edge_authority_exact": True,
        "edge_run_scope_exact": True,
        "derived_edge_traceability_exact": True,
        "no_cartesian_expansion": True,
        "no_direct_multistage_shortcut": True,
        "no_global_transitive_closure_materialized": True,
        "capture_completeness_gate_enforced": True,
        "no_second_authority_store": True,
    }
    validation_material = {
        "graph_id": graph.metadata.graph_id,
        "status": "PASS",
        "gates": gates,
        "counts": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "primitive_edges": sum(
                row.primitive_or_derived == "primitive" for row in graph.edges
            ),
            "derived_edges": sum(
                row.primitive_or_derived == "derived" for row in graph.edges
            ),
        },
    }
    validation = GraphValidation(
        **validation_material,
        validation_sha256=canonical_hash(validation_material),
    )
    return ValidatedGenerationFactGraph(graph, validation, capture_audit)


def load_contracts() -> dict[str, Any]:
    return {
        "graph_schema": load_contract("graph_schema.json"),
        "graph_profile": load_contract("graph_profile.json"),
        "relation_type_registry": load_contract("relation_type_registry.json"),
        "relation_lifting_contract": load_contract(
            "relation_lifting_contract.json"
        ),
        "graph_query_contract": load_contract("graph_query_contract.json"),
    }

