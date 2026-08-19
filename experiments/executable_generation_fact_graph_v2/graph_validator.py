from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .canonical_graph import canonical_hash, graph_id, load_contract
from .graph_compiler import (
    compile_executable_generation_fact_graph_v2,
)
from .graph_model import (
    ExecutableGenerationFactGraphV2,
    GraphValidationV2,
    ValidatedGenerationFactGraphV2,
)


class GraphValidationErrorV2(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(
            reason_code + (":" + detail if detail else "")
        )


def _unique(
    rows: list[dict[str, Any]], field: str, reason: str
) -> None:
    values = [row[field] for row in rows]
    if len(values) != len(set(values)):
        raise GraphValidationErrorV2(reason)


def _compare_by_id(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    id_field: str,
    missing_reason: str,
    extra_reason: str,
    content_reason: str,
) -> None:
    actual_by_id = {row[id_field]: row for row in actual}
    expected_by_id = {row[id_field]: row for row in expected}
    missing = sorted(set(expected_by_id) - set(actual_by_id))
    if missing:
        raise GraphValidationErrorV2(missing_reason, missing[0])
    extra = sorted(set(actual_by_id) - set(expected_by_id))
    if extra:
        raise GraphValidationErrorV2(extra_reason, extra[0])
    for identity in sorted(expected_by_id):
        if actual_by_id[identity] != expected_by_id[identity]:
            raise GraphValidationErrorV2(content_reason, identity)


def _relation_signature(
    row: dict[str, Any],
) -> str:
    return (
        row["source_node_kind"] + "->" + row["target_node_kind"]
    )


def _relation_endpoint_identity(
    edge: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    source = nodes[edge["source_node_id"]]
    target = nodes[edge["target_node_id"]]

    def native(node: dict[str, Any], expected: str) -> str:
        if node["node_kind"] == "generation_fact":
            candidates = {
                node["generation_binding_id"],
                node.get("native_fact_id"),
            }
        else:
            candidates = {
                node["concrete_occurrence_instance_id"],
                node.get("generation_occurrence_id"),
            }
        if expected not in candidates:
            raise GraphValidationErrorV2(
                "NATIVE_ENDPOINT_IDENTITY_MISMATCH", expected
            )
        return expected

    return (
        native(source, edge["native_source_id"]),
        native(target, edge["native_target_id"]),
    )


def reject_acyclic_relation_family_cycles(
    relation_rows: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
) -> None:
    adjacency_by_family: dict[
        str, dict[str, set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    nodes_by_family: dict[str, set[str]] = defaultdict(set)
    for edge in relation_rows:
        spec = registry.get(edge["relation_type"])
        if spec is None:
            raise GraphValidationErrorV2("RELATION_TYPE_UNKNOWN")
        policy = spec.get("cycle_policy")
        if policy not in {"ACYCLIC", "ALLOW"}:
            raise GraphValidationErrorV2(
                "RELATION_CYCLE_POLICY_INVALID",
                edge["relation_type"],
            )
        if policy == "ALLOW":
            continue
        family = spec.get("acyclic_family_id")
        if not family:
            raise GraphValidationErrorV2(
                "ACYCLIC_FAMILY_ID_REQUIRED",
                edge["relation_type"],
            )
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        adjacency_by_family[family][source].add(target)
        nodes_by_family[family].update((source, target))

    for family in sorted(nodes_by_family):
        adjacency = adjacency_by_family[family]
        state: dict[str, int] = {}

        def visit(node_id: str) -> None:
            marker = state.get(node_id, 0)
            if marker == 1:
                raise GraphValidationErrorV2(
                    "ACYCLIC_RELATION_FAMILY_CYCLE",
                    family + ":" + node_id,
                )
            if marker == 2:
                return
            state[node_id] = 1
            for target_id in sorted(adjacency[node_id]):
                visit(target_id)
            state[node_id] = 2

        for node_id in sorted(nodes_by_family[family]):
            visit(node_id)


def reject_symmetric_double_write(
    relation_rows: list[dict[str, Any]],
) -> None:
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for edge in relation_rows:
        if edge["relation_semantics"] != "symmetric":
            continue
        left, right = sorted(
            (edge["source_node_id"], edge["target_node_id"])
        )
        key = (
            edge["relation_type"],
            edge["original_relation_id"],
        )
        previous = seen.get(key)
        if previous == (left, right):
            raise GraphValidationErrorV2(
                "SYMMETRIC_RELATION_INSTANCE_DOUBLE_WRITTEN",
                edge["original_relation_id"],
            )
        seen[key] = (left, right)


def _validate_capture_gate(capture_audit: dict[str, Any]) -> None:
    if capture_audit.get("status") == "CAPTURE_PARTIAL" and (
        capture_audit.get("concurrency_inference_allowed", False)
    ):
        raise GraphValidationErrorV2(
            "CAPTURE_PARTIAL_CONCURRENCY_GATE_BYPASSED"
        )
    for scope in capture_audit.get("scopes", []):
        if scope.get("status") == "CAPTURE_PARTIAL" and scope.get(
            "concurrency_inference_allowed", False
        ):
            raise GraphValidationErrorV2(
                "CAPTURE_PARTIAL_CONCURRENCY_GATE_BYPASSED",
                scope.get("scope_id", ""),
            )


def validate_executable_generation_fact_graph_v2(
    graph: ExecutableGenerationFactGraphV2,
    validated_snapshots: list[dict[str, Any]],
    validated_primitive_relation_store: dict[str, Any],
    occurrence_endpoint_catalog: dict[str, Any],
    capture_audit: dict[str, Any],
    contracts: dict[str, Any],
) -> ValidatedGenerationFactGraphV2:
    document = graph.to_dict()
    try:
        Draft202012Validator(contracts["graph_schema"]).validate(
            document
        )
    except ValidationError as exc:
        raise GraphValidationErrorV2(
            "GRAPH_SCHEMA_INVALID", exc.json_path
        ) from exc

    fact_rows = document["fact_nodes"]
    occurrence_rows = document["occurrence_nodes"]
    incidence_rows = document["incidence_edges"]
    relation_rows = document["relation_edges"]
    reject_symmetric_double_write(relation_rows)
    _unique(fact_rows, "graph_node_id", "DUPLICATE_FACT_NODE")
    _unique(
        fact_rows,
        "generation_binding_id",
        "DUPLICATE_BINDING_FACT_NODE",
    )
    _unique(
        occurrence_rows,
        "graph_node_id",
        "DUPLICATE_OCCURRENCE_NODE",
    )
    _unique(
        occurrence_rows,
        "concrete_occurrence_instance_id",
        "DUPLICATE_OCCURRENCE_INSTANCE",
    )
    _unique(
        incidence_rows, "graph_edge_id", "DUPLICATE_INCIDENCE_EDGE"
    )
    _unique(
        relation_rows, "graph_edge_id", "DUPLICATE_RELATION_EDGE"
    )
    _unique(
        relation_rows,
        "original_relation_id",
        "DUPLICATE_PRIMITIVE_RELATION",
    )

    facts_by_id = {row["graph_node_id"]: row for row in fact_rows}
    occurrences_by_id = {
        row["graph_node_id"]: row for row in occurrence_rows
    }
    all_nodes = {**facts_by_id, **occurrences_by_id}
    incidence_by_fact: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    incidence_by_occurrence: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)
    for edge in incidence_rows:
        if edge["source_occurrence_node_id"] not in occurrences_by_id:
            raise GraphValidationErrorV2(
                "INCIDENCE_OCCURRENCE_ENDPOINT_MISSING"
            )
        if edge["target_fact_node_id"] not in facts_by_id:
            raise GraphValidationErrorV2(
                "INCIDENCE_FACT_ENDPOINT_MISSING"
            )
        occurrence = occurrences_by_id[
            edge["source_occurrence_node_id"]
        ]
        fact = facts_by_id[edge["target_fact_node_id"]]
        if (
            occurrence["concrete_occurrence_instance_id"]
            != fact["concrete_occurrence_instance_id"]
            or edge["source_concrete_occurrence_instance_id"]
            != fact["concrete_occurrence_instance_id"]
        ):
            raise GraphValidationErrorV2(
                "INCIDENCE_FACT_OCCURRENCE_MISMATCH"
            )
        incidence_by_fact[edge["target_fact_node_id"]].append(edge)
        incidence_by_occurrence[
            edge["source_occurrence_node_id"]
        ].append(edge)
    if any(len(incidence_by_fact[row["graph_node_id"]]) != 1 for row in fact_rows):
        raise GraphValidationErrorV2(
            "EVERY_FACT_REQUIRES_EXACTLY_ONE_INCIDENCE"
        )

    registry = contracts["relation_type_registry"]["relations"]
    evidence_ids = {
        row["evidence_id"]
        for row in document["metadata"]["relation_evidence_records"]
        if "evidence_id" in row
    }
    evidence_ids.update(
        row["evidence_id"]
        for item in validated_snapshots
        for row in item["snapshot"].tables.evidence_records
    )
    for edge in relation_rows:
        if edge["source_node_id"] not in all_nodes:
            raise GraphValidationErrorV2(
                "RELATION_SOURCE_ENDPOINT_MISSING"
            )
        if edge["target_node_id"] not in all_nodes:
            raise GraphValidationErrorV2(
                "RELATION_TARGET_ENDPOINT_MISSING"
            )
        if (
            all_nodes[edge["source_node_id"]]["node_kind"]
            != edge["source_node_kind"]
        ):
            raise GraphValidationErrorV2(
                "RELATION_SOURCE_ENDPOINT_KIND_MISMATCH"
            )
        if (
            all_nodes[edge["target_node_id"]]["node_kind"]
            != edge["target_node_kind"]
        ):
            raise GraphValidationErrorV2(
                "RELATION_TARGET_ENDPOINT_KIND_MISMATCH"
            )
        spec = registry.get(edge["relation_type"])
        if spec is None:
            raise GraphValidationErrorV2("RELATION_TYPE_UNKNOWN")
        if _relation_signature(edge) not in spec[
            "allowed_endpoint_signatures"
        ]:
            raise GraphValidationErrorV2(
                "RELATION_ENDPOINT_SIGNATURE_MISMATCH"
            )
        if edge["relation_semantics"] != spec["semantics"]:
            raise GraphValidationErrorV2(
                "RELATION_SEMANTICS_MISMATCH"
            )
        _relation_endpoint_identity(edge, all_nodes)
        if not set(edge["evidence_refs"]) <= evidence_ids:
            raise GraphValidationErrorV2(
                "RELATION_EVIDENCE_CLOSURE_MISMATCH"
            )
        if edge["primitive_or_derived"] == "derived" and (
            not edge["rule_id"] or not edge["input_relation_refs"]
        ):
            raise GraphValidationErrorV2(
                "DERIVED_RELATION_NOT_TRACEABLE"
            )
        if edge["execution_run_id"] != graph.metadata.execution_run_id:
            raise GraphValidationErrorV2(
                "RELATION_RUN_SCOPE_MISMATCH"
            )
    reject_acyclic_relation_family_cycles(relation_rows, registry)

    expected = compile_executable_generation_fact_graph_v2(
        validated_snapshots,
        validated_primitive_relation_store,
        occurrence_endpoint_catalog,
        capture_audit,
        contracts["graph_profile"],
        contracts["relation_type_registry"],
    ).to_dict()
    _compare_by_id(
        fact_rows,
        expected["fact_nodes"],
        id_field="generation_binding_id",
        missing_reason="FACT_NODE_MISSING",
        extra_reason="FACT_NODE_FABRICATED",
        content_reason="FACT_NODE_CONTENT_MISMATCH",
    )
    _compare_by_id(
        occurrence_rows,
        expected["occurrence_nodes"],
        id_field="concrete_occurrence_instance_id",
        missing_reason="OCCURRENCE_NODE_MISSING",
        extra_reason="OCCURRENCE_NODE_FABRICATED",
        content_reason="OCCURRENCE_NODE_CONTENT_MISMATCH",
    )
    _compare_by_id(
        incidence_rows,
        expected["incidence_edges"],
        id_field="target_generation_binding_id",
        missing_reason="INCIDENCE_EDGE_MISSING",
        extra_reason="INCIDENCE_EDGE_FABRICATED",
        content_reason="INCIDENCE_EDGE_CONTENT_MISMATCH",
    )
    _compare_by_id(
        relation_rows,
        expected["relation_edges"],
        id_field="original_relation_id",
        missing_reason="PRIMITIVE_RELATION_MISSING",
        extra_reason="PRIMITIVE_RELATION_FABRICATED",
        content_reason="PRIMITIVE_RELATION_CONTENT_MISMATCH",
    )
    for field, value in expected["metadata"].items():
        if field == "graph_id":
            continue
        if document["metadata"].get(field) != value:
            raise GraphValidationErrorV2(
                "GRAPH_METADATA_MISMATCH", field
            )
    if document["metadata"]["graph_id"] != graph_id(document):
        raise GraphValidationErrorV2("GRAPH_HASH_MISMATCH")
    if (
        document["metadata"]["graph_id"]
        != expected["metadata"]["graph_id"]
    ):
        raise GraphValidationErrorV2(
            "GRAPH_EXPECTED_HASH_MISMATCH"
        )
    _validate_capture_gate(capture_audit)

    reordered_store = copy.deepcopy(validated_primitive_relation_store)
    reordered_store["relations"] = list(
        reversed(reordered_store.get("relations", []))
    )
    reordered_store["evidence"] = list(
        reversed(reordered_store.get("evidence", []))
    )
    reordered_catalog = copy.deepcopy(occurrence_endpoint_catalog)
    reordered_catalog["occurrences"] = list(
        reversed(reordered_catalog["occurrences"])
    )
    reordered = compile_executable_generation_fact_graph_v2(
        list(reversed(validated_snapshots)),
        reordered_store,
        reordered_catalog,
        capture_audit,
        contracts["graph_profile"],
        contracts["relation_type_registry"],
    )
    if reordered.to_dict() != expected:
        raise GraphValidationErrorV2(
            "INPUT_REORDERING_CHANGED_CANONICAL_GRAPH"
        )

    incidence_counts = Counter(
        edge["source_occurrence_node_id"] for edge in incidence_rows
    )
    zero_fact = sum(
        incidence_counts[row["graph_node_id"]] == 0
        for row in occurrence_rows
    )
    one_fact = sum(
        incidence_counts[row["graph_node_id"]] == 1
        for row in occurrence_rows
    )
    multi_fact = sum(
        incidence_counts[row["graph_node_id"]] > 1
        for row in occurrence_rows
    )
    gates = {
        "graph_schema_valid": True,
        "canonical_graph_exact": True,
        "input_reordering_invariant": True,
        "every_binding_exactly_one_fact_node": True,
        "fact_content_exact": True,
        "fact_identity_preserved": True,
        "fact_multiplicity_preserved": True,
        "every_referenced_occurrence_exactly_one_node": True,
        "occurrence_content_exact": True,
        "occurrence_identity_preserved": True,
        "zero_fact_occurrence_allowed": True,
        "multi_fact_occurrence_preserved": True,
        "every_fact_exactly_one_incidence": True,
        "incidence_exact": True,
        "no_fake_incidence": True,
        "every_primitive_relation_exactly_once": True,
        "primitive_endpoint_kind_exact": True,
        "primitive_endpoint_identity_exact": True,
        "primitive_relation_type_exact": True,
        "primitive_payload_exact": True,
        "primitive_evidence_exact": True,
        "primitive_authority_exact": True,
        "no_relation_drop": True,
        "no_relation_fabrication": True,
        "no_forced_lifting": True,
        "no_cartesian_expansion": True,
        "derived_edges_traceable": True,
        "capture_completeness_gate_enforced": True,
        "no_global_transitive_closure": True,
        "no_second_authority_store": True,
    }
    validation_material = {
        "graph_id": graph.metadata.graph_id,
        "status": "PASS",
        "gates": gates,
        "counts": {
            "fact_nodes": len(fact_rows),
            "occurrence_nodes": len(occurrence_rows),
            "zero_fact_occurrences": zero_fact,
            "one_fact_occurrences": one_fact,
            "multi_fact_occurrences": multi_fact,
            "incidence_edges": len(incidence_rows),
            "primitive_relation_edges": sum(
                row["primitive_or_derived"] == "primitive"
                for row in relation_rows
            ),
            "derived_relation_edges": sum(
                row["primitive_or_derived"] == "derived"
                for row in relation_rows
            ),
        },
    }
    validation = GraphValidationV2(
        **validation_material,
        validation_sha256=canonical_hash(validation_material),
    )
    return ValidatedGenerationFactGraphV2(
        graph=graph,
        validation=validation,
        capture_audit=capture_audit,
    )


def load_contracts() -> dict[str, Any]:
    return {
        "graph_schema": load_contract("graph_schema.json"),
        "graph_profile": load_contract("graph_profile.json"),
        "node_type_registry": load_contract("node_type_registry.json"),
        "relation_type_registry": load_contract(
            "relation_type_registry.json"
        ),
        "traversal_policies": load_contract(
            "traversal_policies.json"
        ),
        "query_contract": load_contract("query_contract.json"),
    }
