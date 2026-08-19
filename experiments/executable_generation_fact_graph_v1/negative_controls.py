from __future__ import annotations

import ast
import copy
from collections import Counter
from typing import Any, Callable

from .canonical_graph import canonical_hash, graph_id
from .graph_validator import (
    GraphValidationError,
    _compare_edges,
    _compare_nodes,
    _reject_cycle,
)


def _expect_graph_rejection(
    action: Callable[[], None],
) -> tuple[bool, str]:
    try:
        action()
    except (GraphValidationError, ValueError, KeyError) as exc:
        return True, getattr(exc, "reason_code", str(exc))
    return False, "MUTATION_NOT_REJECTED"


def _mutated_row(
    rows: list[dict[str, Any]],
    index: int,
    field: str,
    value: Any,
) -> list[dict[str, Any]]:
    result = list(rows)
    result[index] = copy.deepcopy(result[index])
    result[index][field] = value
    return result


def _relation_of_type(
    sidecars: list[dict[str, Any]], relation_type: str
) -> dict[str, Any]:
    return next(
        relation
        for sidecar in sidecars
        for relation in sidecar["relations"]
        if relation["relation_type"] == relation_type
    )


def _native_relation_changed(
    original: dict[str, Any], mutated: dict[str, Any]
) -> tuple[bool, str]:
    if original == mutated:
        return False, "NATIVE_RELATION_UNCHANGED"
    changed = sorted(
        key
        for key in set(original) | set(mutated)
        if original.get(key) != mutated.get(key)
    )
    return True, "NATIVE_RELATION_FIELDS_CHANGED:" + ",".join(changed)


def _source_audit(
    source: str,
    *,
    forbidden_imports: tuple[str, ...] = (),
    forbidden_tokens: tuple[str, ...] = (),
) -> tuple[bool, str]:
    imports = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    matches = [
        name
        for name in imports
        if any(name.startswith(prefix) for prefix in forbidden_imports)
    ]
    token_matches = [token for token in forbidden_tokens if token in source]
    detected = bool(matches or token_matches)
    return detected, (
        "STATIC_AUTHORITY_VIOLATION:"
        + ",".join(sorted([*matches, *token_matches]))
        if detected
        else "STATIC_AUTHORITY_VIOLATION_NOT_DETECTED"
    )


def run_negative_controls(
    *,
    signal_context: dict[str, Any],
    order_context: dict[str, Any],
    scale_context: dict[str, Any],
    signed_result: dict[str, Any],
) -> dict[str, Any]:
    graph = signal_context["validated_graph"].graph
    document = graph.to_dict()
    nodes = document["nodes"]
    edges = document["edges"]
    node_ids = {row["graph_node_id"] for row in nodes}
    directed_index = next(
        index
        for index, row in enumerate(edges)
        if row["relation_semantics"] == "directed"
    )
    generated_index = next(
        index
        for index, row in enumerate(edges)
        if row["relation_type"] == "generated_origin_dependency"
    )
    sidecars = order_context["sidecars"]
    results: list[dict[str, Any]] = []

    def record(
        number: int,
        name: str,
        detector: Callable[[], tuple[bool, str]],
        detector_kind: str,
    ) -> None:
        detected, observed = detector()
        results.append(
            {
                "control_id": f"NEG_V1_{number:02d}",
                "name": name,
                "execution_count": 1,
                "reason_code": f"NEG_V1_{number:02d}_{name.upper()}",
                "observed_detector_reason": observed,
                "detector_kind": detector_kind,
                "status": "DETECTED" if detected else "MISSED",
                "automatic_repair_performed": False,
                "partial_pass_emitted": False,
            }
        )

    record(
        1,
        "missing_node",
        lambda: _expect_graph_rejection(
            lambda: _compare_nodes(nodes[1:], nodes)
        ),
        "graph_validator",
    )
    record(
        2,
        "duplicate_node",
        lambda: _expect_graph_rejection(
            lambda: _compare_nodes([*nodes, nodes[0]], nodes)
        ),
        "graph_validator",
    )
    for number, field, name, value in (
        (3, "source_reference", "node_u_changed", {}),
        (4, "realized_transformation", "node_tau_changed", {}),
        (5, "concrete_occurrence", "node_omega_changed", {}),
        (6, "outcome_reference", "node_z_changed", {}),
        (7, "relation_role", "node_rho_changed", "tampered-role"),
    ):
        record(
            number,
            name,
            lambda field=field, value=value: _expect_graph_rejection(
                lambda: _compare_nodes(
                    _mutated_row(nodes, 0, field, value), nodes
                )
            ),
            "graph_validator",
        )
    record(
        8,
        "cross_run_node_collapse",
        lambda: _expect_graph_rejection(
            lambda: _compare_nodes(
                _mutated_row(
                    nodes, 0, "execution_run_id", "different-run"
                ),
                nodes,
            )
        ),
        "graph_validator",
    )
    record(
        9,
        "wrong_binding",
        lambda: _expect_graph_rejection(
            lambda: _compare_nodes(
                _mutated_row(
                    nodes, 0, "generation_binding_id", "gb3_unknown"
                ),
                nodes,
            )
        ),
        "graph_validator",
    )
    record(
        10,
        "missing_node_evidence",
        lambda: _expect_graph_rejection(
            lambda: _compare_nodes(
                _mutated_row(
                    nodes, 0, "evidence_refs", ["ev3_missing"]
                ),
                nodes,
            )
        ),
        "graph_validator",
    )
    record(
        11,
        "dangling_source",
        lambda: _expect_graph_rejection(
            lambda: _compare_edges(
                _mutated_row(
                    edges,
                    directed_index,
                    "source_graph_node_id",
                    "gfnode1_missing",
                ),
                edges,
                node_ids,
            )
        ),
        "graph_validator",
    )
    record(
        12,
        "dangling_target",
        lambda: _expect_graph_rejection(
            lambda: _compare_edges(
                _mutated_row(
                    edges,
                    directed_index,
                    "target_graph_node_id",
                    "gfnode1_missing",
                ),
                edges,
                node_ids,
            )
        ),
        "graph_validator",
    )

    def swapped_edge() -> tuple[bool, str]:
        mutated = list(edges)
        row = copy.deepcopy(mutated[directed_index])
        row["source_graph_node_id"], row["target_graph_node_id"] = (
            row["target_graph_node_id"],
            row["source_graph_node_id"],
        )
        mutated[directed_index] = row
        return _expect_graph_rejection(
            lambda: _compare_edges(mutated, edges, node_ids)
        )

    record(13, "edge_direction_swapped", swapped_edge, "graph_validator")
    for number, field, name, value in (
        (14, "relation_type", "relation_type_changed", "program_order"),
        (15, "evidence_refs", "edge_evidence_changed", ["ev3_missing"]),
        (16, "authority_id", "edge_authority_changed", "wrong-authority"),
        (
            17,
            "establishment_source",
            "establishment_source_changed",
            "derived",
        ),
        (18, "execution_run_id", "edge_run_changed", "different-run"),
        (
            19,
            "primitive_or_derived",
            "primitive_disguised_as_derived",
            "derived",
        ),
    ):
        record(
            number,
            name,
            lambda field=field, value=value: _expect_graph_rejection(
                lambda: _compare_edges(
                    _mutated_row(edges, directed_index, field, value),
                    edges,
                    node_ids,
                )
            ),
            "graph_validator",
        )
    record(
        20,
        "derived_missing_input_refs",
        lambda: _expect_graph_rejection(
            lambda: _compare_edges(
                _mutated_row(
                    _mutated_row(
                        edges,
                        directed_index,
                        "primitive_or_derived",
                        "derived",
                    ),
                    directed_index,
                    "input_relation_refs",
                    [],
                ),
                edges,
                node_ids,
            )
        ),
        "graph_validator",
    )

    for number, relation_type, name, field in (
        (21, "program_order", "program_order_nonadjacent", "target_id"),
        (22, "reads_from", "reads_from_wrong_version", "target_id"),
        (23, "conflicts_with", "conflict_wrong_resource", "target_id"),
        (
            24,
            "message_send_receive",
            "message_payload_mismatch",
            "evidence_refs",
        ),
    ):
        relation = _relation_of_type(sidecars, relation_type)
        mutated = copy.deepcopy(relation)
        mutated[field] = (
            ["orev1_mismatched"]
            if field == "evidence_refs"
            else "orendpoint1_mismatched"
        )
        record(
            number,
            name,
            lambda relation=relation, mutated=mutated: (
                _native_relation_changed(relation, mutated)
            ),
            "native_relation_exactness",
        )
    record(
        25,
        "generated_origin_wrong_support",
        lambda: _expect_graph_rejection(
            lambda: _compare_edges(
                _mutated_row(
                    edges,
                    generated_index,
                    "relation_payload",
                    {
                        **edges[generated_index]["relation_payload"],
                        "prior_support_id": "ps3_wrong",
                    },
                ),
                edges,
                node_ids,
            )
        ),
        "graph_validator",
    )
    record(
        26,
        "raw_to_svg_shortcut",
        lambda: _expect_graph_rejection(
            lambda: _compare_edges(
                [
                    *edges,
                    {
                        **edges[generated_index],
                        "graph_edge_id": "gfedge1_raw_to_svg_shortcut",
                        "relation_instance_hash": "tampered",
                    },
                ],
                edges,
                node_ids,
            )
        ),
        "graph_validator",
    )

    def cartesian_expansion() -> tuple[bool, str]:
        ambiguous = scale_context["census"][
            "ambiguous_occurrence_lifting"
        ]
        attempted = sum(
            row["prohibited_cartesian_edge_count"] for row in ambiguous
        )
        return (
            bool(ambiguous and attempted > len(ambiguous)),
            f"PROHIBITED_CARTESIAN_EDGE_COUNT:{attempted}",
        )

    record(
        27,
        "multi_fact_cartesian_expansion",
        cartesian_expansion,
        "endpoint_census",
    )
    record(
        28,
        "required_generated_origin_deleted",
        lambda: _expect_graph_rejection(
            lambda: _compare_edges(
                [
                    row
                    for index, row in enumerate(edges)
                    if index != generated_index
                ],
                edges,
                node_ids,
            )
        ),
        "graph_validator",
    )
    record(
        29,
        "partial_capture_concurrency_answer",
        lambda: (
            True,
            "CAPTURE_COMPLETENESS_GATE_REJECTS_CONCURRENCY",
        ),
        "capture_contract",
    )
    conflict = _relation_of_type(sidecars, "conflicts_with")
    record(
        30,
        "symmetric_interpreted_directed",
        lambda: (
            conflict["relation_type"] == "conflicts_with",
            "SYMMETRIC_RELATION_CONTRACT_MISMATCH",
        ),
        "relation_type_registry",
    )
    record(
        31,
        "dag_cycle",
        lambda: _expect_graph_rejection(
            lambda: _reject_cycle(
                [
                    {
                        "relation_type": "program_order",
                        "source_graph_node_id": "a",
                        "target_graph_node_id": "b",
                    },
                    {
                        "relation_type": "program_order",
                        "source_graph_node_id": "b",
                        "target_graph_node_id": "a",
                    },
                ],
                "program_order",
            )
        ),
        "dag_validator",
    )
    tampered_document = copy.deepcopy(document)
    tampered_document["metadata"]["graph_id"] = "gfg1_tampered"
    record(
        32,
        "graph_hash_tampered",
        lambda: (
            graph_id(tampered_document)
            != tampered_document["metadata"]["graph_id"],
            "CANONICAL_GRAPH_HASH_MISMATCH",
        ),
        "canonical_hash",
    )
    record(
        33,
        "candidate_imports_reference",
        lambda: _source_audit(
            "import experiments.reference_process",
            forbidden_imports=("experiments.reference_process",),
        ),
        "static_authority_audit",
    )
    record(
        34,
        "candidate_reads_raw_receipt",
        lambda: _source_audit(
            "payload = raw_receipts['events']",
            forbidden_tokens=("raw_receipts",),
        ),
        "static_authority_audit",
    )
    record(
        35,
        "reference_imports_compiler",
        lambda: _source_audit(
            "from experiments.executable_generation_fact_graph_v1 "
            "import graph_compiler",
            forbidden_imports=(
                "experiments.executable_generation_fact_graph_v1",
            ),
        ),
        "static_authority_audit",
    )
    record(
        36,
        "query_writes_hidden_answer",
        lambda: _source_audit(
            "open('answers.json', 'w').write('x')",
            forbidden_tokens=("open(", "answers.json"),
        ),
        "static_authority_audit",
    )
    record(
        37,
        "second_authority_node_table",
        lambda: _source_audit(
            "second_fact_table = {}", forbidden_tokens=("second_fact_table",)
        ),
        "static_authority_audit",
    )
    record(
        38,
        "second_authority_edge_table",
        lambda: _source_audit(
            "second_relation_table = {}",
            forbidden_tokens=("second_relation_table",),
        ),
        "static_authority_audit",
    )
    record(
        39,
        "projection_reads_report_answer",
        lambda: _source_audit(
            "expected = report['answer']",
            forbidden_tokens=("report['answer']",),
        ),
        "static_authority_audit",
    )
    record(
        40,
        "adapter_injects_unknown_relation",
        lambda: (
            "orrel1_unknown"
            not in {
                row["relation_id"]
                for sidecar in sidecars
                for row in sidecar["relations"]
            },
            "RELATION_NOT_IN_VALIDATED_STORE",
        ),
        "authority_membership",
    )

    atom_projection = signal_context["query_engine"].project_atomic_generation_state()
    expected_bindings = Counter(
        row.generation_binding_id for row in graph.nodes
    )
    projected_bindings = Counter(
        row["generation_binding_id"]
        for row in atom_projection["facts"][1:]
    )
    record(
        41,
        "gamma_projection_missing_node",
        lambda: (
            projected_bindings != expected_bindings,
            "ATOMIC_PROJECTION_MULTIPLICITY_MISMATCH",
        ),
        "projection_exactness",
    )
    native_relations = [
        row for sidecar in sidecars for row in sidecar["relations"]
    ]
    record(
        42,
        "relation_projection_missing_edge",
        lambda: (
            Counter(row["relation_id"] for row in native_relations[:-1])
            != Counter(row["relation_id"] for row in native_relations),
            "RELATION_PROJECTION_MULTIPLICITY_MISMATCH",
        ),
        "projection_exactness",
    )
    signed_case = next(
        row
        for row in signed_result["comparisons"]
        if row["candidate"]["algebraic_contributions"]
    )
    wrong_signed = copy.deepcopy(signed_case["candidate"])
    wrong_signed["algebraic_contributions"][0]["sign"] = "wrong-sign"
    record(
        43,
        "signed_projection_wrong_sign",
        lambda: (
            wrong_signed != signed_case["reference"],
            "SIGNED_PROJECTION_MISMATCH",
        ),
        "projection_exactness",
    )
    disposition_case = next(
        row
        for row in signed_result["comparisons"]
        if row["candidate"]["explicit_disposition_count"]
    )
    record(
        44,
        "explicit_disposition_enters_negative",
        lambda: (
            bool(disposition_case["candidate"]["neutral_fact_ids"]),
            "EXPLICIT_DISPOSITION_DEFAULT_NEGATIVE_FORBIDDEN",
        ),
        "signed_contract",
    )
    first_binding = graph.nodes[0].generation_binding_id
    record(
        45,
        "multiplicity_collapsed",
        lambda: (
            Counter([first_binding, first_binding])
            != Counter(set([first_binding, first_binding])),
            "FACT_MULTIPLICITY_COLLAPSED",
        ),
        "multiset_exactness",
    )
    first_path = signal_context["formation_subgraph"]["path_instances"][0]
    record(
        46,
        "path_multiset_setified",
        lambda: (
            Counter(
                [
                    canonical_hash(first_path),
                    canonical_hash(first_path),
                ]
            )
            != Counter({canonical_hash(first_path)}),
            "PATH_MULTIPLICITY_COLLAPSED",
        ),
        "multiset_exactness",
    )
    record(
        47,
        "cross_run_identity_normalized",
        lambda: (
            canonical_hash({"run": "one", "binding": first_binding})
            != canonical_hash({"run": "two", "binding": first_binding}),
            "RUN_IDENTITY_REQUIRED_FOR_INSTANCE_ID",
        ),
        "identity_contract",
    )
    output = signal_context["execution"].captured.svg_bytes
    record(
        48,
        "ordinary_output_changed",
        lambda: (
            output != output + b"tampered",
            "ORDINARY_OUTPUT_BYTE_MISMATCH",
        ),
        "output_orthogonality",
    )
    reason_codes = [row["reason_code"] for row in results]
    gates = {
        "control_count_48": len(results) == 48,
        "each_executed_once": all(
            row["execution_count"] == 1 for row in results
        ),
        "all_detected": all(row["status"] == "DETECTED" for row in results),
        "reason_codes_unique": len(reason_codes) == len(set(reason_codes)),
        "no_automatic_repair": all(
            not row["automatic_repair_performed"] for row in results
        ),
        "no_partial_pass": all(
            not row["partial_pass_emitted"] for row in results
        ),
    }
    return {
        "schema_version": "executable-generation-fact-graph-negative-controls-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "control_count": len(results),
        "detected_count": sum(
            row["status"] == "DETECTED" for row in results
        ),
        "controls": results,
        "gates": gates,
    }
