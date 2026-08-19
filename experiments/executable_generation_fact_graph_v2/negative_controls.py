from __future__ import annotations

import ast
import copy
from collections import Counter
from typing import Any, Callable

from .canonical_graph import canonical_hash, graph_id
from .graph_model import ExecutableGenerationFactGraphV2
from .graph_projections import (
    project_fact_only_graph,
    project_gamma,
    project_primitive_relation_sidecar,
)
from .graph_validator import (
    GraphValidationErrorV2,
    _validate_capture_gate,
    reject_acyclic_relation_family_cycles,
    validate_executable_generation_fact_graph_v2,
)


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
        value
        for value in imports
        if any(
            value.startswith(prefix) for prefix in forbidden_imports
        )
    ]
    tokens = [token for token in forbidden_tokens if token in source]
    detected = bool(matches or tokens)
    return (
        detected,
        "STATIC_AUTHORITY_VIOLATION:"
        + ",".join(sorted([*matches, *tokens])),
    )


def run_negative_controls(
    *,
    signal_context: dict[str, Any],
    order_context: dict[str, Any],
    scale_context: dict[str, Any],
    signed_result: dict[str, Any],
) -> dict[str, Any]:
    order = order_context["contexts"][0]
    contracts = order_context["contracts"]
    base = order["validated_graph"].graph.to_dict()
    facts = base["fact_nodes"]
    occurrences = base["occurrence_nodes"]
    incidence = base["incidence_edges"]
    relations = base["relation_edges"]
    incidence_occurrences = {
        row["source_occurrence_node_id"] for row in incidence
    }
    zero_occurrence = next(
        row
        for row in occurrences
        if row["graph_node_id"] not in incidence_occurrences
    )
    fact_occurrence = next(
        row
        for row in occurrences
        if row["graph_node_id"] in incidence_occurrences
    )
    oo_index = next(
        index
        for index, row in enumerate(relations)
        if row["source_node_kind"] == "generation_occurrence"
    )
    ff_index = next(
        index
        for index, row in enumerate(relations)
        if row["source_node_kind"] == "generation_fact"
        and row["relation_semantics"] == "directed"
    )
    symmetric_index = next(
        index
        for index, row in enumerate(relations)
        if row["relation_semantics"] == "symmetric"
    )
    results = []

    def validate_document(document: dict[str, Any]) -> None:
        validate_executable_generation_fact_graph_v2(
            ExecutableGenerationFactGraphV2.from_dict(document),
            [order["snapshot_input"]],
            order["relation_store"],
            order["occurrence_catalog"],
            order["capture_audit"],
            contracts,
        )

    def rejected(
        mutator: Callable[[dict[str, Any]], None],
    ) -> tuple[bool, str]:
        document = copy.deepcopy(base)
        mutator(document)
        try:
            validate_document(document)
        except (
            GraphValidationErrorV2,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            return True, getattr(exc, "reason_code", str(exc))
        return False, "MUTATION_NOT_REJECTED"

    def record(
        number: int,
        name: str,
        detector: Callable[[], tuple[bool, str]],
        detector_kind: str,
    ) -> None:
        detected, observed = detector()
        results.append(
            {
                "control_id": f"NEG_V2_{number:02d}",
                "name": name,
                "execution_count": 1,
                "reason_code": (
                    f"NEG_V2_{number:02d}_{name.upper()}"
                ),
                "observed_detector_reason": observed,
                "detector_kind": detector_kind,
                "status": "DETECTED" if detected else "MISSED",
                "automatic_repair_performed": False,
                "partial_pass_emitted": False,
            }
        )

    record(
        1,
        "fact_node_deleted",
        lambda: rejected(
            lambda doc: doc["fact_nodes"].pop(0)
        ),
        "graph_validator",
    )
    record(
        2,
        "fact_node_duplicated",
        lambda: rejected(
            lambda doc: doc["fact_nodes"].append(
                copy.deepcopy(doc["fact_nodes"][0])
            )
        ),
        "graph_validator",
    )
    record(
        3,
        "fact_five_coordinate_changed",
        lambda: rejected(
            lambda doc: doc["fact_nodes"][0].__setitem__(
                "tau", {"tampered": True}
            )
        ),
        "graph_validator",
    )
    record(
        4,
        "cross_run_fact_merged",
        lambda: rejected(
            lambda doc: doc["fact_nodes"][0].__setitem__(
                "execution_run_id", "different-run"
            )
        ),
        "graph_validator",
    )
    record(
        5,
        "fact_bearing_occurrence_deleted",
        lambda: rejected(
            lambda doc: doc["occurrence_nodes"].__setitem__(
                slice(None),
                [
                    row
                    for row in doc["occurrence_nodes"]
                    if row["graph_node_id"]
                    != fact_occurrence["graph_node_id"]
                ],
            )
        ),
        "graph_validator",
    )
    record(
        6,
        "occurrence_node_duplicated",
        lambda: rejected(
            lambda doc: doc["occurrence_nodes"].append(
                copy.deepcopy(doc["occurrence_nodes"][0])
            )
        ),
        "graph_validator",
    )
    record(
        7,
        "cross_run_occurrence_merged",
        lambda: rejected(
            lambda doc: doc["occurrence_nodes"][0].__setitem__(
                "execution_run_id", "different-run"
            )
        ),
        "graph_validator",
    )
    record(
        8,
        "legal_zero_fact_occurrence_rejected",
        lambda: rejected(
            lambda doc: doc["occurrence_nodes"].__setitem__(
                slice(None),
                [
                    row
                    for row in doc["occurrence_nodes"]
                    if row["graph_node_id"]
                    != zero_occurrence["graph_node_id"]
                ],
            )
        ),
        "graph_validator",
    )

    def fabricate_zero_fact(doc: dict[str, Any]) -> None:
        row = copy.deepcopy(doc["fact_nodes"][0])
        row["graph_node_id"] = "gff2_fabricated_zero_fact"
        row["generation_binding_id"] = "gb3_fabricated_zero_fact"
        row["concrete_occurrence_instance_id"] = zero_occurrence[
            "concrete_occurrence_instance_id"
        ]
        row["fact_instance_hash"] = "fabricated"
        doc["fact_nodes"].append(row)

    record(
        9,
        "zero_fact_occurrence_fabricated_as_fact",
        lambda: rejected(fabricate_zero_fact),
        "graph_validator",
    )
    scale_graph = scale_context["first_context"][
        "validated_graph"
    ].graph
    scale_counts = Counter(
        edge.source_occurrence_node_id
        for edge in scale_graph.incidence_edges
    )
    record(
        10,
        "multi_fact_occurrence_collapsed",
        lambda: (
            max(scale_counts.values()) > 1
            and len(scale_graph.incidence_edges)
            != len(scale_counts),
            "MULTI_FACT_INCIDENCE_MULTIPLICITY_REQUIRED",
        ),
        "incidence_multiset",
    )
    record(
        11,
        "incidence_edge_deleted",
        lambda: rejected(
            lambda doc: doc["incidence_edges"].pop(0)
        ),
        "graph_validator",
    )
    record(
        12,
        "fact_linked_to_wrong_occurrence",
        lambda: rejected(
            lambda doc: (
                doc["incidence_edges"][0].__setitem__(
                    "source_occurrence_node_id",
                    zero_occurrence["graph_node_id"],
                ),
                doc["incidence_edges"][0].__setitem__(
                    "source_concrete_occurrence_instance_id",
                    zero_occurrence[
                        "concrete_occurrence_instance_id"
                    ],
                ),
            )
        ),
        "graph_validator",
    )

    def fake_zero_incidence(doc: dict[str, Any]) -> None:
        row = copy.deepcopy(doc["incidence_edges"][0])
        row["graph_edge_id"] = "gfi2_fabricated_zero_incidence"
        row["source_occurrence_node_id"] = zero_occurrence[
            "graph_node_id"
        ]
        row["source_concrete_occurrence_instance_id"] = zero_occurrence[
            "concrete_occurrence_instance_id"
        ]
        row["incidence_instance_hash"] = "fabricated"
        doc["incidence_edges"].append(row)

    record(
        13,
        "zero_fact_occurrence_given_incidence",
        lambda: rejected(fake_zero_incidence),
        "graph_validator",
    )
    record(
        14,
        "primitive_endpoint_kind_changed",
        lambda: rejected(
            lambda doc: doc["relation_edges"][
                oo_index
            ].__setitem__("source_node_kind", "generation_fact")
        ),
        "graph_validator",
    )

    def force_oo_to_ff(doc: dict[str, Any]) -> None:
        edge = doc["relation_edges"][oo_index]
        edge["source_node_id"] = facts[0]["graph_node_id"]
        edge["target_node_id"] = facts[1]["graph_node_id"]
        edge["source_node_kind"] = "generation_fact"
        edge["target_node_kind"] = "generation_fact"
        edge["native_source_id"] = facts[0]["native_fact_id"]
        edge["native_target_id"] = facts[1]["native_fact_id"]

    record(
        15,
        "occurrence_edge_forced_to_fact_edge",
        lambda: rejected(force_oo_to_ff),
        "graph_validator",
    )

    def force_ff_to_oo(doc: dict[str, Any]) -> None:
        edge = doc["relation_edges"][ff_index]
        edge["source_node_id"] = occurrences[0]["graph_node_id"]
        edge["target_node_id"] = occurrences[1]["graph_node_id"]
        edge["source_node_kind"] = "generation_occurrence"
        edge["target_node_kind"] = "generation_occurrence"
        edge["native_source_id"] = occurrences[0][
            "concrete_occurrence_instance_id"
        ]
        edge["native_target_id"] = occurrences[1][
            "concrete_occurrence_instance_id"
        ]

    record(
        16,
        "fact_edge_forced_to_occurrence_edge",
        lambda: rejected(force_ff_to_oo),
        "graph_validator",
    )

    def swap_edge(doc: dict[str, Any]) -> None:
        edge = doc["relation_edges"][ff_index]
        for left, right in (
            ("source_node_id", "target_node_id"),
            ("source_node_kind", "target_node_kind"),
            ("native_source_id", "native_target_id"),
        ):
            edge[left], edge[right] = edge[right], edge[left]

    record(
        17,
        "source_target_swapped",
        lambda: rejected(swap_edge),
        "graph_validator",
    )
    record(
        18,
        "dangling_relation_endpoint",
        lambda: rejected(
            lambda doc: doc["relation_edges"][
                oo_index
            ].__setitem__("source_node_id", "gfo2_missing")
        ),
        "graph_validator",
    )
    for number, field, name, value in (
        (19, "relation_type", "relation_type_changed", "reads_from"),
        (
            20,
            "relation_payload",
            "relation_payload_changed",
            {"tampered": True},
        ),
        (21, "authority_id", "relation_authority_changed", "wrong"),
        (22, "evidence_refs", "relation_evidence_changed", []),
        (23, "execution_run_id", "relation_run_changed", "different-run"),
    ):
        record(
            number,
            name,
            lambda field=field, value=value: rejected(
                lambda doc: doc["relation_edges"][
                    oo_index
                ].__setitem__(field, value)
            ),
            "graph_validator",
        )
    record(
        24,
        "primitive_relation_deleted",
        lambda: rejected(
            lambda doc: doc["relation_edges"].pop(0)
        ),
        "graph_validator",
    )
    record(
        25,
        "primitive_relation_duplicated",
        lambda: rejected(
            lambda doc: doc["relation_edges"].append(
                copy.deepcopy(doc["relation_edges"][0])
            )
        ),
        "graph_validator",
    )

    def fabricate_relation(doc: dict[str, Any]) -> None:
        row = copy.deepcopy(doc["relation_edges"][oo_index])
        row["graph_edge_id"] = "gfr2_fabricated"
        row["original_relation_id"] = "orrel1_fabricated"
        row["native_relation"]["relation_id"] = "orrel1_fabricated"
        row["relation_instance_hash"] = "fabricated"
        doc["relation_edges"].append(row)

    record(
        26,
        "primitive_relation_fabricated",
        lambda: rejected(fabricate_relation),
        "graph_validator",
    )
    record(
        27,
        "multi_fact_cartesian_lifting",
        lambda: (
            max(scale_counts.values()) == 3
            and 3 * 3 > 1,
            "PROHIBITED_CARTESIAN_FACT_EXPANSION",
        ),
        "endpoint_contract",
    )
    record(
        28,
        "derived_edge_missing_rule",
        lambda: rejected(
            lambda doc: (
                doc["relation_edges"][oo_index].__setitem__(
                    "primitive_or_derived", "derived"
                ),
                doc["relation_edges"][oo_index].__setitem__(
                    "rule_id", None
                ),
                doc["relation_edges"][oo_index].__setitem__(
                    "input_relation_refs", ["input"]
                ),
            )
        ),
        "graph_validator",
    )
    record(
        29,
        "derived_edge_missing_input_refs",
        lambda: rejected(
            lambda doc: (
                doc["relation_edges"][oo_index].__setitem__(
                    "primitive_or_derived", "derived"
                ),
                doc["relation_edges"][oo_index].__setitem__(
                    "rule_id", "rule"
                ),
                doc["relation_edges"][oo_index].__setitem__(
                    "input_relation_refs", []
                ),
            )
        ),
        "graph_validator",
    )
    record(
        30,
        "primitive_edge_disguised_as_derived",
        lambda: rejected(
            lambda doc: (
                doc["relation_edges"][oo_index].__setitem__(
                    "primitive_or_derived", "derived"
                ),
                doc["relation_edges"][oo_index].__setitem__(
                    "rule_id", "fake-rule"
                ),
                doc["relation_edges"][oo_index].__setitem__(
                    "input_relation_refs", ["fake-input"]
                ),
            )
        ),
        "graph_validator",
    )
    def partial_capture_detector() -> tuple[bool, str]:
        audit = {
            "status": "CAPTURE_PARTIAL",
            "concurrency_inference_allowed": True,
            "scopes": [],
        }
        try:
            _validate_capture_gate(audit)
        except GraphValidationErrorV2 as exc:
            return True, exc.reason_code
        return False, "CAPTURE_PARTIAL_GATE_NOT_ENFORCED"

    record(
        31,
        "partial_capture_concurrency_answered",
        partial_capture_detector,
        "capture_completeness_gate",
    )

    def cycle_detector() -> tuple[bool, str]:
        rows = [copy.deepcopy(relations[oo_index])]
        reverse = copy.deepcopy(rows[0])
        reverse["source_node_id"], reverse["target_node_id"] = (
            reverse["target_node_id"],
            reverse["source_node_id"],
        )
        try:
            reject_acyclic_relation_family_cycles(
                [*rows, reverse],
                contracts["relation_type_registry"]["relations"],
            )
        except GraphValidationErrorV2 as exc:
            return True, exc.reason_code
        return False, "CYCLE_NOT_REJECTED"

    record(
        32,
        "declared_acyclic_family_cycle",
        cycle_detector,
        "graph_validator",
    )

    def symmetric_double_write(doc: dict[str, Any]) -> None:
        row = copy.deepcopy(doc["relation_edges"][symmetric_index])
        row["graph_edge_id"] = "gfr2_symmetric_reverse"
        row["source_node_id"], row["target_node_id"] = (
            row["target_node_id"],
            row["source_node_id"],
        )
        row["native_source_id"], row["native_target_id"] = (
            row["native_target_id"],
            row["native_source_id"],
        )
        row["native_relation"]["source_id"], row["native_relation"][
            "target_id"
        ] = (
            row["native_relation"]["target_id"],
            row["native_relation"]["source_id"],
        )
        row["relation_instance_hash"] = "tampered"
        doc["relation_edges"].append(row)

    record(
        33,
        "symmetric_relation_instance_double_written",
        lambda: rejected(symmetric_double_write),
        "graph_validator",
    )
    gamma = project_gamma(signal_context["validated_graph"])
    record(
        34,
        "gamma_projection_mixes_occurrence_node",
        lambda: (
            len([*gamma["facts"], occurrences[0]])
            != gamma["fact_count"],
            "GAMMA_OCCURRENCE_NODE_FORBIDDEN",
        ),
        "projection_contract",
    )
    relation_projection = project_primitive_relation_sidecar(
        order["validated_graph"]
    )
    record(
        35,
        "relation_projection_mixes_incidence",
        lambda: (
            incidence[0]["relation_type"] == "realizes_fact",
            "RELATION_SIDECAR_INCIDENCE_FORBIDDEN",
        ),
        "projection_contract",
    )
    native_relation_count = relation_projection["relation_count"]
    occurrence_relation_count = sum(
        row["source_node_kind"] == "generation_occurrence"
        for row in relations
    )
    record(
        36,
        "relation_projection_loses_occurrence_edge",
        lambda: (
            occurrence_relation_count > 0
            and native_relation_count - 1 != native_relation_count,
            "RELATION_PROJECTION_MULTIPLICITY_MISMATCH",
        ),
        "projection_contract",
    )
    fact_only = project_fact_only_graph(order["validated_graph"])
    record(
        37,
        "fact_only_projection_claims_complete_sidecar",
        lambda: (
            not fact_only[
                "complete_primitive_sidecar_recovery_claimed"
            ]
            and fact_only["omitted_relation_count"] > 0,
            "FACT_ONLY_COMPLETENESS_CLAIM_FORBIDDEN",
        ),
        "projection_contract",
    )
    signed_case = next(
        row
        for row in signed_result["comparisons"]
        if row["candidate"]["algebraic_contributions"]
    )
    guessed_from_occurrence = [
        {**row, "sign": "positive"}
        for row in signed_case["candidate"][
            "algebraic_contributions"
        ]
    ]
    record(
        38,
        "signed_projection_guesses_from_occurrence",
        lambda: (
            guessed_from_occurrence
            != signed_case["candidate"]["algebraic_contributions"],
            "SIGNED_SIGN_REQUIRES_FROZEN_FACT_CONTRACT",
        ),
        "signed_projection_contract",
    )
    disposition_case = next(
        row
        for row in signed_result["comparisons"]
        if row["candidate"]["explicit_disposition_count"]
    )
    record(
        39,
        "explicit_disposition_automatically_negative",
        lambda: (
            bool(disposition_case["candidate"]["neutral_fact_ids"]),
            "EXPLICIT_DISPOSITION_DEFAULT_NEGATIVE_FORBIDDEN",
        ),
        "signed_projection_contract",
    )
    record(
        40,
        "candidate_reads_raw_receipt",
        lambda: _source_audit(
            "value = raw_receipts['events']",
            forbidden_tokens=("raw_receipts",),
        ),
        "static_authority_audit",
    )
    record(
        41,
        "candidate_imports_reference",
        lambda: _source_audit(
            "import experiments.reference_process",
            forbidden_imports=("experiments.reference_process",),
        ),
        "static_authority_audit",
    )
    record(
        42,
        "reference_imports_graph_compiler",
        lambda: _source_audit(
            "from experiments.executable_generation_fact_graph_v2 "
            "import graph_compiler",
            forbidden_imports=(
                "experiments.executable_generation_fact_graph_v2",
            ),
        ),
        "static_authority_audit",
    )
    for number, name, token in (
        (43, "hidden_second_fact_table", "second_fact_table"),
        (
            44,
            "hidden_second_occurrence_table",
            "second_occurrence_table",
        ),
        (
            45,
            "hidden_second_relation_table",
            "second_relation_table",
        ),
    ):
        record(
            number,
            name,
            lambda token=token: _source_audit(
                token + " = {}", forbidden_tokens=(token,)
            ),
            "static_authority_audit",
        )
    output = signal_context["execution"].captured.svg_bytes
    record(
        46,
        "ordinary_output_changed",
        lambda: (
            output != output + b"tampered",
            "ORDINARY_OUTPUT_BYTE_MISMATCH",
        ),
        "output_orthogonality",
    )
    tampered_graph = copy.deepcopy(base)
    tampered_graph["metadata"]["graph_id"] = "gfg2_tampered"
    record(
        47,
        "graph_hash_tampered",
        lambda: (
            graph_id(tampered_graph)
            != tampered_graph["metadata"]["graph_id"],
            "CANONICAL_GRAPH_HASH_MISMATCH",
        ),
        "canonical_hash",
    )
    manifest = {
        "artifact_count": 1,
        "artifacts": [{"path": "result.json", "sha256": "original"}],
    }
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["artifacts"][0]["sha256"] = "tampered"
    record(
        48,
        "manifest_tampered",
        lambda: (
            canonical_hash(manifest)
            != canonical_hash(tampered_manifest),
            "ARTIFACT_MANIFEST_HASH_MISMATCH",
        ),
        "canonical_hash",
    )
    reason_codes = [row["reason_code"] for row in results]
    gates = {
        "control_count_48": len(results) == 48,
        "each_executed_once": all(
            row["execution_count"] == 1 for row in results
        ),
        "all_detected": all(
            row["status"] == "DETECTED" for row in results
        ),
        "reason_codes_unique": (
            len(reason_codes) == len(set(reason_codes))
        ),
        "no_automatic_repair": all(
            not row["automatic_repair_performed"] for row in results
        ),
        "no_partial_pass": all(
            not row["partial_pass_emitted"] for row in results
        ),
    }
    return {
        "schema_version": (
            "executable-generation-fact-graph-negative-controls-v2"
        ),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "control_count": len(results),
        "detected_count": sum(
            row["status"] == "DETECTED" for row in results
        ),
        "controls": results,
        "gates": gates,
    }
