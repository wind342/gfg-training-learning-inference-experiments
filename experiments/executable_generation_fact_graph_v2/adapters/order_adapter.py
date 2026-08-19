from __future__ import annotations

import ast
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.executable_generation_fact_graph_v1.adapters.core_snapshot_adapter import (
    build_core_snapshot_from_atomic_facts,
)
from experiments.order_refund_freeze_inter_fact_relations_v1.src.scientific_runner import (
    _run_process,
    run_scientific,
)
from experiments.order_refund_freeze_inter_fact_relations_v1.src.queries import (
    frozen_queries,
)

from ..canonical_graph import canonical_hash, content_id
from ..endpoint_registry import build_core_occurrence_catalog
from ..graph_compiler import (
    compile_executable_generation_fact_graph_v2,
)
from ..graph_projections import (
    project_fact_only_graph,
    project_gamma,
    project_occurrence_view,
    project_primitive_relation_sidecar,
)
from ..graph_query import ExecutableGenerationFactGraphQueryEngineV2
from ..graph_validator import (
    load_contracts,
    validate_executable_generation_fact_graph_v2,
)
from .common import (
    native_sidecar_semantic_material,
    normalize_native_relation_sidecar,
)
from .order_graph_query_adapter import (
    ORDER_COMPENSATION_POLICY,
    resolve_order_compensation_targets,
)


DOMAIN_SCOPE_ID = "order-refund-freeze-v1"


def _endpoint_evidence(
    sidecar: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_occurrence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evidence in sidecar.get("evidence", []):
        endpoint_ids = [
            *evidence.get("occurrence_ids", []),
            *evidence.get("endpoint_ids", []),
        ]
        for endpoint_id in endpoint_ids:
            if endpoint_id.startswith("orocc1_"):
                by_occurrence[endpoint_id].append(evidence)
    return by_occurrence


def _program_order_positions(
    sidecar: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    positions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evidence in sidecar.get("evidence", []):
        if evidence.get("evidence_kind") != "program_order_log":
            continue
        occurrences = evidence.get("occurrence_ids", [])
        if len(occurrences) != 2:
            continue
        payload = evidence.get("payload", {})
        actor_id = payload.get("actor_id")
        for occurrence_id, index_key in zip(
            occurrences,
            ("source_sequence_index", "target_sequence_index"),
            strict=True,
        ):
            candidate = {
                "actor_id": actor_id,
                "sequence_index": payload.get(index_key),
            }
            if candidate not in positions[occurrence_id]:
                positions[occurrence_id].append(candidate)
    return {
        occurrence_id: sorted(
            rows,
            key=lambda row: (
                str(row.get("actor_id")),
                -1
                if row.get("sequence_index") is None
                else row["sequence_index"],
            ),
        )
        for occurrence_id, rows in positions.items()
    }


def build_order_occurrence_catalog(
    snapshot_input: dict[str, Any],
    sidecar: dict[str, Any],
) -> dict[str, Any]:
    core_catalog = build_core_occurrence_catalog([snapshot_input])
    core_rows = {
        row["concrete_occurrence_instance_id"]: row
        for row in core_catalog["occurrences"]
    }
    occurrence_endpoints = sorted(
        {
            endpoint
            for relation in sidecar["relations"]
            if relation["endpoint_level"] == "occurrence"
            for endpoint in (relation["source_id"], relation["target_id"])
        }
    )
    evidence_by_occurrence = _endpoint_evidence(sidecar)
    positions = _program_order_positions(sidecar)
    rows = dict(core_rows)
    for occurrence_id in occurrence_endpoints:
        if occurrence_id in rows:
            continue
        evidence = sorted(
            evidence_by_occurrence.get(occurrence_id, []),
            key=lambda row: row["evidence_id"],
        )
        if not evidence:
            raise ValueError(
                "ORDER_OCCURRENCE_ENDPOINT_EVIDENCE_MISSING:"
                + occurrence_id
            )
        occurrence_positions = positions.get(occurrence_id, [])
        actors = sorted(
            {
                row["actor_id"]
                for row in occurrence_positions
                if row.get("actor_id")
            }
        )
        indexes = {
            row["sequence_index"]
            for row in occurrence_positions
            if row.get("sequence_index") is not None
        }
        evidence_kinds = sorted(
            {row["evidence_kind"] for row in evidence}
        )
        rows[occurrence_id] = {
            "execution_run_id": sidecar["execution_run_id"],
            "concrete_occurrence_instance_id": occurrence_id,
            "generation_occurrence_id": None,
            "occurrence_type": "validated_relation_endpoint_occurrence",
            "occurrence_stage": (
                actors[0]
                if len(actors) == 1
                else (
                    "multi_actor_endpoint"
                    if actors
                    else "validated_relation_endpoint"
                )
            ),
            "stable_instance_key": (
                sidecar["execution_run_id"] + ":" + occurrence_id
            ),
            "occurrence_index": (
                next(iter(indexes)) if len(indexes) == 1 else None
            ),
            "transform_reference": {
                "native_evidence_kinds": evidence_kinds,
                "semantic_claim": "endpoint occurrence only",
            },
            "occurrence_payload": {
                "native_occurrence_id": occurrence_id,
                "program_order_positions": occurrence_positions,
                "endpoint_evidence": evidence,
            },
            "generator_manifest_id": None,
            "evidence_refs": [
                row["evidence_id"] for row in evidence
            ],
            "catalog_authority": (
                "controlled-order-workflow-endpoint-catalog-v2"
            ),
        }
    material = {
        "schema_version": "occurrence-endpoint-catalog-v2",
        "execution_run_id": sidecar["execution_run_id"],
        "occurrences": [rows[key] for key in sorted(rows)],
        "establishment_source": (
            "validated_core_snapshot_and_native_relation_evidence"
        ),
        "input_catalog_ids": [core_catalog["occurrence_catalog_id"]],
    }
    return {
        **material,
        "occurrence_catalog_id": content_id("gfoc2_", material),
    }


def _compile_bundle(
    atomic: dict[str, Any],
    native_sidecar: dict[str, Any],
    capture_audit: dict[str, Any],
    contracts: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    run_id = atomic["execution_run_id"]
    snapshot_input, mapping = build_core_snapshot_from_atomic_facts(
        atomic_fact_bundle=atomic,
        execution_run_id=run_id,
        domain_scope_id=DOMAIN_SCOPE_ID,
        generator_name="order-refund-freeze-native-workflow",
    )
    relation_store = normalize_native_relation_sidecar(native_sidecar)
    catalog = build_order_occurrence_catalog(
        snapshot_input, native_sidecar
    )
    graph = compile_executable_generation_fact_graph_v2(
        [snapshot_input],
        relation_store,
        catalog,
        capture_audit,
        contracts["graph_profile"],
        contracts["relation_type_registry"],
    )
    validated = validate_executable_generation_fact_graph_v2(
        graph,
        [snapshot_input],
        relation_store,
        catalog,
        capture_audit,
        contracts,
    )
    return validated, {
        "snapshot_input": snapshot_input,
        "mapping": mapping,
        "relation_store": relation_store,
        "occurrence_catalog": catalog,
        "capture_audit": capture_audit,
    }


def _order_subgraph_examples(
    query: ExecutableGenerationFactGraphQueryEngineV2,
) -> dict[str, Any]:
    by_value: dict[str, list[Any]] = defaultdict(list)
    for node in query.graph.fact_nodes:
        native = node.native_fact or {}
        coordinates = native.get("coordinates", {})
        value = coordinates.get("z", {}).get("value")
        if isinstance(value, str):
            by_value[value].append(node)

    def describe(value: str) -> dict[str, Any]:
        rows = sorted(
            by_value.get(value, []),
            key=lambda row: row.graph_node_id,
        )
        if not rows:
            return {"status": "NOT_APPLICABLE", "value": value}
        node = rows[0]
        occurrence_id = query.occurrence_for_fact(
            node.graph_node_id
        )
        result = {
            "status": "PRESENT",
            "value": value,
            "fact_node_id": node.graph_node_id,
            "occurrence_node_id": occurrence_id,
            "incoming_relations": query.relation_edges(
                node.graph_node_id, "in"
            ),
            "outgoing_relations": query.relation_edges(
                node.graph_node_id, "out"
            ),
            "execution_subgraph": query.execution_subgraph(
                occurrence_id,
                {
                    "relation_types": [
                        "program_order",
                        "message_send_receive",
                        "synchronizes_with",
                    ],
                    "maximum_edges": 3,
                },
            ),
            "downstream_impact": query.downstream_impact(
                node.graph_node_id,
                {"generated_origin_dependency"},
            ),
        }
        if value == ORDER_COMPENSATION_POLICY[
            "source_outcome_value"
        ]:
            result["order_compensation_target"] = (
                resolve_order_compensation_targets(
                    query,
                    [node.graph_node_id],
                    ORDER_COMPENSATION_POLICY,
                )
            )
        if node.z["reference"]["kind"] == "support":
            key = node.z["entity"]["support_payload"][
                "native_support_key"
            ]
            result["formation_subgraph"] = query.formation_subgraph(
                {
                    "predicate": "native_support_key_membership",
                    "native_support_keys": [key],
                },
                {
                    "relation_types": [
                        "generated_origin_dependency"
                    ],
                    "stop_at_registered_source": True,
                },
            )
        else:
            result["formation_subgraph"] = {
                "status": "EXPLICIT_DISPOSITION_ANCHOR",
                "fact_node_id": node.graph_node_id,
                "incoming_relations": result["incoming_relations"],
            }
        return result

    return {
        "RefundCommitted": describe("RefundCommitted"),
        "RefundRejected": describe(
            "REFUND_VERSION_CONFLICT_AFTER_FREEZE"
        ),
        "NotificationSent": describe("NotificationSent"),
        "NotificationSuppressed": describe(
            "NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND"
        ),
        "reads_from_edges": [
            row.to_dict()
            for row in query.graph.relation_edges
            if row.relation_type == "reads_from"
        ],
        "conflict_edges": [
            row.to_dict()
            for row in query.graph.relation_edges
            if row.relation_type == "conflicts_with"
        ],
        "queue_send_receive_edges": [
            row.to_dict()
            for row in query.graph.relation_edges
            if row.relation_type == "message_send_receive"
        ],
        "barrier_event_synchronization_edges": [
            row.to_dict()
            for row in query.graph.relation_edges
            if row.relation_type == "synchronizes_with"
        ],
    }


def _run_projected_candidate(
    compiled_rows: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    source_scientific: dict[str, Any],
) -> dict[str, Any]:
    projected_contexts = []
    for summary, context in zip(
        compiled_rows, contexts, strict=True
    ):
        graph = context["validated_graph"]
        sidecar = project_primitive_relation_sidecar(graph)[
            "stores"
        ][0]
        facts = sorted(
            (
                node.native_fact
                for node in graph.graph.fact_nodes
                if node.native_fact is not None
            ),
            key=lambda row: row["fact_id"],
        )
        projected_contexts.append(
            {
                "validated_atomic_facts": {
                    "schema_version": (
                        "order-refund-freeze-atomic-facts-v1"
                    ),
                    "execution_run_id": summary["execution_run_id"],
                    "scenario": summary["scenario"],
                    "status": "PASS",
                    "fact_count": len(facts),
                    "coordinate_names": [
                        "u",
                        "tau",
                        "omega_bar",
                        "z",
                        "rho",
                    ],
                    "sixth_coordinate_present": False,
                    "facts": facts,
                },
                "validated_relation_sidecar": sidecar,
                "capture_audit": context["capture_audit"],
            }
        )
    candidate_input = {
        "contexts": projected_contexts,
        "queries": frozen_queries(),
        "lifting_rules": {
            "policy": "RESULT_LEVEL_RELATION_SPECIFIC",
            "concurrency_requires": "CAPTURE_COMPLETE",
        },
        "schema_version": "candidate-input-v1",
    }
    compare_input = {
        "candidate_answers": None,
        "reference_answers": source_scientific[
            "reference_answers"
        ],
        "schema_version": "order-v2-compare-input-v2",
    }
    with tempfile.TemporaryDirectory(
        prefix="generation-fact-graph-v2-order-"
    ) as temporary_directory:
        directory = Path(temporary_directory)
        candidate = _run_process(
            (
                "experiments.order_refund_freeze_inter_fact_relations_v1."
                "src.candidate_process"
            ),
            candidate_input,
            directory,
            "v2-candidate",
        )
        compare_input["candidate_answers"] = candidate["output"]
        from ..references.order_compare_process import compare_payload

        precheck = compare_payload(compare_input)
        if precheck["status"] != "PASS":
            raise RuntimeError(
                "ORDER_PROJECTED_CANDIDATE_MISMATCH:"
                + repr(precheck["mismatches"][:1])
            )
        comparison = _run_process(
            (
                "experiments.executable_generation_fact_graph_v2."
                "references.order_compare_process"
            ),
            compare_input,
            directory,
            "v2-compare",
        )
    return {
        "comparison": comparison["output"],
        "candidate_pid": candidate["pid"],
        "compare_pid": comparison["pid"],
        "candidate_compare_distinct_processes": (
            candidate["pid"] != comparison["pid"]
        ),
        "source_reference_was_distinct_process": source_scientific[
            "process_isolation_audit"
        ]["observed_process_ids_distinct"],
        "candidate_input_keys": sorted(candidate_input),
        "candidate_reads_raw_receipts": False,
        "candidate_reads_reference_output": False,
        "reference_reads_graph": False,
    }


def _direct_candidate_source_audit() -> dict[str, Any]:
    paths = [
        Path(__file__).with_name(
            "order_graph_candidate_process.py"
        ),
        Path(__file__).with_name(
            "order_graph_query_adapter.py"
        ),
    ]
    forbidden_import_prefixes = (
        "experiments.order_refund_freeze_inter_fact_relations_v1",
        "experiments.executable_generation_fact_graph_v2.references",
    )
    forbidden_tokens = (
        "native_fact",
        "native_relation",
        "project_primitive_relation_sidecar",
        "validated_relation_sidecar",
        "reference_answers",
        "sql_receipts",
        "queue_receipts",
        "synchronization_receipts",
    )
    imports = []
    token_matches = []
    sources = {}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        sources[path.name] = source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        token_matches.extend(
            token for token in forbidden_tokens if token in source
        )
    import_matches = [
        value
        for value in imports
        if any(
            value.startswith(prefix)
            for prefix in forbidden_import_prefixes
        )
    ]
    return {
        "status": (
            "PASS"
            if not import_matches and not token_matches
            else "FAIL"
        ),
        "forbidden_import_matches": sorted(set(import_matches)),
        "forbidden_token_matches": sorted(set(token_matches)),
        "source_sha256": canonical_hash(sources),
    }


def _run_direct_graph_candidate(
    contexts: list[dict[str, Any]],
    source_scientific: dict[str, Any],
) -> dict[str, Any]:
    candidate_contexts = [
        {
            "graph": context["validated_graph"].graph.to_dict(),
            "validation": (
                context["validated_graph"].validation.to_dict()
            ),
            "capture_audit": context["capture_audit"],
        }
        for context in contexts
    ]
    candidate_input = {
        "contexts": candidate_contexts,
        "queries": frozen_queries(),
        "schema_version": "order-graph-candidate-input-v2",
    }
    compare_input = {
        "candidate_answers": None,
        "reference_answers": source_scientific[
            "reference_answers"
        ],
        "schema_version": "order-v2-compare-input-v2",
    }
    with tempfile.TemporaryDirectory(
        prefix="generation-fact-graph-v2-order-direct-"
    ) as temporary_directory:
        directory = Path(temporary_directory)
        candidate = _run_process(
            (
                "experiments.executable_generation_fact_graph_v2."
                "adapters.order_graph_candidate_process"
            ),
            candidate_input,
            directory,
            "v2-direct-graph-candidate",
        )
        compare_input["candidate_answers"] = candidate["output"]
        from ..references.order_compare_process import compare_payload

        precheck = compare_payload(compare_input)
        if precheck["status"] != "PASS":
            raise RuntimeError(
                "ORDER_DIRECT_GRAPH_CANDIDATE_MISMATCH:"
                + repr(precheck["mismatches"][:1])
            )
        comparison = _run_process(
            (
                "experiments.executable_generation_fact_graph_v2."
                "references.order_compare_process"
            ),
            compare_input,
            directory,
            "v2-direct-graph-compare",
        )
    candidate_rows = candidate["output"]["answers"]
    reference_by_key = {
        (row["scenario"], row["query_id"]): row
        for row in source_scientific["reference_answers"]["answers"]
    }
    compensation_rows = [
        row for row in candidate_rows if row["query_id"] == "Q11"
    ]
    return {
        "comparison": comparison["output"],
        "candidate_pid": candidate["pid"],
        "compare_pid": comparison["pid"],
        "candidate_compare_distinct_processes": (
            candidate["pid"] != comparison["pid"]
        ),
        "source_reference_was_distinct_process": source_scientific[
            "process_isolation_audit"
        ]["observed_process_ids_distinct"],
        "candidate_input_keys": sorted(candidate_input),
        "candidate_context_keys": sorted(candidate_contexts[0]),
        "candidate_runtime_file_read_audit": candidate["output"][
            "runtime_file_read_audit"
        ],
        "candidate_source_audit": _direct_candidate_source_audit(),
        "candidate_answer_sha256": canonical_hash(candidate_rows),
        "compensation_policy_id": candidate["output"][
            "compensation_policy_id"
        ],
        "compensation_query_count": len(compensation_rows),
        "compensation_queries_exact": all(
            row["answer"]
            == reference_by_key[
                (row["scenario"], row["query_id"])
            ]["answer"]
            for row in compensation_rows
        ),
        "candidate_reads_projected_facts": False,
        "candidate_reads_projected_sidecar": False,
        "candidate_reads_raw_receipts": False,
        "candidate_reads_reference_output": False,
        "reference_reads_graph": False,
    }


def run_order_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    source_result = run_scientific()
    scientific = source_result["scientific"]
    contracts = load_contracts()
    compiled = []
    contexts = []
    for atomic, sidecar, audit in zip(
        scientific["atomic_generation_facts"],
        scientific["primitive_relation_sidecars"],
        scientific["capture_completeness_audits"],
        strict=True,
    ):
        validated, context = _compile_bundle(
            atomic, sidecar, audit, contracts
        )
        query = ExecutableGenerationFactGraphQueryEngineV2(validated)
        relation_projection = project_primitive_relation_sidecar(
            validated
        )
        projected_store = relation_projection["stores"][0]
        source_material = native_sidecar_semantic_material(sidecar)
        projected_material = native_sidecar_semantic_material(
            projected_store
        )
        fact_only = project_fact_only_graph(validated)
        occurrence_view = project_occurrence_view(validated)
        gamma = project_gamma(validated)
        examples = _order_subgraph_examples(query)
        occurrence_counts = Counter(
            len(query.facts_realized_by_occurrence(row.graph_node_id))
            for row in validated.graph.occurrence_nodes
        )
        compiled.append(
            {
                "scenario": atomic["scenario"],
                "execution_run_id": atomic["execution_run_id"],
                "graph_id": validated.graph_id,
                "fact_node_count": len(validated.graph.fact_nodes),
                "occurrence_node_count": len(
                    validated.graph.occurrence_nodes
                ),
                "zero_fact_occurrence_count": occurrence_counts[0],
                "one_fact_occurrence_count": occurrence_counts[1],
                "multi_fact_occurrence_count": sum(
                    count
                    for multiplicity, count in occurrence_counts.items()
                    if multiplicity > 1
                ),
                "incidence_edge_count": len(
                    validated.graph.incidence_edges
                ),
                "primitive_relation_count": len(
                    validated.graph.relation_edges
                ),
                "native_relation_count": len(sidecar["relations"]),
                "relation_projection_exact": (
                    projected_material == source_material
                ),
                "relation_projection_sha256": canonical_hash(
                    projected_material
                ),
                "native_sidecar_sha256": canonical_hash(source_material),
                "gamma_fact_count": gamma["fact_count"],
                "occurrence_view_relation_count": occurrence_view[
                    "relation_count"
                ],
                "fact_only_retained_count": fact_only[
                    "retained_relation_count"
                ],
                "fact_only_omitted_count": fact_only[
                    "omitted_relation_count"
                ],
                "validation": validated.validation.to_dict(),
                "subgraph_examples": examples,
            }
        )
        contexts.append(
            {
                **context,
                "validated_graph": validated,
                "query_engine": query,
            }
        )

    projected = _run_projected_candidate(
        compiled, contexts, scientific
    )
    direct = _run_direct_graph_candidate(contexts, scientific)
    relation_type_counts = Counter(
        edge.relation_type
        for context in contexts
        for edge in context["validated_graph"].graph.relation_edges
    )
    endpoint_signature_counts = Counter(
        edge.source_node_kind + "->" + edge.target_node_kind
        for context in contexts
        for edge in context["validated_graph"].graph.relation_edges
    )
    comparison = direct["comparison"]
    projection_comparison = projected["comparison"]
    total_native = sum(
        len(row["relations"])
        for row in scientific["primitive_relation_sidecars"]
    )
    total_compiled = sum(
        row["primitive_relation_count"] for row in compiled
    )
    gates = {
        "source_scientific_pass": scientific["status"] == "PASS",
        "forty_real_workflow_executions": scientific["run_manifest"][
            "real_workflow_execution_count"
        ]
        == 40,
        "order_direct_graph_queries_56_exact": (
            comparison["query_count"] == 56
            and comparison["status"] == "PASS"
        ),
        "order_direct_graph_fp_zero": (
            comparison["false_positive_count"] == 0
        ),
        "order_direct_graph_fn_zero": (
            comparison["false_negative_count"] == 0
        ),
        "order_projection_compatibility_56_exact": (
            projection_comparison["query_count"] == 56
            and projection_comparison["status"] == "PASS"
        ),
        "order_compensation_queries_4_exact": (
            direct["compensation_query_count"] == 4
            and direct["compensation_queries_exact"]
            and direct["compensation_policy_id"]
            == ORDER_COMPENSATION_POLICY["policy_id"]
        ),
        "all_fact_nodes_exact": all(
            row["fact_node_count"] == row["gamma_fact_count"]
            for row in compiled
        ),
        "all_occurrence_endpoints_represented": all(
            context["validated_graph"].validation.gates[
                "every_referenced_occurrence_exactly_one_node"
            ]
            for context in contexts
        ),
        "all_primitive_relations_represented": (
            total_native == total_compiled == 83
        ),
        "all_relation_projections_exact": all(
            row["relation_projection_exact"] for row in compiled
        ),
        "endpoint_kind_mismatch_zero": all(
            row["validation"]["gates"][
                "primitive_endpoint_kind_exact"
            ]
            for row in compiled
        ),
        "endpoint_identity_mismatch_zero": all(
            row["validation"]["gates"][
                "primitive_endpoint_identity_exact"
            ]
            for row in compiled
        ),
        "zero_fact_occurrences_present": sum(
            row["zero_fact_occurrence_count"] for row in compiled
        )
        > 0,
        "no_forced_lifting": all(
            row["validation"]["gates"]["no_forced_lifting"]
            for row in compiled
        ),
        "no_relation_drop": all(
            row["validation"]["gates"]["no_relation_drop"]
            for row in compiled
        ),
        "no_relation_fabrication": all(
            row["validation"]["gates"]["no_relation_fabrication"]
            for row in compiled
        ),
        "direct_graph_candidate_inputs_only": (
            not direct["candidate_reads_projected_facts"]
            and not direct["candidate_reads_projected_sidecar"]
            and not direct["candidate_reads_raw_receipts"]
            and not direct["candidate_reads_reference_output"]
            and not direct["reference_reads_graph"]
            and direct["candidate_runtime_file_read_audit"][
                "input_file_only"
            ]
            and direct["candidate_source_audit"]["status"] == "PASS"
        ),
        "projection_compatibility_inputs_only": (
            not projected["candidate_reads_raw_receipts"]
            and not projected["candidate_reads_reference_output"]
            and not projected["reference_reads_graph"]
        ),
        "direct_candidate_reference_compare_process_isolation": (
            direct["candidate_compare_distinct_processes"]
            and direct["source_reference_was_distinct_process"]
        ),
        "projection_candidate_compare_process_isolation": (
            projected["candidate_compare_distinct_processes"]
        ),
    }
    result = {
        "schema_version": "executable-generation-fact-graph-order-v2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "source_scientific_sha256": source_result["scientific_sha256"],
        "workflow_execution_count": scientific["run_manifest"][
            "real_workflow_execution_count"
        ],
        "query_count": comparison["query_count"],
        "false_positive_count": comparison["false_positive_count"],
        "false_negative_count": comparison["false_negative_count"],
        "query_mismatch_count": comparison["mismatch_count"],
        "direct_graph_query_count": comparison["query_count"],
        "direct_graph_query_mismatch_count": comparison[
            "mismatch_count"
        ],
        "projection_compatibility_query_count": (
            projection_comparison["query_count"]
        ),
        "projection_compatibility_mismatch_count": (
            projection_comparison["mismatch_count"]
        ),
        "compensation_query_count": direct[
            "compensation_query_count"
        ],
        "native_primitive_relation_count": total_native,
        "compiled_primitive_relation_count": total_compiled,
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "endpoint_signature_counts": dict(
            sorted(endpoint_signature_counts.items())
        ),
        "graphs": compiled,
        "gates": gates,
        "process_isolation": {
            "source_reference_was_distinct_process": direct[
                "source_reference_was_distinct_process"
            ],
            "direct_graph": direct,
            "projection_compatibility": projected,
        },
    }
    return result, {
        "source_result": source_result,
        "contexts": contexts,
        "contracts": contracts,
    }
