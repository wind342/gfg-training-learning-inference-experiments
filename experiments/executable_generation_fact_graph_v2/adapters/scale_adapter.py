from __future__ import annotations

import gc
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from experiments.executable_generation_fact_graph_v1.adapters.core_snapshot_adapter import (
    build_core_snapshot_from_atomic_facts,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.common import (
    canonical_sha256,
    write_json,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.mixed_dag import (
    build_mixed_dag,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.capture_auditor import (
    audit_capture,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.runner import (
    PeakRssSampler,
    _run_process,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.semantic_evidence_validator import (
    validate_primitive_store,
)

from ..canonical_graph import canonical_hash, content_id
from ..graph_compiler import (
    compile_executable_generation_fact_graph_v2,
)
from ..graph_projections import project_primitive_relation_sidecar
from ..graph_validator import (
    load_contracts,
    validate_executable_generation_fact_graph_v2,
)
from .common import (
    native_sidecar_semantic_material,
    normalize_native_relation_sidecar,
)


DOMAIN_SCOPE_ID = "inter-fact-relations-scale-v1"


def _native_sidecar(
    primitive_store: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "inter-fact-relation-hardening-v1",
        "execution_run_id": primitive_store["execution_run_id"],
        "status": primitive_store["status"],
        "primitive_relation_count": primitive_store[
            "primitive_relation_count"
        ],
        "relations": primitive_store["primitive_relations"],
        "evidence": primitive_store["evidence"],
    }


def _build_occurrence_catalog(
    *,
    receipts: dict[str, Any],
    primitive_store: dict[str, Any],
    snapshot_input: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    evidence_by_occurrence: dict[str, set[str]] = defaultdict(set)
    for evidence in primitive_store["evidence"]:
        for occurrence_id in evidence.get("occurrence_ids", []):
            evidence_by_occurrence[occurrence_id].add(
                evidence["evidence_id"]
            )
    snapshot_occurrences = {
        row["generation_occurrence_id"]: row
        for row in snapshot_input[
            "snapshot"
        ].tables.generation_occurrences
    }
    rows = []
    for receipt in receipts["occurrences"]:
        native_id = receipt["concrete_occurrence_instance_id"]
        core_id = mapping["occurrence_to_core_occurrence"].get(
            native_id
        )
        if core_id is None:
            raise ValueError(
                "SCALE_OCCURRENCE_CORE_ID_MISSING:" + native_id
            )
        core_occurrence = snapshot_occurrences[core_id]
        rows.append(
            {
                "execution_run_id": receipts["execution_run_id"],
                "concrete_occurrence_instance_id": native_id,
                "generation_occurrence_id": core_id,
                "occurrence_type": "controlled_executor_occurrence",
                "occurrence_stage": receipt["operation"],
                "stable_instance_key": (
                    receipts["execution_run_id"] + ":" + native_id
                ),
                "occurrence_index": receipt["sequence_index"],
                "transform_reference": {
                    "operation": receipt["operation"]
                },
                "occurrence_payload": {
                    "native_occurrence_receipt": receipt
                },
                "generator_manifest_id": core_occurrence[
                    "generator_manifest_id"
                ],
                "evidence_refs": sorted(
                    evidence_by_occurrence.get(native_id, set())
                ),
                "catalog_authority": (
                    "controlled-executor-runtime-receipts-v2"
                ),
            }
        )
    material = {
        "schema_version": "occurrence-endpoint-catalog-v2",
        "execution_run_id": receipts["execution_run_id"],
        "occurrences": sorted(
            rows,
            key=lambda row: row[
                "concrete_occurrence_instance_id"
            ],
        ),
        "establishment_source": (
            "validated_controlled_executor_runtime_receipts"
        ),
    }
    return {
        **material,
        "occurrence_catalog_id": content_id("gfoc2_", material),
    }


def _build_graph_context(scale: str) -> dict[str, Any]:
    workload = build_mixed_dag(scale)
    builder = workload["builder"]
    receipts = builder.runtime_receipts()
    capture_contract = builder.capture_contract()
    primitive_store = validate_primitive_store(
        builder.primitive_store(), receipts
    )
    capture_audit = audit_capture(
        capture_contract, receipts, primitive_store
    )
    atomic = {
        "schema_version": "scale-atomic-fact-bundle-v2",
        "execution_run_id": builder.run_id,
        "facts": receipts["facts"],
    }
    snapshot_input, mapping = build_core_snapshot_from_atomic_facts(
        atomic_fact_bundle=atomic,
        execution_run_id=builder.run_id,
        domain_scope_id=DOMAIN_SCOPE_ID,
        generator_name="controlled-mixed-dag-executor",
    )
    native_sidecar = _native_sidecar(primitive_store)
    relation_store = normalize_native_relation_sidecar(native_sidecar)
    occurrence_catalog = _build_occurrence_catalog(
        receipts=receipts,
        primitive_store=primitive_store,
        snapshot_input=snapshot_input,
        mapping=mapping,
    )
    contracts = load_contracts()
    compile_started = time.perf_counter()
    graph = compile_executable_generation_fact_graph_v2(
        [snapshot_input],
        relation_store,
        occurrence_catalog,
        capture_audit,
        contracts["graph_profile"],
        contracts["relation_type_registry"],
    )
    compile_elapsed = time.perf_counter() - compile_started
    validation_started = time.perf_counter()
    validated = validate_executable_generation_fact_graph_v2(
        graph,
        [snapshot_input],
        relation_store,
        occurrence_catalog,
        capture_audit,
        contracts,
    )
    validation_elapsed = time.perf_counter() - validation_started
    projection = project_primitive_relation_sidecar(validated)[
        "stores"
    ][0]
    projection_exact = (
        native_sidecar_semantic_material(projection)
        == native_sidecar_semantic_material(native_sidecar)
    )
    return {
        "scale": scale,
        "workload": workload,
        "receipts": receipts,
        "capture_contract": capture_contract,
        "primitive_store": primitive_store,
        "native_sidecar": native_sidecar,
        "relation_store": relation_store,
        "capture_audit": capture_audit,
        "snapshot_input": snapshot_input,
        "mapping": mapping,
        "occurrence_catalog": occurrence_catalog,
        "contracts": contracts,
        "validated_graph": validated,
        "compile_elapsed_seconds": compile_elapsed,
        "validation_elapsed_seconds": validation_elapsed,
        "relation_projection_exact": projection_exact,
    }


def _run_isolated_queries(
    context: dict[str, Any],
) -> dict[str, Any]:
    graph = context["validated_graph"]
    workload = context["workload"]
    with tempfile.TemporaryDirectory(
        prefix="generation-fact-graph-v2-scale-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        candidate_input_path = temporary / "candidate-input.json"
        candidate_output_path = temporary / "candidate-output.json"
        reference_input_path = temporary / "reference-input.json"
        reference_output_path = temporary / "reference-output.json"
        compare_input_path = temporary / "compare-input.json"
        compare_output_path = temporary / "compare-output.json"
        candidate_input = {
            "execution_run_id": graph.graph.metadata.execution_run_id,
            "graph": graph.graph.to_dict(),
            "validation": graph.validation.to_dict(),
            "capture_audit": context["capture_audit"],
            "queries": workload["queries"],
            "schema_version": "scale-graph-candidate-input-v2",
        }
        reference_input = {
            "execution_run_id": graph.graph.metadata.execution_run_id,
            "runtime_receipts": context["receipts"],
            "capture_contract": context["capture_contract"],
            "queries": workload["queries"],
            "reference_mode": (
                "eager"
                if context["scale"] == "small"
                else "lazy_oracle"
            ),
            "schema_version": "reference-input-v1",
        }
        write_json(candidate_input_path, candidate_input)
        write_json(reference_input_path, reference_input)
        candidate = _run_process(
            (
                "experiments.executable_generation_fact_graph_v2."
                "adapters.scale_candidate_process"
            ),
            candidate_input_path,
            candidate_output_path,
        )
        reference = _run_process(
            (
                "experiments.inter_fact_relations_v0_hardening_scale_v1."
                "src.reference_process"
            ),
            reference_input_path,
            reference_output_path,
        )
        compare_input = {
            "candidate": candidate["result"],
            "reference": reference["result"],
            "query_manifest_sha256": workload[
                "query_manifest_sha256"
            ],
        }
        write_json(compare_input_path, compare_input)
        comparison = _run_process(
            (
                "experiments.inter_fact_relations_v0_hardening_scale_v1."
                "src.compare_process"
            ),
            compare_input_path,
            compare_output_path,
        )
        return {
            "candidate_pid": candidate["pid"],
            "reference_pid": reference["pid"],
            "compare_pid": comparison["pid"],
            "candidate_elapsed_seconds": candidate[
                "elapsed_seconds"
            ],
            "candidate_peak_rss_bytes": candidate["peak_rss_bytes"],
            "reference_elapsed_seconds": reference[
                "elapsed_seconds"
            ],
            "reference_peak_rss_bytes": reference["peak_rss_bytes"],
            "comparison": comparison["result"],
            "candidate_metrics": candidate["result"]["metrics"],
            "reference_metrics": reference["result"]["metrics"],
            "candidate_reference_distinct_processes": (
                candidate["pid"] != reference["pid"]
            ),
            "compare_distinct_process": comparison["pid"]
            not in {candidate["pid"], reference["pid"]},
            "candidate_input_keys": sorted(candidate_input),
            "reference_input_keys": sorted(reference_input),
            "candidate_reads_runtime_receipts": False,
            "candidate_reads_reference_output": False,
            "reference_reads_graph": False,
            "reference_reads_candidate_output": False,
        }


def _scientific_material(
    context: dict[str, Any],
    isolated: dict[str, Any],
) -> dict[str, Any]:
    graph = context["validated_graph"]
    counts = graph.validation.counts
    endpoint_signatures = Counter(
        edge.source_node_kind + "->" + edge.target_node_kind
        for edge in graph.graph.relation_edges
    )
    index_material = {
        "fact_native_ids": sorted(
            row.native_fact_id for row in graph.graph.fact_nodes
        ),
        "occurrence_native_ids": sorted(
            row.concrete_occurrence_instance_id
            for row in graph.graph.occurrence_nodes
        ),
        "relation_ids": sorted(
            row.original_relation_id
            for row in graph.graph.relation_edges
        ),
    }
    return {
        "schema_version": "executable-generation-fact-graph-scale-v2",
        "scale": context["scale"],
        "execution_run_id": graph.graph.metadata.execution_run_id,
        "graph_id": graph.graph_id,
        "graph_canonical_sha256": canonical_hash(
            graph.graph.to_dict()
        ),
        "fact_node_count": counts["fact_nodes"],
        "occurrence_node_count": counts["occurrence_nodes"],
        "zero_fact_occurrence_count": counts[
            "zero_fact_occurrences"
        ],
        "one_fact_occurrence_count": counts["one_fact_occurrences"],
        "multi_fact_occurrence_count": counts[
            "multi_fact_occurrences"
        ],
        "incidence_edge_count": counts["incidence_edges"],
        "primitive_relation_edge_count": counts[
            "primitive_relation_edges"
        ],
        "derived_relation_edge_count": counts[
            "derived_relation_edges"
        ],
        "endpoint_signature_counts": dict(
            sorted(endpoint_signatures.items())
        ),
        "relation_projection_exact": context[
            "relation_projection_exact"
        ],
        "query_count": isolated["comparison"]["query_count"],
        "query_comparison_status": isolated["comparison"]["status"],
        "query_mismatch_count": isolated["comparison"][
            "mismatch_count"
        ],
        "false_positive_count": isolated["comparison"][
            "false_positive_count"
        ],
        "false_negative_count": isolated["comparison"][
            "false_negative_count"
        ],
        "global_transitive_closure_materialized": False,
        "serialized_graph_bytes": len(
            canonical_bytes(graph.graph.to_dict())
        ),
        "canonical_index_key_bytes": len(
            canonical_bytes(index_material)
        ),
        "candidate_reference_distinct_processes": isolated[
            "candidate_reference_distinct_processes"
        ],
        "compare_distinct_process": isolated[
            "compare_distinct_process"
        ],
        "candidate_inputs_only": (
            not isolated["candidate_reads_runtime_receipts"]
            and not isolated["candidate_reads_reference_output"]
        ),
        "reference_isolated": (
            not isolated["reference_reads_graph"]
            and not isolated["reference_reads_candidate_output"]
        ),
        "validation_sha256": graph.validation.validation_sha256,
    }


def run_scale_graph(
    scale: str = "large",
) -> tuple[dict[str, Any], dict[str, Any]]:
    with PeakRssSampler() as sampler:
        first = _build_graph_context(scale)
        first_queries_started = time.perf_counter()
        first_isolated = _run_isolated_queries(first)
        first_query_elapsed = (
            time.perf_counter() - first_queries_started
        )
        first_scientific = _scientific_material(
            first, first_isolated
        )
        second = _build_graph_context(scale)
        second_queries_started = time.perf_counter()
        second_isolated = _run_isolated_queries(second)
        second_query_elapsed = (
            time.perf_counter() - second_queries_started
        )
        second_scientific = _scientific_material(
            second, second_isolated
        )

    deterministic_fields = {
        key: value
        for key, value in first_scientific.items()
        if key
        not in {
            "serialized_graph_bytes",
            "canonical_index_key_bytes",
        }
    }
    second_deterministic_fields = {
        key: value
        for key, value in second_scientific.items()
        if key
        not in {
            "serialized_graph_bytes",
            "canonical_index_key_bytes",
        }
    }
    two_run_exact = (
        canonical_bytes(deterministic_fields)
        == canonical_bytes(second_deterministic_fields)
        and first_scientific["serialized_graph_bytes"]
        == second_scientific["serialized_graph_bytes"]
        and first_scientific["canonical_index_key_bytes"]
        == second_scientific["canonical_index_key_bytes"]
    )
    gates = {
        "occurrence_count_10000": (
            scale != "large"
            or first_scientific["occurrence_node_count"] == 10_000
        ),
        "fact_count_30000": (
            scale != "large"
            or first_scientific["fact_node_count"] == 30_000
        ),
        "incidence_count_exact": (
            first_scientific["incidence_edge_count"]
            == first_scientific["fact_node_count"]
        ),
        "all_occurrences_multi_fact": (
            first_scientific["multi_fact_occurrence_count"]
            == first_scientific["occurrence_node_count"]
        ),
        "relation_projection_exact": first_scientific[
            "relation_projection_exact"
        ],
        "queries_exact": (
            first_scientific["query_comparison_status"] == "PASS"
            and second_scientific["query_comparison_status"] == "PASS"
        ),
        "false_positive_zero": (
            first_scientific["false_positive_count"] == 0
            and second_scientific["false_positive_count"] == 0
        ),
        "false_negative_zero": (
            first_scientific["false_negative_count"] == 0
            and second_scientific["false_negative_count"] == 0
        ),
        "two_run_scientific_exact": two_run_exact,
        "no_global_transitive_closure": not first_scientific[
            "global_transitive_closure_materialized"
        ],
        "process_isolation": (
            first_scientific[
                "candidate_reference_distinct_processes"
            ]
            and first_scientific["compare_distinct_process"]
            and first_scientific["candidate_inputs_only"]
            and first_scientific["reference_isolated"]
        ),
    }
    diagnostics = {
        "first": {
            "compile_elapsed_seconds": first[
                "compile_elapsed_seconds"
            ],
            "validation_elapsed_seconds": first[
                "validation_elapsed_seconds"
            ],
            "query_elapsed_seconds": first_query_elapsed,
            "candidate_peak_rss_bytes": first_isolated[
                "candidate_peak_rss_bytes"
            ],
            "reference_peak_rss_bytes": first_isolated[
                "reference_peak_rss_bytes"
            ],
        },
        "second": {
            "compile_elapsed_seconds": second[
                "compile_elapsed_seconds"
            ],
            "validation_elapsed_seconds": second[
                "validation_elapsed_seconds"
            ],
            "query_elapsed_seconds": second_query_elapsed,
            "candidate_peak_rss_bytes": second_isolated[
                "candidate_peak_rss_bytes"
            ],
            "reference_peak_rss_bytes": second_isolated[
                "reference_peak_rss_bytes"
            ],
        },
        "peak_parent_plus_children_rss_bytes": sampler.peak_rss_bytes,
        "performance_claim": "DIAGNOSTIC_ONLY",
    }
    result = {
        **first_scientific,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "second_graph_id": second_scientific["graph_id"],
        "two_run_scientific_exact": two_run_exact,
        "scientific_sha256": canonical_sha256(first_scientific),
        "gates": gates,
        "diagnostics": diagnostics,
    }
    del second
    gc.collect()
    return result, {
        "first_context": first,
        "first_isolated_queries": first_isolated,
        "second_isolated_queries": second_isolated,
    }
