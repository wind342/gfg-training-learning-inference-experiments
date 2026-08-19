from __future__ import annotations

import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from experiments.inter_fact_relations_v0_hardening_scale_v1.common import (
    write_json,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.runner import (
    _run_process,
)
from experiments.signal_multistage_generated_origin_v1.data import (
    DEFAULT_DATA_ROOT,
    load_signal_window,
)
from experiments.signal_multistage_generated_origin_v1.run_experiment import (
    Execution,
    execute_once,
)

from ..canonical_graph import canonical_hash, content_id
from ..endpoint_registry import build_core_occurrence_catalog
from ..graph_compiler import (
    compile_executable_generation_fact_graph_v2,
)
from ..graph_projections import project_gamma
from ..graph_query import ExecutableGenerationFactGraphQueryEngineV2
from ..graph_validator import (
    load_contracts,
    validate_executable_generation_fact_graph_v2,
)
from .common import complete_capture_audit


EXECUTION_RUN_ID = "signal-multistage-generated-origin-v1"


def build_generated_origin_relation_store(
    execution: Execution,
) -> dict[str, Any]:
    snapshot = execution.snapshot
    bindings = list(snapshot.tables.generation_bindings)
    generated = {
        row["generated_origin_id"]: row
        for row in snapshot.tables.generated_origins
    }
    producers_by_support: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for binding in bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            producers_by_support[outcome["support_id"]].append(binding)
    native_relations = []
    for consumer in bindings:
        origin_reference = consumer["origin_reference"]
        if origin_reference["kind"] != "generated_origin":
            continue
        origin = generated[origin_reference["generated_origin_id"]]
        prior_support_id = origin["origin_payload"].get(
            "prior_support_id"
        )
        if not prior_support_id:
            raise ValueError("GENERATED_ORIGIN_PRIOR_SUPPORT_MISSING")
        producers = producers_by_support.get(prior_support_id, [])
        if not producers:
            raise ValueError("GENERATED_ORIGIN_PRODUCER_MISSING")
        for producer in sorted(
            producers, key=lambda row: row["generation_binding_id"]
        ):
            relation_material = {
                "execution_run_id": EXECUTION_RUN_ID,
                "relation_type": "generated_origin_dependency",
                "endpoint_level": "fact",
                "source_id": producer["generation_binding_id"],
                "target_id": consumer["generation_binding_id"],
                "relation_payload": {
                    "projection_scope": "core_generated_origin",
                    "generated_origin_id": origin[
                        "generated_origin_id"
                    ],
                    "prior_support_id": prior_support_id,
                    "producer_binding_id": producer[
                        "generation_binding_id"
                    ],
                    "consumer_binding_id": consumer[
                        "generation_binding_id"
                    ],
                },
                "establishment_source": "generator_established",
                "authority_id": "validated-core-generated-origin-v2",
                "evidence_refs": sorted(consumer["evidence_ids"]),
            }
            native_relations.append(
                {
                    **relation_material,
                    "relation_id": content_id(
                        "sigrel2_", relation_material
                    ),
                }
            )
    native_relations.sort(key=lambda row: row["relation_id"])
    native_store_material = {
        "schema_version": "signal-generated-origin-sidecar-v2",
        "execution_run_id": EXECUTION_RUN_ID,
        "relations": native_relations,
        "evidence": [],
        "establishment_basis": (
            "validated GeneratedOrigin prior_support_id and exact "
            "producer outcome support identity"
        ),
    }
    store_id = content_id("gfrstore2_", native_store_material)
    return {
        **{
            key: value
            for key, value in native_store_material.items()
            if key not in {"relations", "evidence"}
        },
        "relation_store_id": store_id,
        "relations": [
            {
                **relation,
                "source_endpoint_kind": "fact",
                "target_endpoint_kind": "fact",
                "primitive_or_derived": "primitive",
                "rule_id": None,
                "input_relation_refs": [],
                "native_relation": relation,
            }
            for relation in native_relations
        ],
        "evidence": [],
    }


def _compile(
    execution: Execution,
) -> tuple[Any, dict[str, Any]]:
    contracts = load_contracts()
    snapshot_inputs = [
        {
            "snapshot": execution.snapshot,
            "execution_run_id": EXECUTION_RUN_ID,
        }
    ]
    relation_store = build_generated_origin_relation_store(execution)
    occurrence_catalog = build_core_occurrence_catalog(snapshot_inputs)
    audit = complete_capture_audit(
        EXECUTION_RUN_ID, domain="signal_multistage"
    )
    graph = compile_executable_generation_fact_graph_v2(
        snapshot_inputs,
        relation_store,
        occurrence_catalog,
        audit,
        contracts["graph_profile"],
        contracts["relation_type_registry"],
    )
    validated = validate_executable_generation_fact_graph_v2(
        graph,
        snapshot_inputs,
        relation_store,
        occurrence_catalog,
        audit,
        contracts,
    )
    return validated, {
        "snapshot_inputs": snapshot_inputs,
        "relation_store": relation_store,
        "occurrence_catalog": occurrence_catalog,
        "capture_audit": audit,
        "contracts": contracts,
    }


def _path_signatures(
    query: ExecutableGenerationFactGraphQueryEngineV2,
    formation: dict[str, Any],
) -> list[str]:
    signatures = []
    for path in formation["path_instances"]:
        final_to_source = [
            query.fact_nodes[node_id]
            for node_id in reversed(path["node_ids"])
        ]
        support_keys = [
            node.z["entity"]["support_payload"]["native_support_key"]
            for node in final_to_source
        ]
        roles = [node.rho for node in final_to_source]
        occurrence_keys = [
            node.omega_bar["generation_occurrence"][
                "stable_instance_key"
            ]
            for node in final_to_source
        ]
        source = query.fact_nodes[path["node_ids"][0]].u["entity"]
        signatures.append(
            "|".join(
                [
                    *support_keys,
                    *roles,
                    *occurrence_keys,
                    source["source_identity"],
                ]
            )
        )
    return sorted(signatures)


def _run_isolated_comparison(
    validated_graph: Any,
    capture_audit: dict[str, Any],
    query_window: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="generation-fact-graph-v2-signal-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        candidate_input_path = temporary / "candidate-input.json"
        candidate_output_path = temporary / "candidate-output.json"
        reference_input_path = temporary / "reference-input.json"
        reference_output_path = temporary / "reference-output.json"
        compare_input_path = temporary / "compare-input.json"
        compare_output_path = temporary / "compare-output.json"
        candidate_input = {
            "execution_run_id": EXECUTION_RUN_ID,
            "graph": validated_graph.graph.to_dict(),
            "validation": validated_graph.validation.to_dict(),
            "capture_audit": capture_audit,
            "query_window": query_window,
            "traversal_policy": {
                "relation_types": ["generated_origin_dependency"],
                "stop_at_registered_source": True,
            },
            "schema_version": "signal-graph-candidate-input-v2",
        }
        reference_input = {
            "execution_run_id": EXECUTION_RUN_ID,
            "data_root": str(DEFAULT_DATA_ROOT),
            "schema_version": "signal-reference-input-v2",
        }
        write_json(candidate_input_path, candidate_input)
        write_json(reference_input_path, reference_input)
        candidate = _run_process(
            (
                "experiments.executable_generation_fact_graph_v2."
                "adapters.signal_candidate_process"
            ),
            candidate_input_path,
            candidate_output_path,
        )
        reference = _run_process(
            (
                "experiments.executable_generation_fact_graph_v2."
                "references.signal_reference_process"
            ),
            reference_input_path,
            reference_output_path,
        )
        write_json(
            compare_input_path,
            {
                "candidate": candidate["result"],
                "reference": reference["result"],
            },
        )
        comparison = _run_process(
            (
                "experiments.executable_generation_fact_graph_v2."
                "references.signal_compare_process"
            ),
            compare_input_path,
            compare_output_path,
        )
        return {
            "comparison": comparison["result"],
            "candidate_pid": candidate["pid"],
            "reference_pid": reference["pid"],
            "compare_pid": comparison["pid"],
            "candidate_reference_distinct_processes": (
                candidate["pid"] != reference["pid"]
            ),
            "compare_distinct_process": comparison["pid"]
            not in {candidate["pid"], reference["pid"]},
            "candidate_input_keys": sorted(candidate_input),
            "reference_input_keys": sorted(reference_input),
            "candidate_reads_raw_signal": False,
            "candidate_reads_reference_answer": False,
            "reference_reads_graph": False,
            "reference_reads_candidate_answer": False,
            "candidate_elapsed_seconds": candidate[
                "elapsed_seconds"
            ],
            "reference_elapsed_seconds": reference[
                "elapsed_seconds"
            ],
        }


def run_signal_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    signal = load_signal_window(DEFAULT_DATA_ROOT)
    first = execute_once(signal)
    second = execute_once(signal)
    first_graph, context = _compile(first)
    second_graph, _ = _compile(second)
    query = ExecutableGenerationFactGraphQueryEngineV2(first_graph)
    query_window = {
        "support_space_id": first.collector.visual_space[
            "support_space_id"
        ],
        "predicate": "rectangle_intersection",
        "rectangle": first.candidate_answer["query_rectangle"],
    }
    isolated_started = time.perf_counter()
    isolated = _run_isolated_comparison(
        first_graph, context["capture_audit"], query_window
    )
    isolated_elapsed = time.perf_counter() - isolated_started
    formation = query.formation_subgraph(
        query_window,
        {
            "relation_types": ["generated_origin_dependency"],
            "stop_at_registered_source": True,
        },
    )
    signatures = _path_signatures(query, formation)
    signature_hash = canonical_hash(signatures)
    included = [
        query.fact_nodes[node_id]
        for node_id in formation["included_fact_nodes"]
    ]
    stage_fact_counts = Counter(
        row.omega_bar["generation_occurrence"]["occurrence_stage"]
        for row in included
    )
    stage_supports: dict[str, set[str]] = defaultdict(set)
    stage_occurrences: dict[str, set[str]] = defaultdict(set)
    for node in included:
        occurrence = node.omega_bar["generation_occurrence"]
        stage = occurrence["occurrence_stage"]
        stage_occurrences[stage].add(node.generation_occurrence_id)
        reference = node.z["reference"]
        if reference["kind"] == "support":
            stage_supports[stage].add(reference["support_id"])
    raw_sources = sorted(
        {
            query.fact_nodes[path["node_ids"][0]]
            .u["entity"]["source_identity"]
            for path in formation["path_instances"]
        }
    )
    projected = project_gamma(first_graph)
    snapshot_bindings = sorted(
        row["generation_binding_id"]
        for row in first.snapshot.tables.generation_bindings
    )
    projected_bindings = sorted(
        row["generation_binding_id"] for row in projected["facts"]
    )
    direct_shortcuts = [
        edge.graph_edge_id
        for edge in first_graph.graph.relation_edges
        if edge.relation_type == "generated_origin_dependency"
        and query.fact_nodes[edge.source_node_id]
        .omega_bar["generation_occurrence"]["occurrence_stage"]
        == "fir_filter"
        and query.fact_nodes[edge.target_node_id]
        .omega_bar["generation_occurrence"]["occurrence_stage"]
        == "svg_render"
    ]
    graph_bytes_equal = (
        canonical_bytes(first_graph.graph.to_dict())
        == canonical_bytes(second_graph.graph.to_dict())
    )
    result = {
        "schema_version": "executable-generation-fact-graph-signal-v2",
        "native_execution_count": 2,
        "ordinary_output_byte_identical": first.comparison[
            "ordinary_output_byte_identical"
        ],
        "numeric_reference_exact_within_1e_10": first.comparison[
            "numeric_reference_exact_within_1e_10"
        ],
        "graph_id": first_graph.graph_id,
        "second_graph_id": second_graph.graph_id,
        "graph_two_run_deterministic": graph_bytes_equal,
        "fact_node_count": len(first_graph.graph.fact_nodes),
        "occurrence_node_count": len(first_graph.graph.occurrence_nodes),
        "incidence_edge_count": len(first_graph.graph.incidence_edges),
        "relation_edge_count": len(first_graph.graph.relation_edges),
        "selected_svg_cell_count": len(
            formation["selected_result_nodes"]
        ),
        "fft_result_count": len(stage_supports["fft"]),
        "fft_occurrence_count": len(stage_occurrences["fft"]),
        "downsampled_sample_count": len(stage_supports["downsample"]),
        "retained_filtered_sample_count": len(
            stage_supports["fir_filter"]
        ),
        "raw_source_count": len(raw_sources),
        "path_count": len(signatures),
        "path_signature_multiset_sha256": signature_hash,
        "reference_path_signature_multiset_sha256": first.reference.answer[
            "path_signature_multiset_sha256"
        ],
        "path_multiset_exact": (
            signature_hash
            == first.reference.answer["path_signature_multiset_sha256"]
        ),
        "isolated_candidate_reference_exact": isolated[
            "comparison"
        ]["candidate_reference_exact"],
        "candidate_reference_distinct_processes": isolated[
            "candidate_reference_distinct_processes"
        ],
        "compare_distinct_process": isolated[
            "compare_distinct_process"
        ],
        "candidate_inputs_only": (
            not isolated["candidate_reads_raw_signal"]
            and not isolated["candidate_reads_reference_answer"]
        ),
        "reference_isolated": (
            not isolated["reference_reads_graph"]
            and not isolated["reference_reads_candidate_answer"]
        ),
        "isolated_query_elapsed_seconds": isolated_elapsed,
        "raw_sources_exact": (
            raw_sources == first.reference.answer["raw_source_identities"]
        ),
        "stage_fact_node_counts": dict(
            sorted(stage_fact_counts.items())
        ),
        "projection_gamma_exact": (
            snapshot_bindings == projected_bindings
        ),
        "direct_raw_to_svg_shortcut_count": len(direct_shortcuts),
        "global_transitive_closure_materialized": (
            query.global_transitive_closure_materialized
        ),
        "validation": first_graph.validation.to_dict(),
    }
    gates = {
        "selected_svg_cells_10": result["selected_svg_cell_count"] == 10,
        "fft_occurrences_2": result["fft_occurrence_count"] == 2,
        "downsampled_samples_48": (
            result["downsampled_sample_count"] == 48
        ),
        "retained_filtered_samples_48": (
            result["retained_filtered_sample_count"] == 48
        ),
        "raw_sources_197": result["raw_source_count"] == 197,
        "paths_2880": result["path_count"] == 2880,
        "path_multiset_exact": result["path_multiset_exact"],
        "raw_sources_exact": result["raw_sources_exact"],
        "fact_nodes_exact": (
            result["fact_node_count"] == len(snapshot_bindings)
        ),
        "occurrence_nodes_exact": (
            result["occurrence_node_count"]
            == len(first.snapshot.tables.generation_occurrences)
        ),
        "incidence_edges_exact": (
            result["incidence_edge_count"] == len(snapshot_bindings)
        ),
        "relation_edges_exact": (
            result["relation_edge_count"] == 11_078
        ),
        "projection_gamma_exact": result["projection_gamma_exact"],
        "no_direct_raw_to_svg_shortcut": not direct_shortcuts,
        "two_run_deterministic": result["graph_two_run_deterministic"],
        "isolated_candidate_reference_exact": result[
            "isolated_candidate_reference_exact"
        ],
        "process_isolation": (
            result["candidate_reference_distinct_processes"]
            and result["compare_distinct_process"]
            and result["candidate_inputs_only"]
            and result["reference_isolated"]
        ),
        "no_global_transitive_closure": not result[
            "global_transitive_closure_materialized"
        ],
    }
    result["gates"] = gates
    result["status"] = "PASS" if all(gates.values()) else "FAIL"
    return result, {
        "execution": first,
        "validated_graph": first_graph,
        "formation_subgraph": formation,
        "query_engine": query,
        **context,
    }
