from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from experiments.signal_multistage_generated_origin_v1.data import (
    DEFAULT_DATA_ROOT,
    load_signal_window,
)
from experiments.signal_multistage_generated_origin_v1.run_experiment import (
    Execution,
    execute_once,
)

from ..graph_compiler import compile_generation_fact_graph
from ..graph_projections import project_atomic_generation_state
from ..graph_query import GenerationFactGraphQueryEngine
from ..graph_validator import (
    load_contracts,
    validate_generation_fact_graph,
)
from ..graph_model import ValidatedGenerationFactGraph
from .common import complete_capture_audit, empty_relation_store


EXECUTION_RUN_ID = "signal-multistage-generated-origin-v1"


def _compile(
    execution: Execution,
) -> tuple[ValidatedGenerationFactGraph, dict[str, Any]]:
    contracts = load_contracts()
    snapshot_inputs = [
        {
            "snapshot": execution.snapshot,
            "execution_run_id": EXECUTION_RUN_ID,
        }
    ]
    relation_store = empty_relation_store(EXECUTION_RUN_ID)
    audit = complete_capture_audit(
        EXECUTION_RUN_ID, domain="signal_multistage"
    )
    graph = compile_generation_fact_graph(
        snapshot_inputs,
        relation_store,
        audit,
        contracts["graph_profile"],
        contracts["relation_lifting_contract"],
        relation_type_registry=contracts["relation_type_registry"],
    )
    validated = validate_generation_fact_graph(
        graph,
        snapshot_inputs,
        relation_store,
        audit,
        contracts,
    )
    return validated, {
        "snapshot_inputs": snapshot_inputs,
        "relation_store": relation_store,
        "capture_audit": audit,
        "contracts": contracts,
    }


def _path_signatures(
    query: GenerationFactGraphQueryEngine,
    formation: dict[str, Any],
) -> list[str]:
    signatures: list[str] = []
    for path in formation["path_instances"]:
        final_to_source = [
            query.nodes[node_id] for node_id in reversed(path["node_ids"])
        ]
        support_keys = [
            node.outcome_reference["entity"]["support_payload"][
                "native_support_key"
            ]
            for node in final_to_source
        ]
        roles = [node.relation_role for node in final_to_source]
        occurrence_keys = [
            node.concrete_occurrence["stable_instance_key"]
            for node in final_to_source
        ]
        source = query.nodes[path["node_ids"][0]].source_reference["entity"]
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


def run_signal_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    signal = load_signal_window(DEFAULT_DATA_ROOT)
    first = execute_once(signal)
    second = execute_once(signal)
    first_graph, first_context = _compile(first)
    second_graph, _ = _compile(second)

    query = GenerationFactGraphQueryEngine(first_graph)
    query_window = {
        "support_space_id": first.collector.visual_space[
            "support_space_id"
        ],
        "predicate": "rectangle_intersection",
        "rectangle": first.candidate_answer["query_rectangle"],
    }
    formation = query.formation_subgraph(
        query_window,
        {
            "relation_types": ["generated_origin_dependency"],
            "stop_at_registered_source": True,
        },
    )
    signatures = _path_signatures(query, formation)
    signature_hash = hashlib.sha256(
        canonical_bytes(signatures)
    ).hexdigest()
    included = [query.nodes[node_id] for node_id in formation["included_nodes"]]
    stage_nodes = Counter(
        row.concrete_occurrence["occurrence_stage"] for row in included
    )
    stage_supports: dict[str, set[str]] = {}
    stage_occurrences: dict[str, set[str]] = {}
    for node in included:
        stage = node.concrete_occurrence["occurrence_stage"]
        stage_occurrences.setdefault(stage, set()).add(
            node.occurrence_identity
        )
        reference = node.outcome_reference["reference"]
        if reference["kind"] == "support":
            stage_supports.setdefault(stage, set()).add(
                reference["support_id"]
            )
    raw_sources = sorted(
        {
            query.nodes[path["node_ids"][0]]
            .source_reference["entity"]["source_identity"]
            for path in formation["path_instances"]
        }
    )
    atom_projection = project_atomic_generation_state(first_graph)
    snapshot_binding_ids = sorted(
        row["generation_binding_id"]
        for row in first.snapshot.tables.generation_bindings
    )
    projected_binding_ids = sorted(
        row["generation_binding_id"] for row in atom_projection["facts"]
    )
    direct_shortcuts = [
        edge.graph_edge_id
        for edge in first_graph.graph.edges
        if edge.relation_type == "generated_origin_dependency"
        and query.nodes[edge.source_graph_node_id]
        .concrete_occurrence["occurrence_stage"]
        == "fir_filter"
        and query.nodes[edge.target_graph_node_id]
        .concrete_occurrence["occurrence_stage"]
        == "svg_render"
    ]
    result = {
        "schema_version": "executable-generation-fact-graph-signal-v1",
        "status": "PASS",
        "native_execution_count": 2,
        "ordinary_output_byte_identical": first.comparison[
            "ordinary_output_byte_identical"
        ],
        "numeric_reference_exact_within_1e_10": first.comparison[
            "numeric_reference_exact_within_1e_10"
        ],
        "graph_id": first_graph.graph_id,
        "second_graph_id": second_graph.graph_id,
        "graph_two_run_deterministic": (
            first_graph.graph.to_dict() == second_graph.graph.to_dict()
        ),
        "node_count": len(first_graph.graph.nodes),
        "edge_count": len(first_graph.graph.edges),
        "selected_svg_cell_count": len(
            formation["selected_result_nodes"]
        ),
        "fft_result_count": len(stage_supports.get("fft", set())),
        "fft_occurrence_count": len(
            stage_occurrences.get("fft", set())
        ),
        "downsampled_sample_count": len(
            stage_supports.get("downsample", set())
        ),
        "retained_filtered_sample_count": len(
            stage_supports.get("fir_filter", set())
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
        "raw_sources_exact": (
            raw_sources == first.reference.answer["raw_source_identities"]
        ),
        "selected_results_exact": (
            len(formation["selected_result_nodes"]) == 10
        ),
        "stage_fact_node_counts": dict(sorted(stage_nodes.items())),
        "atomic_projection_exact": (
            snapshot_binding_ids == projected_binding_ids
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
        "atomic_projection_exact": result["atomic_projection_exact"],
        "no_direct_raw_to_svg_shortcut": not direct_shortcuts,
        "two_run_deterministic": result["graph_two_run_deterministic"],
    }
    result["gates"] = gates
    result["status"] = "PASS" if all(gates.values()) else "FAIL"
    return result, {
        "execution": first,
        "validated_graph": first_graph,
        "formation_subgraph": formation,
        "query_engine": query,
        **first_context,
    }
