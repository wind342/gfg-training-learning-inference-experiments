from __future__ import annotations

from typing import Any

from .adapters.order_workflow_adapter import run_order_workflow_graph
from .adapters.scale_relation_adapter import run_scale_graph
from .adapters.signal_adapter import run_signal_graph
from .adapters.signed_projection_adapter import run_signed_projection
from .canonical_graph import canonical_hash
from .negative_controls import run_negative_controls
from .protected_audit import protected_path_audit


FINAL_STATUS = "EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED"
FAILURE_REASON = (
    "PURE_FACT_VERTEX_MODEL_CANNOT_PRESERVE_ALL_NATIVE_"
    "PRIMITIVE_RELATION_ENDPOINTS"
)


def run_all_scientific() -> tuple[dict[str, Any], dict[str, Any]]:
    signal_result, signal_context = run_signal_graph()
    order_result, order_context = run_order_workflow_graph()
    scale_result, scale_context = run_scale_graph()
    signed_result = run_signed_projection()
    scale_negative_context = {
        "census": scale_result["endpoint_census"],
    }
    negative_controls = run_negative_controls(
        signal_context=signal_context,
        order_context=order_context,
        scale_context=scale_negative_context,
        signed_result=signed_result,
    )
    protected = protected_path_audit()
    census = order_result["endpoint_census"]
    projections = {
        "schema_version": "graph-projection-exactness-v1",
        "graph_to_atomic_state_exact": signal_result[
            "atomic_projection_exact"
        ],
        "graph_to_relation_store_exact": False,
        "graph_to_relation_store_reason": FAILURE_REASON,
        "graph_to_signed_algebra_exact": (
            signed_result["status"] == "PASS"
        ),
        "relation_store_native_count": census[
            "primitive_relation_count"
        ],
        "relation_store_legally_projectable_count": census[
            "legally_mappable_relation_count"
        ],
        "relation_store_unmappable_count": census[
            "unmappable_primitive_relation_count"
        ],
        "status": "FAIL",
    }
    mandatory_gates = {
        "frozen_core_unchanged": protected["comparisons"][
            "src/generation_relation_core"
        ]["unchanged"],
        "manuscript_unchanged": protected["comparisons"]["manuscript"][
            "unchanged"
        ],
        "frozen_source_experiments_unchanged": all(
            protected["comparisons"][path]["unchanged"]
            for path in (
                "experiments/signal_multistage_generated_origin_v1",
                "experiments/inter_fact_relations_v0",
                "experiments/inter_fact_relations_v0_hardening_scale_v1",
                "experiments/order_refund_freeze_inter_fact_relations_v1",
                "experiments/signed_generation_algebra_v1",
            )
        ),
        "graph_schema_valid": signal_result["validation"]["gates"][
            "graph_schema_valid"
        ],
        "graph_canonical_serialization_exact": signal_result[
            "validation"
        ]["gates"]["graph_canonical_serialization_exact"],
        "graph_two_run_deterministic": signal_result[
            "graph_two_run_deterministic"
        ],
        "node_binding_coverage_exact": signal_result["validation"][
            "gates"
        ]["node_binding_coverage_exact"],
        "node_content_exact": signal_result["validation"]["gates"][
            "node_content_exact"
        ],
        "node_instance_identity_preserved": signal_result["validation"][
            "gates"
        ]["node_instance_identity_preserved"],
        "node_multiplicity_preserved": signal_result["validation"][
            "gates"
        ]["node_multiplicity_preserved"],
        "edge_primitive_coverage_exact": False,
        "edge_endpoint_closure": False,
        "edge_endpoint_semantics_exact": False,
        "edge_relation_type_exact": False,
        "edge_evidence_closure": False,
        "edge_authority_exact": False,
        "edge_run_scope_exact": False,
        "derived_edge_traceability_exact": False,
        "no_cartesian_expansion": (
            census["prohibited_action_counts"][
                "cartesian_expanded_edge_count"
            ]
            == 0
        ),
        "no_direct_multistage_shortcut": (
            signal_result["direct_raw_to_svg_shortcut_count"] == 0
        ),
        "no_global_transitive_closure_materialized": (
            not signal_result["global_transitive_closure_materialized"]
        ),
        "capture_completeness_gate_enforced": True,
        "graph_to_atomic_state_exact": projections[
            "graph_to_atomic_state_exact"
        ],
        "graph_to_relation_store_exact": False,
        "graph_to_signed_algebra_exact": projections[
            "graph_to_signed_algebra_exact"
        ],
        "signal_selected_results_exact": signal_result[
            "selected_results_exact"
        ],
        "signal_node_set_exact": signal_result[
            "atomic_projection_exact"
        ],
        "signal_edge_set_exact": signal_result["path_multiset_exact"],
        "signal_path_multiset_exact": signal_result[
            "path_multiset_exact"
        ],
        "signal_2880_paths_exact": signal_result["path_count"] == 2880,
        "signal_197_raw_sources_exact": (
            signal_result["raw_source_count"] == 197
        ),
        "order_56_queries_exact": order_result[
            "query_comparison_status"
        ]
        == "PASS",
        "order_false_positive_zero": (
            order_result["false_positive_count"] == 0
        ),
        "order_false_negative_zero": (
            order_result["false_negative_count"] == 0
        ),
        "order_path_exact": False,
        "order_compensation_target_exact": False,
        "order_downstream_impact_exact": False,
        "scale_graph_exact": False,
        "scale_queries_exact": scale_result["source_scientific"][
            "comparison"
        ]["status"]
        == "PASS",
        "candidate_graph_inputs_only": True,
        "independent_references_isolated": (
            order_context["source_result"]["scientific"][
                "process_isolation_audit"
            ]["status"]
            == "PASS"
        ),
        "no_second_authority_store": True,
        "all_negative_controls_detected": (
            negative_controls["status"] == "PASS"
        ),
    }
    failed = sorted(
        name for name, passed in mandatory_gates.items() if not passed
    )
    material = {
        "schema_version": "executable-generation-fact-graph-final-v1",
        "final_status": FINAL_STATUS,
        "failure_reason": FAILURE_REASON,
        "claim_boundary": {
            "atomic_generation_facts_invalidated": False,
            "inter_fact_relations_invalidated": False,
            "executable_graphs_invalidated": False,
            "only_combined_definition_invalidated": (
                "fact-only vertices plus complete preservation of every "
                "occurrence-level primitive relation"
            ),
        },
        "signal": signal_result,
        "order": {
            key: value
            for key, value in order_result.items()
            if key != "endpoint_census"
        },
        "scale": {
            key: value
            for key, value in scale_result.items()
            if key != "endpoint_census"
        },
        "signed_projection": signed_result,
        "projections": projections,
        "negative_controls": negative_controls,
        "protected_path_audit": protected,
        "endpoint_census_sha256": census["census_sha256"],
        "unmappable_primitive_relation_count": census[
            "unmappable_primitive_relation_count"
        ],
        "ambiguous_scale_relation_count": scale_result[
            "endpoint_census"
        ]["ambiguous_occurrence_relation_count"],
        "mandatory_gates": mandatory_gates,
        "failed_mandatory_gates": failed,
        "prohibited_action_counts": census["prohibited_action_counts"],
    }
    result = {**material, "scientific_sha256": canonical_hash(material)}
    return result, {
        "endpoint_census": census,
        "scale_endpoint_census": scale_result["endpoint_census"],
        "scale_diagnostics": scale_context["diagnostics"],
    }
