from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .adapters.order_adapter import run_order_graph
from .adapters.scale_adapter import run_scale_graph
from .adapters.signal_adapter import run_signal_graph
from .adapters.signed_projection_adapter import run_signed_projection
from .canonical_graph import canonical_hash
from .negative_controls import run_negative_controls
from .protected_audit import protected_path_audit


EXPERIMENT_ROOT = Path(__file__).resolve().parent
V1_RESULT = (
    EXPERIMENT_ROOT.parent
    / "executable_generation_fact_graph_v1"
    / "artifacts"
    / "v1_final_result.json"
)
FINAL_STATUS_SUPPORTED = (
    "EXECUTABLE_GENERATION_FACT_GRAPH_V2_SUPPORTED"
)
FINAL_STATUS_NOT_SUPPORTED = (
    "EXECUTABLE_GENERATION_FACT_GRAPH_V2_NOT_SUPPORTED"
)


def _without(
    value: dict[str, Any], *keys: str
) -> dict[str, Any]:
    return {
        key: row for key, row in value.items() if key not in keys
    }


def run_all_scientific() -> tuple[dict[str, Any], dict[str, Any]]:
    signal_result, signal_context = run_signal_graph()
    order_result, order_context = run_order_graph()
    scale_result, scale_context = run_scale_graph("large")
    signed_result, signed_context = run_signed_projection()
    negative_controls = run_negative_controls(
        signal_context=signal_context,
        order_context=order_context,
        scale_context=scale_context,
        signed_result=signed_result,
    )
    protected = protected_path_audit()
    v1 = json.loads(V1_RESULT.read_text(encoding="utf-8"))

    order_validation_rows = [
        row["validation"] for row in order_result["graphs"]
    ]
    projection_result = {
        "schema_version": "graph-projection-exactness-v2",
        "projection_gamma_exact": signal_result[
            "projection_gamma_exact"
        ],
        "projection_occurrence_view_exact": all(
            row["occurrence_view_relation_count"]
            == row["fact_only_omitted_count"]
            for row in order_result["graphs"]
        ),
        "projection_relation_sidecar_exact": (
            all(
                row["relation_projection_exact"]
                for row in order_result["graphs"]
            )
            and scale_result["relation_projection_exact"]
        ),
        "projection_fact_only_boundary_honest": all(
            row["fact_only_omitted_count"] > 0
            and row["fact_only_retained_count"]
            + row["fact_only_omitted_count"]
            == row["primitive_relation_count"]
            for row in order_result["graphs"]
        ),
        "projection_signed_exact": signed_result["status"] == "PASS",
    }
    projection_result["status"] = (
        "PASS"
        if all(
            value
            for key, value in projection_result.items()
            if key
            not in {
                "schema_version",
                "status",
            }
        )
        else "FAIL"
    )

    relation_type_counts = Counter(
        order_result["relation_type_counts"]
    )
    endpoint_census = {
        "schema_version": "graph-v2-endpoint-type-census",
        "order_native_primitive_relation_count": order_result[
            "native_primitive_relation_count"
        ],
        "order_relation_type_counts": order_result[
            "relation_type_counts"
        ],
        "order_endpoint_signature_counts": order_result[
            "endpoint_signature_counts"
        ],
        "order_fact_node_count": sum(
            row["fact_node_count"] for row in order_result["graphs"]
        ),
        "order_occurrence_node_count": sum(
            row["occurrence_node_count"]
            for row in order_result["graphs"]
        ),
        "order_zero_fact_occurrence_count": sum(
            row["zero_fact_occurrence_count"]
            for row in order_result["graphs"]
        ),
        "order_multi_fact_occurrence_count": sum(
            row["multi_fact_occurrence_count"]
            for row in order_result["graphs"]
        ),
        "scale_fact_node_count": scale_result["fact_node_count"],
        "scale_occurrence_node_count": scale_result[
            "occurrence_node_count"
        ],
        "scale_multi_fact_occurrence_count": scale_result[
            "multi_fact_occurrence_count"
        ],
        "scale_endpoint_signature_counts": scale_result[
            "endpoint_signature_counts"
        ],
        "v1_unmappable_primitive_relation_count": v1[
            "unmappable_primitive_relation_count"
        ],
        "v2_unmapped_primitive_relation_count": (
            order_result["native_primitive_relation_count"]
            - order_result["compiled_primitive_relation_count"]
        ),
    }
    endpoint_census["census_sha256"] = canonical_hash(
        endpoint_census
    )

    mandatory_gates = {
        "v1_falsification_preserved": (
            v1["final_status"]
            == "EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED"
            and v1["unmappable_primitive_relation_count"] == 55
            and protected["comparisons"][
                "experiments/executable_generation_fact_graph_v1"
            ]["unchanged"]
        ),
        "frozen_core_unchanged": protected["comparisons"][
            "src/generation_relation_core"
        ]["unchanged"],
        "manuscript_unchanged": protected["comparisons"][
            "manuscript"
        ]["unchanged"],
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
        ]
        and all(
            row["gates"]["graph_schema_valid"]
            for row in order_validation_rows
        ),
        "canonical_graph_exact": signal_result["validation"]["gates"][
            "canonical_graph_exact"
        ],
        "two_run_deterministic": (
            signal_result["graph_two_run_deterministic"]
            and scale_result["two_run_scientific_exact"]
        ),
        "every_binding_exactly_one_fact_node": all(
            row["gates"]["every_binding_exactly_one_fact_node"]
            for row in order_validation_rows
        ),
        "fact_content_exact": all(
            row["gates"]["fact_content_exact"]
            for row in order_validation_rows
        ),
        "fact_identity_preserved": all(
            row["gates"]["fact_identity_preserved"]
            for row in order_validation_rows
        ),
        "fact_multiplicity_preserved": all(
            row["gates"]["fact_multiplicity_preserved"]
            for row in order_validation_rows
        ),
        "every_referenced_occurrence_exactly_one_node": all(
            row["gates"][
                "every_referenced_occurrence_exactly_one_node"
            ]
            for row in order_validation_rows
        ),
        "occurrence_content_exact": all(
            row["gates"]["occurrence_content_exact"]
            for row in order_validation_rows
        ),
        "occurrence_identity_preserved": all(
            row["gates"]["occurrence_identity_preserved"]
            for row in order_validation_rows
        ),
        "zero_fact_occurrences_preserved": endpoint_census[
            "order_zero_fact_occurrence_count"
        ]
        > 0,
        "multi_fact_occurrences_preserved": scale_result[
            "multi_fact_occurrence_count"
        ]
        == 10_000,
        "every_fact_exactly_one_incidence": all(
            row["gates"]["every_fact_exactly_one_incidence"]
            for row in order_validation_rows
        ),
        "incidence_exact": all(
            row["gates"]["incidence_exact"]
            for row in order_validation_rows
        ),
        "no_fake_incidence": all(
            row["gates"]["no_fake_incidence"]
            for row in order_validation_rows
        ),
        "every_primitive_relation_exactly_once": order_result[
            "native_primitive_relation_count"
        ]
        == order_result["compiled_primitive_relation_count"],
        "primitive_endpoint_kind_exact": order_result["gates"][
            "endpoint_kind_mismatch_zero"
        ],
        "primitive_endpoint_identity_exact": order_result["gates"][
            "endpoint_identity_mismatch_zero"
        ],
        "primitive_relation_type_exact": all(
            row["gates"]["primitive_relation_type_exact"]
            for row in order_validation_rows
        ),
        "primitive_payload_exact": all(
            row["gates"]["primitive_payload_exact"]
            for row in order_validation_rows
        ),
        "primitive_evidence_exact": all(
            row["gates"]["primitive_evidence_exact"]
            for row in order_validation_rows
        ),
        "primitive_authority_exact": all(
            row["gates"]["primitive_authority_exact"]
            for row in order_validation_rows
        ),
        "no_relation_drop": order_result["gates"]["no_relation_drop"],
        "no_relation_fabrication": order_result["gates"][
            "no_relation_fabrication"
        ],
        "no_forced_lifting": order_result["gates"][
            "no_forced_lifting"
        ],
        "no_cartesian_expansion": all(
            row["gates"]["no_cartesian_expansion"]
            for row in order_validation_rows
        ),
        "derived_edges_traceable": all(
            row["gates"]["derived_edges_traceable"]
            for row in order_validation_rows
        ),
        "capture_completeness_gate_enforced": all(
            row["gates"]["capture_completeness_gate_enforced"]
            for row in order_validation_rows
        ),
        "no_global_transitive_closure": (
            not signal_result[
                "global_transitive_closure_materialized"
            ]
            and not scale_result[
                "global_transitive_closure_materialized"
            ]
        ),
        **{
            key: value
            for key, value in projection_result.items()
            if key.startswith("projection_")
        },
        "signal_fact_nodes_exact": signal_result["gates"][
            "fact_nodes_exact"
        ],
        "signal_occurrence_nodes_exact": signal_result["gates"][
            "occurrence_nodes_exact"
        ],
        "signal_edges_exact": (
            signal_result["gates"]["incidence_edges_exact"]
            and signal_result["gates"]["relation_edges_exact"]
        ),
        "signal_2880_paths_exact": signal_result["path_count"] == 2880,
        "signal_197_sources_exact": signal_result[
            "raw_source_count"
        ]
        == 197,
        "order_all_occurrence_endpoints_represented": order_result[
            "gates"
        ]["all_occurrence_endpoints_represented"],
        "order_all_primitive_relations_represented": order_result[
            "gates"
        ]["all_primitive_relations_represented"],
        "order_direct_graph_queries_56_exact": order_result[
            "gates"
        ]["order_direct_graph_queries_56_exact"],
        "order_projection_compatibility_56_exact": order_result[
            "gates"
        ]["order_projection_compatibility_56_exact"],
        "order_compensation_queries_4_exact": order_result[
            "gates"
        ]["order_compensation_queries_4_exact"],
        "order_direct_graph_fp_zero": order_result["gates"][
            "order_direct_graph_fp_zero"
        ],
        "order_direct_graph_fn_zero": order_result["gates"][
            "order_direct_graph_fn_zero"
        ],
        "order_endpoint_kind_mismatch_zero": order_result["gates"][
            "endpoint_kind_mismatch_zero"
        ],
        "order_endpoint_identity_mismatch_zero": order_result[
            "gates"
        ]["endpoint_identity_mismatch_zero"],
        "scale_graph_exact": scale_result["status"] == "PASS",
        "scale_queries_exact": scale_result["gates"][
            "queries_exact"
        ],
        "candidate_inputs_only": (
            signal_result["candidate_inputs_only"]
            and order_result["gates"][
                "direct_graph_candidate_inputs_only"
            ]
            and scale_result["candidate_inputs_only"]
        ),
        "references_isolated": (
            signal_result["reference_isolated"]
            and order_result["process_isolation"][
                "source_reference_was_distinct_process"
            ]
            and scale_result["reference_isolated"]
        ),
        "no_second_authority_store": all(
            row["gates"]["no_second_authority_store"]
            for row in order_validation_rows
        ),
        "all_negative_controls_detected": negative_controls[
            "status"
        ]
        == "PASS",
        "v2_focused_tests_passed": False,
        "core_tests_passed": False,
        "full_repository_tests_passed": False,
    }
    failed = sorted(
        key for key, value in mandatory_gates.items() if not value
    )

    signal_scientific = _without(
        signal_result, "isolated_query_elapsed_seconds"
    )
    order_scientific = json.loads(json.dumps(order_result))
    for key in ("candidate_pid", "compare_pid"):
        order_scientific["process_isolation"].pop(key, None)
    for section in (
        "direct_graph",
        "projection_compatibility",
    ):
        for key in ("candidate_pid", "compare_pid"):
            order_scientific["process_isolation"][section].pop(
                key,
                None,
            )
    scale_scientific = _without(scale_result, "diagnostics")
    protected_scientific = protected
    scientific_material = {
        "schema_version": "executable-generation-fact-graph-final-v2",
        "graph_definition": "G_e=(V_F,V_O,E_I,E_R;Sigma)",
        "v1_final_status": v1["final_status"],
        "v1_failure_reason": v1["failure_reason"],
        "v1_unmappable_relation_count": v1[
            "unmappable_primitive_relation_count"
        ],
        "signal": signal_scientific,
        "order": order_scientific,
        "scale": scale_scientific,
        "signed_projection": signed_result,
        "projections": projection_result,
        "negative_controls": negative_controls,
        "protected_path_audit": protected_scientific,
        "endpoint_census": endpoint_census,
        "mandatory_gates": mandatory_gates,
        "failed_mandatory_gates": failed,
    }
    scientific_sha256 = canonical_hash(scientific_material)
    result = {
        **scientific_material,
        "final_status": (
            FINAL_STATUS_SUPPORTED
            if not failed
            else FINAL_STATUS_NOT_SUPPORTED
        ),
        "scientific_sha256": scientific_sha256,
        "diagnostics": {
            "signal_isolated_query_elapsed_seconds": signal_result[
                "isolated_query_elapsed_seconds"
            ],
            "scale": scale_result["diagnostics"],
            "performance_claim": "DIAGNOSTIC_ONLY",
        },
    }
    return result, {
        "signal_context": signal_context,
        "order_context": order_context,
        "scale_context": scale_context,
        "signed_context": signed_context,
        "endpoint_census": endpoint_census,
        "projection_result": projection_result,
        "protected_path_audit": protected,
    }
