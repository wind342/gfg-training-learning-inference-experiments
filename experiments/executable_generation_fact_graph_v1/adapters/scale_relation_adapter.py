from __future__ import annotations

from typing import Any

from generation_relation_core.canonical import canonical_bytes

from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.mixed_dag import (
    build_mixed_dag,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.capture_auditor import (
    audit_capture,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.semantic_evidence_validator import (
    validate_primitive_store,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.runner import (
    run_scale,
)

from ..endpoint_census import census_primitive_relation_endpoints


def run_scale_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    first = run_scale("large")
    second = run_scale("large")
    workload = build_mixed_dag("large")
    builder = workload["builder"]
    receipts = builder.runtime_receipts()
    primitive_store = validate_primitive_store(
        builder.primitive_store(), receipts
    )
    capture_audit = audit_capture(
        builder.capture_contract(), receipts, primitive_store
    )
    fact_bundle = {
        "execution_run_id": builder.run_id,
        "facts": receipts["facts"],
    }
    sidecar = {
        "execution_run_id": builder.run_id,
        "relations": primitive_store["primitive_relations"],
    }
    census = census_primitive_relation_endpoints(
        [fact_bundle], [sidecar]
    )
    comparison = first["scientific"]["comparison"]
    gates = {
        "occurrence_count_10000": (
            first["scientific"]["occurrence_count"] == 10_000
        ),
        "fact_count_30000": (
            first["scientific"]["fact_count"] == 30_000
        ),
        "candidate_reference_exact": comparison["status"] == "PASS",
        "false_positive_zero": comparison["false_positive_count"] == 0,
        "false_negative_zero": comparison["false_negative_count"] == 0,
        "two_scientific_runs_deterministic": (
            canonical_bytes(first["scientific"])
            == canonical_bytes(second["scientific"])
        ),
        "capture_profiles_exact": (
            capture_audit["overall_status"] == "CAPTURE_PARTIAL"
            and {
                row["scope_id"]: (
                    row["status"],
                    tuple(row["reason_codes"]),
                    row["concurrency_inference_allowed"],
                )
                for row in capture_audit["scopes"]
            }
            == {
                "capture-complete": ("CAPTURE_COMPLETE", (), True),
                "capture-incomplete": (
                    "CAPTURE_PARTIAL",
                    ("CAPTURE_UNKNOWN_EDGE_PRESENT",),
                    False,
                ),
            }
        ),
        "endpoint_census_complete": (
            census["primitive_relation_count"]
            == len(primitive_store["primitive_relations"])
        ),
        "no_cartesian_expansion_performed": (
            census["prohibited_action_counts"][
                "cartesian_expanded_edge_count"
            ]
            == 0
        ),
        "ambiguous_occurrence_lifting_detected": (
            census["ambiguous_occurrence_relation_count"] > 0
        ),
    }
    result = {
        "schema_version": "executable-generation-fact-graph-scale-v1",
        "status": (
            "EXPECTED_FALSIFICATION_OBSERVED"
            if all(gates.values())
            else "AUDIT_FAILURE"
        ),
        "source_scientific": first["scientific"],
        "endpoint_census": census,
        "graph_compilation_status": "NOT_SUPPORTED_FAIL_CLOSED",
        "graph_compilation_reason": (
            "MULTI_FACT_OCCURRENCE_HAS_NO_UNIQUE_FACT_ENDPOINT"
        ),
        "complete_graph_compilation_attempted": False,
        "gates": gates,
    }
    return result, {
        "receipts": receipts,
        "primitive_store": primitive_store,
        "capture_audit": capture_audit,
        "workload": workload,
        "diagnostics": {
            "first": first["diagnostics"],
            "second": second["diagnostics"],
            "excluded_from_scientific_hash": True,
        },
    }
