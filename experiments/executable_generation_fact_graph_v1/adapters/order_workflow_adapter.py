from __future__ import annotations

from typing import Any

from experiments.order_refund_freeze_inter_fact_relations_v1.src.scientific_runner import (
    run_scientific,
)

from ..endpoint_census import census_primitive_relation_endpoints
from .core_snapshot_adapter import (
    build_core_snapshot_from_atomic_facts,
    normalize_relation_store,
)


def run_order_workflow_graph() -> tuple[dict[str, Any], dict[str, Any]]:
    source_result = run_scientific()
    scientific = source_result["scientific"]
    atomic = scientific["atomic_generation_facts"]
    sidecars = scientific["primitive_relation_sidecars"]
    audits = scientific["capture_completeness_audits"]
    census = census_primitive_relation_endpoints(atomic, sidecars)
    snapshots = []
    normalization_attempts = []
    for fact_bundle, sidecar, audit in zip(
        atomic, sidecars, audits, strict=True
    ):
        run_id = fact_bundle["execution_run_id"]
        snapshot_input, mapping = build_core_snapshot_from_atomic_facts(
            atomic_fact_bundle=fact_bundle,
            execution_run_id=run_id,
            domain_scope_id="order-refund-freeze-v1",
            generator_name="order-refund-freeze-native-workflow",
        )
        snapshots.append(
            {
                "scenario": fact_bundle["scenario"],
                "snapshot_input": snapshot_input,
                "mapping": mapping,
                "capture_audit": audit,
            }
        )
        _, partial_audit = normalize_relation_store(
            native_sidecar=sidecar,
            mapping=mapping,
            require_complete=False,
        )
        fail_closed = False
        reason = None
        try:
            normalize_relation_store(
                native_sidecar=sidecar,
                mapping=mapping,
                require_complete=True,
            )
        except ValueError as exc:
            fail_closed = True
            reason = str(exc)
        normalization_attempts.append(
            {
                "scenario": fact_bundle["scenario"],
                "execution_run_id": run_id,
                "native_relation_count": partial_audit[
                    "native_relation_count"
                ],
                "normalized_relation_count": partial_audit[
                    "normalized_relation_count"
                ],
                "unmapped_relation_count": partial_audit[
                    "unmapped_relation_count"
                ],
                "complete_graph_compilation_attempted": False,
                "fail_closed": fail_closed,
                "reason": reason,
            }
        )
    comparison = scientific["query_comparison"]
    gates = {
        "forty_real_workflow_executions": (
            scientific["run_manifest"]["real_workflow_execution_count"] == 40
        ),
        "query_count_56": comparison["query_count"] == 56,
        "candidate_reference_exact": comparison["status"] == "PASS",
        "false_positive_zero": comparison["false_positive_count"] == 0,
        "false_negative_zero": comparison["false_negative_count"] == 0,
        "endpoint_census_complete": (
            census["primitive_relation_count"]
            == sum(len(row["relations"]) for row in sidecars)
        ),
        "no_relation_discarded": (
            census["prohibited_action_counts"][
                "discarded_relation_count"
            ]
            == 0
        ),
        "no_fact_fabricated": (
            census["prohibited_action_counts"][
                "fabricated_fact_node_count"
            ]
            == 0
        ),
        "no_relation_reattached": (
            census["prohibited_action_counts"][
                "reattached_relation_count"
            ]
            == 0
        ),
        "no_cartesian_expansion": (
            census["prohibited_action_counts"][
                "cartesian_expanded_edge_count"
            ]
            == 0
        ),
        "pure_fact_model_falsified": (
            census["unmappable_primitive_relation_count"] > 0
            and not census["pure_fact_vertex_model_supported"]
        ),
        "all_compilers_fail_closed_before_relation_loss": all(
            row["fail_closed"] for row in normalization_attempts
        ),
    }
    result = {
        "schema_version": "executable-generation-fact-graph-order-v1",
        "status": (
            "EXPECTED_FALSIFICATION_OBSERVED"
            if all(gates.values())
            else "AUDIT_FAILURE"
        ),
        "source_scientific_status": scientific["status"],
        "source_scientific_sha256": source_result["scientific_sha256"],
        "workflow_execution_count": scientific["run_manifest"][
            "real_workflow_execution_count"
        ],
        "query_count": comparison["query_count"],
        "false_positive_count": comparison["false_positive_count"],
        "false_negative_count": comparison["false_negative_count"],
        "query_comparison_status": comparison["status"],
        "endpoint_census": census,
        "normalization_attempts": normalization_attempts,
        "graph_compilation_status": "NOT_SUPPORTED_FAIL_CLOSED",
        "graph_compilation_reason": (
            "PURE_FACT_VERTEX_MODEL_CANNOT_PRESERVE_ALL_NATIVE_"
            "PRIMITIVE_RELATION_ENDPOINTS"
        ),
        "gates": gates,
    }
    return result, {
        "source_result": source_result,
        "atomic_fact_bundles": atomic,
        "sidecars": sidecars,
        "capture_audits": audits,
        "snapshot_contexts": snapshots,
    }
