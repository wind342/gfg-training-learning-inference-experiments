from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .artifact_io import canonical_json_bytes
from .checkpoint_native_validation import validate_checkpoint_native_dependency
from .gradient_dependency_comparison import compare_snapshots_with_native_oracle
from .gradient_intervention_oracle import run_gradient_intervention_oracle
from .gradient_oracle_isolation import run_gradient_oracle_process_isolation
from .gradient_oracle_negative_controls import run_gradient_oracle_negative_controls
from .lineage_v2 import run_v2_query_comparison
from .science import run_complete_science


EXPERIMENT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"


def _v1_preservation(v1: dict[str, Any]) -> dict[str, Any]:
    artifact_rows = []
    for name, payload in sorted(v1["artifacts"].items()):
        expected = canonical_json_bytes(payload)
        path = ARTIFACT_ROOT / f"{name}.json"
        actual = path.read_bytes() if path.exists() else b""
        artifact_rows.append({
            "actual_sha256": hashlib.sha256(actual).hexdigest(),
            "byte_exact": actual == expected,
            "expected_sha256": hashlib.sha256(expected).hexdigest(),
            "path": f"artifacts/{name}.json",
        })
    summary = v1["scientific_summary"]
    projection = summary["projection_aggregate"]
    query = summary["query_comparison"]
    preservation_gates = {
        "candidate_native_exact_for_five_workloads": (
            projection["exact_workload_count"] == projection["workload_count"] == 5
        ),
        "checkpoint_divergence_preserved": (
            summary["checkpoint_status"]
            == "CHECKPOINT_RECOMPUTATION_DIVERGENCE_LOCALIZATION_SUPPORTED"
        ),
        "core_zero_change": summary["protected_scope"]["core_zero_change"],
        "native_aggregate_edge_slots_33": projection["native_edge_count"] == 33,
        "native_aggregate_nodes_33": projection["native_node_count"] == 33,
        "original_32_negative_controls_pass": (
            summary["negative_controls_all_detected"]
            and summary["negative_controls_count"] == 32
        ),
        "output_orthogonality": (
            summary["output_orthogonality_status"]
            == "TRAINING_OUTPUT_ORTHOGONALITY_SUPPORTED"
        ),
        "query_comparison_preserved": query["all_exact"],
        "strict_projection_counterexamples_preserved": (
            summary["strict_projection_status"]
            == "PYTORCH_AUTOGRAD_STRICT_PROJECTION_SUPPORTED"
        ),
        "zero_gradient_and_unused_distinguished": (
            summary["zero_gradient_status"]
            == "ZERO_GRADIENT_PARTICIPATION_DISTINCTION_SUPPORTED"
        ),
    }
    preserved = all(row["byte_exact"] for row in artifact_rows) and all(
        preservation_gates.values()
    )
    return {
        "artifact_comparison": artifact_rows,
        "frozen_pr17_head": "19eac2a1c5435b378a19c6b37d17a2d275cf794c",
        "preservation_gates": preservation_gates,
        "status": (
            "V1_SCIENTIFIC_RESULTS_PRESERVED"
            if preserved
            else "V1_SCIENTIFIC_RESULT_REGRESSION_BLOCK"
        ),
        "v1_artifact_mismatch_count": sum(
            not row["byte_exact"] for row in artifact_rows
        ),
        "v1_status": "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED",
    }


def _observation_artifacts(oracle: dict[str, Any]) -> dict[str, Any]:
    baselines = oracle["baseline_observations"]
    node_trace = {
        key: {
            "execution_order": row["observation"]["backward"]["execution_order"],
            "executions": row["observation"]["backward"]["executions"],
            "status": row["observation"]["status"],
        }
        for key, row in sorted(baselines.items())
    }
    slot_trace = {
        key: {
            "gradient_slots": row["observation"]["backward"]["gradient_slots"],
            "leaf_gradient_hooks": row["observation"]["backward"][
                "leaf_gradient_hooks"
            ],
        }
        for key, row in sorted(baselines.items())
    }
    pack_trace = {
        key: row["observation"]["saved_tensors"]["pack_trace"]
        for key, row in sorted(baselines.items())
    }
    unpack_trace = {
        key: row["observation"]["saved_tensors"]["unpack_trace"]
        for key, row in sorted(baselines.items())
    }
    assignments = {}
    ordering_rows = {}
    all_ordered = True
    all_assigned = True
    for key, row in sorted(baselines.items()):
        events = row["observation"]["hook_ordering"]
        by_node: dict[str, dict[str, int]] = {}
        for event in events:
            node_id = event.get("native_node_id")
            if node_id is None:
                continue
            if event["event"] == "node_prehook":
                by_node.setdefault(node_id, {})["pre"] = event["ordinal"]
            elif event["event"] == "node_posthook":
                by_node.setdefault(node_id, {})["post"] = event["ordinal"]
        rows = []
        for unpack in unpack_trace[key]:
            node_id = unpack["native_node_id"]
            bounds = by_node.get(node_id, {})
            ordered = (
                unpack["assigned"]
                and bounds.get("pre", -1) < unpack["unpack_ordinal"]
                < bounds.get("post", -1)
            )
            rows.append({
                "native_node_id": node_id,
                "ordered_pre_unpack_post": ordered,
                "token_key": unpack["token_key"],
                "unpack_ordinal": unpack["unpack_ordinal"],
            })
            all_ordered = all_ordered and ordered
            all_assigned = all_assigned and unpack["assigned"]
        assignments[key] = rows
        ordering_rows[key] = {
            "event_trace": events,
            "ordered_assignment_count": sum(
                item["ordered_pre_unpack_post"] for item in rows
            ),
            "unpack_count": len(rows),
        }
    baseline_exact = all(
        row["baseline_gradients_exact"] and row["baseline_ordinary_bytes_exact"]
        for row in baselines.values()
    )
    status = (
        "NATIVE_SAVED_TENSOR_CONSUMPTION_OBSERVATION_SUPPORTED"
        if all_assigned and all_ordered and baseline_exact
        else "NATIVE_SAVED_TENSOR_CONSUMPTION_OBSERVATION_NOT_ESTABLISHED"
    )
    return {
        "backward_hook_ordering_probe": {
            "all_pre_unpack_post_ordered": all_ordered,
            "workloads": ordering_rows,
        },
        "native_backward_gradient_slot_trace": {
            "workloads": slot_trace,
        },
        "native_backward_node_execution_trace": {
            "status": (
                "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED"
                if all(
                    row["observation"]["status"]
                    == "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED"
                    for row in baselines.values()
                )
                else "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_NOT_ESTABLISHED"
            ),
            "workloads": node_trace,
        },
        "saved_tensor_node_assignment": {
            "all_assigned": all_assigned,
            "status": status,
            "workloads": assignments,
        },
        "saved_tensor_pack_trace": {
            "checkpoint_recomputation_replay_equivalence": oracle[
                "checkpoint_recomputation_replay_equivalence"
            ],
            "workloads": pack_trace,
        },
        "saved_tensor_unpack_trace": {
            "status": status,
            "workloads": unpack_trace,
        },
    }


def run_complete_hardening_science() -> dict[str, Any]:
    v1 = run_complete_science()
    preservation = _v1_preservation(v1)
    oracle = run_gradient_intervention_oracle()
    snapshots = v1["artifacts"]["validated_core_snapshots"]
    comparison = compare_snapshots_with_native_oracle(snapshots, oracle)
    v2 = run_v2_query_comparison(oracle)
    checkpoint = validate_checkpoint_native_dependency(oracle, v2)
    isolation = run_gradient_oracle_process_isolation(snapshots)
    negative = run_gradient_oracle_negative_controls(
        comparison,
        oracle,
        v2,
        isolation,
    )
    observation = _observation_artifacts(oracle)
    saved_relations = [
        row
        for row in oracle["native_gradient_dependency_oracle"]["relations"]
        if "saved_tensor" in row["dependency_kind"]
    ]
    source_relations = [
        row
        for row in oracle["native_gradient_dependency_oracle"]["relations"]
        if "registered_source_replay" in row["dependency_kind"]
    ]
    artifacts = {
        **observation,
        **v2,
        **checkpoint,
        **isolation,
        "gradient_dependency_native_oracle_exact_comparison": comparison,
        "gradient_oracle_negative_control_accounting": {
            key: value for key, value in negative.items() if key != "controls"
        },
        "gradient_oracle_negative_controls": {"controls": negative["controls"]},
        "native_gradient_dependency_oracle": oracle[
            "native_gradient_dependency_oracle"
        ],
        "native_saved_tensor_to_gradient_relations": {
            "relation_count": len(saved_relations),
            "relations": saved_relations,
            "status": oracle["saved_tensor_interventions"]["status"],
        },
        "native_source_to_gradient_relations": {
            "relation_count": len(source_relations),
            "relations": source_relations,
            "status": oracle["source_replay_interventions"]["status"],
        },
        "saved_tensor_gradient_interventions": oracle[
            "saved_tensor_interventions"
        ],
        "source_replay_gradient_interventions": oracle[
            "source_replay_interventions"
        ],
        "v1_scientific_result_preservation": preservation,
    }
    component_status = {
        "checkpoint_native_validation": checkpoint[
            "checkpoint_divergence_native_oracle_validation"
        ]["status"],
        "core_native_exact": comparison["status"],
        "native_backward_observation": observation[
            "native_backward_node_execution_trace"
        ]["status"],
        "native_gradient_oracle": oracle["native_gradient_dependency_oracle"][
            "status"
        ],
        "negative_controls": negative["status"],
        "process_isolation": isolation["gradient_oracle_process_isolation"][
            "status"
        ],
        "saved_tensor_intervention": oracle["saved_tensor_interventions"][
            "status"
        ],
        "source_replay": oracle["source_replay_interventions"]["status"],
        "v1_preservation": preservation["status"],
        "v2_lineage": v2["bidirectional_training_lineage_v2"]["status"],
    }
    supported = all([
        component_status["checkpoint_native_validation"]
        == "CHECKPOINT_DIVERGENCE_NATIVE_ORACLE_VALIDATED_SUPPORTED",
        component_status["core_native_exact"]
        == "GRADIENT_DEPENDENCY_NATIVE_ORACLE_EXACT_SUPPORTED",
        component_status["native_backward_observation"]
        == "NATIVE_BACKWARD_NODE_EXECUTION_OBSERVATION_SUPPORTED",
        component_status["native_gradient_oracle"]
        == "NATIVE_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
        component_status["negative_controls"]
        == "GRADIENT_ORACLE_NEGATIVE_CONTROLS_SUPPORTED",
        component_status["process_isolation"]
        == "GRADIENT_ORACLE_PROCESS_ISOLATION_SUPPORTED",
        component_status["saved_tensor_intervention"]
        == "NATIVE_SAVED_TENSOR_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
        component_status["source_replay"]
        == "NATIVE_SOURCE_REPLAY_GRADIENT_DEPENDENCY_ORACLE_SUPPORTED",
        component_status["v1_preservation"] == "V1_SCIENTIFIC_RESULTS_PRESERVED",
        component_status["v2_lineage"]
        == "BIDIRECTIONAL_TRAINING_UPDATE_LINEAGE_NATIVE_ORACLE_VALIDATED_SUPPORTED",
    ])
    summary = {
        "component_status": component_status,
        "core_native_relation_count": comparison["core_relation_count"],
        "mismatch_counts": {
            key: comparison[key]
            for key in (
                "duplicate_identity_collapse",
                "false_negative",
                "false_positive",
                "graph_topology_mismatch",
                "identity_mismatch",
                "multiplicity_mismatch",
                "saved_tensor_witness_missing",
                "source_replay_witness_missing",
                "target_gradient_mismatch",
                "unrepresented_native_dependency",
                "unsupported_core_dependency_count",
            )
        },
        "native_relation_count": comparison["native_relation_count"],
        "new_negative_control_count": negative["control_count"],
        "original_negative_control_count": v1["scientific_summary"][
            "negative_controls_count"
        ],
        "status": (
            "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_EVIDENCE_HARDENING_SUPPORTED"
            if supported
            else "PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED_GRADIENT_DEPENDENCY_INDEPENDENCE_NOT_ESTABLISHED"
        ),
        "v2_forward_queries": v2["query_exact_comparison_v2"][
            "forward_query_count"
        ],
        "v2_reverse_queries": v2["query_exact_comparison_v2"][
            "reverse_query_count"
        ],
    }
    return {
        "artifacts": artifacts,
        "hardening_summary": summary,
    }
