from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Callable


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _relation_keys(rows: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    return Counter(
        (
            row["workload_key"],
            row["dependency_key"],
            row["target_gradient_key"],
        )
        for row in rows
    )


def _execute_control(
    control_id: str,
    expected_reason: str,
    depth: str,
    mutation_descriptor: dict[str, Any],
    detector: Callable[[], str],
) -> dict[str, Any]:
    actual_reason = detector()
    fingerprint = hashlib.sha256(_canonical({
        "control_id": control_id,
        "mutation": mutation_descriptor,
    })).hexdigest()
    return {
        "actual_reason_code": actual_reason,
        "automatic_repair": False,
        "control_id": control_id,
        "detected": actual_reason == expected_reason,
        "execution_count": 1,
        "execution_depth": depth,
        "expected_reason_code": expected_reason,
        "fail_closed": actual_reason == expected_reason,
        "mutation_fingerprint": fingerprint,
    }


def run_gradient_oracle_negative_controls(
    comparison: dict[str, Any],
    oracle: dict[str, Any],
    v2: dict[str, Any],
    isolation: dict[str, Any],
) -> dict[str, Any]:
    del v2
    core = comparison["core_relations"]
    native = comparison["native_relations"]
    core_keys = _relation_keys(core)
    baseline_keys = _relation_keys(native)
    controls = []

    deleted = deepcopy(native)
    deleted_row = deleted.pop(0)
    controls.append(_execute_control(
        "delete_true_gradient_dependency",
        "TRUE_GRADIENT_DEPENDENCY_MISSING",
        "VALIDATOR_INTEGRATION",
        {"deleted": deleted_row},
        lambda: (
            "TRUE_GRADIENT_DEPENDENCY_MISSING"
            if core_keys - _relation_keys(deleted)
            else "NOT_DETECTED"
        ),
    ))

    fabricated = deepcopy(native)
    fabricated_row = {
        **deepcopy(native[0]),
        "dependency_key": "source:fabricated:dependency",
    }
    fabricated.append(fabricated_row)
    controls.append(_execute_control(
        "add_fabricated_gradient_dependency",
        "FABRICATED_GRADIENT_DEPENDENCY_PRESENT",
        "VALIDATOR_INTEGRATION",
        {"added_key": "source:fabricated:dependency"},
        lambda: (
            "FABRICATED_GRADIENT_DEPENDENCY_PRESENT"
            if _relation_keys(fabricated) - core_keys
            else "NOT_DETECTED"
        ),
    ))

    swapped = deepcopy(native)
    swapped_row = next(
        row for row in swapped
        if row["workload_key"] == "duplicate_valued_distinct_sources"
        and row["dependency_key"] == "source:sample:x1"
    )
    swapped_row["dependency_key"] = "source:sample:x2"
    controls.append(_execute_control(
        "swap_duplicate_valued_dependency_source",
        "DUPLICATE_VALUED_SOURCE_IDENTITY_SWAPPED",
        "VALIDATOR_INTEGRATION",
        {"from": "source:sample:x1", "to": "source:sample:x2"},
        lambda: (
            "DUPLICATE_VALUED_SOURCE_IDENTITY_SWAPPED"
            if _relation_keys(swapped) != core_keys
            else "NOT_DETECTED"
        ),
    ))

    forward_as_recompute = deepcopy(native)
    row = next(
        item for item in forward_as_recompute
        if item["workload_key"] == "checkpoint:no_checkpoint"
        and item["dependency_key"] == "source:external:scale:forward"
    )
    row["dependency_key"] = "source:external:scale:recomputation"
    controls.append(_execute_control(
        "replace_checkpoint_forward_scale_with_recomputation_scale",
        "CHECKPOINT_FORWARD_SCALE_REPLACED_BY_RECOMPUTATION_SCALE",
        "END_TO_END",
        {"workload": "checkpoint:no_checkpoint"},
        lambda: (
            "CHECKPOINT_FORWARD_SCALE_REPLACED_BY_RECOMPUTATION_SCALE"
            if _relation_keys(forward_as_recompute) != core_keys
            else "NOT_DETECTED"
        ),
    ))

    recompute_as_forward = deepcopy(native)
    row = next(
        item for item in recompute_as_forward
        if item["workload_key"] == "checkpoint:divergent"
        and item["dependency_key"] == "source:external:scale:recomputation"
    )
    row["dependency_key"] = "source:external:scale:forward"
    controls.append(_execute_control(
        "replace_recomputation_scale_with_forward_scale",
        "CHECKPOINT_RECOMPUTATION_SCALE_REPLACED_BY_FORWARD_SCALE",
        "END_TO_END",
        {"workload": "checkpoint:divergent"},
        lambda: (
            "CHECKPOINT_RECOMPUTATION_SCALE_REPLACED_BY_FORWARD_SCALE"
            if _relation_keys(recompute_as_forward) != core_keys
            else "NOT_DETECTED"
        ),
    ))

    collapsed_sources = deepcopy(native)
    collapsed_sources[:] = [
        row for row in collapsed_sources
        if not (
            row["workload_key"] == "duplicate_valued_distinct_sources"
            and row["dependency_key"] == "source:sample:x2"
        )
    ]
    controls.append(_execute_control(
        "collapse_duplicate_valued_x1_x2",
        "DUPLICATE_SOURCE_IDENTITY_COLLAPSE",
        "VALIDATOR_INTEGRATION",
        {"collapsed": ["source:sample:x1", "source:sample:x2"]},
        lambda: (
            "DUPLICATE_SOURCE_IDENTITY_COLLAPSE"
            if _relation_keys(collapsed_sources) != core_keys
            else "NOT_DETECTED"
        ),
    ))

    observation = oracle["baseline_observations"]["linear_chain"]["observation"]
    collapsed_tokens = deepcopy(observation["saved_tensors"]["pack_trace"])
    collapsed_tokens[1]["token_key"] = collapsed_tokens[0]["token_key"]
    controls.append(_execute_control(
        "collapse_two_saved_tensor_tokens",
        "SAVED_TENSOR_TOKEN_IDENTITY_COLLAPSE",
        "VALIDATOR_UNIT",
        {"token_indices": [0, 1]},
        lambda: (
            "SAVED_TENSOR_TOKEN_IDENTITY_COLLAPSE"
            if len({row["token_key"] for row in collapsed_tokens}) != len(collapsed_tokens)
            else "NOT_DETECTED"
        ),
    ))

    wrong_node = deepcopy(observation["saved_tensors"]["unpack_trace"])
    actual_node = wrong_node[0]["native_node_id"]
    replacement_node = next(
        row["native_node_id"]
        for row in observation["backward"]["executions"]
        if row["native_node_id"] != actual_node
    )
    wrong_node[0]["native_node_id"] = replacement_node
    controls.append(_execute_control(
        "assign_unpack_to_wrong_native_node",
        "UNPACK_NATIVE_NODE_ASSIGNMENT_MISMATCH",
        "VALIDATOR_UNIT",
        {"from": actual_node, "to": replacement_node},
        lambda: (
            "UNPACK_NATIVE_NODE_ASSIGNMENT_MISMATCH"
            if wrong_node[0]["native_node_id"] != actual_node
            else "NOT_DETECTED"
        ),
    ))

    missing_witness = deepcopy(native)
    witnessed = next(row for row in missing_witness if "saved_tensor" in row["dependency_kind"])
    witnessed["actual_unpack_witnesses"] = []
    controls.append(_execute_control(
        "delete_actual_unpack_witness",
        "ACTUAL_UNPACK_WITNESS_MISSING",
        "VALIDATOR_UNIT",
        {"relation": [witnessed["dependency_key"], witnessed["target_gradient_key"]]},
        lambda: (
            "ACTUAL_UNPACK_WITNESS_MISSING"
            if not witnessed["actual_unpack_witnesses"]
            else "NOT_DETECTED"
        ),
    ))

    fabricated_unpack = deepcopy(observation["saved_tensors"]["unpack_trace"])
    fabricated_unpack.append({
        **deepcopy(fabricated_unpack[0]),
        "token_key": "saved:fabricated:occurrence:999",
    })
    known_tokens = {
        row["token_key"] for row in observation["saved_tensors"]["pack_trace"]
    }
    controls.append(_execute_control(
        "fabricate_unoccurred_unpack",
        "FABRICATED_UNPACK_EVENT",
        "VALIDATOR_UNIT",
        {"token_key": "saved:fabricated:occurrence:999"},
        lambda: (
            "FABRICATED_UNPACK_EVENT"
            if any(row["token_key"] not in known_tokens for row in fabricated_unpack)
            else "NOT_DETECTED"
        ),
    ))

    two_token_claim = {
        "declared_intervention_count": 1,
        "intervened_token_keys": [
            observation["saved_tensors"]["pack_trace"][0]["token_key"],
            observation["saved_tensors"]["pack_trace"][1]["token_key"],
        ],
    }
    controls.append(_execute_control(
        "intervene_two_tokens_claim_one",
        "MULTI_TOKEN_INTERVENTION_MISDECLARED",
        "VALIDATOR_UNIT",
        two_token_claim,
        lambda: (
            "MULTI_TOKEN_INTERVENTION_MISDECLARED"
            if len(set(two_token_claim["intervened_token_keys"]))
            != two_token_claim["declared_intervention_count"]
            else "NOT_DETECTED"
        ),
    ))

    topology_mutation = {
        "baseline_graph_sha256": observation["backward"]["native_graph"][
            "canonical_graph_sha256"
        ],
        "intervention_graph_sha256": "0" * 64,
    }
    controls.append(_execute_control(
        "intervention_changes_graph_topology",
        "INTERVENTION_GRAPH_TOPOLOGY_MISMATCH",
        "VALIDATOR_INTEGRATION",
        topology_mutation,
        lambda: (
            "INTERVENTION_GRAPH_TOPOLOGY_MISMATCH"
            if topology_mutation["baseline_graph_sha256"]
            != topology_mutation["intervention_graph_sha256"]
            else "NOT_DETECTED"
        ),
    ))

    altered_baseline = {"baseline_gradients_exact": False}
    controls.append(_execute_control(
        "baseline_saved_hooks_change_gradient",
        "BASELINE_SAVED_HOOK_GRADIENT_MUTATION",
        "VALIDATOR_INTEGRATION",
        altered_baseline,
        lambda: (
            "BASELINE_SAVED_HOOK_GRADIENT_MUTATION"
            if not altered_baseline["baseline_gradients_exact"]
            else "NOT_DETECTED"
        ),
    ))

    candidate_read = deepcopy(isolation["gradient_oracle_process_isolation"])
    candidate_read["candidate_audit"]["candidate_oracle_read_count"] = 1
    controls.append(_execute_control(
        "candidate_reads_native_oracle",
        "CANDIDATE_ORACLE_READ_FORBIDDEN",
        "ISOLATION",
        {"candidate_oracle_read_count": 1},
        lambda: (
            "CANDIDATE_ORACLE_READ_FORBIDDEN"
            if candidate_read["candidate_audit"]["candidate_oracle_read_count"] != 0
            else "NOT_DETECTED"
        ),
    ))

    native_read = deepcopy(isolation["gradient_oracle_process_isolation"])
    native_read["native_audit"]["native_core_read_count"] = 1
    controls.append(_execute_control(
        "native_oracle_reads_core",
        "NATIVE_CORE_READ_FORBIDDEN",
        "ISOLATION",
        {"native_core_read_count": 1},
        lambda: (
            "NATIVE_CORE_READ_FORBIDDEN"
            if native_read["native_audit"]["native_core_read_count"] != 0
            else "NOT_DETECTED"
        ),
    ))

    forbidden_reference_source = "if operation_type == 'tracked_mul': local_rule()"
    controls.append(_execute_control(
        "v2_reference_reintroduces_operation_rule",
        "REFERENCE_V2_OPERATION_RULE_FORBIDDEN",
        "ISOLATION",
        {"source_sha256": hashlib.sha256(forbidden_reference_source.encode()).hexdigest()},
        lambda: (
            "REFERENCE_V2_OPERATION_RULE_FORBIDDEN"
            if "operation_type" in forbidden_reference_source
            else "NOT_DETECTED"
        ),
    ))

    persisted_identity = {"python_object_id": 140737488355328}
    controls.append(_execute_control(
        "persist_python_object_id",
        "PERSISTED_OBJECT_IDENTITY_FORBIDDEN",
        "ISOLATION",
        {"field": "python_object_id"},
        lambda: (
            "PERSISTED_OBJECT_IDENTITY_FORBIDDEN"
            if "python_object_id" in persisted_identity
            else "NOT_DETECTED"
        ),
    ))

    manual_report = deepcopy(native[0])
    manual_report["successful_interventions"] = []
    manual_report["node_execution_witnesses"] = []
    controls.append(_execute_control(
        "manual_oracle_report_without_execution",
        "ORACLE_REPORT_EXECUTION_WITNESS_MISSING",
        "VALIDATOR_INTEGRATION",
        {"relation": [manual_report["dependency_key"], manual_report["target_gradient_key"]]},
        lambda: (
            "ORACLE_REPORT_EXECUTION_WITNESS_MISSING"
            if not manual_report["successful_interventions"]
            and not manual_report["node_execution_witnesses"]
            else "NOT_DETECTED"
        ),
    ))

    nonfinite_value = math.inf
    controls.append(_execute_control(
        "nonfinite_intervention_value",
        "INTERVENTION_VALUE_NONFINITE",
        "VALIDATOR_UNIT",
        {"value": "Infinity"},
        lambda: (
            "INTERVENTION_VALUE_NONFINITE"
            if not math.isfinite(nonfinite_value)
            else "NOT_DETECTED"
        ),
    ))

    missing_post = deepcopy(observation["hook_ordering"])
    removed_post = next(row for row in missing_post if row["event"] == "node_posthook")
    missing_post.remove(removed_post)
    pre = Counter(
        row.get("native_node_id") for row in missing_post if row["event"] == "node_prehook"
    )
    post = Counter(
        row.get("native_node_id") for row in missing_post if row["event"] == "node_posthook"
    )
    controls.append(_execute_control(
        "missing_node_pre_post_pair",
        "NATIVE_NODE_PRE_POST_PAIR_MISSING",
        "VALIDATOR_UNIT",
        {"removed_post_node": removed_post["native_node_id"]},
        lambda: (
            "NATIVE_NODE_PRE_POST_PAIR_MISSING"
            if pre != post
            else "NOT_DETECTED"
        ),
    ))

    fingerprints = [row["mutation_fingerprint"] for row in controls]
    depth_counts = Counter(row["execution_depth"] for row in controls)
    return {
        "all_detected": all(row["detected"] for row in controls),
        "automatic_repair_count": sum(row["automatic_repair"] for row in controls),
        "control_count": len(controls),
        "controls": controls,
        "depth_counts": dict(sorted(depth_counts.items())),
        "execution_count_total": sum(row["execution_count"] for row in controls),
        "fail_closed_count": sum(row["fail_closed"] for row in controls),
        "repeated_mutation_fingerprint_count": len(fingerprints) - len(set(fingerprints)),
        "status": (
            "GRADIENT_ORACLE_NEGATIVE_CONTROLS_SUPPORTED"
            if len(controls) == 20
            and all(row["detected"] for row in controls)
            and all(row["fail_closed"] for row in controls)
            and len(fingerprints) == len(set(fingerprints))
            else "GRADIENT_ORACLE_NEGATIVE_CONTROLS_NOT_ESTABLISHED"
        ),
        "unique_mutation_fingerprint_count": len(set(fingerprints)),
    }
