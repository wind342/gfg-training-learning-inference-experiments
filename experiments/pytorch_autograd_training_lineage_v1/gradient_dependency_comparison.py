from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any


RelationKey = tuple[str, str, str]


def _snapshot_items(
    snapshots: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    result = []
    for workload, payload in sorted(snapshots["standard_workloads"].items()):
        result.append((workload, payload))
    for mode, payload in sorted(snapshots["checkpoint"].items()):
        result.append((f"checkpoint:{mode}", payload))
    return result


def extract_core_gradient_dependencies(
    snapshots: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for workload_key, snapshot in _snapshot_items(snapshots):
        tables = snapshot["tables"]
        semantic_keys = {
            row["source_information_id"]: row["source_payload"]["source_ref"]
            for row in tables["source_information_records"]
        }
        semantic_keys.update({
            row["support_id"]: row["support_payload"]["support_key"]
            for row in tables["perceptual_support_records"]
        })
        semantic_keys.update({
            row["generated_origin_id"]: row["origin_payload"]["support_key"]
            for row in tables["generated_origins"]
        })
        for binding in tables["generation_bindings"]:
            if not binding["relation_role"].startswith("gradient_value_dependency|"):
                continue
            origin = binding["origin_reference"]
            origin_id = origin.get("source_information_id", origin.get("generated_origin_id"))
            target_id = binding["outcome_reference"]["support_id"]
            result.append({
                "binding_id": binding["generation_binding_id"],
                "dependency_key": semantic_keys[origin_id],
                "relation_role": binding["relation_role"],
                "target_gradient_key": semantic_keys[target_id],
                "workload_key": workload_key,
            })
    return sorted(
        result,
        key=lambda row: (
            row["workload_key"],
            row["dependency_key"],
            row["target_gradient_key"],
            row["relation_role"],
        ),
    )


def _key(row: dict[str, Any]) -> RelationKey:
    return (
        row["workload_key"],
        row["dependency_key"],
        row["target_gradient_key"],
    )


def _expanded(counter: Counter[RelationKey]) -> list[dict[str, str]]:
    result = []
    for (workload, dependency, target), count in sorted(counter.items()):
        for _ in range(count):
            result.append({
                "dependency_key": dependency,
                "target_gradient_key": target,
                "workload_key": workload,
            })
    return result


def _identity_mismatches(
    core_only: Counter[RelationKey],
    native_only: Counter[RelationKey],
) -> int:
    core_by_target: dict[tuple[str, str], int] = defaultdict(int)
    native_by_target: dict[tuple[str, str], int] = defaultdict(int)
    for (workload, _dependency, target), count in core_only.items():
        core_by_target[(workload, target)] += count
    for (workload, _dependency, target), count in native_only.items():
        native_by_target[(workload, target)] += count
    return sum(
        min(core_by_target[key], native_by_target[key])
        for key in set(core_by_target) | set(native_by_target)
    )


def _target_mismatches(
    core_only: Counter[RelationKey],
    native_only: Counter[RelationKey],
) -> int:
    core_by_dependency: dict[tuple[str, str], int] = defaultdict(int)
    native_by_dependency: dict[tuple[str, str], int] = defaultdict(int)
    for (workload, dependency, _target), count in core_only.items():
        core_by_dependency[(workload, dependency)] += count
    for (workload, dependency, _target), count in native_only.items():
        native_by_dependency[(workload, dependency)] += count
    return sum(
        min(core_by_dependency[key], native_by_dependency[key])
        for key in set(core_by_dependency) | set(native_by_dependency)
    )


def compare_gradient_dependencies(
    core_relations: list[dict[str, Any]],
    native_relations: list[dict[str, Any]],
    *,
    checkpoint_replay_equivalence: list[dict[str, Any]],
) -> dict[str, Any]:
    core_counter = Counter(_key(row) for row in core_relations)
    native_counter = Counter(_key(row) for row in native_relations)
    core_only = core_counter - native_counter
    native_only = native_counter - core_counter
    multiplicity_mismatch = sum(
        abs(core_counter[key] - native_counter[key])
        for key in set(core_counter) | set(native_counter)
        if core_counter[key] != native_counter[key]
    )
    native_by_key = {_key(row): row for row in native_relations}
    duplicate_native_key_count = len(native_relations) - len(native_by_key)
    unsupported_core = []
    saved_witness_missing = 0
    source_replay_witness_missing = 0
    for core in core_relations:
        native = native_by_key.get(_key(core))
        if native is None:
            unsupported_core.append(core)
            continue
        kinds = set(native["dependency_kind"])
        has_unpack = bool(native["actual_unpack_witnesses"])
        has_intervention = bool(native["successful_interventions"])
        has_saved = "saved_tensor" in kinds and has_unpack and has_intervention
        has_source = "registered_source_replay" in kinds and has_intervention
        if "saved_tensor" in kinds and not has_saved:
            saved_witness_missing += 1
        if "registered_source_replay" in kinds and not has_source:
            source_replay_witness_missing += 1
        if not (has_saved or has_source):
            unsupported_core.append(core)

    graph_topology_mismatch = sum(
        not row["graph_topology_exact"] for row in checkpoint_replay_equivalence
    )
    duplicate_identity_relations = {
        (row["dependency_key"], row["target_gradient_key"])
        for row in native_relations
        if row["workload_key"] == "duplicate_valued_distinct_sources"
    }
    duplicate_identity_collapse = int(not all([
        ("source:sample:x1", "step_0:gradient:parameter:w")
        in duplicate_identity_relations,
        ("source:sample:x2", "step_0:gradient:parameter:w")
        in duplicate_identity_relations,
        ("source:parameter:w:before", "step_0:gradient:input:x1")
        in duplicate_identity_relations,
        ("source:parameter:w:before", "step_0:gradient:input:x2")
        in duplicate_identity_relations,
    ]))
    exact = all([
        not core_only,
        not native_only,
        multiplicity_mismatch == 0,
        duplicate_native_key_count == 0,
        not unsupported_core,
        saved_witness_missing == 0,
        source_replay_witness_missing == 0,
        graph_topology_mismatch == 0,
        duplicate_identity_collapse == 0,
    ])
    return {
        "core_relation_count": sum(core_counter.values()),
        "duplicate_identity_collapse": duplicate_identity_collapse,
        "duplicate_native_key_count": duplicate_native_key_count,
        "exact": exact,
        "false_negative": sum(native_only.values()),
        "false_negative_relations": _expanded(native_only),
        "false_positive": sum(core_only.values()),
        "false_positive_relations": _expanded(core_only),
        "graph_topology_mismatch": graph_topology_mismatch,
        "identity_mismatch": _identity_mismatches(core_only, native_only),
        "multiplicity_mismatch": multiplicity_mismatch,
        "native_relation_count": sum(native_counter.values()),
        "saved_tensor_witness_missing": saved_witness_missing,
        "source_replay_witness_missing": source_replay_witness_missing,
        "status": (
            "GRADIENT_DEPENDENCY_NATIVE_ORACLE_EXACT_SUPPORTED"
            if exact
            else "GRADIENT_DEPENDENCY_NATIVE_ORACLE_EXACT_NOT_ESTABLISHED"
        ),
        "target_gradient_mismatch": _target_mismatches(core_only, native_only),
        "unrepresented_native_dependency": sum(native_only.values()),
        "unsupported_core_dependencies": deepcopy(unsupported_core),
        "unsupported_core_dependency_count": len(unsupported_core),
    }


def compare_snapshots_with_native_oracle(
    snapshots: dict[str, Any],
    oracle_result: dict[str, Any],
) -> dict[str, Any]:
    core = extract_core_gradient_dependencies(snapshots)
    native = oracle_result["native_gradient_dependency_oracle"]["relations"]
    comparison = compare_gradient_dependencies(
        core,
        native,
        checkpoint_replay_equivalence=oracle_result[
            "checkpoint_recomputation_replay_equivalence"
        ],
    )
    return {
        **comparison,
        "core_relations": core,
        "native_relations": native,
    }
