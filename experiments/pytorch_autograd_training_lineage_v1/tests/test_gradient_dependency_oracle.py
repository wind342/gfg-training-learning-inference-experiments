from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.pytorch_autograd_training_lineage_v1.artifact_io import (
    build_artifact_manifest,
    canonical_json_bytes,
    verify_artifact_manifest,
    write_json,
)
from experiments.pytorch_autograd_training_lineage_v1.gradient_dependency_comparison import (
    compare_gradient_dependencies,
)
from experiments.pytorch_autograd_training_lineage_v1.hardening_science import (
    EXPERIMENT_ROOT,
    run_complete_hardening_science,
)
from experiments.pytorch_autograd_training_lineage_v1.native_backward_dependency_oracle import (
    observe_native_backward,
)
from experiments.pytorch_autograd_training_lineage_v1.native_oracle_workloads import (
    NativeTrainingSpec,
    run_native_oracle_training_step,
)


@pytest.fixture(scope="session")
def hardening_runs() -> tuple[dict, dict]:
    # Both values are rebuilt from actual PyTorch executions in this test process.
    return run_complete_hardening_science(), run_complete_hardening_science()


def _artifacts(hardening_runs: tuple[dict, dict]) -> dict:
    return hardening_runs[0]["artifacts"]


def _comparison(hardening_runs: tuple[dict, dict]) -> dict:
    return _artifacts(hardening_runs)[
        "gradient_dependency_native_oracle_exact_comparison"
    ]


def test_baseline_saved_tensor_hooks_do_not_change_gradients() -> None:
    observation = observe_native_backward(NativeTrainingSpec("linear_chain"))
    assert observation["baseline_ordinary_bytes_exact"]
    assert observation["baseline_gradients_exact"]
    assert observation["saved_tensors"]["pack_trace"]
    assert observation["saved_tensors"]["unpack_trace"]


def test_native_node_pre_post_hooks_pair_exactly() -> None:
    observation = observe_native_backward(NativeTrainingSpec("branch_and_merge"))
    executions = observation["backward"]["executions"]
    assert observation["backward"]["all_executed_nodes_paired"]
    assert len(executions) == len(observation["backward"]["execution_order"])
    assert all(row["prehook_ordinal"] < row["posthook_ordinal"] for row in executions)


def test_unpack_events_attach_to_actual_executed_nodes() -> None:
    observation = observe_native_backward(NativeTrainingSpec("linear_chain"))
    executed = {
        row["native_node_id"] for row in observation["backward"]["executions"]
    }
    unpacked = observation["saved_tensors"]["unpack_trace"]
    assert unpacked
    assert observation["unassigned_unpack_count"] == 0
    assert all(row["assigned"] and row["native_node_id"] in executed for row in unpacked)


def test_native_oracle_source_has_no_operation_specific_rule_table() -> None:
    paths = (
        "native_backward_dependency_oracle.py",
        "saved_tensor_observer.py",
        "gradient_intervention_oracle.py",
    )
    source = "\n".join(
        (EXPERIMENT_ROOT / name).read_text(encoding="utf-8") for name in paths
    )
    forbidden = (
        "operation_type",
        "tracked_matmul",
        "tracked_mul",
        "tracked_relu",
        "tracked_pow",
        "tracked_sin",
        "_gradient_dependency_refs",
        "._saved_",
        "._raw_saved_",
    )
    assert not [token for token in forbidden if token in source]


def test_v2_reference_has_no_operation_specific_rule_table() -> None:
    source = (EXPERIMENT_ROOT / "independent_reference_v2.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "operation_type",
        "tracked_matmul",
        "tracked_mul",
        "tracked_relu",
        "tracked_pow",
        "tracked_sin",
        "UNDECLARED_LOCAL_GRADIENT_RULE",
        "REFERENCE_LOCAL_GRADIENT_RULE",
    )
    assert not [token for token in forbidden if token in source]


def test_candidate_process_cannot_read_native_oracle(
    hardening_runs: tuple[dict, dict],
) -> None:
    artifacts = _artifacts(hardening_runs)
    isolation = artifacts["gradient_oracle_process_isolation"]
    trace = artifacts["gradient_oracle_runtime_dependency_trace"]["candidate"]
    assert isolation["candidate_audit"]["candidate_oracle_read_count"] == 0
    assert not any(
        Path(row["path"]).stem
        in {
            "gradient_intervention_oracle",
            "native_backward_dependency_oracle",
            "native_oracle_workloads",
            "saved_tensor_observer",
        }
        for row in trace
    )


def test_native_oracle_process_cannot_read_core(
    hardening_runs: tuple[dict, dict],
) -> None:
    artifacts = _artifacts(hardening_runs)
    isolation = artifacts["gradient_oracle_process_isolation"]
    trace = artifacts["gradient_oracle_runtime_dependency_trace"]["native"]
    assert isolation["native_audit"]["native_core_read_count"] == 0
    assert not any(
        Path(row["path"]).stem
        in {"candidate_projection", "core_capture", "independent_reference"}
        for row in trace
    )


def test_saved_tensor_token_identity_is_stable_without_persisted_object_ids() -> None:
    first = observe_native_backward(NativeTrainingSpec("linear_chain"))
    second = observe_native_backward(NativeTrainingSpec("linear_chain"))
    first_trace = first["saved_tensors"]["pack_trace"]
    second_trace = second["saved_tensors"]["pack_trace"]
    assert first_trace == second_trace
    serialized = json.dumps(first_trace, sort_keys=True)
    assert "python_object_id" not in serialized
    assert "object_address" not in serialized


def test_one_token_intervention_only_intervenes_declared_token(
    hardening_runs: tuple[dict, dict],
) -> None:
    attempts = _artifacts(hardening_runs)[
        "saved_tensor_gradient_interventions"
    ]["attempts"]
    assert attempts
    for row in attempts:
        assert row["only_declared_token_intervened"]
        assert all(
            application["token_key"] == row["token_key"]
            for application in row["intervention_applications"]
        )
        gradient_keys = set(row["baseline_gradients"])
        assert set(row["changed_target_gradients"]) | set(
            row["unchanged_target_gradients"]
        ) == gradient_keys


def test_graph_topology_remains_exact_under_every_intervention(
    hardening_runs: tuple[dict, dict],
) -> None:
    artifacts = _artifacts(hardening_runs)
    saved = artifacts["saved_tensor_gradient_interventions"]["attempts"]
    source = artifacts["source_replay_gradient_interventions"]["attempts"]
    assert all(row["graph_topology_exact"] for row in [*saved, *source])


def test_source_replay_preserves_source_identity(
    hardening_runs: tuple[dict, dict],
) -> None:
    attempts = _artifacts(hardening_runs)[
        "source_replay_gradient_interventions"
    ]["attempts"]
    assert attempts
    assert all(row["source_identity_preserved"] for row in attempts)
    assert all(row["other_saved_tensors_frozen"] for row in attempts)


def test_native_oracle_and_core_relation_sets_are_exact(
    hardening_runs: tuple[dict, dict],
) -> None:
    comparison = _comparison(hardening_runs)
    core = {
        (row["workload_key"], row["dependency_key"], row["target_gradient_key"])
        for row in comparison["core_relations"]
    }
    native = {
        (row["workload_key"], row["dependency_key"], row["target_gradient_key"])
        for row in comparison["native_relations"]
    }
    assert core == native
    assert len(core) == len(native) == 29


def test_missing_core_dependency_is_detected(
    hardening_runs: tuple[dict, dict],
) -> None:
    baseline = _comparison(hardening_runs)
    core = deepcopy(baseline["core_relations"])
    core.pop()
    result = compare_gradient_dependencies(
        core,
        baseline["native_relations"],
        checkpoint_replay_equivalence=[],
    )
    assert not result["exact"]
    assert result["false_negative"] == 1


def test_fabricated_core_dependency_is_detected(
    hardening_runs: tuple[dict, dict],
) -> None:
    baseline = _comparison(hardening_runs)
    core = deepcopy(baseline["core_relations"])
    core.append({
        **core[0],
        "dependency_key": "source:fabricated:test",
    })
    result = compare_gradient_dependencies(
        core,
        baseline["native_relations"],
        checkpoint_replay_equivalence=[],
    )
    assert not result["exact"]
    assert result["false_positive"] == 1


def test_swapped_duplicate_valued_identity_is_detected(
    hardening_runs: tuple[dict, dict],
) -> None:
    baseline = _comparison(hardening_runs)
    core = deepcopy(baseline["core_relations"])
    row = next(
        item
        for item in core
        if item["workload_key"] == "duplicate_valued_distinct_sources"
        and item["dependency_key"] == "source:sample:x1"
    )
    row["dependency_key"] = "source:sample:x2"
    result = compare_gradient_dependencies(
        core,
        baseline["native_relations"],
        checkpoint_replay_equivalence=[],
    )
    assert not result["exact"]
    assert result["identity_mismatch"] >= 1


def test_checkpoint_scale_two_dependency_is_independently_observed(
    hardening_runs: tuple[dict, dict],
) -> None:
    relations = _artifacts(hardening_runs)[
        "native_source_to_gradient_relations"
    ]["relations"]
    relation = next(
        row
        for row in relations
        if row["workload_key"] == "checkpoint:divergent"
        and row["dependency_key"] == "source:external:scale:recomputation"
        and row["target_gradient_key"] == "step_0:gradient:parameter:p"
    )
    assert relation["successful_interventions"]
    assert relation["node_execution_witnesses"]


def test_zero_gradient_and_unused_remain_distinct_in_real_pytorch() -> None:
    run = run_native_oracle_training_step(
        NativeTrainingSpec("zero_gradient_and_unused_sources")
    )
    gradients = run.ordinary_result["gradients"]
    assert gradients["parameter:p_zero"]["value"] == [0.0, 0.0]
    assert gradients["parameter:p_unused"] is None


def test_complete_v2_forward_queries_are_exact(
    hardening_runs: tuple[dict, dict],
) -> None:
    artifacts = _artifacts(hardening_runs)
    queries = artifacts["bidirectional_training_lineage_v2"]["forward_queries"]
    comparison = artifacts["query_exact_comparison_v2"]
    assert queries and all(row["query"]["paths"] == row["reference_paths"] for row in queries)
    assert comparison["forward_exact_count"] == comparison["forward_query_count"]


def test_complete_v2_reverse_queries_are_exact(
    hardening_runs: tuple[dict, dict],
) -> None:
    artifacts = _artifacts(hardening_runs)
    queries = artifacts["bidirectional_training_lineage_v2"]["reverse_queries"]
    comparison = artifacts["query_exact_comparison_v2"]
    assert queries and all(row["query"]["paths"] == row["reference_paths"] for row in queries)
    assert comparison["reverse_exact_count"] == comparison["reverse_query_count"]


def test_old_v1_scientific_results_are_byte_exactly_preserved(
    hardening_runs: tuple[dict, dict],
) -> None:
    preservation = _artifacts(hardening_runs)["v1_scientific_result_preservation"]
    assert preservation["v1_artifact_mismatch_count"] == 0
    assert all(row["byte_exact"] for row in preservation["artifact_comparison"])
    assert all(preservation["preservation_gates"].values())


def test_two_complete_hardening_runs_are_deterministic(
    hardening_runs: tuple[dict, dict],
) -> None:
    first, second = hardening_runs
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_rebuilt_artifact_manifest_rehashes_exactly(
    hardening_runs: tuple[dict, dict],
    tmp_path: Path,
) -> None:
    root = tmp_path / "hardening-manifest-probe"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    for name, payload in sorted(_artifacts(hardening_runs).items()):
        write_json(artifacts / f"{name}.json", payload)
    manifest = build_artifact_manifest(root)
    verification = verify_artifact_manifest(root, manifest)
    assert verification["verified"]
    assert verification["checked_count"] == manifest["artifact_count"]
