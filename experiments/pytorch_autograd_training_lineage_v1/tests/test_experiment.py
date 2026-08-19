from __future__ import annotations

import ast
from collections import Counter
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.pytorch_autograd_training_lineage_v1.artifact_io import (
    build_artifact_manifest,
    verify_artifact_manifest,
)
from experiments.pytorch_autograd_training_lineage_v1.isolation_audit import build_isolation_audit
from experiments.pytorch_autograd_training_lineage_v1.native_graph import observe_native_autograd_graph
from experiments.pytorch_autograd_training_lineage_v1.pipeline import TrainingSpec, run_training_step
from experiments.pytorch_autograd_training_lineage_v1.science import (
    EXPERIMENT_ROOT,
    _protected_scope,
    run_complete_science,
)


@pytest.fixture(scope="session")
def science_result() -> dict:
    return run_complete_science()


def test_official_cpu_torch_executes() -> None:
    assert torch.__version__ == "2.13.0+cpu"
    assert torch.version.cuda is None
    result = run_training_step(TrainingSpec(workload="linear_chain"))
    assert result.ordinary_result["device"] == "cpu"
    assert result.ordinary_result["dtype"] == "torch.float64"


def test_native_graph_traversal_reproduces_frozen_graph() -> None:
    first = run_training_step(
        TrainingSpec(workload="branch_and_merge"),
        native_observer=observe_native_autograd_graph,
    ).native_observation
    second = run_training_step(
        TrainingSpec(workload="branch_and_merge"),
        native_observer=observe_native_autograd_graph,
    ).native_observation
    assert first == second
    assert first["canonical_graph_sha256"] == second["canonical_graph_sha256"]


def test_candidate_graph_exact_against_native_graph(science_result: dict) -> None:
    comparison = science_result["artifacts"]["autograd_projection_exact_comparison"]
    assert comparison["aggregate"]["exact_workload_count"] == 5
    for row in comparison["workloads"].values():
        assert row["canonical_bytes_exact"]
        assert row["edge_mismatch"] == 0
        assert row["node_type_mismatch"] == 0


def test_candidate_cannot_read_grad_fn_or_next_functions() -> None:
    audit = build_isolation_audit()
    assert audit["candidate_forbidden_attributes"] == []
    assert "grad_fn" not in audit["candidate"]["attributes"]
    assert "next_functions" not in audit["candidate"]["attributes"]


def test_branch_fanout_and_shared_nodes_preserved(science_result: dict) -> None:
    graph = science_result["artifacts"]["native_autograd_graph"]["workloads"]["branch_and_merge"]
    candidate = science_result["artifacts"]["core_projected_autograd_graph"]["workloads"]["branch_and_merge"]
    assert graph == candidate
    assert graph["shared_node_count"] > 0
    assert any(row["is_shared"] for row in graph["nodes"])


def test_shared_tensor_edge_multiplicity_preserved(science_result: dict) -> None:
    graph = science_result["artifacts"]["native_autograd_graph"]["workloads"]["shared_tensor_reuse"]
    multiplicity = Counter((row["source_node_id"], row["target_node_id"]) for row in graph["edges"])
    assert max(multiplicity.values()) == 2
    assert science_result["artifacts"]["autograd_projection_exact_comparison"]["workloads"]["shared_tensor_reuse"]["multiplicity_mismatch"] is False


def test_duplicate_valued_source_identity_preserved(science_result: dict) -> None:
    snapshot = science_result["artifacts"]["validated_core_snapshots"]["standard_workloads"]["duplicate_valued_distinct_sources"]
    sources = [
        row for row in snapshot["tables"]["source_information_records"]
        if row["source_payload"]["source_role"] == "training_sample"
    ]
    assert len(sources) == 2
    assert sources[0]["source_payload"]["tensor"]["value"] == sources[1]["source_payload"]["tensor"]["value"]
    assert sources[0]["source_information_id"] != sources[1]["source_information_id"]


def test_duplicate_source_does_not_expand_to_peer_gradient(science_result: dict) -> None:
    queries = science_result["artifacts"]["forward_training_source_queries"]["queries"]
    x1 = next(
        row for row in queries
        if row["workload"] == "duplicate_valued_distinct_sources"
        and row["source_ref"] == "source:sample:x1"
    )
    assert "step_0:gradient:parameter:w" in x1["query"]["outcome_keys"]
    assert "step_0:gradient:input:x2" not in x1["query"]["outcome_keys"]


@pytest.mark.parametrize("pair_name", [
    "different_sample_identity_equal_value",
    "different_evidence_context_equal_computation",
    "checkpoint_recomputation_external_state",
])
def test_autograd_projection_strictness_pairs(science_result: dict, pair_name: str) -> None:
    pairs = science_result["artifacts"]["autograd_strict_projection_counterexamples"]["pairs"]
    pair = next(row for row in pairs if row["pair"] == pair_name)
    assert pair["gamma_different"]
    assert pair["graph_equal"]


def test_reverse_parameter_update_query_exact(science_result: dict) -> None:
    queries = science_result["artifacts"]["reverse_parameter_update_queries"]["queries"]
    parameter_queries = [row for row in queries if ":parameter:" in row["support_key"]]
    assert parameter_queries
    assert all(row["exact"] for row in parameter_queries)
    assert all(row["query"]["path_count"] > 0 for row in parameter_queries)


def test_forward_sample_query_exact(science_result: dict) -> None:
    queries = science_result["artifacts"]["forward_training_source_queries"]["queries"]
    sample_queries = [row for row in queries if row["source_ref"].startswith("source:sample:")]
    assert sample_queries
    assert all(row["exact"] for row in sample_queries)
    assert all(any("parameter:" in key for key in row["query"]["outcome_keys"]) for row in sample_queries)


def test_no_fabricated_sample_parameter_shortcut(science_result: dict) -> None:
    snapshots = science_result["artifacts"]["validated_core_snapshots"]["standard_workloads"]
    for snapshot in snapshots.values():
        tables = snapshot["tables"]
        sample_ids = {
            row["source_information_id"] for row in tables["source_information_records"]
            if row["source_payload"]["source_role"] == "training_sample"
        }
        parameter_after_ids = {
            row["support_id"] for row in tables["perceptual_support_records"]
            if row["support_payload"]["support_kind"] == "parameter_after_step"
        }
        assert not any(
            binding["origin_reference"].get("source_information_id") in sample_ids
            and binding["outcome_reference"].get("support_id") in parameter_after_ids
            for binding in tables["generation_bindings"]
        )


def test_generated_origin_multistage_paths_valid(science_result: dict) -> None:
    snapshots = science_result["artifacts"]["validated_core_snapshots"]["standard_workloads"]
    for snapshot in snapshots.values():
        tables = snapshot["tables"]
        support_ids = {row["support_id"] for row in tables["perceptual_support_records"]}
        generated = {row["generated_origin_id"]: row for row in tables["generated_origins"]}
        used = {
            row["origin_reference"]["generated_origin_id"]
            for row in tables["generation_bindings"]
            if row["origin_reference"]["kind"] == "generated_origin"
        }
        assert set(generated) == used
        assert all(row["origin_payload"]["source_support_id"] in support_ids for row in generated.values())


def test_optimizer_parameter_versions_remain_distinct(science_result: dict) -> None:
    snapshot = science_result["artifacts"]["validated_core_snapshots"]["standard_workloads"]["linear_chain"]
    tables = snapshot["tables"]
    before_ids = {
        row["source_information_id"] for row in tables["source_information_records"]
        if row["source_payload"]["source_role"] == "parameter_before_step"
    }
    after_ids = {
        row["support_id"] for row in tables["perceptual_support_records"]
        if row["support_payload"]["support_kind"] == "parameter_after_step"
    }
    assert before_ids
    assert after_ids
    assert before_ids.isdisjoint(after_ids)


def test_official_optimizer_result_matches_formula(science_result: dict) -> None:
    verification = science_result["artifacts"]["optimizer_verification"]
    assert verification["all_official_results_match_independent_formula"]
    assert all(
        row.get("formula_exact", row.get("actual_step_skipped", False))
        for row in verification["rows"]
    )


def test_checkpoint_original_and_recompute_contexts_distinct(science_result: dict) -> None:
    result = science_result["artifacts"]["checkpoint_divergence_localization"]
    assert result["checks"]["recomputation_actually_executed"]
    assert result["checks"]["original_and_recomputation_occurrences_distinct"]


def test_stable_checkpoint_equals_no_checkpoint_gradient(science_result: dict) -> None:
    comparison = science_result["artifacts"]["checkpoint_gradient_comparison"]
    assert comparison["stable_gradient"] == comparison["no_checkpoint_gradient"]
    assert comparison["checks"]["no_checkpoint_stable_parameter_update_exact"]


def test_divergent_checkpoint_gives_finite_wrong_gradient(science_result: dict) -> None:
    comparison = science_result["artifacts"]["checkpoint_gradient_comparison"]
    assert comparison["divergent_gradient"] != comparison["stable_gradient"]
    assert all(math.isfinite(value) for value in comparison["divergent_gradient"])
    assert comparison["checks"]["default_determinism_check_did_not_raise"]


def test_stable_divergent_native_graph_exact_equal(science_result: dict) -> None:
    graph = science_result["artifacts"]["checkpoint_graph_equality"]
    assert graph["stable_divergent_exact"]
    assert len(graph["canonical_graph_sha256"]) == 64


def test_divergent_reverse_trace_reaches_scale_two(science_result: dict) -> None:
    trace = science_result["artifacts"]["checkpoint_divergence_reverse_trace"]
    assert "source:external:scale:recomputation" in trace["source_keys"]
    assert any("backward_recomputation" in key for key in trace["occurrence_keys"])


def test_forward_trace_from_scale_two_reaches_update(science_result: dict) -> None:
    trace = science_result["artifacts"]["checkpoint_divergence_forward_trace"]
    assert "step_0:gradient:parameter:p" in trace["outcome_keys"]
    assert "step_0:parameter:p:after" in trace["outcome_keys"]
    assert "step_0:optimizer_state:after" in trace["outcome_keys"]


def test_zero_gradient_differs_from_unused(science_result: dict) -> None:
    result = science_result["artifacts"]["zero_gradient_vs_nonparticipation"]
    assert result["p_zero_classification"] == "PARTICIPATED_WITH_ZERO_DERIVATIVE"
    assert result["p_unused_classification"] == "DID_NOT_PARTICIPATE"
    assert all(result["checks"].values())


def test_output_orthogonality_for_all_workloads(science_result: dict) -> None:
    result = science_result["artifacts"]["output_orthogonality"]
    assert result["all_workloads_exact"]
    for workload in result["workloads"].values():
        assert len(set(workload["all_ordinary_sha256"].values())) == 1
        assert workload["core_snapshot_core_dual_exact"]


def test_collector_feedback_negative_control_executes_end_to_end(science_result: dict) -> None:
    controls = science_result["artifacts"]["negative_controls"]["controls"]
    row = next(item for item in controls if item["control_id"] == "NC05")
    assert row["executed_depth"] == "END_TO_END"
    assert row["observed_reason_code"] == "COLLECTOR_CALLBACK_MUST_BE_WRITE_ONLY"
    assert row["execution_count"] == 1


def test_all_negative_controls_fail_closed(science_result: dict) -> None:
    controls = science_result["artifacts"]["negative_controls"]["controls"]
    assert len(controls) == 32
    assert all(row["detected"] and row["fail_closed"] for row in controls)
    assert all(row["automatic_repair"] is False for row in controls)


def test_negative_control_accounting_is_honest(science_result: dict) -> None:
    accounting = science_result["artifacts"]["negative_control_accounting"]
    assert accounting["execution_count_total"] == 32
    assert accounting["unique_mutation_fingerprint_count"] == 32
    assert accounting["repeated_mutation_fingerprint_count"] == 0
    assert accounting["depth_counts"] == {
        "END_TO_END": 1,
        "ISOLATION": 8,
        "VALIDATOR_INTEGRATION": 2,
        "VALIDATOR_UNIT": 21,
    }


def test_two_run_determinism() -> None:
    run_1 = EXPERIMENT_ROOT / "artifacts" / "runs" / "run_1" / "scientific_reports.json"
    run_2 = EXPERIMENT_ROOT / "artifacts" / "runs" / "run_2" / "scientific_reports.json"
    if run_1.is_file() and run_2.is_file():
        assert run_1.read_bytes() == run_2.read_bytes()
    else:
        first = run_training_step(
            TrainingSpec(workload="branch_and_merge"),
            native_observer=observe_native_autograd_graph,
        )
        second = run_training_step(
            TrainingSpec(workload="branch_and_merge"),
            native_observer=observe_native_autograd_graph,
        )
        assert first.ordinary_bytes == second.ordinary_bytes
        assert first.native_observation == second.native_observation


def test_core_protected_paths_unchanged() -> None:
    protected = _protected_scope()
    assert protected["core_zero_change"]
    assert protected["changed_existing_experiment_paths"] == []
    assert protected["pytorch_specific_core_field_count"] == 0


def test_artifact_manifest_rehash() -> None:
    manifest_path = EXPERIMENT_ROOT / "artifacts" / "artifact_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else build_artifact_manifest(EXPERIMENT_ROOT)
    )
    result = verify_artifact_manifest(EXPERIMENT_ROOT, manifest)
    assert result["verified"], result["mismatches"]


def test_capture_modes_use_one_shared_run_training_step() -> None:
    source = (EXPERIMENT_ROOT / "modes.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_training_step"
    ]
    assert len(calls) == 4
    assert all(len(node.args) == 1 for node in calls)


def test_frozen_function_gradcheck(science_result: dict) -> None:
    assert science_result["artifacts"]["gradcheck"]["passed"]


def test_native_observer_has_no_core_import() -> None:
    audit = build_isolation_audit()
    assert audit["native_core_imports"] == []
    assert audit["reference_core_imports"] == []
