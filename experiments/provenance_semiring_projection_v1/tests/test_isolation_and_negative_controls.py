from __future__ import annotations

from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.isolation_audit import evaluate_isolation
from experiments.provenance_semiring_projection_v1.src.negative_controls import run_negative_controls


def test_original_45_and_added_25_unique_mutations_fail_closed_exactly_once() -> None:
    artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
    report, classification = run_negative_controls(artifact_root)
    assert report["status"] == "ALL_NEGATIVE_CONTROLS_FAILED_CLOSED"
    assert report["actual_control_count"] == 70
    assert report["passed_control_count"] == 70
    assert report["unique_fingerprint_count"] == 70
    assert report["original_control_count"] == report["original_passed_control_count"] == 45
    assert report["hardening_control_count"] == report["hardening_passed_control_count"] == 25
    assert report["automatic_repair_count"] == 0
    assert classification["all_execution_counts_one"] is True
    assert classification["all_fingerprints_unique"] is True
    assert classification["original_controls"] == {"actual": 45, "passed": 45, "required": 45}
    assert classification["hardening_controls"] == {"actual": 25, "passed": 25, "required": 25}


def test_added_hardening_controls_have_exact_machine_reasons_and_honest_depths() -> None:
    artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
    report, _classification = run_negative_controls(artifact_root)
    added = report["controls"][45:]
    assert [item["control_id"] for item in added] == [f"NC{number}" for number in range(46, 71)]
    assert all(item["expected_reason_code"] == item["actual_reason_code"] for item in added)
    assert all(item["honest_depth"] and item["execution_count"] == 1 for item in added)
    assert all(item["automatic_repair"] is False for item in added)


def test_materialized_isolation_traces_recompute_to_zero_targets() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    artifact_root = Path(__file__).resolve().parents[1] / "artifacts"
    traces = artifact_root / "evidence" / "isolation_probe" / "traces"
    authority, static, classification, direct_independence = evaluate_isolation(repo_root, traces)
    assert authority["status"] == "ISOLATION_SUPPORTED"
    assert all(value == 0 for value in authority["target_counts"].values())
    assert static["status"] == "SUPPORTED"
    assert classification["status"] == "SUPPORTED"
    assert direct_independence["status"] == "DIRECT_LOWER_K_INDEPENDENCE_SUPPORTED"
