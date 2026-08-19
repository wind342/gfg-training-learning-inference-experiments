from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.five_profile_unified_projection_proof.src import runner
from experiments.five_profile_unified_projection_proof.src.result_validation import MECHANISMS


def _passing_tests():
    return {
        "status": "PASS",
        "all_passed": True,
        "totals": {"tests": 8, "failures": 0, "errors": 0, "skipped": 0},
        "suites": {name: {"status": "PASS", "counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}} for name in ("core", *MECHANISMS, "three_profile_unified", "five_profile_unified")},
    }


def test_runner_calls_all_five_mechanisms(monkeypatch, tmp_path, complete_results):
    called = []

    def execute(mechanism, _repo, _runtime):
        called.append(mechanism)
        return complete_results[mechanism]

    monkeypatch.setattr(runner, "run_test_suites", lambda *_args: _passing_tests())
    results, tests = runner.execute_pass(1, tmp_path, tmp_path / "runtime", tmp_path / "artifacts", execute)
    assert tuple(called) == MECHANISMS
    assert set(results) == set(MECHANISMS)
    assert tests["all_passed"]


def test_runner_stops_on_missing_experiment(monkeypatch, tmp_path, complete_results):
    def execute(mechanism, _repo, _runtime):
        if mechanism == MECHANISMS[2]:
            raise FileNotFoundError(mechanism)
        return complete_results[mechanism]

    monkeypatch.setattr(runner, "run_test_suites", lambda *_args: _passing_tests())
    with pytest.raises(FileNotFoundError):
        runner.execute_pass(1, tmp_path, tmp_path / "runtime", tmp_path / "artifacts", execute)


def test_old_summary_is_removed_not_used(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    fake = artifacts / "five_profile_summary.json"
    fake.write_text(json.dumps({"status": runner.SUPPORTED_STATUS}), encoding="utf-8")
    runner._prepare_artifacts(artifacts)
    assert not fake.exists()


def test_subprocess_contract_passes_no_candidate_or_native_artifact(monkeypatch, tmp_path, result_factory):
    captured = {}

    class Process:
        returncode = 0
        stdout = "structured"

    def fake_run(command, **_kwargs):
        captured["command"] = command
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result_factory(MECHANISMS[0])), encoding="utf-8")
        return Process()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.execute_mechanism_subprocess(MECHANISMS[0], tmp_path, tmp_path / "runtime")
    joined = " ".join(captured["command"]).lower()
    assert "candidate.provn" not in joined
    assert "native_reference" not in joined
    assert "artifacts" not in joined


def test_two_complete_runs_have_identical_canonical_summary(monkeypatch, tmp_path, complete_results):
    monkeypatch.setattr(runner, "run_test_suites", lambda *_args: _passing_tests())
    monkeypatch.setattr(runner, "build_manifest", lambda *_args, **_kwargs: {"core_changed_files": 0})
    result = runner.run_all(tmp_path, tmp_path / "artifacts", executor=lambda mechanism, _repo, _runtime: complete_results[mechanism])
    determinism = result["summary"]["determinism"]
    assert determinism["run_1_hash"] == determinism["run_2_hash"]
    assert result["summary"]["all_five_executed"]

