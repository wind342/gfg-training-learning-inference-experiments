from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..scripts.run_tests import run_test_suites
from .common import canonical_sha256, file_sha256, write_json
from .manifest import build_manifest
from .result_validation import MECHANISMS, ResultValidationError, stable_scientific_result, validate_complete_result_set, validate_mechanism_result


SUPPORTED_STATUS = "FIVE_PROFILE_EXACT_STRICT_PROJECTION_SUPPORTED"
FAILED_STATUS = "FIVE_PROFILE_EXACT_STRICT_PROJECTION_FAILED"
RATINGS = {
    "database_which_lineage": "C",
    "source_map": "B",
    "opentelemetry": "B",
    "w3c_prov_generation_profile": "C",
    "pytorch_autograd_dependency_profile": "A",
}
SCOPE_STATEMENT = "Within five explicitly frozen and limited profiles/workloads, complete target representations are exactly equal and valid non-injectivity witnesses exist."
NON_CLAIMS = [
    "coverage of an entire external standard or arbitrary program",
    "Core is the unique or minimal complete ontology",
    "the frozen crosswalks are unique",
    "all five mechanisms received equally strong third-party system confirmation",
]


class UnifiedRunError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execute_mechanism_subprocess(mechanism: str, repo: Path, runtime_dir: Path) -> dict[str, Any]:
    output = runtime_dir / "structured_result.json"
    command = [
        sys.executable,
        "-m",
        "experiments.five_profile_unified_projection_proof.src.mechanism_entry",
        "--mechanism",
        mechanism,
        "--repo",
        str(repo),
        "--run-dir",
        str(runtime_dir / "work"),
        "--output",
        str(output),
    ]
    process = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    receipt = {
        "mechanism": mechanism,
        "command": [
            "python",
            "-m",
            "experiments.five_profile_unified_projection_proof.src.mechanism_entry",
            "--mechanism",
            mechanism,
            "--repo",
            "<REPO>",
            "--run-dir",
            "<RUNTIME>",
            "--output",
            "<RUNTIME>/structured_result.json",
        ],
        "exit_code": process.returncode,
        "structured_result_exists": output.is_file(),
        "stdout_tail": process.stdout[-8000:],
    }
    if process.returncode != 0 or not output.is_file():
        raise UnifiedRunError(f"{mechanism} failed without an acceptable structured result: {receipt}")
    result = json.loads(output.read_text(encoding="utf-8"))
    validate_mechanism_result(result, expected_mechanism=mechanism)
    receipt["structured_result_sha256"] = file_sha256(output)
    result["execution_receipt"] = receipt
    return result


Executor = Callable[[str, Path, Path], dict[str, Any]]


def execute_pass(index: int, repo: Path, runtime_root: Path, artifacts: Path, executor: Executor = execute_mechanism_subprocess) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    run_runtime = runtime_root / f"run_{index}"
    run_artifacts = artifacts / "runs" / f"run_{index}"
    results: dict[str, dict[str, Any]] = {}
    for mechanism in MECHANISMS:
        mechanism_runtime = run_runtime / mechanism
        mechanism_runtime.mkdir(parents=True, exist_ok=True)
        result = executor(mechanism, repo, mechanism_runtime)
        validate_mechanism_result(result, expected_mechanism=mechanism)
        results[mechanism] = result
        write_json(run_artifacts / "mechanisms" / f"{mechanism}.json", result)
    validate_complete_result_set(results)
    tests = run_test_suites(repo, run_runtime / "test_receipts")
    write_json(run_artifacts / "test_results.json", tests)
    if not tests["all_passed"]:
        raise UnifiedRunError(f"test run {index} failed")
    return results, tests


def _normalized_tests(tests: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": tests["status"],
        "totals": tests["totals"],
        "suites": {name: {"status": row["status"], "counts": row["counts"]} for name, row in tests["suites"].items()},
    }


def _canonical_summary(results: dict[str, dict[str, Any]], tests: dict[str, Any]) -> dict[str, Any]:
    return {
        "mechanisms": {name: stable_scientific_result(result) for name, result in results.items()},
        "external_independence_ratings": RATINGS,
        "scope_statement": SCOPE_STATEMENT,
        "non_claims": NON_CLAIMS,
        "tests": _normalized_tests(tests),
    }


def _prepare_artifacts(artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    runs = artifacts / "runs"
    if runs.exists():
        shutil.rmtree(runs)
    for name in ("five_profile_summary.json", "unified_manifest.json", "test_results.json", "determinism.json"):
        path = artifacts / name
        if path.exists():
            path.unlink()


def run_all(repo: Path, artifacts: Path, *, executor: Executor = execute_mechanism_subprocess) -> dict[str, Any]:
    started = _utc_now()
    _prepare_artifacts(artifacts)
    runtime_root = repo / "data_private" / "five_profile_unified_projection_proof"
    runtime_root.mkdir(parents=True, exist_ok=True)
    first, first_tests = execute_pass(1, repo, runtime_root, artifacts, executor)
    second, second_tests = execute_pass(2, repo, runtime_root, artifacts, executor)
    canonical_one = _canonical_summary(first, first_tests)
    canonical_two = _canonical_summary(second, second_tests)
    hash_one = canonical_sha256(canonical_one)
    hash_two = canonical_sha256(canonical_two)
    write_json(artifacts / "runs/run_1/canonical_summary.json", canonical_one)
    write_json(artifacts / "runs/run_2/canonical_summary.json", canonical_two)
    if hash_one != hash_two or canonical_one != canonical_two:
        raise UnifiedRunError("two complete unified runs have different canonical summaries")
    final_results = deepcopy(first)
    for mechanism in MECHANISMS:
        mechanism_hash_one = canonical_sha256(stable_scientific_result(first[mechanism]))
        mechanism_hash_two = canonical_sha256(stable_scientific_result(second[mechanism]))
        if mechanism_hash_one != mechanism_hash_two:
            raise UnifiedRunError(f"{mechanism} differs across complete runs")
        final_results[mechanism]["determinism"] = {
            "checked": True,
            "status": "PASS",
            "run_1_hash": mechanism_hash_one,
            "run_2_hash": mechanism_hash_two,
        }
    validate_complete_result_set(final_results)
    combined_tests = {"run_1": first_tests, "run_2": second_tests, "all_passed": first_tests["all_passed"] and second_tests["all_passed"]}
    write_json(artifacts / "test_results.json", combined_tests)
    determinism = {"status": "PASS", "canonical_summaries_equal": True, "run_1_hash": hash_one, "run_2_hash": hash_two}
    write_json(artifacts / "determinism.json", determinism)
    summary = {
        "status": SUPPORTED_STATUS,
        "all_five_executed": set(final_results) == set(MECHANISMS),
        "all_p1_passed": all(result["p1"]["status"] == "PASS" for result in final_results.values()),
        "all_p2_passed": all(result["p2"]["status"] == "PASS" for result in final_results.values()),
        "core_changed_files": 0,
        "mechanisms": final_results,
        "external_independence_ratings": RATINGS,
        "scope_statement": SCOPE_STATEMENT,
        "non_claims": NON_CLAIMS,
        "determinism": determinism,
        "tests": {"all_passed": combined_tests["all_passed"], "run_1_totals": first_tests["totals"], "run_2_totals": second_tests["totals"]},
    }
    if not all((summary["all_five_executed"], summary["all_p1_passed"], summary["all_p2_passed"], combined_tests["all_passed"])):
        raise UnifiedRunError("unified success predicate is false")
    write_json(artifacts / "five_profile_summary.json", summary)
    manifest = build_manifest(repo, artifacts, final_results, started_at=started, ended_at=_utc_now())
    if manifest["core_changed_files"] != 0:
        raise UnifiedRunError(f"Core changed files must be zero: {manifest['core_changed_paths']}")
    write_json(artifacts / "unified_manifest.json", manifest)
    return {"summary": summary, "manifest": manifest}


def write_failure(artifacts: Path, error: BaseException) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json(artifacts / "five_profile_summary.json", {
        "status": FAILED_STATUS,
        "all_five_executed": False,
        "all_p1_passed": False,
        "all_p2_passed": False,
        "error_type": type(error).__name__,
        "error": str(error),
    })
