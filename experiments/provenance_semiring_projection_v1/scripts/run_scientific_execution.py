from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.provenance_semiring_projection_v1.src.ordinary_execution import execute_ordinary
from experiments.provenance_semiring_projection_v1.src.workloads import load_workloads


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"scientific component failed: {command}\n{completed.stdout}\n{completed.stderr}")
    return completed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _test_inventory(test_root: Path) -> tuple[list[str], str]:
    names = []
    digest = hashlib.sha256()
    for path in sorted(test_root.glob("test_*.py")):
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + data)
        tree = ast.parse(data.decode("utf-8"), filename=path.as_posix())
        names.extend(f"{path.name}::{node.name}" for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
    return sorted(names), digest.hexdigest()


def _write_execution_manifest(artifact_root: Path) -> dict[str, Any]:
    manifest_path = artifact_root / "artifact_manifest.json"
    files = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file() and path.resolve() != manifest_path.resolve() and "__pycache__" not in path.parts:
            files.append({
                "path": path.relative_to(artifact_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha(path),
            })
    manifest = {
        "schema_version": "complete-hardening-execution-manifest-v1",
        "status": "ALL_FILES_REHASHED",
        "scope_root": ".",
        "self_reference_exclusion": "artifact_manifest.json",
        "file_count": len(files),
        "artifact_file_count": len(files),
        "rehash_mismatch_count": 0,
        "files": files,
    }
    _write(manifest_path, manifest)
    return manifest


def _manifest_rehash(artifact_root: Path, manifest: dict[str, Any]) -> bool:
    return all(
        (artifact_root / item["path"]).is_file()
        and (artifact_root / item["path"]).stat().st_size == item["size"]
        and _sha(artifact_root / item["path"]) == item["sha256"]
        for item in manifest["files"]
    )


def run_execution(repo_root: Path, experiment_root: Path, run_dir: Path, work_root: Path) -> None:
    artifacts_root = experiment_root / "artifacts"
    resolved_work = work_root.resolve()
    if resolved_work.parent != artifacts_root.resolve() or resolved_work.name != "_scientific_work":
        raise ValueError("scientific work root must be the exact experiment artifact scratch path")
    if resolved_work.exists():
        shutil.rmtree(resolved_work)
    resolved_work.mkdir(parents=True)
    python = sys.executable
    components = [
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_p1_exact", "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_p2_strictness", "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_formal_semantics_audit", "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_native_oracle_selftest", "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_p3_hierarchy", "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_database_which_bridge", "--repo-root", str(repo_root), "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_lower_strictness", "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_v1_preservation", "--repo-root", str(repo_root), "--artifact-root", str(resolved_work)],
        [python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_negative_controls", "--artifact-root", str(resolved_work)],
    ]
    for command in components:
        _run(command, cwd=repo_root)

    core_corpus = _read(resolved_work / "evidence" / "core_validated_snapshot_corpus.json")
    captured_by_case = {(item["workload_id"], item["variant"]): item for item in core_corpus["results"]}
    ordinary_cases = []
    for workload in load_workloads():
        variants = list(workload.get("queries", {"default": None}))
        for variant in variants:
            requested = None if variant == "default" else variant
            output, measurements = execute_ordinary(workload, variant=requested)
            captured = captured_by_case[(workload["id"], variant)]
            captured_bytes = captured["ordinary_output_utf8"].encode("utf-8")
            ordinary_cases.append({
                "workload_id": workload["id"],
                "variant": variant,
                "capture_on_off_bytes_equal": output == captured_bytes,
                "ordinary_sha256": hashlib.sha256(output).hexdigest(),
                "measurements": measurements,
            })

    tests = _run([python, "-m", "pytest", "experiments/provenance_semiring_projection_v1/tests", "-q"], cwd=repo_root)
    match = re.search(r"(\d+) passed", tests.stdout)
    if match is None:
        raise RuntimeError("pytest output did not report passed count")
    test_names, test_tree_sha = _test_inventory(experiment_root / "tests")
    test_results = {
        "schema_version": "scientific-test-results-v1",
        "status": "PASSED",
        "exit_code": tests.returncode,
        "passed_count": int(match.group(1)),
        "failed_count": 0,
        "collected_test_area_count": len(test_names),
        "collected_test_areas": test_names,
        "test_source_tree_sha256": test_tree_sha,
        "performance_fields_excluded": ["duration", "wall_clock", "pytest_timing"],
    }
    _write(resolved_work / "test_results.json", test_results)
    execution_manifest = _write_execution_manifest(resolved_work)
    _run(
        [
            python,
            "-m",
            "experiments.provenance_semiring_projection_v1.scripts.run_report_statistics",
            "--artifact-root",
            str(resolved_work),
            "--report",
            str(experiment_root / "EXPERIMENT_REPORT.md"),
            "--rendered-report-output",
            str(resolved_work / "generated_experiment_report.md"),
        ],
        cwd=repo_root,
    )
    _run(
        [
            python,
            "-m",
            "experiments.provenance_semiring_projection_v1.scripts.run_isolation_audit",
            "--repo-root",
            str(repo_root),
            "--artifact-root",
            str(resolved_work),
        ],
        cwd=repo_root,
    )

    authority = _read(resolved_work / "authority_isolation.json")
    classification = _read(resolved_work / "persisted_artifact_classification.json")
    direct_independence = _read(resolved_work / "direct_lower_k_independence_v2.json")
    component_inventory = []
    for path in sorted(resolved_work.rglob("*")):
        if path.is_file():
            component_inventory.append({
                "path": path.relative_to(resolved_work).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha(path),
            })
    rehash_match = all(item["sha256"] == _sha(resolved_work / item["path"]) and item["size"] == (resolved_work / item["path"]).stat().st_size for item in component_inventory)
    manifest_rehash_match = _manifest_rehash(resolved_work, execution_manifest)
    p1 = _read(resolved_work / "nx_exact_comparison.json")
    p1_v2 = _read(resolved_work / "native_candidate_nx_exact_comparison_v2.json")
    p2 = _read(resolved_work / "nx_strictness_counterexamples.json")
    p3 = _read(resolved_work / "hierarchical_projection_exact_comparison.json")
    oracle = _read(resolved_work / "independent_native_polynomial_oracle.json")
    algebra_independence = _read(resolved_work / "native_candidate_algebra_independence.json")
    formal = _read(resolved_work / "formal_target_semantics_audit.json")
    flat = _read(resolved_work / "flat_support_view_exact_comparison.json")
    hierarchy = _read(resolved_work / "two_level_unification_hierarchy_v2.json")
    which = _read(resolved_work / "nx_to_existing_which_lineage.json")
    lower = _read(resolved_work / "lower_projection_strictness_constructions.json")
    joint = _read(resolved_work / "joint_lower_projection_strictness.json")
    negative = _read(resolved_work / "negative_controls.json")
    report_statistics = _read(resolved_work / "report_statistics.json")
    report_consistency = _read(resolved_work / "report_artifact_consistency.json")
    v1_preservation = _read(resolved_work / "v1_result_preservation.json")
    candidate = _read(resolved_work / "core_projected_nx_polynomials.json")
    gates = {
        "all_ordinary_capture_bytes_equal": all(item["capture_on_off_bytes_equal"] for item in ordinary_cases),
        "p1_exact": p1["status"] == "EXACT_SUPPORTED",
        "p2_strict": p2["status"] == "STRICTNESS_SUPPORTED",
        "v1_results_preserved": v1_preservation["status"] == "PR19_V1_RESULTS_PRESERVED",
        "independent_native_oracle": oracle["status"] == "INDEPENDENT_NATIVE_POLYNOMIAL_PRIMITIVES_SUPPORTED",
        "independent_native_candidate_exact": p1_v2["status"] == "INDEPENDENT_NATIVE_NX_ORACLE_EXACT_SUPPORTED",
        "native_candidate_algebra_independence": algebra_independence["status"] == "NATIVE_CANDIDATE_ALGEBRA_INDEPENDENCE_SUPPORTED" and all(value == 0 for value in algebra_independence["counts"].values()),
        "formal_target_semantics": formal["status"] == "FORMAL_TARGET_SEMANTICS_CLASSIFIED" and formal["formal_algebraic_target_count"] >= 3,
        "p3_hierarchy": p3["status"] == "FORMAL_PROJECTION_HIERARCHY_EXACT_SUPPORTED",
        "flat_support_task_projection": flat["status"] == "FLAT_SOURCE_SUPPORT_VIEW_EXACT_PROJECTION_SUPPORTED",
        "formal_two_level_hierarchy": hierarchy["status"] == "TWO_LEVEL_FORMAL_HIERARCHY_SUPPORTED",
        "existing_which": which["status"] == "THREE_WAY_EXACT_SUPPORTED",
        "lower_strictness": lower["status"] == "LOWER_PROJECTION_STRICTNESS_SUPPORTED",
        "joint_lower_strictness": joint["status"] == "JOINT_LOWER_PROJECTION_STRICTNESS_SUPPORTED",
        "isolation": authority["status"] == "ISOLATION_SUPPORTED" and all(value == 0 for value in authority["target_counts"].values()),
        "direct_lower_independence": direct_independence["status"] == "DIRECT_LOWER_K_INDEPENDENCE_SUPPORTED" and all(value == 0 for value in direct_independence["counts"].values()),
        "negative_controls": negative["status"] == "ALL_NEGATIVE_CONTROLS_FAILED_CLOSED" and negative["original_passed_control_count"] == 45 and negative["hardening_passed_control_count"] >= 20,
        "report_statistics": report_statistics["status"] == "REPORT_STATISTICS_DERIVED_FROM_ARTIFACTS" and report_consistency["status"] == "REPORT_STATISTICS_EXACT_AGAINST_ARTIFACTS",
        "tests": test_results["status"] == "PASSED" and len(test_names) >= 36,
        "execution_manifest_rehash": manifest_rehash_match,
        "artifact_rehash": rehash_match,
    }
    report = {
        "schema_version": "complete-formal-semantics-hardening-execution-v1",
        "run_identity": "repeated-complete-formal-semantics-hardening-execution-v1",
        "status": "COMPLETE_HARDENING_RUN_SUPPORTED" if all(gates.values()) else "NOT_ESTABLISHED",
        "gates": gates,
        "ordinary_execution": {"case_count": len(ordinary_cases), "cases": ordinary_cases},
        "p1": p1,
        "independent_native_candidate_exact": p1_v2,
        "snapshot_identities": [
            {"workload_id": item["workload_id"], "variant": item["variant"], "snapshot_id": item["snapshot_id"]}
            for item in candidate["results"]
        ],
        "p2": p2,
        "v1_result_preservation": v1_preservation,
        "independent_native_polynomial_oracle": oracle,
        "native_candidate_algebra_independence": algebra_independence,
        "formal_target_semantics": formal,
        "p3": p3,
        "flat_support_task_projection": flat,
        "two_level_hierarchy": hierarchy,
        "existing_which": which,
        "lower_strictness": lower,
        "joint_lower_strictness": joint,
        "authority_isolation": {
            "status": authority["status"],
            "target_counts": authority["target_counts"],
            "authorized_core_validation_git_subprocess_count": authority["authorized_core_validation_git_subprocess_count"],
        },
        "direct_lower_k_independence": direct_independence,
        "persisted_classification": {
            "status": classification["status"],
            "unclassified_file_count": len(classification["unclassified_files"]),
            "hidden_registry_count": len(classification["hidden_registry_hits"]),
            "relation_answer_store_count": len(classification["relation_answer_store_hits"]),
        },
        "negative_controls": {
            "status": negative["status"],
            "actual_control_count": negative["actual_control_count"],
            "passed_control_count": negative["passed_control_count"],
            "unique_fingerprint_count": negative["unique_fingerprint_count"],
            "automatic_repair_count": negative["automatic_repair_count"],
            "original_control_count": negative["original_control_count"],
            "original_passed_control_count": negative["original_passed_control_count"],
            "hardening_control_count": negative["hardening_control_count"],
            "hardening_passed_control_count": negative["hardening_passed_control_count"],
            "fingerprints": [item["mutation_fingerprint"] for item in negative["controls"]],
        },
        "report_statistics": report_statistics,
        "report_artifact_consistency": report_consistency,
        "artifact_materialization": {
            "file_count": len(component_inventory),
            "independent_rehash_match": rehash_match,
            "execution_manifest_rehash_match": manifest_rehash_match,
            "files": component_inventory,
        },
        "scientific_determinism_exclusions": [],
        "performance_fields_not_recorded": True,
    }
    _write(run_dir / "scientific_reports.json", report)
    _write(run_dir / "test_results.json", test_results)
    shutil.rmtree(resolved_work)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one complete materialized scientific execution")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "experiments" / "provenance_semiring_projection_v1"
    run_execution(repo_root, experiment_root, args.run_dir.resolve(), experiment_root / "artifacts" / "_scientific_work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
