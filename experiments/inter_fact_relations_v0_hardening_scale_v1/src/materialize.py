from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..common import (
    EXPERIMENT_BRANCH,
    EXPERIMENT_ROOT,
    FROZEN_SOURCE_COMMIT,
    PROTECTED_TREE_HASHES,
    REPOSITORY_ROOT,
    canonical_sha256,
    file_sha256,
    git,
    load_json,
    write_json,
)
from .runner import (
    optional_scale_guard,
    run_scientific,
    source_isolation_audit,
)


ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"
BEFORE_REPAIR_HEAD = "3bc650889441f844c6e79b08aff2af41f1613b3e"
BEFORE_REPAIR_METRICS = {
    "small": {
        "total_elapsed_seconds": 0.7784118000417948,
        "capture_audit_elapsed_seconds": 0.0004790000384673476,
        "peak_parent_plus_children_rss_bytes": 76021760,
    },
    "medium": {
        "total_elapsed_seconds": 0.9117837999947369,
        "capture_audit_elapsed_seconds": 0.008171900059096515,
        "peak_parent_plus_children_rss_bytes": 85094400,
    },
    "large": {
        "total_elapsed_seconds": 6.452021200093441,
        "capture_audit_elapsed_seconds": 0.2314889000263065,
        "peak_parent_plus_children_rss_bytes": 455606272,
    },
}


def baseline_environment() -> dict[str, Any]:
    return {
        "status": "PASS",
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "branch": EXPERIMENT_BRANCH,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "dependencies": {
            "jsonschema": importlib.metadata.version("jsonschema"),
            "psutil": importlib.metadata.version("psutil"),
            "pytest": importlib.metadata.version("pytest"),
        },
        "protected_tree_hashes": PROTECTED_TREE_HASHES,
    }


def _git_lines(*args: str) -> list[str]:
    output = git(*args)
    return output.splitlines() if output else []


def protected_path_audit() -> dict[str, Any]:
    rows = []
    for path, expected_hash in PROTECTED_TREE_HASHES.items():
        current_hash = git("rev-parse", f"HEAD:{path}")
        committed_changes = _git_lines(
            "diff", "--name-only", FROZEN_SOURCE_COMMIT, "HEAD", "--", path
        )
        worktree_changes = _git_lines("diff", "--name-only", "--", path)
        untracked = _git_lines(
            "ls-files", "--others", "--exclude-standard", "--", path
        )
        rows.append(
            {
                "path": path,
                "expected_tree_hash": expected_hash,
                "current_commit_tree_hash": current_hash,
                "committed_change_count": len(committed_changes),
                "worktree_change_count": len(worktree_changes),
                "untracked_change_count": len(untracked),
                "status": (
                    "PASS"
                    if current_hash == expected_hash
                    and not committed_changes
                    and not worktree_changes
                    and not untracked
                    else "FAIL"
                ),
            }
        )
    all_committed = _git_lines(
        "diff", "--name-only", FROZEN_SOURCE_COMMIT, "HEAD"
    )
    all_worktree = _git_lines("diff", "--name-only")
    all_untracked = _git_lines("ls-files", "--others", "--exclude-standard")
    all_changes = sorted(set(all_committed + all_worktree + all_untracked))
    namespace = "experiments/inter_fact_relations_v0_hardening_scale_v1/"
    namespace_only = all(path.startswith(namespace) for path in all_changes)
    core_copy = (
        EXPERIMENT_ROOT / "experimental_core_copy"
    ).exists() or any(
        "experimental_core_copy/" in path.replace("\\", "/")
        for path in all_changes
    )
    status = (
        all(row["status"] == "PASS" for row in rows)
        and namespace_only
        and not core_copy
    )
    return {
        "status": "PASS" if status else "FAIL",
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "protected_paths": rows,
        "changed_file_count": len(all_changes),
        "changes_limited_to_new_namespace": namespace_only,
        "allowed_namespace": namespace,
        "formal_core_changed_files": 0
        if all(
            row["status"] == "PASS"
            for row in rows
            if row["path"] != "experiments/inter_fact_relations_v0"
        )
        else 1,
        "v0_experiment_changed_files": 0
        if next(
            row
            for row in rows
            if row["path"] == "experiments/inter_fact_relations_v0"
        )["status"]
        == "PASS"
        else 1,
        "experimental_core_copy_present": core_copy,
    }


def _normalized_pytest_output(output: str) -> str:
    normalized = re.sub(r"\s+in\s+\d+(?:\.\d+)?s", "", output)
    normalized = re.sub(r"\s+in\s+\d+:\d+:\d+", "", normalized)
    lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
    return "\n".join(lines[-40:])


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {}
    for name in ("passed", "failed", "skipped", "xfailed", "xpassed", "error"):
        match = re.search(rf"(\d+)\s+{name}", output)
        counts[name] = int(match.group(1)) if match else 0
    return counts


def _run_pytest(label: str, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "-ra"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    combined = "\n".join(
        value for value in (completed.stdout, completed.stderr) if value
    )
    counts = _pytest_counts(combined)
    return {
        "label": label,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "return_code": completed.returncode,
        "counts": counts,
        "normalized_output_tail": _normalized_pytest_output(combined),
    }


def test_results() -> dict[str, Any]:
    suites = [
        _run_pytest(
            "hardening_scale_v1",
            ["experiments/inter_fact_relations_v0_hardening_scale_v1/tests"],
        ),
        _run_pytest("frozen_core", ["tests/core"]),
        _run_pytest("full_repository", []),
    ]
    return {
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in suites) else "FAIL"
        ),
        "suites": suites,
        "elapsed_time_excluded": True,
    }


def repair_before_after(
    run1: dict[str, Any], tests: dict[str, Any]
) -> dict[str, Any]:
    after_metrics = {
        row["scale"]: {
            "total_elapsed_seconds": row["total_elapsed_seconds"],
            "capture_audit_elapsed_seconds": row[
                "capture_audit_elapsed_seconds"
            ],
            "peak_parent_plus_children_rss_bytes": row[
                "peak_parent_plus_children_rss_bytes"
            ],
        }
        for row in run1["diagnostics"]
    }
    comparisons = []
    for scale in ("small", "medium", "large"):
        before = BEFORE_REPAIR_METRICS[scale]
        after = after_metrics[scale]
        comparisons.append(
            {
                "scale": scale,
                "before": before,
                "after": after,
                "delta": {
                    key: after[key] - before[key] for key in sorted(before)
                },
            }
        )
    suite_by_label = {row["label"]: row for row in tests["suites"]}
    after_controls = run1["scientific"]["negative_controls"]
    return {
        "status": (
            "PASS"
            if tests["status"] == "PASS"
            and after_controls["status"] == "PASS"
            and after_controls["control_count"] == 31
            else "FAIL"
        ),
        "before_repair_head": BEFORE_REPAIR_HEAD,
        "after_code_commit": (
            "7ed4749f06fdf39517858b456fda9cd724d6579e"
        ),
        "diagnostic_comparability": (
            "SAME_VERSIONED_FIXTURES_AND_SCALES_DIFFERENT_PUBLICATION_RUN"
        ),
        "controlled_benchmark_claimed": False,
        "metrics_excluded_from_scientific_hash": True,
        "scale_comparisons": comparisons,
        "validation": {
            "before": {
                "status": "PASS_WITH_UNTESTED_ADJACENCY_GAP",
                "negative_controls": "30/30 PASS",
                "program_order_same_count_wrong_adjacency": "NOT_TESTED",
                "hardening_scale_tests": "11 passed",
                "frozen_core_tests": "33 passed",
                "full_repository_tests": "121 passed, 5 skipped",
            },
            "after": {
                "status": "PASS",
                "negative_controls": (
                    f"{after_controls['passed_count']}/"
                    f"{after_controls['control_count']} PASS"
                ),
                "program_order_same_count_wrong_adjacency": (
                    "FAIL_CLOSED:"
                    "PROGRAM_ORDER_ADJACENCY_SET_MISMATCH"
                ),
                "hardening_scale_tests": (
                    f"{suite_by_label['hardening_scale_v1']['counts']['passed']}"
                    " passed"
                ),
                "frozen_core_tests": (
                    f"{suite_by_label['frozen_core']['counts']['passed']}"
                    " passed"
                ),
                "full_repository_tests": (
                    f"{suite_by_label['full_repository']['counts']['passed']}"
                    " passed, "
                    f"{suite_by_label['full_repository']['counts']['skipped']}"
                    " skipped"
                ),
            },
        },
    }


def _scale_by_name(run: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        row for row in run["scientific"]["scale_results"] if row["scale"] == name
    )


def _artifact_payloads(
    run1: dict[str, Any],
    run2: dict[str, Any],
    optional_guard_result: dict[str, Any],
    protected: dict[str, Any],
    tests: dict[str, Any],
) -> dict[str, Any]:
    scientific1 = run1["scientific"]
    small = _scale_by_name(run1, "small")
    medium = _scale_by_name(run1, "medium")
    large = _scale_by_name(run1, "large")
    isolation_source = source_isolation_audit()
    isolation = {
        "status": (
            "PASS"
            if isolation_source["status"] == "PASS"
            and all(
                row["process_isolation"]["status"] == "PASS"
                for row in scientific1["scale_results"]
            )
            else "FAIL"
        ),
        "source_isolation": isolation_source,
        "scale_process_isolation": [
            {
                "scale": row["scale"],
                **row["process_isolation"],
            }
            for row in scientific1["scale_results"]
        ],
    }
    output = {
        "status": (
            "PASS"
            if all(
                row["ordinary_output"]["status"] == "PASS"
                for row in scientific1["scale_results"]
            )
            else "FAIL"
        ),
        "capture_modes": [
            "relation_capture_disabled",
            "primitive_capture_enabled",
            "full_relation_resolution_enabled",
        ],
        "scale_results": [
            {
                "scale": row["scale"],
                **row["ordinary_output"],
            }
            for row in scientific1["scale_results"]
        ],
    }
    determinism = {
        "status": (
            "PASS"
            if run1["scientific_sha256"] == run2["scientific_sha256"]
            and run1["scientific"] == run2["scientific"]
            else "FAIL"
        ),
        "run_1_scientific_sha256": run1["scientific_sha256"],
        "run_2_scientific_sha256": run2["scientific_sha256"],
        "scientific_results_byte_equal": (
            json.dumps(run1["scientific"], sort_keys=True, separators=(",", ":"))
            == json.dumps(
                run2["scientific"], sort_keys=True, separators=(",", ":")
            )
        ),
        "diagnostic_exclusions": [
            "elapsed_seconds",
            "peak_rss_bytes",
            "available_bytes_at_guard",
            "process_id",
            "temporary_path",
        ],
        "exclusion_count": 5,
    }
    acceptance = {
        "protected_paths_unchanged": protected["status"] == "PASS",
        "primitive_semantic_validators_pass": scientific1[
            "primitive_semantic_validation"
        ]["status"]
        == "PASS",
        "capture_completeness_machine_audited": small[
            "capture_scope_statuses"
        ][0]["status"]
        == "CAPTURE_COMPLETE",
        "reads_from_controlled_fixture": scientific1["reads_from_versions"][
            "reads_from_relation_count"
        ]
        == 1,
        "multi_fact_selective_lifting": not scientific1[
            "multi_fact_occurrence"
        ]["false_dependency_pairs"],
        "strict_run_identity_distinguished": not scientific1[
            "run_identity_comparison"
        ]["concrete_run_scoped_gamma_equal"],
        "small_exact": small["comparison"]["mismatch_count"] == 0,
        "medium_exact": medium["comparison"]["mismatch_count"] == 0,
        "large_occurrence_floor": large["occurrence_count"] >= 10_000,
        "large_fact_floor": 30_000 <= large["fact_count"] <= 50_000,
        "large_query_floor": large["query_count"] >= 20_000,
        "large_fp_zero": large["comparison"]["false_positive_count"] == 0,
        "large_fn_zero": large["comparison"]["false_negative_count"] == 0,
        "large_no_full_closure": not large["candidate_metrics"][
            "full_transitive_closure_materialized"
        ],
        "thirty_one_negative_controls": scientific1["negative_controls"][
            "passed_count"
        ]
        == 31,
        "ordinary_output_orthogonal": output["status"] == "PASS",
        "process_isolation": isolation["status"] == "PASS",
        "scientific_determinism": determinism["status"] == "PASS",
        "tests_no_new_failure": tests["status"] == "PASS",
    }
    overall_pass = all(acceptance.values())
    summary = {
        "status": (
            "INTER_FACT_RELATIONS_HARDENING_SCALE_V1_SUPPORTED"
            if overall_pass
            else "INTER_FACT_RELATIONS_HARDENING_SCALE_V1_FAILED"
        ),
        "acceptance": acceptance,
        "scientific_sha256": run1["scientific_sha256"],
        "core_modification_required": False,
        "formal_theory_status": "NOT_PART_OF_FROZEN_THEORY",
        "claim_atlas_status": "NOT_ADDED",
        "exact_gamma_equality": scientific1["run_identity_comparison"][
            "exact_gamma_equality_status"
        ],
        "large": {
            "occurrence_count": large["occurrence_count"],
            "fact_count": large["fact_count"],
            "primitive_relation_count": large["primitive_relation_count"],
            "query_count": large["query_count"],
            "false_positive_count": large["comparison"][
                "false_positive_count"
            ],
            "false_negative_count": large["comparison"][
                "false_negative_count"
            ],
            "full_closure_materialized": large["candidate_metrics"][
                "full_transitive_closure_materialized"
            ],
        },
    }
    return {
        "baseline_environment.json": baseline_environment(),
        "protected_path_audit.json": protected,
        "primitive_semantic_validation.json": scientific1[
            "primitive_semantic_validation"
        ],
        "capture_completeness_audit.json": {
            "status": "PASS",
            "audits": run1["capture_audits"],
        },
        "run_identity_comparison.json": scientific1[
            "run_identity_comparison"
        ],
        "selective_lifting_results.json": scientific1[
            "multi_fact_occurrence"
        ],
        "small_full_comparison.json": small,
        "medium_query_comparison.json": medium,
        "large_query_comparison.json": large,
        "performance_metrics.json": {
            "status": "PASS",
            "diagnostics_excluded_from_scientific_hash": True,
            "scale_metrics": run1["diagnostics"],
            "optional_scale_guard": optional_guard_result,
        },
        "repair_before_after.json": repair_before_after(run1, tests),
        "output_orthogonality.json": output,
        "process_isolation_audit.json": isolation,
        "negative_controls.json": scientific1["negative_controls"],
        "determinism.json": determinism,
        "test_results.json": tests,
        "experiment_summary.json": summary,
    }


def _write_artifacts(payloads: dict[str, Any]) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, payload in sorted(payloads.items()):
        write_json(ARTIFACT_ROOT / name, payload)
    entries = []
    for name in sorted(payloads):
        path = ARTIFACT_ROOT / name
        entries.append(
            {
                "path": name,
                "byte_size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    independently_rehashed = all(
        (ARTIFACT_ROOT / row["path"]).stat().st_size == row["byte_size"]
        and file_sha256(ARTIFACT_ROOT / row["path"]) == row["sha256"]
        for row in entries
    )
    manifest = {
        "status": "PASS" if independently_rehashed else "FAIL",
        "declared_artifact_count": len(entries),
        "artifacts": entries,
        "independently_rehashed": independently_rehashed,
        "manifest_self_excluded": True,
        "canonical_json": True,
    }
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest)
    loaded = load_json(ARTIFACT_ROOT / "artifact_manifest.json")
    if loaded != manifest:
        raise RuntimeError("artifact manifest canonical round-trip failed")
    return manifest


def main() -> int:
    guard = optional_scale_guard()
    run1 = run_scientific(guard)
    run2 = run_scientific(guard)
    protected = protected_path_audit()
    tests = test_results()
    payloads = _artifact_payloads(
        run1, run2, guard, protected, tests
    )
    manifest = _write_artifacts(payloads)
    summary = payloads["experiment_summary.json"]
    result = {
        "status": summary["status"],
        "scientific_sha256": summary["scientific_sha256"],
        "artifact_count": manifest["declared_artifact_count"],
        "manifest_status": manifest["status"],
        "test_status": tests["status"],
        "large": summary["large"],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return (
        0
        if summary["status"]
        == "INTER_FACT_RELATIONS_HARDENING_SCALE_V1_SUPPORTED"
        and manifest["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
