from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..common import (
    BRANCH,
    EXPERIMENT_ROOT,
    REPAIRED_BASE_HEAD,
    REPOSITORY_ROOT,
    SCENARIOS,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    git,
    load_json,
    write_json,
)
from .negative_controls import run_negative_controls
from .scientific_runner import run_scientific
from .sqlite_runtime import INITIAL_STATE_PATH, SCHEMA_PATH


ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"
ALLOWED_NAMESPACE = "experiments/order_refund_freeze_inter_fact_relations_v1/"
PROTECTED_PATHS = (
    "src/generation_relation_core",
    "protocol/core_v3",
    "compat/v2",
    "tests/core",
    "experiments/inter_fact_relations_v0",
    "experiments/inter_fact_relations_v0_hardening_scale_v1",
    "claims",
    "claim_atlas",
)


def baseline_environment() -> dict[str, Any]:
    return {
        "status": "PASS",
        "repaired_base_head": REPAIRED_BASE_HEAD,
        "branch": BRANCH,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "diagnostic_environment_not_in_scientific_hash": True,
    }


def dependency_identity() -> dict[str, Any]:
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "opentelemetry-api",
            "opentelemetry-sdk",
            "pytest",
        )
    }
    return {
        "status": "PASS",
        "packages": packages,
        "requirements_lock_sha256": file_sha256(
            EXPERIMENT_ROOT / "requirements.lock"
        ),
        "official_opentelemetry_sdk_used": True,
    }


def scenario_protocol() -> dict[str, Any]:
    return {
        "status": "PASS",
        "scenarios": list(SCENARIOS),
        "initial_order": {
            "order_id": "order-001",
            "amount_cents": 5000,
            "status": "OPEN",
            "version": 7,
        },
        "capture_modes": ["disabled", "enabled"],
        "repeats_per_scenario_per_mode": 5,
        "minimum_real_workflow_executions": 40,
        "schedule_basis": "Barrier/Event gates; no wall-clock causality",
        "components": [
            "SQLite file",
            "WAL",
            "RefundWorker process",
            "FreezeWorker or duplicate RefundWorker process",
            "NotificationWorker process",
            "multiprocessing.Queue",
            "multiprocessing.Barrier",
            "multiprocessing.Event",
        ],
    }


def _git_lines(*args: str) -> list[str]:
    value = git(*args)
    return value.splitlines() if value else []


def protected_path_audit() -> dict[str, Any]:
    rows = []
    for path in PROTECTED_PATHS:
        committed = _git_lines(
            "diff", "--name-only", REPAIRED_BASE_HEAD, "HEAD", "--", path
        )
        worktree = _git_lines("diff", "--name-only", "--", path)
        untracked = _git_lines(
            "ls-files", "--others", "--exclude-standard", "--", path
        )
        rows.append(
            {
                "path": path,
                "committed_change_count": len(committed),
                "worktree_change_count": len(worktree),
                "untracked_change_count": len(untracked),
                "status": (
                    "PASS"
                    if not committed and not worktree and not untracked
                    else "FAIL"
                ),
            }
        )
    all_changes = sorted(
        set(
            _git_lines("diff", "--name-only", REPAIRED_BASE_HEAD, "HEAD")
            + _git_lines("diff", "--name-only")
            + _git_lines("ls-files", "--others", "--exclude-standard")
        )
    )
    namespace_only = all(
        path.startswith(ALLOWED_NAMESPACE) for path in all_changes
    )
    core_copy = any(
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
        "repaired_base_head": REPAIRED_BASE_HEAD,
        "allowed_namespace": ALLOWED_NAMESPACE,
        "changes_limited_to_allowed_namespace": namespace_only,
        "protected_paths": rows,
        "core_changed_files": 0
        if rows[0]["status"] == "PASS"
        and rows[1]["status"] == "PASS"
        and rows[2]["status"] == "PASS"
        and rows[3]["status"] == "PASS"
        else 1,
        "stage_a_experiment_changed_files": 0
        if next(
            row
            for row in rows
            if row["path"]
            == "experiments/inter_fact_relations_v0_hardening_scale_v1"
        )["status"]
        == "PASS"
        else 1,
        "experimental_core_copy_present": core_copy,
        "claim_atlas_changed": any(
            row["status"] != "PASS"
            for row in rows
            if row["path"] in {"claims", "claim_atlas"}
        ),
    }


def _pytest_counts(output: str) -> dict[str, int]:
    values = {}
    for name in ("passed", "failed", "skipped", "xfailed", "xpassed", "error"):
        match = re.search(rf"(\d+)\s+{name}", output)
        values[name] = int(match.group(1)) if match else 0
    return values


def _run_pytest(label: str, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "-ra"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    combined = "\n".join(
        item for item in (completed.stdout, completed.stderr) if item
    )
    normalized = re.sub(r"\s+in\s+\d+(?:\.\d+)?s", "", combined)
    normalized = re.sub(r"\s+in\s+\d+:\d+:\d+", "", normalized)
    lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
    return {
        "label": label,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "return_code": completed.returncode,
        "counts": _pytest_counts(combined),
        "normalized_output_tail": "\n".join(lines[-50:]),
    }


def test_results() -> dict[str, Any]:
    suites = [
        _run_pytest(
            "order_refund_freeze_experiment",
            ["experiments/order_refund_freeze_inter_fact_relations_v1/tests"],
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


def _payloads(
    run1: dict[str, Any],
    run2: dict[str, Any],
    controls: dict[str, Any],
    tests: dict[str, Any],
    protected: dict[str, Any],
) -> dict[str, Any]:
    scientific = run1["scientific"]
    deterministic = (
        run1["scientific_sha256"] == run2["scientific_sha256"]
        and canonical_bytes(run1["scientific"])
        == canonical_bytes(run2["scientific"])
    )
    determinism = {
        "status": "PASS" if deterministic else "FAIL",
        "run_1_scientific_sha256": run1["scientific_sha256"],
        "run_2_scientific_sha256": run2["scientific_sha256"],
        "canonical_bytes_equal": deterministic,
        "diagnostic_exclusions": [
            "SQLite DB binary hash",
            "SQLite WAL binary hash and salt",
            "process ID",
            "temporary path",
            "wall clock",
            "test elapsed time",
        ],
    }
    query_comparison = scientific["query_comparison"]
    witness = scientific["paired_business_view_witness"]
    q07_rows = [
        row
        for row in query_comparison["comparisons"]
        if row["query_id"] == "Q07"
    ]
    q14_d = next(
        row
        for row in query_comparison["comparisons"]
        if row["query_id"] == "Q14"
        and row["scenario"] == "IDEMPOTENT_DUPLICATE_REFUND"
    )
    transaction_rows = [
        receipt
        for scenario in scientific["transaction_receipts"]
        for receipt in scenario["receipts"]
    ]
    queue_rows = [
        receipt
        for scenario in scientific["queue_receipts"]
        for receipt in scenario["receipts"]
    ]
    sync_rows = [
        receipt
        for scenario in scientific["synchronization_receipts"]
        for receipt in scenario["receipts"]
    ]
    acceptance = {
        "real_sqlite_wal": all(
            identity["wal_present"]
            for identity in run1["diagnostics"]["sqlite_binary_identities"]
        ),
        "real_multiprocessing": all(
            row["process_count"] >= 4
            for row in scientific["run_manifest"]["runs"]
        ),
        "real_queue_put_get": (
            any(row["operation"] == "put" for row in queue_rows)
            and any(row["operation"] == "get" for row in queue_rows)
        ),
        "real_barrier_and_event": (
            any(row["sync_type"] == "Barrier" for row in sync_rows)
            and any(row["sync_type"] == "Event" for row in sync_rows)
        ),
        "real_commit_and_rollback": (
            any(
                row["transaction_outcome"] == "COMMIT"
                for row in transaction_rows
            )
            and any(
                row["transaction_outcome"] == "ROLLBACK"
                for row in transaction_rows
            )
        ),
        "forty_real_workflow_executions": (
            scientific["run_manifest"]["real_workflow_execution_count"] >= 40
        ),
        "capture_orthogonality": (
            scientific["business_output_orthogonality"]["status"] == "PASS"
        ),
        "capture_complete": all(
            row["status"] == "CAPTURE_COMPLETE"
            for row in scientific["capture_completeness_audits"]
        ),
        "candidate_reference_exact": query_comparison["status"] == "PASS",
        "false_positive_zero": query_comparison["false_positive_count"] == 0,
        "false_negative_zero": query_comparison["false_negative_count"] == 0,
        "paired_business_view_equal": witness[
            "ordinary_business_view_equal"
        ],
        "paired_formation_answer_different": not witness[
            "formation_answer_equal"
        ],
        "notification_sent_origin_is_refund_committed": all(
            row["candidate_answer"]
            in {
                "RefundCommitted",
                "NOT_APPLICABLE_NO_NOTIFICATION_SENT",
            }
            for row in q07_rows
        ),
        "idempotent_single_refund_and_notification": (
            q14_d["candidate_answer"]["refund_row_count"] == 1
            and q14_d["candidate_answer"]["notification_sent_count"] == 1
            and not q14_d["candidate_answer"]["second_refund_formed"]
            and not q14_d["candidate_answer"]["second_notification_formed"]
        ),
        "native_trace_profile_honest": all(
            export["forbidden_sidecar_field_count"] == 0
            and export["profile"] == "CONVENTIONAL_NATIVE_TRACE_V1"
            for export in scientific["native_trace_export"]
        ),
        "explicit_dispositions_complete": scientific[
            "explicit_disposition_results"
        ]["missing_disposition_count"]
        == 0,
        "process_isolation": scientific["process_isolation_audit"]["status"]
        == "PASS",
        "thirty_negative_controls": controls["passed_count"] == 30,
        "scientific_determinism": determinism["status"] == "PASS",
        "tests_no_new_failure": tests["status"] == "PASS",
        "protected_paths_unchanged": protected["status"] == "PASS",
    }
    summary = {
        "status": (
            "ORDER_REFUND_FREEZE_INTER_FACT_RELATIONS_V1_SUPPORTED"
            if all(acceptance.values())
            else "ORDER_REFUND_FREEZE_INTER_FACT_RELATIONS_V1_FAILED"
        ),
        "acceptance": acceptance,
        "scientific_sha256": run1["scientific_sha256"],
        "query_count": query_comparison["query_count"],
        "false_positive_count": query_comparison["false_positive_count"],
        "false_negative_count": query_comparison["false_negative_count"],
        "core_modification_required": False,
        "formal_theory_status": "NOT_PART_OF_FROZEN_THEORY",
        "claim_atlas_status": "NOT_ADDED",
        "concurrency_scope": "CONTROLLED_CAPTURE_SCOPE_ONLY",
        "general_concurrency_theory_established": False,
        "sqlite_or_otel_replacement_claimed": False,
    }
    sqlite_identities = run1["diagnostics"]["sqlite_binary_identities"]
    sqlite_identity = {
        "status": "PASS",
        "sqlite_version": sqlite3.sqlite_version,
        "journal_mode": "wal",
        "schema_sha256": file_sha256(SCHEMA_PATH),
        "initial_state_sha256": file_sha256(INITIAL_STATE_PATH),
        "representative_binary_identities": [
            {
                "scenario": scenario,
                **identity,
            }
            for scenario, identity in zip(SCENARIOS, sqlite_identities)
        ],
        "binary_hashes_excluded_from_scientific_hash": True,
    }
    return {
        "baseline_environment.json": baseline_environment(),
        "sqlite_identity.json": sqlite_identity,
        "dependency_identity.json": dependency_identity(),
        "scenario_protocol.json": scenario_protocol(),
        "run_manifest.json": scientific["run_manifest"],
        "business_output_orthogonality.json": scientific[
            "business_output_orthogonality"
        ],
        "transaction_receipts.json": scientific["transaction_receipts"],
        "queue_receipts.json": scientific["queue_receipts"],
        "synchronization_receipts.json": scientific[
            "synchronization_receipts"
        ],
        "atomic_generation_facts.json": scientific[
            "atomic_generation_facts"
        ],
        "primitive_relation_sidecar.json": scientific[
            "primitive_relation_sidecars"
        ],
        "capture_completeness_audit.json": scientific[
            "capture_completeness_audits"
        ],
        "native_trace_export.json": scientific["native_trace_export"],
        "native_trace_answerability.json": scientific[
            "native_trace_answerability"
        ],
        "atomic_fact_answerability.json": scientific[
            "atomic_fact_answerability"
        ],
        "candidate_answers.json": scientific["candidate_answers"],
        "reference_answers.json": scientific["reference_answers"],
        "query_comparison.json": query_comparison,
        "result_impact_analysis.json": scientific[
            "result_impact_analysis"
        ],
        "paired_business_view_witness.json": witness,
        "explicit_disposition_results.json": scientific[
            "explicit_disposition_results"
        ],
        "process_isolation_audit.json": scientific[
            "process_isolation_audit"
        ],
        "negative_controls.json": controls,
        "determinism.json": determinism,
        "test_results.json": tests,
        "protected_path_audit.json": protected,
        "canonical_table_dumps.json": scientific["canonical_table_dumps"],
        "experiment_summary.json": summary,
    }


def _write_artifacts(payloads: dict[str, Any]) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, payload in sorted(payloads.items()):
        write_json(ARTIFACT_ROOT / name, payload)
    rows = [
        {
            "path": name,
            "byte_size": (ARTIFACT_ROOT / name).stat().st_size,
            "sha256": file_sha256(ARTIFACT_ROOT / name),
        }
        for name in sorted(payloads)
    ]
    valid = all(
        (ARTIFACT_ROOT / row["path"]).stat().st_size == row["byte_size"]
        and file_sha256(ARTIFACT_ROOT / row["path"]) == row["sha256"]
        for row in rows
    )
    manifest = {
        "status": "PASS" if valid else "FAIL",
        "declared_artifact_count": len(rows),
        "artifacts": rows,
        "independently_rehashed": valid,
        "manifest_self_excluded": True,
        "canonical_json": True,
    }
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest)
    if load_json(ARTIFACT_ROOT / "artifact_manifest.json") != manifest:
        raise RuntimeError("ARTIFACT_MANIFEST_ROUND_TRIP_FAILURE")
    return manifest


def main() -> int:
    run1 = run_scientific()
    run2 = run_scientific()
    controls = run_negative_controls()
    protected = protected_path_audit()
    tests = test_results()
    payloads = _payloads(run1, run2, controls, tests, protected)
    manifest = _write_artifacts(payloads)
    summary = payloads["experiment_summary.json"]
    output = {
        "status": summary["status"],
        "scientific_sha256": summary["scientific_sha256"],
        "artifact_count": manifest["declared_artifact_count"] + 1,
        "manifest_status": manifest["status"],
        "negative_controls": (
            f"{controls['passed_count']}/{controls['control_count']}"
        ),
        "query_count": summary["query_count"],
        "false_positive_count": summary["false_positive_count"],
        "false_negative_count": summary["false_negative_count"],
        "test_status": tests["status"],
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return (
        0
        if summary["status"]
        == "ORDER_REFUND_FREEZE_INTER_FACT_RELATIONS_V1_SUPPORTED"
        and manifest["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
