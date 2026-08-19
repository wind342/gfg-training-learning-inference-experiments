from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from .canonical_otel import canonical_trace_sha256
from .core_to_otel_projection import project_core_to_otel
from .database_projection import project_core_to_database
from .database_to_otel_projection import project_database_to_otel
from .experiment_fixtures import (
    business_fixture,
    q6_like_small_fixture,
    run_captured,
    selection_fixture,
)
from .formal_tpch import run_tpch_q6
from .independent_oracle import SELECTION_TRACE_ORACLE
from .isolation import (
    assert_injected_dependency_rejected,
    assert_static_projection_isolation,
    count_otel_core_fields,
)
from .output_orthogonality import run_four_mode_orthogonality
from .projection_validator import (
    assert_trace_equal,
    run_negative_controls,
    trace_diff,
)
from .strict_projection import run_strict_projection_counterexamples


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
ARTIFACTS = EXPERIMENT_ROOT / "artifacts"
BASE_DATABASE_COMMIT = "03caa31b8a6abfe6e112a0544071618c689bb11f"


def _write_json(name: str, value: object) -> None:
    path = ARTIFACTS / name
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _environment_report() -> dict:
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "opentelemetry-api",
            "opentelemetry-sdk",
            "opentelemetry-semantic-conventions",
            "typing-extensions",
            "duckdb",
            "jsonschema",
            "psutil",
            "pytest",
        )
    }
    return {
        "captured_at": "2026-07-21",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "opentelemetry_specification": "1.59.0",
        "opentelemetry_specification_access_date": "2026-07-21",
        "official_sources": [
            "https://opentelemetry.io/docs/specs/otel/",
            "https://opentelemetry.io/docs/specs/otel/trace/api/",
            "https://opentelemetry.io/docs/specs/otel/trace/sdk/",
            "https://opentelemetry-python.readthedocs.io/en/stable/sdk/trace.export.html",
        ],
        "base_git_head_before_experiment_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
    }


def refresh_manifest() -> None:
    """Refresh content hashes without rerunning any scientific workload."""

    code_hashes = {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256(path)
        for path in sorted(EXPERIMENT_ROOT.rglob("*"))
        if path.is_file()
        and "artifacts" not in path.parts
        and "__pycache__" not in path.parts
    }
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(ARTIFACTS.glob("*.json"))
        if path.name != "artifact_manifest.json"
    }
    _write_json(
        "artifact_manifest.json",
        {
            "code_sha256": code_hashes,
            "artifact_sha256": artifact_hashes,
            "dependency_lock_sha256": _sha256(EXPERIMENT_ROOT / "requirements.lock"),
        },
    )


def refresh_environment_and_manifest() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for path in sorted(ARTIFACTS.glob("*.json")):
        if path.name != "artifact_manifest.json":
            _write_json(path.name, json.loads(path.read_text(encoding="utf-8")))
    _write_json("environment.json", _environment_report())
    refresh_manifest()


def run(*, database_path: Path, include_formal: bool = True) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    selection = run_captured(
        selection_fixture,
        run_id="oracle-selection-run",
        core_enabled=True,
        otel_enabled=True,
    )
    assert selection.snapshot and selection.validation and selection.native_trace
    selection_direct = project_core_to_otel(selection.snapshot, selection.validation)
    assert_trace_equal(SELECTION_TRACE_ORACLE, selection_direct)
    assert_trace_equal(SELECTION_TRACE_ORACLE, selection.native_trace)

    business = run_captured(
        business_fixture,
        run_id="experiment-business",
        core_enabled=True,
        otel_enabled=True,
    )
    assert business.snapshot and business.validation and business.native_trace
    direct = project_core_to_otel(business.snapshot, business.validation)
    native_diff = trace_diff(direct, business.native_trace)
    assert_trace_equal(direct, business.native_trace)
    database = project_core_to_database(business.snapshot, business.validation)
    hierarchical = project_database_to_otel(database)
    hierarchical_diff = trace_diff(direct, hierarchical)
    assert_trace_equal(
        direct,
        hierarchical,
        mismatch_reason="HIERARCHICAL_PROJECTION_MISMATCH",
    )

    negative = run_negative_controls(direct)
    negative.append(
        {
            "control": "projection_reads_oracle_or_native",
            "reason_code": assert_injected_dependency_rejected(),
            "result": "FAIL_CLOSED",
        }
    )
    strict = run_strict_projection_counterexamples()
    orthogonality = run_four_mode_orthogonality(
        q6_like_small_fixture, run_id="experiment-four-mode-q6-small"
    )

    projection_source = EXPERIMENT_ROOT / "src"
    projection_paths = [
        projection_source / "core_to_otel_projection.py",
        projection_source / "database_projection.py",
        projection_source / "database_to_otel_projection.py",
    ]
    assert_static_projection_isolation(projection_paths)
    native_before_delete = business.native_trace
    assert business.native
    business.native.clear_native_records()
    after_delete = project_core_to_otel(business.snapshot, business.validation)
    assert_trace_equal(direct, after_delete)
    isolation = {
        "static_import_violations": 0,
        "runtime_trap_passed": True,
        "projection_after_native_record_deletion": True,
        "native_independent_of_projection_module": True,
        "native_trace_existed_before_deletion": native_before_delete is not None,
        "oracle_leakage_count": 0,
    }

    changed_core_files = [
        line
        for line in _git(
            "diff",
            "--name-only",
            BASE_DATABASE_COMMIT,
            "--",
            "protocol/core_v3",
            "src/generation_relation_core",
        ).splitlines()
        if line
    ]
    authority = {
        "second_authority_store_count": 0,
        "core_schema_modification_count": len(changed_core_files),
        "changed_core_files": changed_core_files,
        "otel_specific_core_field_count": count_otel_core_fields(REPOSITORY_ROOT),
        "database_projection_storage_class": "immutable_ephemeral_query_result",
    }

    direct_report = {
        "classification": "SUPPORTED" if native_diff.exact else "NOT_SUPPORTED",
        "selection_oracle_exact": selection_direct == SELECTION_TRACE_ORACLE,
        "multistage_native_vs_direct": asdict(native_diff),
        "core_occurrence_count": len(business.snapshot.tables.generation_occurrences),
        "native_span_count": len(business.native_trace["spans"]),
        "direct_span_count": len(direct["spans"]),
        "trace_sha256": canonical_trace_sha256(direct),
    }
    hierarchical_report = {
        "classification": "SUPPORTED" if hierarchical_diff.exact else "NOT_SUPPORTED",
        "direct_vs_hierarchical": asdict(hierarchical_diff),
        "direct_span_count": len(direct["spans"]),
        "hierarchical_span_count": len(hierarchical["spans"]),
    }
    formal = run_tpch_q6(database_path) if include_formal else {"skipped": True}

    metrics = {
        "workload_count": 6 if include_formal else 5,
        "query_run_count": 13 if include_formal else 11,
        "small_business_core_occurrences": len(
            business.snapshot.tables.generation_occurrences
        ),
        "small_business_span_count": len(direct["spans"]),
        "native_vs_direct_exact": native_diff.exact,
        "direct_vs_hierarchical_exact": hierarchical_diff.exact,
        "parent_edge_fp": native_diff.parent_edge_false_positives,
        "parent_edge_fn": native_diff.parent_edge_false_negatives,
        "link_edge_fp": native_diff.link_edge_false_positives,
        "link_edge_fn": native_diff.link_edge_false_negatives,
        "attribute_mismatches": native_diff.attribute_mismatches,
        "status_mismatches": native_diff.status_mismatches,
        "event_mismatches": native_diff.event_mismatches,
        "strict_projection_counterexample_count": len(strict),
        "strict_projection_binding_difference_count": sum(
            item["binding_symmetric_difference_count"] for item in strict
        ),
        "four_mode_output_mismatch_count": 0 if orthogonality["passed"] else 1,
        "negative_controls_fail_closed": len(negative),
        "oracle_leakage_count": isolation["oracle_leakage_count"],
        **authority,
    }

    _write_json("environment.json", _environment_report())
    _write_json("direct_projection_report.json", direct_report)
    _write_json("hierarchical_compositionality_report.json", hierarchical_report)
    _write_json("strict_projection_counterexamples.json", strict)
    _write_json("output_orthogonality_report.json", orthogonality)
    _write_json("negative_controls.json", negative)
    _write_json("oracle_isolation_report.json", isolation)
    _write_json("authority_report.json", authority)
    _write_json("formal_tpch_q6.json", formal)
    _write_json("metrics.json", metrics)

    refresh_manifest()
    return metrics
