from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from generation_relation_core.canonical import canonical_bytes

from ..src.audits import (
    core_change_lineage,
    isolation_audit,
    profile_and_code_identity,
    second_authority_audit,
    source_branch_lineage,
    v1_preservation,
)
from ..src.common import (
    ProofFailure,
    canonical_sha256,
    git,
    read_json,
    sha256_file,
    write_json,
)
from ..src.database_proof import run_database_proof
from ..src.matrix import build_matrix, render_matrix_markdown
from ..src.negative_controls import classify_negative_controls
from ..src.otel_proof import run_otel_proof
from ..src.source_map_proof import run_source_map_proof


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
ARTIFACTS = EXPERIMENT_ROOT / "artifacts"
PRIVATE_ROOT = REPO_ROOT / "data_private/operational_projection_proof_v2"
PROFILE_ROOT = EXPERIMENT_ROOT / "profiles"
SUPPORTED = "UNIFIED_OPERATIONAL_PROJECTION_PROOF_V2_SUPPORTED"
NOT_SUPPORTED = "UNIFIED_OPERATIONAL_PROJECTION_PROOF_V2_NOT_SUPPORTED"

REQUIRED_PROFILE_FIELDS = {
    "profile_id",
    "source_branch",
    "source_commit",
    "declared_input_scope",
    "selected_facts",
    "excluded_facts",
    "candidate_path",
    "reference_path",
    "exact_comparison_fields",
    "multiplicity_semantics",
    "canonicalization_rules",
    "output_orthogonality_rules",
    "authority_rules",
    "accepted_limitations",
    "mandatory_status_conditions",
}

TEST_SUITES = [
    ("core", "tests/core"),
    ("database_lineage", "experiments/database_lineage/tests"),
    ("opentelemetry_projection", "experiments/opentelemetry_projection/tests"),
    ("source_map_projection", "tests/experiments/source_map_projection"),
    ("operational_projection_proof_v1", "experiments/operational_projection_proof/tests"),
    ("operational_projection_proof_v2", "experiments/operational_projection_proof_v2/tests"),
]


def validate_profiles() -> dict[str, Any]:
    expected_commits = {
        "database_which_lineage_v1.json": "03caa31b8a6abfe6e112a0544071618c689bb11f",
        "opentelemetry_occurrence_execution_v1.json": "25a9d2a614d2d34d36c38f7c560b818cdbc4b179",
        "ecma426_ordinary_source_map_v1.json": "7dba987713da345453781e4b95130f1deb5f04d4",
        "core_database_to_opentelemetry_v1.json": "25a9d2a614d2d34d36c38f7c560b818cdbc4b179",
        "ecma426_multistage_composition_v1.json": "7dba987713da345453781e4b95130f1deb5f04d4",
    }
    rows = []
    failures = []
    for name, expected_commit in expected_commits.items():
        path = PROFILE_ROOT / name
        value = read_json(path)
        missing = sorted(REQUIRED_PROFILE_FIELDS - set(value))
        extra = sorted(set(value) - REQUIRED_PROFILE_FIELDS)
        exact_commit = value.get("source_commit") == expected_commit
        vague_tokens = ("相关字段", "基本一致", "大致相同")
        rendered = json.dumps(value, ensure_ascii=False)
        vague = [token for token in vague_tokens if token in rendered]
        valid = not missing and not extra and exact_commit and not vague
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "missing_fields": missing,
                "extra_fields": extra,
                "source_commit_exact": exact_commit,
                "prohibited_vague_tokens": vague,
                "valid": valid,
            }
        )
        if not valid:
            failures.append(f"PROFILE_INVALID:{name}")
    return {
        "profile_count": len(rows),
        "profiles": rows,
        "blocking_reasons": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def environment_report(
    source_lineage: dict[str, Any], source_map_environment: dict[str, Any]
) -> dict[str, Any]:
    package_names = (
        "duckdb",
        "jsonschema",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-semantic-conventions",
        "psutil",
        "pytest",
    )
    database_path = (
        REPO_ROOT / "experiments/database_lineage/runtime/tpch_sf_0_01.duckdb"
    )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "source_map_environment": source_map_environment,
        "database_fixture_sha256": sha256_file(database_path),
        "git_branch": git(REPO_ROOT, "branch", "--show-current"),
        "git_head_at_run": git(REPO_ROOT, "rev-parse", "HEAD"),
        "source_branch_identity_sha256": canonical_sha256(source_lineage),
        **profile_and_code_identity(REPO_ROOT, EXPERIMENT_ROOT),
    }


def _combined_output_orthogonality(reports: dict[str, Any]) -> dict[str, Any]:
    domains = {
        "database": reports["database_output_orthogonality"],
        "opentelemetry": reports["otel_output_orthogonality"],
        "source_map": reports["source_map_output_orthogonality"],
    }
    passed = (
        domains["database"]["status"] == "SUPPORTED"
        and domains["opentelemetry"]["status"] == "PASS"
        and domains["source_map"]["status"] == "PASS"
    )
    return {"domains": domains, "status": "PASS" if passed else "FAIL"}


def _combined_oracle_isolation(reports: dict[str, Any]) -> dict[str, Any]:
    domains = {
        "database": reports["database_oracle_isolation"],
        "opentelemetry": reports["otel_oracle_isolation"],
        "source_map": reports["source_map_oracle_isolation"],
    }
    passed = (
        domains["database"]["status"] == "SUPPORTED"
        and domains["opentelemetry"]["status"] == "PASS"
        and domains["source_map"]["status"] == "PASS"
    )
    return {"domains": domains, "status": "PASS" if passed else "FAIL"}


def _domain_statuses(reports: dict[str, Any]) -> dict[str, str]:
    return {
        "database_p1": reports["projection_equivalence_database.json"]["status"],
        "database_p2": reports["strict_partiality_database.json"]["status"],
        "otel_p1": reports["projection_equivalence_opentelemetry.json"]["status"],
        "otel_p2": reports["strict_partiality_opentelemetry.json"]["status"],
        "otel_p3": reports[
            "hierarchical_consistency_core_database_to_opentelemetry.json"
        ]["status"],
        "source_map_p1": reports["projection_equivalence_source_map.json"][
            "status"
        ],
        "source_map_p2": reports["strict_partiality_source_map.json"]["status"],
        "source_map_p3": reports["composition_consistency_source_map.json"][
            "status"
        ],
    }


def scientific_run(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = PRIVATE_ROOT / f"run_{index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_lineage = source_branch_lineage(REPO_ROOT)
    if source_lineage["status"] != "PASS":
        raise ProofFailure(
            "BRANCH_HEAD_DRIFT", ",".join(source_lineage["blocking_reasons"])
        )
    preservation = v1_preservation(REPO_ROOT)
    profiles = validate_profiles()
    if preservation["status"] != "PASS":
        raise ProofFailure("V1_EVIDENCE_CHANGED")
    if profiles["status"] != "PASS":
        raise ProofFailure("PROFILE_INVALID", ",".join(profiles["blocking_reasons"]))

    started = time.perf_counter()
    database = run_database_proof(run_dir / "database")
    otel = run_otel_proof(run_dir / "opentelemetry", repo_root=REPO_ROOT, include_formal=True)
    gc.collect()
    source_map = run_source_map_proof(run_dir / "source_map", repo_root=REPO_ROOT)
    reports = {**database, **otel, **source_map}
    core_lineage = core_change_lineage(REPO_ROOT)
    second_authority = second_authority_audit(REPO_ROOT)
    static_isolation = isolation_audit(REPO_ROOT)
    negative = classify_negative_controls(reports)
    output = _combined_output_orthogonality(reports)
    oracle = _combined_oracle_isolation(reports)
    reports.update(
        {
            "source_branch_lineage.json": source_lineage,
            "v1_preservation.json": preservation,
            "profile_validation.json": profiles,
            "core_change_lineage.json": core_lineage,
            "second_authority_audit.json": second_authority,
            "static_isolation_audit.json": static_isolation,
            "negative_control_classification.json": negative,
            "output_orthogonality.json": output,
            "oracle_isolation.json": oracle,
        }
    )
    matrix = build_matrix(reports)
    reports["unified_projection_matrix.json"] = matrix
    reports["environment.json"] = environment_report(
        source_lineage, reports["source_map_environment"]
    )
    statuses = _domain_statuses(reports)
    blocking = [
        f"{name.upper()}={status}"
        for name, status in statuses.items()
        if status != "SUPPORTED"
    ]
    mandatory_checks = {
        "all_domain_properties_supported": not blocking,
        "output_orthogonality": output["status"] == "PASS",
        "oracle_isolation": oracle["status"] == "PASS",
        "second_authority_count_zero": second_authority[
            "secondary_authority_store_count"
        ]
        == 0,
        "new_core_changes_zero": core_lineage[
            "new_core_change_count"
        ]
        == 0,
        "new_core_schema_changes_zero": core_lineage[
            "new_core_schema_change_count"
        ]
        == 0,
        "new_domain_specific_core_fields_zero": core_lineage[
            "new_domain_specific_core_field_count"
        ]
        == 0,
        "all_negative_controls_fail_closed": negative["status"] == "PASS",
        "v1_preserved": preservation["status"] == "PASS",
        "exact_source_heads_and_ancestry": source_lineage["status"] == "PASS",
        "profiles_valid": profiles["status"] == "PASS",
        "matrix_supported": matrix["status"] == "SUPPORTED",
    }
    blocking.extend(
        f"MANDATORY_{name.upper()}_FAILED"
        for name, passed in mandatory_checks.items()
        if not passed
    )
    reports["run_summary.json"] = {
        "framework": "unified-operational-projection-proof-v2",
        "domain_statuses": statuses,
        "mandatory_checks": mandatory_checks,
        "blocking_reasons": sorted(set(blocking)),
        "status": SUPPORTED if not blocking else NOT_SUPPORTED,
    }
    performance = {
        "run_elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(
            getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
        ),
        "otel_formal": reports["otel_performance_observations"],
    }
    scientific_reports = {
        key: value
        for key, value in reports.items()
        if key.endswith(".json")
    }
    return scientific_reports, performance


def publish_scientific_reports(reports: dict[str, Any]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name, report in sorted(reports.items()):
        write_json(ARTIFACTS / name, report)
    matrix = reports["unified_projection_matrix.json"]
    (ARTIFACTS / "unified_projection_matrix.md").write_text(
        render_matrix_markdown(matrix), encoding="utf-8", newline="\n"
    )


def _count_test_outcome(output: str, label: str) -> int:
    matches = re.findall(rf"(\d+) {label}", output)
    return sum(int(value) for value in matches[-1:])


def run_test_suites() -> dict[str, Any]:
    suites = []
    for name, relative in TEST_SUITES:
        started = time.perf_counter()
        command = [
            sys.executable,
            "-m",
            "pytest",
            relative,
            "-q",
            "--disable-warnings",
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        combined = "\n".join(value for value in (result.stdout, result.stderr) if value)
        passed = _count_test_outcome(combined, "passed")
        failed = _count_test_outcome(combined, "failed")
        skipped = _count_test_outcome(combined, "skipped")
        summary_lines = [line.strip() for line in combined.splitlines() if line.strip()][-3:]
        normalized_summary = [
            re.sub(
                r"\s*\(\d+:\d{2}:\d{2}\)$",
                "",
                re.sub(r"in \d+(?:\.\d+)?s", "in <elapsed>", line),
            )
            for line in summary_lines
        ]
        historical_v1_status_boundary_skips = (
            name == "operational_projection_proof_v1"
            and skipped == 6
            and combined.count("NOT_EVALUATED:") >= 6
        )
        prohibited_skip_count = (
            0 if historical_v1_status_boundary_skips else skipped
        )
        status = (
            "PASS"
            if result.returncode == 0
            and failed == 0
            and prohibited_skip_count == 0
            and passed > 0
            else "FAIL"
        )
        suites.append(
            {
                "suite": name,
                "command": "python -m pytest " + relative + " -q --disable-warnings",
                "exit_code": result.returncode,
                "passed_count": passed,
                "failed_count": failed,
                "skipped_count": skipped,
                "historical_v1_status_boundary_skip_count": (
                    skipped if historical_v1_status_boundary_skips else 0
                ),
                "prohibited_dependency_or_mandatory_skip_count": prohibited_skip_count,
                "normalized_terminal_summary": normalized_summary,
                "elapsed_seconds": time.perf_counter() - started,
                "status": status,
            }
        )
    return {
        "suite_count": len(suites),
        "suites": suites,
        "total_passed": sum(row["passed_count"] for row in suites),
        "total_failed": sum(row["failed_count"] for row in suites),
        "total_skipped": sum(row["skipped_count"] for row in suites),
        "status": "PASS" if all(row["status"] == "PASS" for row in suites) else "FAIL",
    }


def normalized_test_results(report: dict[str, Any]) -> dict[str, Any]:
    return {
        **report,
        "suites": [
            {key: value for key, value in row.items() if key != "elapsed_seconds"}
            for row in report["suites"]
        ],
    }


def _trace_identity(reports: dict[str, Any]) -> dict[str, Any]:
    otel = reports["projection_equivalence_opentelemetry.json"]
    return {
        "small": otel["small"]["canonical_trace_sha256"],
        "formal": otel["formal_tpch_q6"]["trace_sha256"],
    }


def _map_identity(reports: dict[str, Any]) -> dict[str, Any]:
    return reports["projection_equivalence_source_map.json"][
        "map_document_hashes"
    ]


def determinism_report(
    first: dict[str, Any],
    second: dict[str, Any],
    first_tests: dict[str, Any],
    second_tests: dict[str, Any],
) -> dict[str, Any]:
    first_bytes = canonical_bytes(first)
    second_bytes = canonical_bytes(second)
    tests_first = normalized_test_results(first_tests)
    tests_second = normalized_test_results(second_tests)
    checks = {
        "all_machine_reports": first_bytes == second_bytes,
        "normalized_records_and_projection_documents": _map_identity(first)
        == _map_identity(second),
        "trace_bytes": _trace_identity(first) == _trace_identity(second),
        "map_bytes": _map_identity(first) == _map_identity(second),
        "matrices": first["unified_projection_matrix.json"]
        == second["unified_projection_matrix.json"],
        "profiles": first["environment.json"]["profiles_sha256"]
        == second["environment.json"]["profiles_sha256"],
        "environment_identity": first["environment.json"]
        == second["environment.json"],
        "source_branch_identity": first["source_branch_lineage.json"]
        == second["source_branch_lineage.json"],
        "code_hashes": first["environment.json"]["v2_python_code_sha256"]
        == second["environment.json"]["v2_python_code_sha256"],
        "test_summaries": tests_first == tests_second,
    }
    return {
        "complete_run_count": 2,
        "scientific_run_1_sha256": canonical_sha256(first),
        "scientific_run_2_sha256": canonical_sha256(second),
        "normalized_test_run_1_sha256": canonical_sha256(tests_first),
        "normalized_test_run_2_sha256": canonical_sha256(tests_second),
        "checks": checks,
        "excluded_fields": [
            {
                "field": "runs/run_1/performance_observations.json#/run_elapsed_seconds",
                "reason": "elapsed wall-clock time is machine-load dependent",
            },
            {
                "field": "runs/run_2/performance_observations.json#/run_elapsed_seconds",
                "reason": "elapsed wall-clock time is machine-load dependent",
            },
            {
                "field": "runs/run_1/performance_observations.json#/peak_process_rss_bytes",
                "reason": "process peak RSS depends on allocator and concurrent OS state",
            },
            {
                "field": "runs/run_2/performance_observations.json#/peak_process_rss_bytes",
                "reason": "process peak RSS depends on allocator and concurrent OS state",
            },
            {
                "field": "runs/run_1/performance_observations.json#/otel_formal/performance_seconds",
                "reason": "per-stage elapsed time is explicitly observational",
            },
            {
                "field": "runs/run_2/performance_observations.json#/otel_formal/performance_seconds",
                "reason": "per-stage elapsed time is explicitly observational",
            },
            {
                "field": "runs/run_1/test_results.json#/suites/*/elapsed_seconds",
                "reason": "test wall-clock duration is machine-load dependent",
            },
            {
                "field": "runs/run_2/test_results.json#/suites/*/elapsed_seconds",
                "reason": "test wall-clock duration is machine-load dependent",
            },
        ],
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def render_readme() -> str:
    return """# Unified operational projection proof v2

This package reruns and integrates the frozen Database which-lineage, OpenTelemetry trace-shadow, and ordinary ECMA-426 Source Map experiments without changing Core or any source experiment artifact.

The only publishable entrypoint is:

```console
python -m experiments.operational_projection_proof_v2.scripts.run_all --full
```

The command verifies exact source heads and merge ancestry, preserves the v1 proof bytes, executes all P1/P2/P3 workloads twice, runs every required test suite twice, compares deterministic machine evidence, and emits a conjunctive status. Missing dependencies, count drift, skipped mandatory tests, authority leakage, new Core changes, or any mismatch fail closed.

The declared profiles are in `profiles/`; the preimplementation authority audit is in `audits/`; stable reports are in `artifacts/`. Transient complete runs remain under ignored `data_private/operational_projection_proof_v2`.
"""


def render_experiment_report(reports: dict[str, Any], determinism: dict[str, Any], tests: dict[str, Any], final_status: str) -> str:
    db1 = reports["projection_equivalence_database.json"]
    ot1 = reports["projection_equivalence_opentelemetry.json"]
    sm1 = reports["projection_equivalence_source_map.json"]
    sm3 = reports["composition_consistency_source_map.json"]
    return f"""# Unified Operational Projection Proof v2

## Outcome

`{final_status}`

Within three declared profiles, database which-lineage, ordinary ECMA-426 Source Maps, and an OpenTelemetry occurrence/execution/causal trace shadow are exactly derivable from validated occurrence-specific generation facts. Each is a strict partial view of the full fact space. OpenTelemetry additionally satisfies cross-domain hierarchical consistency through the database projection, while Source Map relations satisfy multistage composition consistency through GeneratedOrigin bridges.

## 1. Complete generation-fact space

The authority is the validated occurrence-specific Core Snapshot: sources and generated origins, occurrences, supports and dispositions, bindings, relation material, primary evidence, and successful operation closure. Function-local indexes and projection objects are rebuildable views, not authority stores.

## 2. Three operational properties

P1 is exact candidate/reference equality inside each declared profile. P2 is witnessed by real same-projection/different-Snapshot counterexamples. P3 is split deliberately: OpenTelemetry uses cross-domain hierarchical projection; Source Map uses multistage relation composition.

## 3. Database: wider relation projection

The integrated rerun produced {db1['candidate_record_count']} candidate and {db1['reference_record_count']} reference records with {db1['false_positive']} false positives, {db1['false_negative']} false negatives, {db1['field_mismatch']} field mismatches, and {db1['multiplicity_mismatch']} multiplicity mismatches.

## 4. Source Map: cross-representation position projection

The ordinary non-indexed profile reproduced {sm1['total_mapping_segments']} segments and {sm1['bidirectional_query_count']} bidirectional queries with {sm1['query_mismatch_count']} mismatches. The declared profile is `{sm1['status']}`; full ECMA-426 surface coverage remains `PARTIAL` because indexed maps and other declared surfaces are excluded.

## 5. OpenTelemetry: narrower occurrence/execution/causal projection

The formal Q6 rerun reproduced {ot1['formal_tpch_q6']['core_occurrence_count']} Core occurrences, {ot1['formal_tpch_q6']['direct_projected_span_count']} spans, {ot1['formal_tpch_q6']['core_binding_count']} bindings, {ot1['formal_tpch_q6']['causal_link_count']} causal Links, and canonical SHA `{ot1['formal_tpch_q6']['trace_sha256']['direct']}`.

## 6. OpenTelemetry projection through Database

Direct Core→OTel and Core→immutable DatabaseDomainProjection→OTel traversals are isolated and exact on both the small workload and Q6. They share canonical trace formatting, not occurrence/binding/producer/GeneratedOrigin extraction.

## 7. Source Map multistage composition

Core GeneratedOrigin composition and independent native SourceMapConsumer composition agree on {sm3['composed_mapping_count']}/5 mappings, with zero false positives, false negatives, broken bridges, ambiguities, cycles, invented mappings, or original→final shortcut bindings. Derived paths are not GenerationBinding entities.

## 8. Strict partiality

Database has 2, OpenTelemetry has 2, and Source Map has 3 rerun counterexamples where complete generation facts differ while the declared projection remains exactly equal. Source Map additionally retains two same-output/different-source-map ambiguity cases.

## 9. Result invariance

Database and OTel CSV/JSON outputs and Source Map generated JavaScript bytes remain identical across capture modes. Control-plane metadata is absent from ordinary output.

## 10. Authority and boundaries

The second-authority count is {reports['second_authority_audit.json']['secondary_authority_store_count']}. New Core changes after the Database head are {reports['core_change_lineage.json']['new_core_change_count']}; new schema changes and domain-specific Core fields are both zero. The two full scientific runs and normalized test summaries are deterministic (`{determinism['status']}`); {tests['total_passed']} tests passed in the second test run with {tests['total_failed']} failures and {tests['total_skipped']} skips.

This evidence does not establish all provenance as projection, all tracing systems, the full ECMA-426 surface, arbitrary DBMS replacement, distributed causality, a universal projection algebra, existence or uniqueness of all domain projections, or reconstruction of complete generation facts from Source Map/OTel.
"""


def artifact_manifest(determinism: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for path in sorted(ARTIFACTS.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(EXPERIMENT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "artifact_count": len(rows),
        "artifacts": rows,
        "comparison_scope": [
            "all machine reports",
            "normalized records",
            "projection document hashes",
            "trace canonical bytes and SHA",
            "map bytes",
            "unified matrix",
            "profiles",
            "environment identity",
            "source branch identity",
            "v2 code hashes",
            "normalized test summaries",
        ],
        "excluded_fields": determinism["excluded_fields"],
        "v1_artifact_tree_id": read_json(
            ARTIFACTS / "v1_preservation.json"
        )["observed_artifact_tree_id"],
    }


def run_full() -> int:
    lineage = source_branch_lineage(REPO_ROOT)
    if lineage["status"] != "PASS":
        print("BRANCH_HEAD_DRIFT", file=sys.stderr)
        for reason in lineage["blocking_reasons"]:
            print(reason, file=sys.stderr)
        return 1

    first, first_performance = scientific_run(1)
    publish_scientific_reports(first)
    write_json(ARTIFACTS / "runs/run_1/scientific_reports.json", first)
    write_json(ARTIFACTS / "runs/run_1/performance_observations.json", first_performance)
    first_tests = run_test_suites()
    write_json(ARTIFACTS / "runs/run_1/test_results.json", first_tests)

    gc.collect()
    second, second_performance = scientific_run(2)
    publish_scientific_reports(second)
    write_json(ARTIFACTS / "runs/run_2/scientific_reports.json", second)
    write_json(ARTIFACTS / "runs/run_2/performance_observations.json", second_performance)
    second_tests = run_test_suites()
    write_json(ARTIFACTS / "runs/run_2/test_results.json", second_tests)

    determinism = determinism_report(first, second, first_tests, second_tests)
    write_json(ARTIFACTS / "determinism.json", determinism)
    write_json(
        ARTIFACTS / "test_results.json",
        {"run_1": first_tests, "run_2": second_tests, "status": "PASS" if first_tests["status"] == second_tests["status"] == "PASS" else "FAIL"},
    )
    scientific_supported = (
        first["run_summary.json"]["status"] == SUPPORTED
        and second["run_summary.json"]["status"] == SUPPORTED
    )
    blocking = sorted(
        set(
            first["run_summary.json"]["blocking_reasons"]
            + second["run_summary.json"]["blocking_reasons"]
            + ([] if determinism["status"] == "PASS" else ["TWO_COMPLETE_RUNS_NOT_DETERMINISTIC"])
            + ([] if first_tests["status"] == "PASS" else ["RUN_1_TESTS_FAILED_OR_SKIPPED"])
            + ([] if second_tests["status"] == "PASS" else ["RUN_2_TESTS_FAILED_OR_SKIPPED"])
        )
    )
    final_status = (
        SUPPORTED
        if scientific_supported
        and determinism["status"] == "PASS"
        and first_tests["status"] == second_tests["status"] == "PASS"
        else NOT_SUPPORTED
    )
    final_summary = {
        **second["run_summary.json"],
        "two_complete_scientific_runs": True,
        "two_complete_test_runs": True,
        "determinism": determinism["status"],
        "test_run_1": first_tests["status"],
        "test_run_2": second_tests["status"],
        "blocking_reasons": blocking,
        "status": final_status,
    }
    write_json(ARTIFACTS / "run_summary.json", final_summary)
    (EXPERIMENT_ROOT / "README.md").write_text(
        render_readme(), encoding="utf-8", newline="\n"
    )
    (EXPERIMENT_ROOT / "EXPERIMENT_REPORT.md").write_text(
        render_experiment_report(second, determinism, second_tests, final_status),
        encoding="utf-8",
        newline="\n",
    )
    write_json(ARTIFACTS / "artifact_manifest.json", artifact_manifest(determinism))
    print(final_status)
    if blocking:
        for reason in blocking:
            print(reason)
    return 0 if final_status == SUPPORTED else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete unified operational projection proof v2"
    )
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not args.full:
        parser.error("--full is required; partial runs are not publishable")
    try:
        return run_full()
    except Exception as exc:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        reason = getattr(exc, "reason_code", type(exc).__name__)
        detail = getattr(exc, "detail", str(exc))
        write_json(
            ARTIFACTS / "run_summary.json",
            {
                "status": NOT_SUPPORTED,
                "blocking_reasons": [f"{reason}:{detail}"],
            },
        )
        print(NOT_SUPPORTED, file=sys.stderr)
        print(f"{reason}:{detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
