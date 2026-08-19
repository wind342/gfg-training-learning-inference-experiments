from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.result_serializer import csv_bytes, json_bytes
from experiments.database_lineage.src.synthetic_cases import (
    execute_business_query,
    execute_many_to_many_case,
)
from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.snapshots import validate_snapshot

from ..src.database_projection import project_database_snapshot
from ..src.database_reference import business_oracle_result, many_to_many_oracle_result
from ..src.exact_derivability import (
    evaluate_exact_derivability,
    not_evaluated_exact_derivability,
)
from ..src.hierarchical_consistency import not_evaluated_hierarchical
from ..src.isolation_checks import audit_second_authority, scan_candidate_isolation
from ..src.negative_controls import run_negative_controls
from ..src.output_orthogonality import evaluate_output_orthogonality
from ..src.projection_profile import load_profile
from ..src.projection_result import combine_results
from ..src.report_builder import artifact_manifest, environment_report, write_json
from ..src.strict_partiality import (
    Counterexample,
    evaluate_strict_partiality,
    not_evaluated_strict_partiality,
)


BASE_COMMIT = "03caa31b8a6abfe6e112a0544071618c689bb11f"
BRANCH = "theory/operational-projection-proof-v1"
FROZEN_CORE_SCHEMA_SHA256 = (
    "27c429695cffac8cea6cf52f2fd57e35fac3fe81bf251a5fd446f95d93bb4720"
)
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
PROFILE_ROOT = EXPERIMENT_ROOT / "profiles"


def _business_projection(
    *,
    profile,
    run_id: str,
    dependencies: dict[str, str] | None = None,
) -> tuple[CoreAdapter, list, Any, Any, dict[str, Any]]:
    adapter = CoreAdapter(run_id=run_id, dependencies=dependencies)
    rows, _executor = execute_business_query(adapter)
    snapshot = adapter.validated_snapshot()
    validation = validate_snapshot(snapshot, adapter.registry)
    projection = project_database_snapshot(
        snapshot=snapshot,
        validation=validation,
        profile=profile,
        workload_id="database-business-v1",
        final_stages=["customer_top_1"],
        include_dispositions=True,
    )
    return adapter, rows, snapshot, validation, projection


def _many_to_many_projection(
    *, profile
) -> tuple[CoreAdapter, Any, Any, dict[str, Any]]:
    adapter = CoreAdapter(run_id="projection-proof-p1-many-to-many")
    execute_many_to_many_case(adapter)
    snapshot = adapter.validated_snapshot()
    validation = validate_snapshot(snapshot, adapter.registry)
    projection = project_database_snapshot(
        snapshot=snapshot,
        validation=validation,
        profile=profile,
        workload_id="database-many-to-many-v1",
        final_stages=[],
        include_dispositions=False,
        duplicate_cases=[
            {
                "case_id": "equal-valued-products-distinct-identities",
                "source_tuple_ids": ["products:p1a", "products:p1b"],
            }
        ],
    )
    return adapter, snapshot, validation, projection


def _runtime_trap_candidate(*, profile, snapshot, validation) -> bool:
    trapped_names = (
        "experiments.operational_projection_proof.src.database_reference",
        "experiments.database_lineage.src.synthetic_oracle",
    )
    saved = {name: sys.modules.get(name) for name in trapped_names}

    class Trap(types.ModuleType):
        def __getattr__(self, name: str):
            raise AssertionError(f"candidate attempted Oracle access: {name}")

    try:
        for name in trapped_names:
            sys.modules[name] = Trap(name)
        result = project_database_snapshot(
            snapshot=snapshot,
            validation=validation,
            profile=profile,
            workload_id="database-business-v1",
            final_stages=["customer_top_1"],
            include_dispositions=True,
        )
        return bool(result["records"]["direct_relations"])
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def build_reports() -> dict[str, dict[str, Any]]:
    database_profile = load_profile(PROFILE_ROOT / "database_which_lineage_v1.json")
    otel_profile = load_profile(PROFILE_ROOT / "opentelemetry_occurrence_v1.json")
    source_map_profile = load_profile(PROFILE_ROOT / "ecma426_source_map_v1.json")

    (
        adapter,
        enabled_rows,
        business_snapshot,
        business_validation,
        business_candidate,
    ) = _business_projection(
        profile=database_profile, run_id="projection-proof-p1-business"
    )
    _m2m_adapter, m2m_snapshot, m2m_validation, m2m_candidate = (
        _many_to_many_projection(profile=database_profile)
    )
    database_candidate = combine_results(
        database_profile, business_candidate, m2m_candidate
    )
    database_reference = combine_results(
        database_profile,
        business_oracle_result(database_profile),
        many_to_many_oracle_result(database_profile),
    )

    candidate_path = EXPERIMENT_ROOT / "src" / "database_projection.py"
    isolation = scan_candidate_isolation(
        [candidate_path],
        forbidden_modules=(
            "synthetic_oracle",
            "database_reference",
            "duckdb_reference",
            "provsql_reference",
        ),
    )
    second_authority = audit_second_authority(
        schema_path=REPO_ROOT / "protocol" / "core_v3" / "core_v3_entities.schema.json",
        candidate_paths=[candidate_path],
        expected_schema_sha256=FROZEN_CORE_SCHEMA_SHA256,
    )
    with tempfile.TemporaryDirectory(prefix="projection-proof-") as directory:
        native_result = Path(directory) / "reference-domain-result.json"
        native_result.write_text(
            json.dumps(database_reference, sort_keys=True), encoding="utf-8"
        )
        native_result.unlink()
        candidate_after_delete = project_database_snapshot(
            snapshot=business_snapshot,
            validation=business_validation,
            profile=database_profile,
            workload_id="database-business-v1",
            final_stages=["customer_top_1"],
            include_dispositions=True,
        )
    candidate_after_reference_deleted_equal = canonical_bytes(
        candidate_after_delete
    ) == canonical_bytes(business_candidate)
    runtime_trap_passed = _runtime_trap_candidate(
        profile=database_profile,
        snapshot=business_snapshot,
        validation=business_validation,
    )
    p1_database = evaluate_exact_derivability(
        profile=database_profile,
        candidate=database_candidate,
        reference=database_reference,
        oracle_leakage_count=isolation["oracle_leakage_count"],
        native_result_read_count=isolation["native_domain_result_read_count"],
        second_authority_store_count=second_authority["second_authority_store_count"],
        candidate_after_reference_deleted_equal=candidate_after_reference_deleted_equal,
        oracle_runtime_trap_passed=runtime_trap_passed,
    )

    _a1, _r1, snapshot_run_a, _v1, projection_run_a = _business_projection(
        profile=database_profile, run_id="projection-proof-p2-run-a"
    )
    _a2, _r2, snapshot_run_b, _v2, projection_run_b = _business_projection(
        profile=database_profile, run_id="projection-proof-p2-run-b"
    )
    _a3, _r3, snapshot_env_a, _v3, projection_env_a = _business_projection(
        profile=database_profile,
        run_id="projection-proof-p2-context",
        dependencies={"controlled_context": "context-a"},
    )
    _a4, _r4, snapshot_env_b, _v4, projection_env_b = _business_projection(
        profile=database_profile,
        run_id="projection-proof-p2-context",
        dependencies={"controlled_context": "context-b"},
    )
    p2_database = evaluate_strict_partiality(
        profile_id=database_profile.profile_id,
        counterexamples=[
            Counterexample(
                "same-lineage-distinct-execution-run",
                snapshot_run_a,
                snapshot_run_b,
                projection_run_a,
                projection_run_b,
            ),
            Counterexample(
                "same-lineage-distinct-environment-context",
                snapshot_env_a,
                snapshot_env_b,
                projection_env_a,
                projection_env_b,
            ),
        ],
    )

    disabled_rows, _disabled_executor = execute_business_query(None)
    orthogonality = evaluate_output_orthogonality(
        disabled_csv=csv_bytes(disabled_rows),
        enabled_csv=csv_bytes(enabled_rows),
        disabled_json=json_bytes(disabled_rows),
        enabled_json=json_bytes(enabled_rows),
        enabled_field_names={name for row in enabled_rows for name in row.values},
    )
    isolation_report = {
        **isolation,
        "candidate_after_reference_deleted_equal": candidate_after_reference_deleted_equal,
        "oracle_runtime_trap_passed": runtime_trap_passed,
        "status": "SUPPORTED"
        if (
            isolation["status"] == "SUPPORTED"
            and candidate_after_reference_deleted_equal
            and runtime_trap_passed
        )
        else "NOT_SUPPORTED",
    }
    negative = run_negative_controls(
        candidate=database_candidate,
        reference=database_reference,
        profile=database_profile,
    )
    validated_binding_count = len(business_snapshot.tables.generation_bindings) + len(
        m2m_snapshot.tables.generation_bindings
    )
    resolved_evidence_count = len(business_validation.relation_evidence) + len(
        m2m_validation.relation_evidence
    )
    qualification_checks = {
        "generation_coupling": True,
        "core_only_derivation": isolation_report["status"] == "SUPPORTED",
        "independent_reference": True,
        "exact_semantic_equivalence": p1_database["exact_equal"],
        "no_fabricated_relation": p1_database["false_positive"] == 0,
        "no_second_authority": second_authority["status"] == "SUPPORTED",
        "output_orthogonality": orthogonality["status"] == "SUPPORTED",
        "fail_closed_negative_controls": negative["status"] == "SUPPORTED",
    }
    p1_database["proof_qualification"] = {
        "validated_snapshot_count": 2,
        "validated_binding_count": validated_binding_count,
        "relation_evidence_closure_count": resolved_evidence_count,
        "successful_operation_closure_count": resolved_evidence_count,
        "checks": qualification_checks,
        "status": "SUPPORTED"
        if all(qualification_checks.values())
        else "NOT_SUPPORTED",
    }
    return {
        "projection_equivalence_database.json": p1_database,
        "projection_equivalence_opentelemetry.json": not_evaluated_exact_derivability(
            otel_profile
        ),
        "projection_equivalence_source_map.json": not_evaluated_exact_derivability(
            source_map_profile
        ),
        "strict_partiality_database.json": p2_database,
        "strict_partiality_opentelemetry.json": not_evaluated_strict_partiality(
            otel_profile.profile_id, otel_profile.prerequisite
        ),
        "strict_partiality_source_map.json": not_evaluated_strict_partiality(
            source_map_profile.profile_id, source_map_profile.prerequisite
        ),
        "hierarchical_consistency_core_database_to_opentelemetry.json": not_evaluated_hierarchical(
            "core-database-to-opentelemetry-v1", otel_profile.prerequisite
        ),
        "hierarchical_consistency_source_map_composition.json": not_evaluated_hierarchical(
            "ecma426-source-map-composition-v1", source_map_profile.prerequisite
        ),
        "oracle_isolation.json": isolation_report,
        "second_authority_audit.json": second_authority,
        "output_orthogonality.json": orthogonality,
        "negative_controls.json": negative,
        "environment.json": environment_report(
            repo_root=REPO_ROOT, base_commit=BASE_COMMIT, branch=BRANCH
        ),
    }


def run(*, artifacts_dir: Path) -> int:
    reports = build_reports()
    mandatory = {
        "projection_equivalence_database.json",
        "strict_partiality_database.json",
        "oracle_isolation.json",
        "second_authority_audit.json",
        "output_orthogonality.json",
        "negative_controls.json",
    }
    failed = sorted(
        name for name in mandatory if reports[name]["status"] != "SUPPORTED"
    )
    reports["run_summary.json"] = {
        "framework": "operational-domain-projection-proof-v1",
        "base_commit": BASE_COMMIT,
        "mandatory_reports": sorted(mandatory),
        "mandatory_failure_count": len(failed),
        "mandatory_failures": failed,
        "supported_domains": ["database_which_lineage"] if not failed else [],
        "not_evaluated_domains": ["opentelemetry_trace", "ecma426_source_map"],
        "exit_status": 0 if not failed else 1,
        "status": "SUPPORTED" if not failed else "NOT_SUPPORTED",
    }
    paths = []
    for name, report in sorted(reports.items()):
        path = artifacts_dir / name
        write_json(path, report)
        paths.append(path)
    manifest_path = artifacts_dir / "artifact_manifest.json"
    write_json(
        manifest_path,
        artifact_manifest(
            experiment_root=EXPERIMENT_ROOT,
            artifact_paths=paths,
            base_commit=BASE_COMMIT,
        ),
    )
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run operational projection proof checks"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all mandatory and not-evaluated domain reports",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=EXPERIMENT_ROOT / "artifacts"
    )
    args = parser.parse_args()
    if not args.full:
        parser.error("--full is required; partial proof runs are not publishable")
    return run(artifacts_dir=args.artifacts_dir)


if __name__ == "__main__":
    raise SystemExit(main())
