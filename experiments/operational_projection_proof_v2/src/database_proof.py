from __future__ import annotations

from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from experiments.database_lineage.src.result_serializer import csv_bytes, json_bytes
from experiments.operational_projection_proof.scripts.run_all import (
    PROFILE_ROOT as V1_PROFILE_ROOT,
    _business_projection,
    _many_to_many_projection,
    build_reports as build_v1_reports,
)
from experiments.operational_projection_proof.src.database_reference import (
    business_oracle_result,
    many_to_many_oracle_result,
)
from experiments.operational_projection_proof.src.projection_profile import load_profile
from experiments.operational_projection_proof.src.projection_result import combine_results

from .common import canonical_sha256, set_comparison, snapshot_document


FROZEN_DATABASE_RECORD_COUNT = 112


def _database_counterexample(
    counterexample_id: str,
    *,
    profile: Any,
    left_run_id: str,
    right_run_id: str,
    left_dependencies: dict[str, str] | None = None,
    right_dependencies: dict[str, str] | None = None,
) -> dict[str, Any]:
    _la, left_rows, left_snapshot, _lv, left_projection = _business_projection(
        profile=profile, run_id=left_run_id, dependencies=left_dependencies
    )
    _ra, right_rows, right_snapshot, _rv, right_projection = _business_projection(
        profile=profile, run_id=right_run_id, dependencies=right_dependencies
    )
    left_tables = left_snapshot.tables
    right_tables = right_snapshot.tables
    bindings = set_comparison(
        left_tables.generation_bindings, right_tables.generation_bindings
    )
    occurrences = set_comparison(
        left_tables.generation_occurrences, right_tables.generation_occurrences
    )
    transforms = set_comparison(
        [row["transform_reference"] for row in left_tables.generation_occurrences],
        [row["transform_reference"] for row in right_tables.generation_occurrences],
    )
    environments = set_comparison(
        left_tables.environment_records, right_tables.environment_records
    )
    left_doc = snapshot_document(left_snapshot)
    right_doc = snapshot_document(right_snapshot)
    projection_equal = canonical_bytes(left_projection) == canonical_bytes(right_projection)
    output_equal = (
        csv_bytes(left_rows) == csv_bytes(right_rows)
        and json_bytes(left_rows) == json_bytes(right_rows)
    )
    complete_equal = canonical_bytes(left_doc) == canonical_bytes(right_doc)
    return {
        "counterexample_id": counterexample_id,
        "left_snapshot_id": left_snapshot.snapshot_id,
        "right_snapshot_id": right_snapshot.snapshot_id,
        "left_snapshot_semantic_sha256": canonical_sha256(left_doc),
        "right_snapshot_semantic_sha256": canonical_sha256(right_doc),
        "complete_snapshot_equal": complete_equal,
        "projection_hash_equal": canonical_sha256(left_projection)
        == canonical_sha256(right_projection),
        "projection_sha256": canonical_sha256(left_projection),
        "binding_set_equal": bindings["equal"],
        "binding_symmetric_difference_count": bindings[
            "symmetric_difference_count"
        ],
        "binding_difference": bindings,
        "occurrence_set_equal": occurrences["equal"],
        "occurrence_symmetric_difference_count": occurrences[
            "symmetric_difference_count"
        ],
        "occurrence_difference": occurrences,
        "transform_context_equal": transforms["equal"],
        "environment_context_equal": environments["equal"],
        "environment_difference": environments,
        "output_equal": output_equal,
        "valid_counterexample": projection_equal and output_equal and not complete_equal,
    }


def run_database_proof(_run_dir: Path) -> dict[str, Any]:
    inherited = build_v1_reports()
    p1 = dict(inherited["projection_equivalence_database.json"])
    profile = load_profile(V1_PROFILE_ROOT / "database_which_lineage_v1.json")
    _adapter, _rows, _snapshot, _validation, business_candidate = (
        _business_projection(
            profile=profile,
            run_id="projection-proof-v2-p1-business-record-identity",
        )
    )
    _m2m_adapter, _m2m_snapshot, _m2m_validation, m2m_candidate = (
        _many_to_many_projection(profile=profile)
    )
    candidate_records = combine_results(
        profile, business_candidate, m2m_candidate
    )
    reference_records = combine_results(
        profile,
        business_oracle_result(profile),
        many_to_many_oracle_result(profile),
    )
    frozen_checks = {
        "candidate_record_count": p1["candidate_record_count"]
        == FROZEN_DATABASE_RECORD_COUNT,
        "reference_record_count": p1["reference_record_count"]
        == FROZEN_DATABASE_RECORD_COUNT,
        "false_positive": p1["false_positive"] == 0,
        "false_negative": p1["false_negative"] == 0,
        "field_mismatch": p1["field_mismatch"] == 0,
        "multiplicity_mismatch": p1["multiplicity_mismatch"] == 0,
        "reference_deletion": p1["candidate_after_reference_deleted_equal"],
        "oracle_runtime_trap": p1["oracle_runtime_trap_passed"],
        "native_result_read_count": p1["native_domain_result_read_count"] == 0,
        "second_authority_store_count": p1["second_authority_store_count"] == 0,
        "output_orthogonality": inherited["output_orthogonality.json"]["status"]
        == "SUPPORTED",
    }
    blocking = []
    if not frozen_checks["candidate_record_count"] or not frozen_checks["reference_record_count"]:
        blocking.append("DATABASE_P1_FROZEN_COUNT_DRIFT")
    blocking.extend(
        f"DATABASE_P1_{name.upper()}_FAILED"
        for name, passed in frozen_checks.items()
        if not passed and name not in {"candidate_record_count", "reference_record_count"}
    )
    p1.update(
        {
            "rerun_on_integrated_branch": True,
            "frozen_expected_record_count": FROZEN_DATABASE_RECORD_COUNT,
            "frozen_checks": frozen_checks,
            "candidate_canonical_sha256": canonical_sha256(candidate_records),
            "reference_canonical_sha256": canonical_sha256(reference_records),
            "canonical_records_hash_equal": canonical_sha256(candidate_records)
            == canonical_sha256(reference_records),
            "section_canonical_sha256": {
                name: canonical_sha256(rows)
                for name, rows in candidate_records["records"].items()
            },
            "blocking_reasons": blocking,
            "status": "SUPPORTED" if not blocking else "NOT_SUPPORTED",
        }
    )

    cases = [
        _database_counterexample(
            "same-lineage-distinct-execution-run",
            profile=profile,
            left_run_id="projection-proof-v2-p2-run-a",
            right_run_id="projection-proof-v2-p2-run-b",
        ),
        _database_counterexample(
            "same-lineage-distinct-environment-context",
            profile=profile,
            left_run_id="projection-proof-v2-p2-context",
            right_run_id="projection-proof-v2-p2-context",
            left_dependencies={"controlled_context": "context-a"},
            right_dependencies={"controlled_context": "context-b"},
        ),
    ]
    p2_supported = len(cases) == 2 and all(row["valid_counterexample"] for row in cases)
    p2 = {
        "profile_id": "database-which-lineage-v1",
        "counterexample_count": len(cases),
        "cases": cases,
        "projection_equality_does_not_imply_complete_fact_equality": p2_supported,
        "status": "SUPPORTED" if p2_supported else "NOT_SUPPORTED",
    }
    return {
        "projection_equivalence_database.json": p1,
        "strict_partiality_database.json": p2,
        "database_output_orthogonality": inherited["output_orthogonality.json"],
        "database_oracle_isolation": inherited["oracle_isolation.json"],
        "database_second_authority": inherited["second_authority_audit.json"],
        "database_negative_controls": inherited["negative_controls.json"],
    }
