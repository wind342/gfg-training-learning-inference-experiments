from __future__ import annotations

from typing import Any

from .projection_profile import ProjectionProfile, require_truthful_status
from .projection_result import compare_exact


def evaluate_exact_derivability(
    *,
    profile: ProjectionProfile,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    oracle_leakage_count: int,
    native_result_read_count: int,
    second_authority_store_count: int,
    candidate_after_reference_deleted_equal: bool,
    oracle_runtime_trap_passed: bool,
) -> dict[str, Any]:
    comparison = compare_exact(candidate, reference, profile)
    isolation_passed = (
        oracle_leakage_count == 0
        and native_result_read_count == 0
        and second_authority_store_count == 0
        and candidate_after_reference_deleted_equal
        and oracle_runtime_trap_passed
    )
    supported = comparison["exact_equal"] and isolation_passed
    status = "SUPPORTED" if supported else "NOT_SUPPORTED"
    require_truthful_status(prerequisite_satisfied=supported, requested_status=status)
    return {
        **comparison,
        "claim_scope": profile.claim_scope,
        "oracle_leakage_count": oracle_leakage_count,
        "native_domain_result_read_count": native_result_read_count,
        "second_authority_store_count": second_authority_store_count,
        "candidate_after_reference_deleted_equal": candidate_after_reference_deleted_equal,
        "oracle_runtime_trap_passed": oracle_runtime_trap_passed,
        "status": status,
    }


def not_evaluated_exact_derivability(profile: ProjectionProfile) -> dict[str, Any]:
    require_truthful_status(
        prerequisite_satisfied=False, requested_status="NOT_EVALUATED"
    )
    return {
        "profile_id": profile.profile_id,
        "candidate_record_count": 0,
        "reference_record_count": 0,
        "false_positive": None,
        "false_negative": None,
        "field_mismatch": None,
        "multiplicity_mismatch": None,
        "exact_equal": None,
        "oracle_leakage_count": None,
        "native_domain_result_read_count": None,
        "second_authority_store_count": None,
        "claim_scope": profile.claim_scope,
        "reason": profile.prerequisite,
        "status": "NOT_EVALUATED",
    }
