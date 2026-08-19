from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from generation_relation_core.canonical import canonical_bytes, sha256_bytes
from generation_relation_core.snapshots import (
    AUTHORITATIVE_TABLE_SPECS,
    ValidatedSnapshot,
)

from .errors import ProjectionProofError
from .projection_profile import require_truthful_status


def _snapshot_payload(snapshot: ValidatedSnapshot) -> dict[str, Any]:
    return {
        "record": snapshot.record,
        "authoritative_tables": {
            name: getattr(snapshot.tables, name) for name in AUTHORITATIVE_TABLE_SPECS
        },
    }


def snapshot_sha256(snapshot: ValidatedSnapshot) -> str:
    return sha256_bytes(canonical_bytes(_snapshot_payload(snapshot)))


def _binding_set(snapshot: ValidatedSnapshot) -> list[dict[str, Any]]:
    return sorted(snapshot.tables.generation_bindings, key=canonical_bytes)


def _transform_context(snapshot: ValidatedSnapshot) -> dict[str, Any]:
    return {
        "occurrences": sorted(
            snapshot.tables.generation_occurrences, key=canonical_bytes
        ),
        "manifests": sorted(snapshot.tables.generator_manifests, key=canonical_bytes),
        "environments": sorted(
            snapshot.tables.environment_records, key=canonical_bytes
        ),
    }


@dataclass(frozen=True)
class Counterexample:
    counterexample_id: str
    left_snapshot: ValidatedSnapshot
    right_snapshot: ValidatedSnapshot
    left_projection: dict[str, Any]
    right_projection: dict[str, Any]


def assert_projection_equality_is_not_complete_equality(
    *,
    projection_equal: bool,
    complete_snapshot_equal: bool,
    inferred_complete_equal: bool,
) -> None:
    if projection_equal and not complete_snapshot_equal and inferred_complete_equal:
        raise ProjectionProofError("PROJECTION_EQUALITY_NOT_COMPLETE_EQUALITY")


def evaluate_strict_partiality(
    *,
    profile_id: str,
    counterexamples: list[Counterexample],
    minimum_count: int = 2,
) -> dict[str, Any]:
    cases = []
    for case in counterexamples:
        complete_equal = canonical_bytes(
            _snapshot_payload(case.left_snapshot)
        ) == canonical_bytes(_snapshot_payload(case.right_snapshot))
        binding_equal = canonical_bytes(
            _binding_set(case.left_snapshot)
        ) == canonical_bytes(_binding_set(case.right_snapshot))
        transform_equal = canonical_bytes(
            _transform_context(case.left_snapshot)
        ) == canonical_bytes(_transform_context(case.right_snapshot))
        projection_equal = canonical_bytes(case.left_projection) == canonical_bytes(
            case.right_projection
        )
        cases.append(
            {
                "counterexample_id": case.counterexample_id,
                "left_snapshot_sha256": snapshot_sha256(case.left_snapshot),
                "right_snapshot_sha256": snapshot_sha256(case.right_snapshot),
                "complete_snapshot_equal": complete_equal,
                "binding_set_equal": binding_equal,
                "transform_context_equal": transform_equal,
                "projection_equal": projection_equal,
                "valid_counterexample": projection_equal
                and not complete_equal
                and not binding_equal
                and not transform_equal,
            }
        )
    supported = len(cases) >= minimum_count and all(
        case["valid_counterexample"] for case in cases
    )
    status = "SUPPORTED" if supported else "NOT_ESTABLISHED"
    require_truthful_status(prerequisite_satisfied=supported, requested_status=status)
    return {
        "profile_id": profile_id,
        "counterexample_count": len(cases),
        "complete_snapshot_equal": all(
            case["complete_snapshot_equal"] for case in cases
        )
        if cases
        else None,
        "binding_set_equal": all(case["binding_set_equal"] for case in cases)
        if cases
        else None,
        "transform_context_equal": all(
            case["transform_context_equal"] for case in cases
        )
        if cases
        else None,
        "projection_equal": all(case["projection_equal"] for case in cases)
        if cases
        else None,
        "cases": cases,
        "interpretation": "Projection equality does not imply complete generation-fact equality.",
        "status": status,
    }


def not_evaluated_strict_partiality(profile_id: str, reason: str) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "counterexample_count": 0,
        "complete_snapshot_equal": None,
        "binding_set_equal": None,
        "transform_context_equal": None,
        "projection_equal": None,
        "interpretation": "Projection equality does not imply complete generation-fact equality.",
        "reason": reason,
        "status": "NOT_EVALUATED",
    }
