from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .errors import ProjectionProofError
from .hierarchical_consistency import (
    assert_acyclic_edges,
    compare_hierarchical,
    require_hierarchical_equality,
)
from .isolation_checks import (
    require_candidate_isolation,
    require_no_second_authority_source,
    scan_source_isolation,
)
from .output_orthogonality import (
    evaluate_output_orthogonality,
    require_output_orthogonality,
)
from .projection_profile import ProjectionProfile, require_truthful_status
from .projection_result import compare_exact, normalize_result
from .strict_partiality import assert_projection_equality_is_not_complete_equality


def _reason(action: Callable[[], None]) -> str:
    try:
        action()
    except ProjectionProofError as exc:
        return exc.reason_code
    return "NO_FAILURE"


def _require_comparison_reason(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    profile: ProjectionProfile,
    reason: str,
) -> None:
    report = compare_exact(candidate, reference, profile)
    if reason not in report["reason_codes"]:
        raise ProjectionProofError("NEGATIVE_CONTROL_DID_NOT_TRIGGER", reason)
    raise ProjectionProofError(reason)


def run_negative_controls(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    profile: ProjectionProfile,
) -> dict[str, Any]:
    controls: list[tuple[str, str, Callable[[], None]]] = []

    extra = deepcopy(candidate)
    extra_row = deepcopy(extra["records"]["direct_relations"][0])
    extra_row["input_tuple_id"] = "negative-control:extra"
    extra["records"]["direct_relations"].append(extra_row)
    controls.append(
        (
            "candidate_extra_relation",
            "FALSE_POSITIVE_RECORD",
            lambda: _require_comparison_reason(
                extra, reference, profile, "FALSE_POSITIVE_RECORD"
            ),
        )
    )

    missing = deepcopy(candidate)
    missing["records"]["direct_relations"].pop()
    controls.append(
        (
            "candidate_missing_relation",
            "FALSE_NEGATIVE_RECORD",
            lambda: _require_comparison_reason(
                missing, reference, profile, "FALSE_NEGATIVE_RECORD"
            ),
        )
    )

    field = deepcopy(candidate)
    field["records"]["backward_lineage"][0]["source_tuple_ids"] = []
    controls.append(
        (
            "field_value_mismatch",
            "FIELD_MISMATCH",
            lambda: _require_comparison_reason(
                field, reference, profile, "FIELD_MISMATCH"
            ),
        )
    )

    multiplicity = deepcopy(candidate)
    multiplicity["records"]["multiplicity"][0]["total_relation_count"] += 1
    controls.append(
        (
            "multiplicity_mismatch",
            "MULTIPLICITY_MISMATCH",
            lambda: _require_comparison_reason(
                multiplicity, reference, profile, "MULTIPLICITY_MISMATCH"
            ),
        )
    )

    duplicate = deepcopy(candidate)
    duplicate["records"]["direct_relations"].append(
        deepcopy(duplicate["records"]["direct_relations"][0])
    )
    controls.append(
        (
            "duplicate_dictionary_overwrite",
            "DUPLICATE_SEMANTIC_KEY",
            lambda: normalize_result(duplicate, profile),
        )
    )

    oracle_source = "from experiments.database_lineage.src import synthetic_oracle\n"
    controls.append(
        (
            "candidate_reads_oracle",
            "ORACLE_LEAKAGE",
            lambda: require_candidate_isolation(
                scan_source_isolation(
                    oracle_source,
                    filename="leaky_candidate.py",
                    forbidden_modules=("synthetic_oracle", "database_reference"),
                )
            ),
        )
    )

    native_source = (
        "from pathlib import Path\nvalue = Path('native.json').read_text()\n"
    )
    controls.append(
        (
            "candidate_reads_native_domain_file",
            "NATIVE_DOMAIN_RESULT_LEAKAGE",
            lambda: require_candidate_isolation(
                scan_source_isolation(
                    native_source,
                    filename="native_leaky_candidate.py",
                    forbidden_modules=("synthetic_oracle", "database_reference"),
                )
            ),
        )
    )

    controls.append(
        (
            "second_authority_table",
            "SECOND_AUTHORITY_STORE",
            lambda: require_no_second_authority_source("lineage_table = {}\n"),
        )
    )

    contaminated = evaluate_output_orthogonality(
        disabled_csv=b"x\n1\n",
        enabled_csv=b"x\n1\n",
        disabled_json=b'[{"x":1}]',
        enabled_json=b'[{"x":1}]',
        enabled_field_names={"x", "lineage"},
    )
    controls.append(
        (
            "control_plane_output_field",
            "CONTROL_PLANE_OUTPUT_CONTAMINATION",
            lambda: require_output_orthogonality(contaminated),
        )
    )

    hierarchy = compare_hierarchical(
        profile_id="negative-hierarchy",
        direct=[{"id": "a"}],
        hierarchical=[{"id": "b"}],
    )
    controls.append(
        (
            "direct_hierarchical_mismatch",
            "HIERARCHICAL_MISMATCH",
            lambda: require_hierarchical_equality(hierarchy),
        )
    )

    controls.append(
        (
            "projection_equality_implies_complete_equality",
            "PROJECTION_EQUALITY_NOT_COMPLETE_EQUALITY",
            lambda: assert_projection_equality_is_not_complete_equality(
                projection_equal=True,
                complete_snapshot_equal=False,
                inferred_complete_equal=True,
            ),
        )
    )

    controls.append(
        (
            "partial_promoted_to_supported",
            "UNSUPPORTED_STATUS_ESCALATION",
            lambda: require_truthful_status(
                prerequisite_satisfied=False, requested_status="SUPPORTED"
            ),
        )
    )

    controls.append(
        (
            "hierarchy_cycle",
            "HIERARCHY_CYCLE",
            lambda: assert_acyclic_edges([("a", "b"), ("b", "a")]),
        )
    )

    results = []
    for control_id, expected, action in controls:
        actual = _reason(action)
        results.append(
            {
                "control_id": control_id,
                "expected_reason_code": expected,
                "actual_reason_code": actual,
                "passed": actual == expected,
            }
        )
    return {
        "negative_control_count": len(results),
        "passed_count": sum(item["passed"] for item in results),
        "failed_count": sum(not item["passed"] for item in results),
        "controls": results,
        "status": "SUPPORTED"
        if all(item["passed"] for item in results)
        else "NOT_SUPPORTED",
    }
