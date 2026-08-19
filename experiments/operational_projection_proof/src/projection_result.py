from __future__ import annotations

from copy import deepcopy
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from .errors import ProjectionProofError
from .projection_profile import ProjectionProfile


def empty_result(profile: ProjectionProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "records": {name: [] for name in profile.sections},
    }


def _record_key(
    record: dict[str, Any], fields: tuple[str, ...], section: str
) -> tuple[Any, ...]:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ProjectionProofError(
            "RECORD_SCHEMA_INVALID", f"{section}:{','.join(missing)}"
        )
    return tuple(canonical_bytes(record[field]) for field in fields)


def normalize_result(
    value: dict[str, Any], profile: ProjectionProfile
) -> dict[str, Any]:
    if (
        set(value) != {"profile_id", "records"}
        or value.get("profile_id") != profile.profile_id
    ):
        raise ProjectionProofError("RESULT_SCHEMA_INVALID", profile.profile_id)
    records = value.get("records")
    if not isinstance(records, dict) or set(records) != set(profile.sections):
        raise ProjectionProofError("RESULT_SCHEMA_INVALID", "sections")
    normalized = empty_result(profile)
    for section, identity_fields in profile.sections.items():
        rows = records[section]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ProjectionProofError("RESULT_SCHEMA_INVALID", section)
        seen: set[tuple[Any, ...]] = set()
        for row in rows:
            key = _record_key(row, identity_fields, section)
            if key in seen:
                raise ProjectionProofError("DUPLICATE_SEMANTIC_KEY", section)
            seen.add(key)
        normalized["records"][section] = sorted(deepcopy(rows), key=canonical_bytes)
    return normalized


def combine_results(
    profile: ProjectionProfile, *values: dict[str, Any]
) -> dict[str, Any]:
    combined = empty_result(profile)
    for value in values:
        normalized = normalize_result(value, profile)
        for section in profile.sections:
            combined["records"][section].extend(normalized["records"][section])
    return normalize_result(combined, profile)


def _changed_fields(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    fields = set(left) | set(right)
    return {
        field
        for field in fields
        if canonical_bytes(left.get(field)) != canonical_bytes(right.get(field))
    }


def compare_exact(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    profile: ProjectionProfile,
) -> dict[str, Any]:
    candidate = normalize_result(candidate, profile)
    reference = normalize_result(reference, profile)
    false_positive = 0
    false_negative = 0
    field_mismatch = 0
    multiplicity_mismatch = 0
    reason_codes: set[str] = set()
    section_results: dict[str, dict[str, Any]] = {}
    for section, identity_fields in profile.sections.items():
        candidate_rows = candidate["records"][section]
        reference_rows = reference["records"][section]
        candidate_by_key = {
            _record_key(row, identity_fields, section): row for row in candidate_rows
        }
        reference_by_key = {
            _record_key(row, identity_fields, section): row for row in reference_rows
        }
        extra = set(candidate_by_key) - set(reference_by_key)
        missing = set(reference_by_key) - set(candidate_by_key)
        section_fp = len(extra)
        section_fn = len(missing)
        section_field = 0
        section_multiplicity = 0
        for key in set(candidate_by_key) & set(reference_by_key):
            left = candidate_by_key[key]
            right = reference_by_key[key]
            if canonical_bytes(left) == canonical_bytes(right):
                continue
            changed = _changed_fields(left, right)
            if changed & profile.multiplicity_fields:
                section_multiplicity += 1
            else:
                section_field += 1
        false_positive += section_fp
        false_negative += section_fn
        field_mismatch += section_field
        multiplicity_mismatch += section_multiplicity
        if section_fp:
            reason_codes.add("FALSE_POSITIVE_RECORD")
        if section_fn:
            reason_codes.add("FALSE_NEGATIVE_RECORD")
        if section_field:
            reason_codes.add("FIELD_MISMATCH")
        if section_multiplicity:
            reason_codes.add("MULTIPLICITY_MISMATCH")
        section_results[section] = {
            "candidate_count": len(candidate_rows),
            "reference_count": len(reference_rows),
            "false_positive": section_fp,
            "false_negative": section_fn,
            "field_mismatch": section_field,
            "multiplicity_mismatch": section_multiplicity,
        }
    exact = not (
        false_positive or false_negative or field_mismatch or multiplicity_mismatch
    )
    return {
        "profile_id": profile.profile_id,
        "candidate_record_count": sum(
            len(rows) for rows in candidate["records"].values()
        ),
        "reference_record_count": sum(
            len(rows) for rows in reference["records"].values()
        ),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "field_mismatch": field_mismatch,
        "multiplicity_mismatch": multiplicity_mismatch,
        "exact_equal": exact,
        "reason_codes": sorted(reason_codes),
        "section_results": section_results,
    }
