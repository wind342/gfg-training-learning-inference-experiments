from __future__ import annotations

import re
from typing import Any


MECHANISMS = (
    "database_which_lineage",
    "source_map",
    "opentelemetry",
    "w3c_prov_generation_profile",
    "pytorch_autograd_dependency_profile",
)

REQUIRED_P1_FIELDS = (
    "status",
    "candidate_record_count",
    "native_record_count",
    "false_positive_count",
    "false_negative_count",
    "field_mismatch_count",
    "multiplicity_mismatch_count",
    "byte_equal",
    "query_count",
    "query_mismatch_count",
)
REQUIRED_P2_FIELDS = (
    "status",
    "witness_count",
    "valid_witness_count",
    "snapshot_distinct",
    "target_equal",
    "witness_summaries",
)
FORBIDDEN_SUCCESS_TOKENS = {"SKIP", "SKIPPED", "BLOCKED", "UNAVAILABLE", "NOT_INSTALLED"}


class ResultValidationError(ValueError):
    pass


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_strings(child)]
    return []


def validate_mechanism_result(result: dict[str, Any], *, expected_mechanism: str | None = None) -> None:
    required = {
        "mechanism",
        "profile_name",
        "source_commit",
        "core_commit",
        "run_status",
        "p1",
        "p2",
        "external_independence",
        "ordinary_output_orthogonality",
        "determinism",
        "artifact_hashes",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ResultValidationError(f"missing mechanism fields: {missing}")
    if expected_mechanism is not None and result["mechanism"] != expected_mechanism:
        raise ResultValidationError(f"mechanism mismatch: {result['mechanism']} != {expected_mechanism}")
    if result["mechanism"] not in MECHANISMS:
        raise ResultValidationError(f"unknown mechanism: {result['mechanism']}")
    if not re.fullmatch(r"[0-9a-f]{40}", result["source_commit"]):
        raise ResultValidationError("source_commit is not a full commit SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", result["core_commit"]):
        raise ResultValidationError("core_commit is not a full commit SHA")
    for section, fields in (("p1", REQUIRED_P1_FIELDS), ("p2", REQUIRED_P2_FIELDS)):
        absent = [field for field in fields if field not in result[section]]
        if absent:
            raise ResultValidationError(f"{section} missing fields: {absent}")
    if result["run_status"] != "PASS" or result["p1"]["status"] != "PASS" or result["p2"]["status"] != "PASS":
        raise ResultValidationError("mechanism, P1, and P2 must all be PASS")
    p2 = result["p2"]
    if p2["witness_count"] < 1 or p2["valid_witness_count"] != p2["witness_count"]:
        raise ResultValidationError("P2 witness set is empty or contains an invalid witness")
    if p2["snapshot_distinct"] is not True or p2["target_equal"] is not True:
        raise ResultValidationError("P2 does not establish distinct snapshots with equal target projection")
    if len(p2["witness_summaries"]) != p2["witness_count"]:
        raise ResultValidationError("P2 witness summaries are incomplete")
    if result["external_independence"].get("rating") not in {"A", "B", "C"}:
        raise ResultValidationError("external independence rating is invalid")
    if result["ordinary_output_orthogonality"].get("status") not in {"PASS", "NOT_APPLICABLE"}:
        raise ResultValidationError("ordinary output orthogonality failed")
    hashes = result["artifact_hashes"]
    if not hashes or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
        raise ResultValidationError("artifact hashes are missing or malformed")
    tokens = {value.upper() for value in _walk_strings(result)}
    forbidden = sorted(tokens & FORBIDDEN_SUCCESS_TOKENS)
    if forbidden:
        raise ResultValidationError(f"forbidden success token(s): {forbidden}")


def validate_complete_result_set(results: dict[str, dict[str, Any]]) -> None:
    if set(results) != set(MECHANISMS):
        missing = sorted(set(MECHANISMS) - set(results))
        extra = sorted(set(results) - set(MECHANISMS))
        raise ResultValidationError(f"mechanism set mismatch; missing={missing}, extra={extra}")
    for mechanism in MECHANISMS:
        validate_mechanism_result(results[mechanism], expected_mechanism=mechanism)


def stable_scientific_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"determinism", "artifact_hashes", "execution_receipt"}
    }

