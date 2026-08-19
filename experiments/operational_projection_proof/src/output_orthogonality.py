from __future__ import annotations

import hashlib
from typing import Iterable

from .errors import ProjectionProofError


FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "tuple_id",
        "origin_id",
        "occurrence_id",
        "binding_id",
        "provenance",
        "lineage",
        "token",
        "stable_tuple_label",
    }
)


def evaluate_output_orthogonality(
    *,
    disabled_csv: bytes,
    enabled_csv: bytes,
    disabled_json: bytes,
    enabled_json: bytes,
    enabled_field_names: Iterable[str],
) -> dict:
    forbidden = sorted(set(enabled_field_names) & FORBIDDEN_OUTPUT_FIELDS)
    csv_equal = disabled_csv == enabled_csv
    json_equal = disabled_json == enabled_json
    supported = csv_equal and json_equal and not forbidden
    return {
        "csv_byte_identical": csv_equal,
        "json_byte_identical": json_equal,
        "csv_sha256": hashlib.sha256(enabled_csv).hexdigest(),
        "json_sha256": hashlib.sha256(enabled_json).hexdigest(),
        "forbidden_fields": forbidden,
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
    }


def require_output_orthogonality(report: dict) -> None:
    if report.get("forbidden_fields"):
        raise ProjectionProofError("CONTROL_PLANE_OUTPUT_CONTAMINATION")
    if not report.get("csv_byte_identical") or not report.get("json_byte_identical"):
        raise ProjectionProofError("OUTPUT_ORTHOGONALITY_MISMATCH")
