from __future__ import annotations

import hashlib

from experiments.database_lineage.src.result_serializer import csv_bytes, json_bytes

from .experiment_fixtures import Workload, ordinary_value_rows, run_captured


FORBIDDEN_OUTPUT_FIELDS = {
    "trace_id",
    "span_id",
    "core_id",
    "tuple_id",
    "lineage",
    "provenance",
    "token",
}


def _type_signature(rows) -> list[dict[str, str]]:
    return [
        {name: type(value).__qualname__ for name, value in row.values.items()}
        for row in rows
    ]


def run_four_mode_orthogonality(workload: Workload, *, run_id: str) -> dict:
    modes = {
        "A_core_off_otel_off": (False, False),
        "B_core_on_otel_off": (True, False),
        "C_core_off_otel_on": (False, True),
        "D_core_on_otel_on": (True, True),
    }
    runs = {
        name: run_captured(
            workload,
            run_id=run_id,
            core_enabled=core_enabled,
            otel_enabled=otel_enabled,
        )
        for name, (core_enabled, otel_enabled) in modes.items()
    }
    csv_payloads = {name: csv_bytes(run.rows) for name, run in runs.items()}
    json_payloads = {name: json_bytes(run.rows) for name, run in runs.items()}
    value_rows = {name: ordinary_value_rows(run.rows) for name, run in runs.items()}
    signatures = {name: _type_signature(run.rows) for name, run in runs.items()}
    schemas = {
        name: [list(row.values) for row in run.rows] for name, run in runs.items()
    }
    value_candidates = list(value_rows.values())
    signature_candidates = list(signatures.values())
    schema_candidates = list(schemas.values())
    output_fields = {
        field for run in runs.values() for row in run.rows for field in row.values
    }
    report = {
        "mode_count": 4,
        "csv_byte_identical": len(set(csv_payloads.values())) == 1,
        "json_byte_identical": len(set(json_payloads.values())) == 1,
        "business_values_equal": all(
            value == value_candidates[0] for value in value_candidates[1:]
        ),
        "business_types_equal": all(
            value == signature_candidates[0] for value in signature_candidates[1:]
        ),
        "schema_and_order_equal": all(
            value == schema_candidates[0] for value in schema_candidates[1:]
        ),
        "forbidden_output_fields": sorted(output_fields & FORBIDDEN_OUTPUT_FIELDS),
        "csv_sha256_by_mode": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in csv_payloads.items()
        },
        "json_sha256_by_mode": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in json_payloads.items()
        },
    }
    report["passed"] = all(
        (
            report["csv_byte_identical"],
            report["json_byte_identical"],
            report["business_values_equal"],
            report["business_types_equal"],
            report["schema_and_order_equal"],
            not report["forbidden_output_fields"],
        )
    )
    return report
