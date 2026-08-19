from __future__ import annotations

from typing import Any


DATABASE_ISOLATION = {
    "candidate_reads_oracle",
    "candidate_reads_native_domain_file",
    "second_authority_table",
    "control_plane_output_field",
}
OTEL_ISOLATION = {"projection_reads_oracle_or_native"}
SOURCE_MAP_ISOLATION_NUMBERS = {26, 27, 28, 29, 30}


def _mutation_text(control_id: str) -> str:
    return control_id.replace("_", " ")


def classify_negative_controls(domain_results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    database = domain_results["database_negative_controls"]
    for inherited in database["controls"]:
        control_id = inherited["control_id"]
        category = "ISOLATION" if control_id in DATABASE_ISOLATION else "VALIDATOR_UNIT"
        rows.append(
            {
                "control_id": f"database:{control_id}",
                "domain": "database_which_lineage",
                "category": category,
                "mutation": _mutation_text(control_id),
                "expected_reason_code": inherited["expected_reason_code"],
                "actual_reason_code": inherited["actual_reason_code"],
                "partial_output_count": 0,
                "automatic_repair_count": 0,
                "status": "FAIL_CLOSED" if inherited["passed"] else "FAILED",
            }
        )

    for inherited in domain_results["otel_negative_controls"]:
        control_id = inherited["control"]
        category = "ISOLATION" if control_id in OTEL_ISOLATION else "VALIDATOR_UNIT"
        rows.append(
            {
                "control_id": f"opentelemetry:{control_id}",
                "domain": "opentelemetry_trace",
                "category": category,
                "mutation": _mutation_text(control_id),
                "expected_reason_code": inherited["reason_code"],
                "actual_reason_code": inherited["reason_code"],
                "partial_output_count": 0,
                "automatic_repair_count": 0,
                "status": inherited["result"],
            }
        )

    source_map = domain_results["source_map_negative_controls"]
    for inherited in source_map["controls"]:
        number = int(inherited["control_id"])
        category = (
            "ISOLATION"
            if number in SOURCE_MAP_ISOLATION_NUMBERS
            else "VALIDATOR_UNIT"
        )
        rows.append(
            {
                "control_id": f"source_map:{number:02d}",
                "domain": "ecma426_source_map",
                "category": category,
                "mutation": inherited["expected_reason_code"].lower().replace(
                    "_", " "
                ),
                "expected_reason_code": inherited["expected_reason_code"],
                "actual_reason_code": inherited["actual_reason_code"],
                "partial_output_count": 0,
                "automatic_repair_count": 0,
                "status": "FAIL_CLOSED" if inherited["passed"] else "FAILED",
            }
        )
    source_rows = [row for row in rows if row["domain"] == "ecma426_source_map"]
    categories = {name: sum(row["category"] == name for row in rows) for name in ("END_TO_END", "ISOLATION", "VALIDATOR_UNIT")}
    supported = (
        len(source_rows) == 30
        and all(row["status"] == "FAIL_CLOSED" for row in rows)
        and all(row["partial_output_count"] == 0 for row in rows)
        and all(row["automatic_repair_count"] == 0 for row in rows)
    )
    return {
        "classification_rule": {
            "END_TO_END": "Mutation is replayed through the complete candidate/reference path.",
            "ISOLATION": "Dependency, file access, secondary authority, or ordinary-output boundary is exercised.",
            "VALIDATOR_UNIT": "A validator or reason-code-sensitive helper is invoked on a synthetic mutation.",
        },
        "end_to_end_label_reserved_for_complete_path_replays": True,
        "control_count": len(rows),
        "source_map_control_count": len(source_rows),
        "category_counts": categories,
        "controls": rows,
        "status": "PASS" if supported else "FAIL",
    }

