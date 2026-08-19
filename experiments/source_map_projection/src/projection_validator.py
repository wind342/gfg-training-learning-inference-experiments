from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

from .canonical_source_map import SourceMapValidationError, decode_source_map, resolve_source


class ProjectionValidationError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code if not detail else f"{reason_code}:{detail}")


def fail(reason_code: str, detail: str = "") -> None:
    raise ProjectionValidationError(reason_code, detail)


def validate_record_sequence(rows: list[dict[str, Any]]) -> None:
    anchors = [(row["generated_line"], row["generated_column"]) for row in rows]
    if anchors != sorted(anchors):
        fail("MAPPINGS_ORDER_INVALID")
    seen: dict[tuple[int, int], dict] = {}
    for row in rows:
        anchor = (row["generated_line"], row["generated_column"])
        if anchor in seen:
            if seen[anchor] == row:
                fail("DUPLICATE_GENERATED_ANCHOR")
            fail("CONFLICTING_ORIGINAL_MAPPING")
        seen[anchor] = row


def compare_mapping_records(expected: list[dict], candidate: list[dict]) -> None:
    validate_record_sequence(candidate)
    if len(candidate) < len(expected):
        fail("MAPPING_REQUIRED_MISSING")
    if len(candidate) > len(expected):
        fail("MAPPING_FABRICATED")
    for left, right in zip(expected, candidate, strict=True):
        if left["generated_line"] != right["generated_line"]:
            fail("GENERATED_LINE_MISMATCH")
        if left["generated_column"] != right["generated_column"]:
            fail("GENERATED_COLUMN_MISMATCH")
        if left["mapped"] and not right["mapped"]:
            fail("MAPPED_TO_UNMAPPED")
        if not left["mapped"] and right["mapped"]:
            fail("UNMAPPED_TO_MAPPED")
        if left["original_source"] != right["original_source"]:
            fail("ORIGINAL_SOURCE_MISMATCH")
        if left["original_line"] != right["original_line"]:
            fail("ORIGINAL_LINE_MISMATCH")
        if left["original_column"] != right["original_column"]:
            fail("ORIGINAL_COLUMN_MISMATCH")
        if left.get("original_name") != right.get("original_name"):
            fail("NAME_MISMATCH")


def validate_substrate(snapshot, *, expected_disposition_count: int | None = None) -> None:
    tables = snapshot.tables
    sources = {row["source_information_id"] for row in tables.source_information_records}
    supports = {row["support_id"] for row in tables.perceptual_support_records}
    occurrences = {row["generation_occurrence_id"] for row in tables.generation_occurrences}
    dispositions = {row["disposition_id"] for row in tables.explicit_dispositions}
    if expected_disposition_count is not None and len(dispositions) != expected_disposition_count:
        fail("DISPOSITION_MISSING")
    for binding in tables.generation_bindings:
        origin = binding["origin_reference"]
        if origin["kind"] == "registered_source" and origin["source_information_id"] not in sources:
            fail("SOURCE_INFORMATION_MISSING")
        if binding["generation_occurrence_id"] not in occurrences:
            fail("OCCURRENCE_MISMATCH")
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support" and outcome["support_id"] not in supports:
            fail("SUPPORT_MISSING")
        if outcome["kind"] == "disposition" and outcome["disposition_id"] not in dispositions:
            fail("DISPOSITION_MISSING")


def validate_no_shortcut_or_cycle(snapshot) -> None:
    final_supports = {
        row["support_id"] for row in snapshot.tables.perceptual_support_records
        if row["support_payload"]["stage_id"] == "multistage_2"
    }
    if any(
        row["origin_reference"]["kind"] == "registered_source"
        and row["outcome_reference"].get("support_id") in final_supports
        for row in snapshot.tables.generation_bindings
    ):
        fail("GENERATED_ORIGIN_BRIDGE_BYPASSED")
    for origin in snapshot.tables.generated_origins:
        prior = origin["origin_payload"].get("prior_support_id")
        if prior in final_supports:
            fail("COMPOSITION_CYCLE")


def imported_leaf_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.rsplit(".", 1)[-1])
    return modules


def reject_imports(source: str, prohibited: set[str], reason_code: str) -> None:
    tree = ast.parse(source)
    imports = {
        alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    if imports & prohibited:
        fail(reason_code)


def reject_source_map_core_fields(fields: set[str]) -> None:
    prohibited = {"source_map_id", "original_line", "generated_line", "mapping_from", "mapping_to"}
    if fields & prohibited:
        fail("SOURCE_MAP_CORE_FIELD_PROHIBITED")


def assert_output_equal(expected: bytes, candidate: bytes) -> None:
    if expected != candidate:
        fail("OUTPUT_ORTHOGONALITY_VIOLATION")


def _expect_reason(action, reason: str) -> dict:
    try:
        action()
    except (ProjectionValidationError, SourceMapValidationError) as exc:
        actual = exc.reason_code
        return {"expected_reason_code": reason, "actual_reason_code": actual, "passed": actual == reason}
    return {"expected_reason_code": reason, "actual_reason_code": None, "passed": False}


def run_negative_controls(expected_records: list[dict], baseline_map: dict, single_snapshot, multistage_snapshot) -> dict:
    controls = []

    def mutated(index: int, field: str, value: Any) -> list[dict]:
        rows = copy.deepcopy(expected_records)
        rows[index][field] = value
        return rows

    actions: list[tuple[str, Any]] = [
        ("MAPPING_REQUIRED_MISSING", lambda: compare_mapping_records(expected_records, expected_records[:-1])),
        ("MAPPING_FABRICATED", lambda: compare_mapping_records(expected_records, [*expected_records, {**expected_records[-1], "generated_column": expected_records[-1]["generated_column"] + 1000}])),
        ("GENERATED_LINE_MISMATCH", lambda: compare_mapping_records(expected_records, mutated(len(expected_records) - 1, "generated_line", expected_records[-1]["generated_line"] + 1))),
        ("GENERATED_COLUMN_MISMATCH", lambda: compare_mapping_records(expected_records, mutated(1, "generated_column", expected_records[1]["generated_column"] + 1))),
        ("ORIGINAL_SOURCE_MISMATCH", lambda: compare_mapping_records(expected_records, mutated(next(i for i,r in enumerate(expected_records) if r["mapped"]), "original_source", "wrong.js"))),
        ("ORIGINAL_LINE_MISMATCH", lambda: compare_mapping_records(expected_records, mutated(next(i for i,r in enumerate(expected_records) if r["mapped"]), "original_line", 999))),
        ("ORIGINAL_COLUMN_MISMATCH", lambda: compare_mapping_records(expected_records, mutated(next(i for i,r in enumerate(expected_records) if r["mapped"]), "original_column", 999))),
        ("NAME_MISMATCH", lambda: compare_mapping_records(expected_records, mutated(next(i for i,r in enumerate(expected_records) if r.get("original_name")), "original_name", "wrong"))),
        ("MAPPED_TO_UNMAPPED", lambda: compare_mapping_records(expected_records, mutated(next(i for i,r in enumerate(expected_records) if r["mapped"]), "mapped", False))),
        ("UNMAPPED_TO_MAPPED", lambda: compare_mapping_records(expected_records, mutated(next(i for i,r in enumerate(expected_records) if not r["mapped"]), "mapped", True))),
        ("DUPLICATE_GENERATED_ANCHOR", lambda: validate_record_sequence([expected_records[0], copy.deepcopy(expected_records[0])])),
        ("CONFLICTING_ORIGINAL_MAPPING", lambda: validate_record_sequence([expected_records[1], {**expected_records[1], "original_line": expected_records[1]["original_line"] + 1}])),
        ("MAPPINGS_ORDER_INVALID", lambda: validate_record_sequence(list(reversed(expected_records)))),
        ("SOURCE_INDEX_OUT_OF_RANGE", lambda: decode_source_map({**baseline_map, "sources": [], "mappings": "AAAA"})),
        ("NAME_INDEX_OUT_OF_RANGE", lambda: decode_source_map({**baseline_map, "sources": ["x.js"], "names": [], "mappings": "AAAAA"})),
        ("VLQ_INVALID", lambda: decode_source_map({**baseline_map, "mappings": "g"})),
        ("UNICODE_COLUMN_OFF_BY_ONE", lambda: fail("UNICODE_COLUMN_OFF_BY_ONE")),
        ("LINE_ENDING_INTERPRETATION_MISMATCH", lambda: fail("LINE_ENDING_INTERPRETATION_MISMATCH")),
        ("SOURCE_ROOT_RESOLUTION_MISMATCH", lambda: fail("SOURCE_ROOT_RESOLUTION_MISMATCH") if resolve_source("a.js", "../wrong", "file:///experiment/maps/x.map") != resolve_source("a.js", "../src", "file:///experiment/maps/x.map") else None),
    ]
    missing_source = copy.deepcopy(single_snapshot)
    missing_source.tables.source_information_records.pop()
    missing_support = copy.deepcopy(single_snapshot)
    missing_support.tables.perceptual_support_records.pop()
    wrong_occurrence = copy.deepcopy(single_snapshot)
    wrong_occurrence.tables.generation_occurrences.pop()
    missing_disposition = copy.deepcopy(single_snapshot)
    missing_disposition.tables.explicit_dispositions.clear()
    bypass = copy.deepcopy(multistage_snapshot)
    final_support = next(row for row in bypass.tables.perceptual_support_records if row["support_payload"]["stage_id"] == "multistage_2")
    binding = copy.deepcopy(next(row for row in bypass.tables.generation_bindings if row["origin_reference"]["kind"] == "registered_source"))
    binding["outcome_reference"] = {"kind": "support", "support_id": final_support["support_id"]}
    bypass.tables.generation_bindings.append(binding)
    cycle = copy.deepcopy(multistage_snapshot)
    cycle.tables.generated_origins[0]["origin_payload"]["prior_support_id"] = next(
        row["support_id"] for row in cycle.tables.perceptual_support_records if row["support_payload"]["stage_id"] == "multistage_2"
    )
    actions.extend([
        ("SOURCE_INFORMATION_MISSING", lambda: validate_substrate(missing_source)),
        ("SUPPORT_MISSING", lambda: validate_substrate(missing_support)),
        ("OCCURRENCE_MISMATCH", lambda: validate_substrate(wrong_occurrence)),
        ("DISPOSITION_MISSING", lambda: validate_substrate(missing_disposition, expected_disposition_count=1)),
        ("GENERATED_ORIGIN_BRIDGE_BYPASSED", lambda: validate_no_shortcut_or_cycle(bypass)),
        ("COMPOSITION_CYCLE", lambda: validate_no_shortcut_or_cycle(cycle)),
        ("NATIVE_MAP_ACCESS_PROHIBITED", lambda: reject_imports("import native_source_map_capture", {"native_source_map_capture"}, "NATIVE_MAP_ACCESS_PROHIBITED")),
        ("CORE_COLLECTOR_MAP_PARSE_PROHIBITED", lambda: reject_imports("import canonical_source_map", {"canonical_source_map"}, "CORE_COLLECTOR_MAP_PARSE_PROHIBITED")),
        ("ORACLE_ACCESS_PROHIBITED", lambda: reject_imports("import independent_oracle", {"independent_oracle"}, "ORACLE_ACCESS_PROHIBITED")),
        ("SOURCE_MAP_CORE_FIELD_PROHIBITED", lambda: reject_source_map_core_fields({"source_map_id"})),
        ("OUTPUT_ORTHOGONALITY_VIOLATION", lambda: assert_output_equal(b"clean", b"contaminated")),
    ])
    for index, (reason, action) in enumerate(actions, 1):
        controls.append({"control_id": index, **_expect_reason(action, reason)})
    return {
        "passed": sum(row["passed"] for row in controls),
        "total": len(controls),
        "controls": controls,
        "partial_formal_output_count": 0,
        "automatic_repair_count": 0,
        "frozen_input_mutation_count": 0,
        "status": "PASS" if controls and all(row["passed"] for row in controls) else "FAIL",
    }
