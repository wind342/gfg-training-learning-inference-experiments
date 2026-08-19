from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

from generation_relation_core.canonical import canonical_bytes


BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
BASE64_INDEX = {char: index for index, char in enumerate(BASE64)}
MAX_I32 = 2_147_483_647


class SourceMapValidationError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code if not detail else f"{reason_code}:{detail}")


def _fail(reason_code: str, detail: str = "") -> None:
    raise SourceMapValidationError(reason_code, detail)


def encode_vlq(value: int) -> str:
    sign = 1 if value < 0 else 0
    remaining = (abs(value) << 1) | sign
    result = []
    while True:
        digit = remaining & 31
        remaining >>= 5
        if remaining:
            digit |= 32
        result.append(BASE64[digit])
        if not remaining:
            return "".join(result)


def decode_vlq_fields(segment: str) -> list[int]:
    if not segment:
        _fail("VLQ_INVALID", "EMPTY_SEGMENT")
    fields: list[int] = []
    value = 0
    shift = 0
    continuation_open = False
    for char in segment:
        digit = BASE64_INDEX.get(char)
        if digit is None:
            _fail("VLQ_INVALID", "NON_BASE64_CHARACTER")
        continuation_open = bool(digit & 32)
        value += (digit & 31) << shift
        if continuation_open:
            shift += 5
            continue
        negative = value & 1
        magnitude = value >> 1
        fields.append(-magnitude if negative else magnitude)
        value = 0
        shift = 0
    if continuation_open:
        _fail("VLQ_INVALID", "MISSING_CONTINUATION_DIGIT")
    return fields


def _optional_string(value: dict, field: str) -> str | None:
    result = value.get(field)
    if result is not None and not isinstance(result, str):
        _fail("SOURCE_MAP_FIELD_TYPE_INVALID", field)
    return result


def _optional_string_list(value: dict, field: str, *, allow_null: bool) -> list[str | None]:
    result = value.get(field, [])
    if not isinstance(result, list):
        _fail("SOURCE_MAP_FIELD_TYPE_INVALID", field)
    for item in result:
        if not isinstance(item, str) and not (allow_null and item is None):
            _fail("SOURCE_MAP_FIELD_TYPE_INVALID", field)
    return result


def resolve_source(source: str | None, source_root: str | None, base_url: str) -> str | None:
    if source is None:
        return None
    prefix = ""
    if source_root is not None:
        prefix = source_root if source_root.endswith("/") else source_root + "/"
    return urljoin(base_url, prefix + source)


@dataclass(frozen=True)
class DecodedSourceMap:
    document: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    def canonical_records_bytes(self) -> bytes:
        return canonical_bytes(list(self.records))


def decode_source_map(
    value: bytes | str | dict[str, Any],
    *,
    base_url: str = "file:///experiment/maps/generated.js.map",
    require_strict_order: bool = False,
) -> DecodedSourceMap:
    if isinstance(value, bytes):
        try:
            document = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail("SOURCE_MAP_JSON_INVALID", type(exc).__name__)
    elif isinstance(value, str):
        try:
            document = json.loads(value)
        except json.JSONDecodeError as exc:
            _fail("SOURCE_MAP_JSON_INVALID", str(exc))
    else:
        document = value
    if not isinstance(document, dict):
        _fail("SOURCE_MAP_JSON_INVALID", "ROOT_NOT_OBJECT")
    if "sections" in document:
        _fail("INDEXED_MAP_OUT_OF_SCOPE")
    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 3:
        _fail("SOURCE_MAP_VERSION_INVALID")
    mappings = document.get("mappings")
    if not isinstance(mappings, str):
        _fail("SOURCE_MAP_FIELD_TYPE_INVALID", "mappings")
    if "sources" not in document:
        _fail("SOURCE_MAP_FIELD_TYPE_INVALID", "sources")
    sources = _optional_string_list(document, "sources", allow_null=True)
    names = _optional_string_list(document, "names", allow_null=False)
    source_root = _optional_string(document, "sourceRoot")
    _optional_string(document, "file")
    sources_content = _optional_string_list(document, "sourcesContent", allow_null=True)
    ignore_list = document.get("ignoreList", [])
    if not isinstance(ignore_list, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ignore_list
    ):
        _fail("SOURCE_MAP_FIELD_TYPE_INVALID", "ignoreList")
    if len(ignore_list) != len(set(ignore_list)) or any(item >= len(sources) for item in ignore_list):
        _fail("SOURCE_INDEX_OUT_OF_RANGE", "ignoreList")

    source_index = 0
    original_line = 0
    original_column = 0
    name_index = 0
    records: list[dict[str, Any]] = []
    previous_anchor: tuple[int, int] | None = None
    for generated_line, group in enumerate(mappings.split(";")):
        generated_column = 0
        if not group:
            continue
        for segment in group.split(","):
            fields = decode_vlq_fields(segment)
            if len(fields) not in (1, 4, 5):
                _fail("MAPPING_SEGMENT_FIELD_COUNT_INVALID", str(len(fields)))
            generated_column += fields[0]
            if generated_column < 0 or generated_column > MAX_I32:
                _fail("GENERATED_COLUMN_OUT_OF_RANGE")
            anchor = (generated_line, generated_column)
            if previous_anchor is not None:
                if anchor == previous_anchor:
                    _fail("DUPLICATE_GENERATED_ANCHOR")
                if require_strict_order and anchor < previous_anchor:
                    _fail("MAPPINGS_ORDER_INVALID")
            previous_anchor = anchor
            if len(fields) == 1:
                records.append({
                    "generated_file": document.get("file"),
                    "generated_line": generated_line,
                    "generated_column": generated_column,
                    "mapped": False,
                    "original_source": None,
                    "resolved_original_source": None,
                    "original_line": None,
                    "original_column": None,
                    "original_name": None,
                })
                continue
            source_index += fields[1]
            original_line += fields[2]
            original_column += fields[3]
            if source_index < 0 or source_index >= len(sources):
                _fail("SOURCE_INDEX_OUT_OF_RANGE")
            if original_line < 0 or original_line > MAX_I32:
                _fail("ORIGINAL_LINE_OUT_OF_RANGE")
            if original_column < 0 or original_column > MAX_I32:
                _fail("ORIGINAL_COLUMN_OUT_OF_RANGE")
            original_name = None
            if len(fields) == 5:
                name_index += fields[4]
                if name_index < 0 or name_index >= len(names):
                    _fail("NAME_INDEX_OUT_OF_RANGE")
                original_name = names[name_index]
            raw_source = sources[source_index]
            records.append({
                "generated_file": document.get("file"),
                "generated_line": generated_line,
                "generated_column": generated_column,
                "mapped": True,
                "original_source": raw_source,
                "resolved_original_source": resolve_source(raw_source, source_root, base_url),
                "original_line": original_line,
                "original_column": original_column,
                "original_name": original_name,
            })
    normalized = dict(document)
    normalized["sources"] = sources
    normalized["names"] = names
    if "sourcesContent" in document:
        normalized["sourcesContent"] = sources_content
    return DecodedSourceMap(normalized, tuple(records))


def encode_source_map(
    records: Iterable[dict[str, Any]],
    *,
    generated_file: str,
    source_root: str | None,
    source_contents: dict[str, str],
) -> dict[str, Any]:
    rows = list(records)
    anchors = [(row["generated_line"], row["generated_column"]) for row in rows]
    if anchors != sorted(anchors):
        _fail("MAPPINGS_ORDER_INVALID")
    if len(anchors) != len(set(anchors)):
        _fail("DUPLICATE_GENERATED_ANCHOR")
    sources: list[str] = []
    names: list[str] = []
    for row in rows:
        if not row["mapped"]:
            continue
        source = row["original_source"]
        if not isinstance(source, str):
            _fail("ORIGINAL_SOURCE_MISMATCH")
        if source not in sources:
            sources.append(source)
        name = row.get("original_name")
        if name is not None and name not in names:
            names.append(name)
    previous_source = 0
    previous_original_line = 0
    previous_original_column = 0
    previous_name = 0
    previous_generated_by_line: dict[int, int] = {}
    lines: list[list[str]] = []
    for row in rows:
        while len(lines) <= row["generated_line"]:
            lines.append([])
        previous_generated_column = previous_generated_by_line.get(row["generated_line"], 0)
        fields = [row["generated_column"] - previous_generated_column]
        previous_generated_by_line[row["generated_line"]] = row["generated_column"]
        if row["mapped"]:
            source = row["original_source"]
            source_index = sources.index(source)
            fields.extend([
                source_index - previous_source,
                row["original_line"] - previous_original_line,
                row["original_column"] - previous_original_column,
            ])
            previous_source = source_index
            previous_original_line = row["original_line"]
            previous_original_column = row["original_column"]
            if row.get("original_name") is not None:
                current_name = names.index(row["original_name"])
                fields.append(current_name - previous_name)
                previous_name = current_name
        lines[row["generated_line"]].append("".join(encode_vlq(field) for field in fields))
    document: dict[str, Any] = {
        "version": 3,
        "file": generated_file,
        "sources": sources,
        "names": names,
        "mappings": ";".join(",".join(line) for line in lines),
        "sourcesContent": [source_contents[source] for source in sources],
    }
    if source_root is not None:
        document["sourceRoot"] = source_root
    decode_source_map(document, require_strict_order=True)
    return document


def source_map_bytes(document: dict[str, Any]) -> bytes:
    return canonical_bytes(document) + b"\n"


def run_official_non_indexed_tests(test_root: Path) -> dict[str, Any]:
    manifest = json.loads((test_root / "source-map-spec-tests.json").read_text(encoding="utf-8"))
    excluded = []
    results = []
    for case in manifest["tests"]:
        name = case["name"]
        if name.startswith("indexMap") or name == "basicMappingWithIndexMap":
            excluded.append({"name": name, "reason": "indexed maps/sections are explicitly out of profile"})
            continue
        path = test_root / "resources" / case["sourceMapFile"]
        try:
            decode_source_map(path.read_bytes(), base_url=path.resolve().as_uri())
            actual = True
            reason = None
        except SourceMapValidationError as exc:
            actual = False
            reason = exc.reason_code
        results.append({
            "name": name,
            "expected_valid": bool(case["sourceMapIsValid"]),
            "actual_valid": actual,
            "reason_code": reason,
            "passed": actual == bool(case["sourceMapIsValid"]),
        })
    passed = sum(row["passed"] for row in results)
    return {
        "official_total": len(manifest["tests"]),
        "applicable_total": len(results),
        "applicable_passed": passed,
        "excluded_total": len(excluded),
        "excluded": excluded,
        "failures": [row for row in results if not row["passed"]],
        "status": "PASS" if passed == len(results) else "FAIL",
    }
