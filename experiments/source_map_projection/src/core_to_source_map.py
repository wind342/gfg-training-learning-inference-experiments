"""Project ordinary ECMA-426 maps only from validated Core Snapshot facts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from generation_relation_core.snapshots import ValidatedSnapshot, validate_snapshot

from .canonical_source_map import encode_source_map


class ProjectionError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code if not detail else f"{reason_code}:{detail}")


def _origin_payload(snapshot: ValidatedSnapshot, origin_reference: dict) -> dict:
    tables = snapshot.tables
    if origin_reference["kind"] == "registered_source":
        row = next((
            item for item in tables.source_information_records
            if item["source_information_id"] == origin_reference["source_information_id"]
        ), None)
        if row is None:
            raise ProjectionError("SOURCE_INFORMATION_MISSING")
        return row["source_payload"]
    row = next((
        item for item in tables.generated_origins
        if item["generated_origin_id"] == origin_reference["generated_origin_id"]
    ), None)
    if row is None:
        raise ProjectionError("GENERATED_ORIGIN_BRIDGE_MISSING")
    return row["origin_payload"]


def project_stage(
    snapshot: ValidatedSnapshot,
    registry,
    stage_id: str,
) -> dict[str, Any]:
    validate_snapshot(snapshot, registry)
    tables = snapshot.tables
    supports = [
        row for row in tables.perceptual_support_records
        if row["support_payload"]["stage_id"] == stage_id
    ]
    if not supports:
        raise ProjectionError("STAGE_SUPPORTS_MISSING", stage_id)
    bindings_by_support: dict[str, list[dict]] = defaultdict(list)
    for binding in tables.generation_bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            bindings_by_support[outcome["support_id"]].append(binding)
    records = []
    source_contents: dict[str, str] = {}
    generated_files = {row["support_payload"]["generated_artifact"] for row in supports}
    source_roots = {row["support_payload"]["source_root"] for row in supports}
    if len(generated_files) != 1 or len(source_roots) != 1:
        raise ProjectionError("STAGE_METADATA_CONFLICT")
    generated_file = next(iter(generated_files))
    source_root = next(iter(source_roots))
    for support in sorted(
        supports,
        key=lambda row: (
            row["support_payload"]["generated_line"],
            row["support_payload"]["generated_column"],
            row["support_id"],
        ),
    ):
        payload = support["support_payload"]
        bindings = bindings_by_support.get(support["support_id"], [])
        anchors = [row for row in bindings if row["relation_role"].startswith("source_map_anchor:")]
        if payload["mapping_eligible"]:
            if len(anchors) != 1:
                raise ProjectionError("CONFLICTING_ORIGINAL_MAPPING", support["support_id"])
            origin = _origin_payload(snapshot, anchors[0]["origin_reference"])
            source_file = origin["source_file"]
            source_contents.setdefault(source_file, origin["source_content"])
            if source_contents[source_file] != origin["source_content"]:
                raise ProjectionError("SOURCE_CONTENT_CONFLICT", source_file)
            start = origin["source_start"]
            records.append({
                "generated_file": generated_file,
                "generated_line": payload["generated_line"],
                "generated_column": payload["generated_column"],
                "mapped": True,
                "original_source": source_file,
                "original_line": start["line"],
                "original_column": start["column"],
                "original_name": origin.get("original_name"),
            })
        else:
            if anchors:
                raise ProjectionError("UNMAPPED_TO_MAPPED", support["support_id"])
            records.append({
                "generated_file": generated_file,
                "generated_line": payload["generated_line"],
                "generated_column": payload["generated_column"],
                "mapped": False,
                "original_source": None,
                "original_line": None,
                "original_column": None,
                "original_name": None,
            })
    anchors = [(row["generated_line"], row["generated_column"]) for row in records]
    if len(anchors) != len(set(anchors)):
        raise ProjectionError("DUPLICATE_GENERATED_ANCHOR")
    document = encode_source_map(
        records,
        generated_file=generated_file,
        source_root=source_root,
        source_contents=source_contents,
    )
    return {
        "stage_id": stage_id,
        "generated_file": generated_file,
        "source_root": source_root,
        "document": document,
        "canonical_records": records,
        "mapping_count": len(records),
        "mapped_count": sum(row["mapped"] for row in records),
        "unmapped_count": sum(not row["mapped"] for row in records),
        "projection_input": "ValidatedSnapshot only",
    }
