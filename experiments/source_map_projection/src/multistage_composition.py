"""Compose final-to-original mappings only through Core GeneratedOrigin bridges."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from urllib.parse import urljoin

from generation_relation_core.snapshots import ValidatedSnapshot, validate_snapshot


class CompositionError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(reason_code if not detail else f"{reason_code}:{detail}")


def compose_core_relations(snapshot: ValidatedSnapshot, registry) -> dict[str, Any]:
    validate_snapshot(snapshot, registry)
    tables = snapshot.tables
    supports = {row["support_id"]: row for row in tables.perceptual_support_records}
    sources = {row["source_information_id"]: row for row in tables.source_information_records}
    origins = {row["generated_origin_id"]: row for row in tables.generated_origins}
    by_support: dict[str, list[dict]] = defaultdict(list)
    for binding in tables.generation_bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            by_support[outcome["support_id"]].append(binding)
    final_supports = [
        row for row in supports.values()
        if row["support_payload"]["stage_id"] == "multistage_2"
    ]
    records = []
    binding_paths = []
    broken = 0
    ambiguity = 0
    cycles = 0
    for final in sorted(final_supports, key=lambda row: (
        row["support_payload"]["generated_line"], row["support_payload"]["generated_column"]
    )):
        tail = [row for row in by_support[final["support_id"]] if row["relation_role"].startswith("source_map_anchor:")]
        if len(tail) != 1:
            ambiguity += 1
            continue
        origin_ref = tail[0]["origin_reference"]
        if origin_ref["kind"] != "generated_origin":
            broken += 1
            continue
        origin = origins.get(origin_ref["generated_origin_id"])
        if origin is None:
            broken += 1
            continue
        prior_id = origin["origin_payload"].get("prior_support_id")
        if prior_id == final["support_id"]:
            cycles += 1
            continue
        prior = supports.get(prior_id)
        if prior is None:
            broken += 1
            continue
        head = [row for row in by_support[prior_id] if row["relation_role"].startswith("source_map_anchor:")]
        if len(head) != 1 or head[0]["origin_reference"]["kind"] != "registered_source":
            ambiguity += 1
            continue
        source = sources[head[0]["origin_reference"]["source_information_id"]]["source_payload"]
        root = prior["support_payload"]["source_root"]
        prefix = "" if root is None else root if root.endswith("/") else root + "/"
        records.append({
            "generated_file": final["support_payload"]["generated_artifact"],
            "generated_line": final["support_payload"]["generated_line"],
            "generated_column": final["support_payload"]["generated_column"],
            "mapped": True,
            "original_source": urljoin("file:///experiment/maps/multistage-1.map", prefix + source["source_file"]),
            "original_line": source["source_start"]["line"],
            "original_column": source["source_start"]["column"],
            "original_name": source.get("original_name"),
        })
        binding_paths.append([head[0]["generation_binding_id"], tail[0]["generation_binding_id"]])
    direct_shortcuts = [
        row for row in tables.generation_bindings
        if row["origin_reference"]["kind"] == "registered_source"
        and row["outcome_reference"].get("support_id") in {item["support_id"] for item in final_supports}
    ]
    return {
        "records": records,
        "binding_paths": binding_paths,
        "composed_mapping_count": len(records),
        "broken_generated_origin_bridge_count": broken,
        "ambiguity_count": ambiguity,
        "cycle_count": cycles,
        "invented_transitive_mapping_count": 0,
        "direct_original_to_final_binding_count": len(direct_shortcuts),
        "status": "PASS" if records and not any((broken, ambiguity, cycles, direct_shortcuts)) else "FAIL",
    }
