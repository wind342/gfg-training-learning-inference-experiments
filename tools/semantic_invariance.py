from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.snapshots import CoreV3Tables


TABLE_NAMES = (
    "source_information_records",
    "generation_occurrences",
    "generated_origins",
    "perceptual_support_records",
    "explicit_dispositions",
    "generation_bindings",
)


def _table_mapping(value: CoreV3Tables | Mapping[str, list[dict]] | Path | str) -> dict[str, list[dict]]:
    if isinstance(value, CoreV3Tables):
        return {name: getattr(value, name) for name in TABLE_NAMES}
    if isinstance(value, Mapping):
        return {name: list(value.get(name, [])) for name in TABLE_NAMES}
    root = Path(value)
    tables: dict[str, list[dict]] = {}
    for name in TABLE_NAMES:
        json_path = root / f"{name}.json"
        jsonl_path = root / f"{name}.jsonl"
        if json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            tables[name] = payload if isinstance(payload, list) else payload.get("records", [])
        elif jsonl_path.is_file():
            tables[name] = [
                json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line
            ]
        else:
            tables[name] = []
    return tables


def normalized_generation_relations(
    value: CoreV3Tables | Mapping[str, list[dict]] | Path | str,
) -> list[dict[str, Any]]:
    tables = _table_mapping(value)
    sources = {row["source_information_id"]: row for row in tables["source_information_records"]}
    occurrences = {
        row["generation_occurrence_id"]: row for row in tables["generation_occurrences"]
    }
    generated = {row["generated_origin_id"]: row for row in tables["generated_origins"]}
    supports = {row["support_id"]: row for row in tables["perceptual_support_records"]}
    dispositions = {row["disposition_id"]: row for row in tables["explicit_dispositions"]}
    normalized = []
    for binding in tables["generation_bindings"]:
        origin_ref = binding["origin_reference"]
        if origin_ref["kind"] == "registered_source":
            source = sources[origin_ref["source_information_id"]]
            origin = {
                "kind": "registered_source",
                "identity": source["source_identity"],
                "parent": source["source_parent_id"],
                "granularity": source["source_granularity"],
                "payload_sha256": source["source_payload_sha256"],
            }
        else:
            row = generated[origin_ref["generated_origin_id"]]
            origin = {
                "kind": "generated_origin",
                "origin_type": row["origin_type"],
                "payload_sha256": row["origin_payload_sha256"],
            }
        occurrence = occurrences[binding["generation_occurrence_id"]]
        occurrence_semantics = {
            "stage": occurrence["occurrence_stage"],
            "type": occurrence["occurrence_type"],
            "stable_instance_key": occurrence["stable_instance_key"],
            "index": occurrence["occurrence_index"],
            "payload_sha256": occurrence["occurrence_payload_sha256"],
        }
        outcome_ref = binding["outcome_reference"]
        if outcome_ref["kind"] == "support":
            support = supports[outcome_ref["support_id"]]
            outcome = {
                "kind": "support",
                "support_space_id": support["support_space_id"],
                "payload_sha256": support["support_payload_sha256"],
                "status": support["support_status"],
            }
        else:
            disposition = dispositions[outcome_ref["disposition_id"]]
            outcome = {
                "kind": "disposition",
                "category": disposition["core_disposition_category"],
                "reason": disposition["domain_reason_code"],
                "payload_sha256": disposition["disposition_payload_sha256"],
            }
        normalized.append(
            {
                "domain_scope_id": binding["domain_scope_id"],
                "origin": origin,
                "occurrence": occurrence_semantics,
                "outcome": outcome,
                "relation_role": binding["relation_role"],
            }
        )
    return sorted(normalized, key=canonical_bytes)


def semantic_digest(value: CoreV3Tables | Mapping[str, list[dict]] | Path | str) -> str:
    return hashlib.sha256(canonical_bytes(normalized_generation_relations(value))).hexdigest()


def compare_semantics(left, right) -> dict[str, Any]:
    left_rows = normalized_generation_relations(left)
    right_rows = normalized_generation_relations(right)
    return {
        "equivalent": canonical_bytes(left_rows) == canonical_bytes(right_rows),
        "left_count": len(left_rows),
        "right_count": len(right_rows),
        "left_sha256": hashlib.sha256(canonical_bytes(left_rows)).hexdigest(),
        "right_sha256": hashlib.sha256(canonical_bytes(right_rows)).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare normalized Core v3 generation-relation semantics.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    report = compare_semantics(args.left, args.right)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
