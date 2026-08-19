from __future__ import annotations

import json
from collections import Counter
from typing import Any


KIND_ORDER = {
    "entity": 0,
    "activity": 1,
    "agent": 2,
    "usage": 3,
    "generation": 4,
    "derivation": 5,
    "association": 6,
}


def sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: (KIND_ORDER[item["kind"]], item["id"]))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_normalized_records(records: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        violations.append("DUPLICATE_RELATION_OR_NODE_IDENTIFIER")
    by_id = {record["id"]: record for record in records}
    by_kind = {kind: {record["id"] for record in records if record["kind"] == kind} for kind in KIND_ORDER}
    generations_by_entity: Counter[str] = Counter()
    for record in records:
        kind = record["kind"]
        if kind == "usage":
            if record["activity"] not in by_kind["activity"] or record["entity"] not in by_kind["entity"]:
                violations.append(f"DANGLING_USAGE:{record['id']}")
        elif kind == "generation":
            if record["activity"] not in by_kind["activity"] or record["entity"] not in by_kind["entity"]:
                violations.append(f"DANGLING_GENERATION:{record['id']}")
            generations_by_entity[record["entity"]] += 1
        elif kind == "derivation":
            required = (
                record["generated_entity"] in by_kind["entity"],
                record["used_entity"] in by_kind["entity"],
                record["activity"] in by_kind["activity"],
                record["generation"] in by_kind["generation"],
                record["usage"] in by_kind["usage"],
            )
            if not all(required):
                violations.append(f"DANGLING_DERIVATION:{record['id']}")
            else:
                generation = by_id[record["generation"]]
                usage = by_id[record["usage"]]
                if (
                    generation["entity"] != record["generated_entity"]
                    or generation["activity"] != record["activity"]
                    or usage["entity"] != record["used_entity"]
                    or usage["activity"] != record["activity"]
                ):
                    violations.append(f"DERIVATION_REFERENCE_MISMATCH:{record['id']}")
        elif kind == "association":
            if record["activity"] not in by_kind["activity"] or record["agent"] not in by_kind["agent"]:
                violations.append(f"DANGLING_ASSOCIATION:{record['id']}")
    for entity, count in generations_by_entity.items():
        if count != 1:
            violations.append(f"UNIQUE_GENERATION:{entity}:{count}")
    return sorted(violations)

