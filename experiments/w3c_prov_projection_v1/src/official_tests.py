from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RELATIONS = {"used", "wasGeneratedBy", "wasDerivedFrom", "wasAssociatedWith"}


def _parts(value: str) -> list[str]:
    return [item.strip() for item in value.split(",")]


def _statements(text: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        match = re.fullmatch(r"([A-Za-z]+)\((.*)\)", line)
        if match is None or match.group(1) not in RELATIONS:
            continue
        kind, inner = match.groups()
        if ";" in inner:
            identifier, body = inner.split(";", 1)
            relation_id = identifier.strip()
        else:
            body = inner
            relation_id = None
        result.append({"kind": kind, "id": relation_id, "values": _parts(body)})
    return result


def _compatible(left: str, right: str) -> bool:
    return left == "-" or right == "-" or left == right


def validate_official_subset_document(text: str) -> tuple[bool, list[str]]:
    """Execute the frozen c23/c24/c50/c51/c53/c55 subset.

    This is intentionally not a general PROV validator. It implements only the
    normalization and consistency checks selected in official_test_inventory.
    """

    reasons: list[str] = []
    statements = _statements(text)
    explicit_types: dict[str, set[str]] = defaultdict(set)
    for raw in text.splitlines():
        line = raw.strip()
        match = re.match(r"(entity|activity|agent)\(([^,;)]+)", line)
        if match:
            explicit_types[match.group(2).strip()].add(match.group(1))
    inferred = {key: set(value) for key, value in explicit_types.items()}
    relation_ids: dict[str, set[str]] = defaultdict(set)
    by_identified: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    generations_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for statement in statements:
        kind = statement["kind"]
        values = statement["values"]
        relation_id = statement["id"]
        if relation_id:
            relation_ids[relation_id].add(kind)
            by_identified[(kind, relation_id)].append(values)
        if kind == "wasGeneratedBy" and len(values) >= 2:
            inferred.setdefault(values[0], set()).add("entity")
            if values[1] != "-":
                inferred.setdefault(values[1], set()).add("activity")
            generations_by_entity[values[0]].append(statement)
        elif kind == "used" and len(values) >= 2:
            if values[0] != "-":
                inferred.setdefault(values[0], set()).add("activity")
            if values[1] != "-":
                inferred.setdefault(values[1], set()).add("entity")
        elif kind == "wasDerivedFrom" and len(values) >= 2:
            inferred.setdefault(values[0], set()).add("entity")
            inferred.setdefault(values[1], set()).add("entity")
            if len(values) == 5 and values[2] != "-":
                inferred.setdefault(values[2], set()).add("activity")
            if len(values) == 5 and values[2] == "-" and (values[3] != "-" or values[4] != "-"):
                reasons.append("C51_UNSPECIFIED_DERIVATION_ACTIVITY")
        elif kind == "wasAssociatedWith":
            if not values or values[0] == "-":
                reasons.append("ASSOCIATION_ACTIVITY_REQUIRED")
            else:
                inferred.setdefault(values[0], set()).add("activity")
            if len(values) > 1 and values[1] != "-":
                inferred.setdefault(values[1], set()).add("agent")
    for identifier, types in inferred.items():
        if "entity" in types and "activity" in types:
            reasons.append(f"C50_TYPE_CONFLICT:{identifier}")
        if "activity" in types and "agent" in types:
            reasons.append(f"C55_TYPE_CONFLICT:{identifier}")
    for relation_id, kinds in relation_ids.items():
        if len(kinds) > 1:
            reasons.append(f"C53_RELATION_OVERLAP:{relation_id}")

    for (kind, relation_id), rows in by_identified.items():
        first = rows[0]
        for other in rows[1:]:
            if kind == "wasDerivedFrom" and len(first) != len(other):
                expanded = first if len(first) > len(other) else other
                if any(value != "-" for value in expanded[2:]):
                    reasons.append(f"C23_DERIVATION_ARITY:{relation_id}")
                    continue
            if kind == "wasAssociatedWith":
                if len(first) != len(other):
                    reasons.append(f"C23_ASSOCIATION_ARITY:{relation_id}")
                    continue
                if first[0] != other[0]:
                    reasons.append(f"C23_ASSOCIATION_ACTIVITY:{relation_id}")
                if not _compatible(first[1], other[1]):
                    reasons.append(f"C23_ASSOCIATION_AGENT:{relation_id}")
                if first[2] != other[2]:
                    reasons.append(f"C23_ASSOCIATION_PLAN:{relation_id}")
                continue
            for index, (left, right) in enumerate(zip(first, other, strict=False)):
                if not _compatible(left, right):
                    reasons.append(f"C23_KEY_CONFLICT:{kind}:{relation_id}:{index}")
    for entity, rows in generations_by_entity.items():
        identified = {row["id"] for row in rows if row["id"]}
        if len(identified) > 1:
            reasons.append(f"C24_UNIQUE_GENERATION:{entity}")
        width = max(len(row["values"]) for row in rows)
        for index in range(1, width):
            concrete = {
                row["values"][index]
                for row in rows
                if index < len(row["values"]) and row["values"][index] != "-"
            }
            if len(concrete) > 1:
                reasons.append(f"C23_GENERATION_CONFLICT:{entity}:{index}")
    return not reasons, sorted(set(reasons))


def run_official_tests(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(f"frozen official tests unavailable: {directory}")
    cases: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.provn")):
        value = path.read_bytes()
        observed_valid, reasons = validate_official_subset_document(value.decode("utf-8"))
        expected_valid = "-FAIL" not in path.stem
        cases.append({
            "id": path.stem,
            "sha256": hashlib.sha256(value).hexdigest(),
            "expected": "VALID" if expected_valid else "INVALID",
            "observed": "VALID" if observed_valid else "INVALID",
            "passed": observed_valid == expected_valid,
            "detected_reasons": reasons,
        })
    passed = sum(case["passed"] for case in cases)
    return {
        "official_test_total": 291,
        "historical_implementation_report_total": 280,
        "applicable_test_count": len(cases),
        "excluded_test_count": 291 - len(cases),
        "passed_applicable_count": passed,
        "failed_applicable_count": len(cases) - passed,
        "cases": cases,
        "exclusion_reasons": [
            "PROV component outside the frozen profile",
            "temporal ordering, Start, End, or Invalidation outside the frozen profile",
            "generic normalization or entailment beyond the selected identified relations",
            "representation outside deterministic PROV-N and qualified PROV-O selected terms",
        ],
        "status": "SUPPORTED" if cases and passed == len(cases) == 53 else "NOT_SUPPORTED",
    }
