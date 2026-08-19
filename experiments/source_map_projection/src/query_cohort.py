from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_query_cohort(records: list[dict[str, Any]], generated_text: str) -> list[dict[str, Any]]:
    cohort: list[dict[str, Any]] = []
    sequence = 0

    def add(payload: dict[str, Any]) -> None:
        nonlocal sequence
        cohort.append({"query_id": f"q{sequence:06d}", **payload})
        sequence += 1

    for row in records:
        add({
            "direction": "generated_to_original",
            "generated_line": row["generated_line"],
            "generated_column": row["generated_column"],
            "bias": "GLB",
            "category": "exact_anchor" if row["mapped"] else "unmapped_anchor",
        })
    by_line: dict[int, list[int]] = defaultdict(list)
    for row in records:
        by_line[row["generated_line"]].append(row["generated_column"])
    for line, columns in sorted(by_line.items()):
        columns = sorted(columns)
        for left, right in zip(columns, columns[1:]):
            if right - left > 1:
                add({
                    "direction": "generated_to_original",
                    "generated_line": line,
                    "generated_column": left + 1,
                    "bias": "GLB",
                    "category": "between_anchors",
                })
    add({
        "direction": "generated_to_original", "generated_line": 0,
        "generated_column": 0, "bias": "GLB", "category": "file_start",
    })
    line_count = len(generated_text.splitlines())
    add({
        "direction": "generated_to_original", "generated_line": line_count + 1,
        "generated_column": 0, "bias": "GLB", "category": "past_file_end",
    })
    for row in records:
        if row["mapped"]:
            add({
                "direction": "original_to_generated",
                "original_source": row["original_source"],
                "original_line": row["original_line"],
                "original_column": row["original_column"],
                "category": "exact_original_anchor",
            })
    add({
        "direction": "original_to_generated", "original_source": "missing-source.js",
        "original_line": 0, "original_column": 0, "category": "missing_source",
    })
    add({
        "direction": "original_to_generated", "original_source": next(
            (row["original_source"] for row in records if row["mapped"]), "missing-source.js"
        ),
        "original_line": -1, "original_column": -1, "category": "invalid_position",
    })
    return cohort


def compare_query_results(native: list[dict], projected: list[dict]) -> dict[str, Any]:
    native_by_id = {row["query_id"]: row for row in native}
    projected_by_id = {row["query_id"]: row for row in projected}
    ids = sorted(set(native_by_id) | set(projected_by_id))
    mismatches = []
    counters = {
        "false_positive": 0,
        "false_negative": 0,
        "ambiguity_mismatch": 0,
        "name_mismatch": 0,
        "source_mismatch": 0,
        "position_mismatch": 0,
    }
    for query_id in ids:
        left = native_by_id.get(query_id)
        right = projected_by_id.get(query_id)
        if left == right:
            continue
        mismatches.append({"query_id": query_id, "native": left, "projected": right})
        if left is None:
            counters["false_positive"] += 1
        elif right is None:
            counters["false_negative"] += 1
        else:
            counters["ambiguity_mismatch"] += int(len(left.get("answers", [])) != len(right.get("answers", [])))
            encoded = repr((left, right))
            counters["name_mismatch"] += int("name" in encoded)
            counters["source_mismatch"] += int("source" in encoded)
            counters["position_mismatch"] += 1
    return {
        "exact_query_count": len(ids) - len(mismatches),
        "total_query_count": len(ids),
        **counters,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
