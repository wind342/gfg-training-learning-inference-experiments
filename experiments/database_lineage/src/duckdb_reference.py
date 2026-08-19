from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb

from .result_serializer import scalar_text


def execute_reference(
    connection: duckdb.DuckDBPyConnection, sql: str
) -> dict[str, Any]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    rows = [dict(zip(columns, record, strict=True)) for record in cursor.fetchall()]
    return {
        "columns": columns,
        "typed_rows": rows,
        "text_rows": [
            {name: scalar_text(value) for name, value in row.items()} for row in rows
        ],
    }


def parse_official_answer(answer: str) -> dict[str, Any]:
    lines = answer.rstrip("\n").splitlines()
    columns = lines[0].split("|")
    return {
        "columns": columns,
        "text_rows": [
            dict(zip(columns, line.split("|"), strict=True)) for line in lines[1:]
        ],
    }


def compare_rows(
    actual: list[dict[str, str]], expected: list[dict[str, str]]
) -> dict[str, Any]:
    mismatches = []
    for index in range(max(len(actual), len(expected))):
        left = actual[index] if index < len(actual) else None
        right = expected[index] if index < len(expected) else None
        if left != right:
            mismatches.append({"row": index, "actual": left, "expected": right})
    return {
        "exact": not mismatches,
        "actual_row_count": len(actual),
        "expected_row_count": len(expected),
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
    }


def compare_official_typed(
    actual: list[dict[str, Any]], expected: list[dict[str, str]]
) -> dict[str, Any]:
    """Compare official display text after parsing to the actual SQL result types.

    This permits only representation differences such as ``380456`` versus
    ``380456.00``. It does not round or apply an epsilon.
    """
    mismatches = []
    display_differences = []
    for index in range(max(len(actual), len(expected))):
        left = actual[index] if index < len(actual) else None
        right = expected[index] if index < len(expected) else None
        if left is None or right is None:
            mismatches.append({"row": index, "actual": left, "expected": right})
            continue
        row_mismatches = {}
        for column, value in left.items():
            text = right.get(column)
            try:
                if isinstance(value, Decimal):
                    parsed: Any = Decimal(text)
                elif isinstance(value, bool):
                    parsed = text.lower() == "true"
                elif isinstance(value, int):
                    parsed = int(text)
                elif isinstance(value, float):
                    parsed = float(text)
                elif isinstance(value, (date, datetime)):
                    parsed = value.__class__.fromisoformat(text)
                elif value is None:
                    parsed = None if text == "" else text
                else:
                    parsed = text
            except (ValueError, TypeError):
                parsed = object()
            if value != parsed:
                row_mismatches[column] = {
                    "actual": scalar_text(value),
                    "expected": text,
                }
            elif scalar_text(value) != text:
                display_differences.append(
                    {
                        "row": index,
                        "column": column,
                        "actual_display": scalar_text(value),
                        "official_display": text,
                    }
                )
        if row_mismatches:
            mismatches.append({"row": index, "columns": row_mismatches})
    return {
        "exact_after_typed_parse": not mismatches,
        "actual_row_count": len(actual),
        "expected_row_count": len(expected),
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
        "display_differences": display_differences[:100],
        "display_difference_count": len(display_differences),
        "normalization_rule": "parse official display text to the exact actual SQL result type; no rounding or epsilon",
    }
