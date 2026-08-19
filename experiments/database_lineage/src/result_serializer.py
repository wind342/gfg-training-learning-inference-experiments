from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence

from .relational_executor import RelationTuple


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def ordinary_rows(rows: Sequence[RelationTuple]) -> list[dict[str, str]]:
    return [
        {name: scalar_text(value) for name, value in row.values.items()} for row in rows
    ]


def csv_bytes(rows: Sequence[RelationTuple]) -> bytes:
    if not rows:
        return b""
    columns = list(rows[0].values)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(ordinary_rows(rows))
    return stream.getvalue().encode("utf-8")


def json_bytes(rows: Sequence[RelationTuple]) -> bytes:
    return (
        json.dumps(
            ordinary_rows(rows),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def output_hashes(rows: Sequence[RelationTuple]) -> dict[str, str | int]:
    csv_payload = csv_bytes(rows)
    json_payload = json_bytes(rows)
    return {
        "row_count": len(rows),
        "csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
        "json_sha256": hashlib.sha256(json_payload).hexdigest(),
        "csv_bytes": len(csv_payload),
        "json_bytes": len(json_payload),
    }
