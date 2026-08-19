from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def compact_witness(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "counterexample_id",
            "id",
            "pair",
            "valid_counterexample",
            "status",
            "left_snapshot_id",
            "right_snapshot_id",
            "first_snapshot_id",
            "second_snapshot_id",
            "gamma_a_snapshot_id",
            "gamma_b_snapshot_id",
            "complete_snapshot_equal",
            "snapshots_differ",
            "gamma_different",
            "projection_hash_equal",
            "map_document_equal",
            "normalized_prov_dm_equal",
            "native_normalized_otel_equal",
            "direct_core_projection_equal",
            "graph_equal",
            "actual_execution_difference",
        )
        if key in row
    }

