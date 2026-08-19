from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_values(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def logical_output_key(workload_id: str, values: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_values(values).encode("utf-8")).hexdigest()
    return f"{workload_id}:value:{digest}"


def variable_for_source(source_identity: str) -> str:
    if not isinstance(source_identity, str) or not source_identity:
        raise ValueError("source identity must be a non-empty string")
    return "x_" + hashlib.sha256(source_identity.encode("utf-8")).hexdigest()
