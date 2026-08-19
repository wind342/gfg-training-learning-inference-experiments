from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Iterable

from .errors import CoreV3Error
from .schema_registry import projection_registry, validate


SAFE_INTEGER = 9_007_199_254_740_991


def _normalize(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if not -SAFE_INTEGER <= value <= SAFE_INTEGER:
            raise CoreV3Error("UNSAFE_INTEGER")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CoreV3Error("NON_FINITE_NUMBER")
        if value == 0:
            return 0
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise CoreV3Error("LONE_SURROGATE") from exc
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CoreV3Error("HASH_OR_ID_MISMATCH", "NON_STRING_OBJECT_KEY")
        return {key: _normalize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize(child) for child in value]
    raise CoreV3Error("UNSUPPORTED_RUNTIME_TYPE", type(value).__name__)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _normalize(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except CoreV3Error:
        raise
    except (TypeError, ValueError) as exc:
        raise CoreV3Error("HASH_OR_ID_MISMATCH", "CANONICAL_JSON_FAILURE") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def canonical_set(values: Iterable[Any]) -> list[Any]:
    encoded = [canonical_bytes(value) for value in values]
    if len(encoded) != len(set(encoded)):
        raise CoreV3Error("DUPLICATE_SET_ELEMENT")
    return [json.loads(value.decode("utf-8")) for value in sorted(encoded)]


def _normalize_declared_sets(entity_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(payload)
    spec = projection_registry().get(entity_type)
    if spec is None:
        raise CoreV3Error("UNKNOWN_ENTITY_TYPE", entity_type)
    for field in spec.get("set_fields", []):
        if field in result:
            if not isinstance(result[field], list):
                raise CoreV3Error("HASH_OR_ID_MISMATCH", f"SET_FIELD_WRONG_TYPE:{field}")
            result[field] = canonical_set(result[field])
    return result


def _verify_declared_set_order(entity_type: str, value: dict[str, Any]) -> None:
    spec = projection_registry()[entity_type]
    for field in spec.get("set_fields", []):
        if field in value and value[field] != canonical_set(value[field]):
            raise CoreV3Error("NON_CANONICAL_SET_ORDER", field)


def _prevalidate_tagged_binding(payload: dict[str, Any]) -> None:
    origin = payload.get("origin_reference")
    if not isinstance(origin, dict):
        raise CoreV3Error("GENERATED_ORIGIN_MISCLASSIFIED")
    if origin.get("kind") == "registered_source":
        if set(origin) != {"kind", "source_information_id"}:
            raise CoreV3Error("GENERATED_ORIGIN_MISCLASSIFIED")
    elif origin.get("kind") == "generated_origin":
        if set(origin) != {"kind", "generated_origin_id"}:
            raise CoreV3Error("GENERATED_ORIGIN_MISCLASSIFIED")
    else:
        raise CoreV3Error("GENERATED_ORIGIN_MISCLASSIFIED")
    outcome = payload.get("outcome_reference")
    if not isinstance(outcome, dict):
        raise CoreV3Error("BINDING_OUTCOME_CARDINALITY_INVALID")
    if outcome.get("kind") == "support":
        if set(outcome) != {"kind", "support_id"}:
            raise CoreV3Error("BINDING_OUTCOME_CARDINALITY_INVALID")
    elif outcome.get("kind") == "disposition":
        if set(outcome) != {"kind", "disposition_id"}:
            raise CoreV3Error("BINDING_OUTCOME_CARDINALITY_INVALID")
    else:
        raise CoreV3Error("BINDING_OUTCOME_CARDINALITY_INVALID")


def projection(entity_type: str, value: dict[str, Any]) -> dict[str, Any]:
    spec = projection_registry().get(entity_type)
    if spec is None:
        raise CoreV3Error("UNKNOWN_ENTITY_TYPE", entity_type)
    excluded = set(spec["excluded_fields"])
    return {key: deepcopy(child) for key, child in value.items() if key not in excluded}


def finalize_entity(entity_type: str, payload: dict[str, Any], *, validate_schema: bool = True) -> dict[str, Any]:
    spec = projection_registry().get(entity_type)
    if spec is None:
        raise CoreV3Error("UNKNOWN_ENTITY_TYPE", entity_type)
    if spec["id_field"] in payload or spec["hash_field"] in payload:
        raise CoreV3Error("HASH_SELF_REFERENCE", entity_type)
    if entity_type == "GenerationBinding":
        _prevalidate_tagged_binding(payload)
    result = _normalize_declared_sets(entity_type, payload)
    digest = sha256_bytes(canonical_bytes(result))
    result[spec["id_field"]] = spec["id_prefix"] + digest
    result[spec["hash_field"]] = digest
    result = _normalize_declared_sets(entity_type, result)
    if validate_schema:
        validate(entity_type, result)
    return result


def verify_entity(entity_type: str, value: dict[str, Any], *, validate_schema: bool = True) -> None:
    if not isinstance(value, dict):
        raise CoreV3Error("HASH_OR_ID_MISMATCH", f"NOT_OBJECT:{entity_type}")
    if validate_schema:
        validate(entity_type, value)
    _verify_declared_set_order(entity_type, value)
    spec = projection_registry()[entity_type]
    digest = sha256_bytes(canonical_bytes(projection(entity_type, value)))
    if value.get(spec["hash_field"]) != digest or value.get(spec["id_field"]) != spec["id_prefix"] + digest:
        reason = "BINDING_ID_OR_HASH_MISMATCH" if entity_type == "GenerationBinding" else "HASH_OR_ID_MISMATCH"
        raise CoreV3Error(reason, entity_type)


def table_hash(rows: Iterable[dict[str, Any]], entity_type: str, *, verify_rows: bool = True) -> str:
    spec = projection_registry()[entity_type]
    items = list(rows)
    ids = [row.get(spec["id_field"]) for row in items]
    if any(value is None for value in ids) or len(ids) != len(set(ids)):
        raise CoreV3Error("DUPLICATE_ENTITY_ID", entity_type)
    if verify_rows:
        for row in items:
            verify_entity(entity_type, row)
    return sha256_bytes(canonical_bytes(sorted(items, key=lambda row: row[spec["id_field"]])))


def canonical_json_file_bytes(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"
