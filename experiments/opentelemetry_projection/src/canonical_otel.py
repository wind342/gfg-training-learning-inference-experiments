from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from generation_relation_core.canonical import canonical_bytes, sha256_bytes

from .projection_errors import ProjectionError


SCHEMA_VERSION = "otel-projection-v1"
_SCHEMA_PATH = Path(__file__).with_name("canonical_otel.schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)

ROOT_ATTRIBUTE_KEYS = frozenset(
    {"logical.order", "span.kind", "execution.kind", "execution.run_id"}
)
OCCURRENCE_ATTRIBUTE_KEYS = frozenset(
    {
        "logical.order",
        "span.kind",
        "operation.type",
        "operation.stage",
        "occurrence.type",
        "occurrence.stable_instance_key",
        "occurrence.index",
        "outcome.kind",
        "transform.operator_type",
        "transform.stage",
        "occurrence.cardinality",
    }
)


def trace_key(run_id: str) -> str:
    return f"trace:query-run:{run_id}"


def root_span_key(run_id: str) -> str:
    return f"span:query-run:{run_id}"


def occurrence_span_key(
    *, occurrence_index: int, occurrence_type: str, stable_instance_key: str
) -> str:
    return (
        f"span:occurrence:{occurrence_index:08d}:"
        f"{occurrence_type}:{stable_instance_key}"
    )


def _validate_attribute_profiles(spans: list[dict[str, Any]]) -> None:
    for span in spans:
        attributes = span["attributes"]
        kind = attributes["span.kind"]
        expected = (
            ROOT_ATTRIBUTE_KEYS if kind == "query_root" else OCCURRENCE_ATTRIBUTE_KEYS
        )
        if set(attributes) != expected:
            raise ProjectionError(
                "ATTRIBUTE_MISMATCH",
                f"{span['span_semantic_key']}:{sorted(set(attributes) ^ expected)}",
            )
        expected_event = (
            "query.execution" if kind == "query_root" else "generation.occurrence"
        )
        if len(span["events"]) != 1 or span["events"][0]["name"] != expected_event:
            raise ProjectionError("EVENT_MISMATCH", span["span_semantic_key"])
        event_attributes = span["events"][0]["attributes"]
        expected_event_keys = (
            {"execution.run_id"}
            if kind == "query_root"
            else {"occurrence.index", "outcome.kind"}
        )
        if set(event_attributes) != expected_event_keys:
            raise ProjectionError("EVENT_MISMATCH", span["span_semantic_key"])


def canonicalize_trace(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize without discarding comparison semantics."""

    candidate = deepcopy(value)
    try:
        _VALIDATOR.validate(candidate)
    except ValidationError as exc:
        raise ProjectionError("PROJECTION_SCHEMA_INVALID", exc.json_path) from exc
    spans = candidate["spans"]
    semantic_keys = [span["span_semantic_key"] for span in spans]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ProjectionError("DUPLICATE_SPAN")
    _validate_attribute_profiles(spans)
    known = set(semantic_keys)
    roots = [span for span in spans if span["parent_semantic_key"] is None]
    if len(roots) != 1 or roots[0]["attributes"]["span.kind"] != "query_root":
        raise ProjectionError("PARENT_EDGE_MISMATCH", "ROOT_CARDINALITY")
    root_key = roots[0]["span_semantic_key"]
    for span in spans:
        parent = span["parent_semantic_key"]
        if parent is not None and parent not in known:
            raise ProjectionError("PARENT_EDGE_MISMATCH", span["span_semantic_key"])
        if span["attributes"]["span.kind"] == "occurrence" and parent != root_key:
            raise ProjectionError("PARENT_EDGE_MISMATCH", span["span_semantic_key"])
        for linked in span["linked_semantic_keys"]:
            if linked not in known:
                raise ProjectionError("LINK_EDGE_MISMATCH", span["span_semantic_key"])
        span["linked_semantic_keys"] = sorted(span["linked_semantic_keys"])
        span["events"] = sorted(
            span["events"],
            key=lambda event: canonical_bytes(event),
        )
    candidate["spans"] = sorted(
        spans,
        key=lambda span: (
            span["attributes"]["logical.order"],
            span["span_semantic_key"],
        ),
    )
    return candidate


def canonical_trace_bytes(value: dict[str, Any]) -> bytes:
    return canonical_bytes(canonicalize_trace(value))


def canonical_trace_sha256(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_trace_bytes(value))
