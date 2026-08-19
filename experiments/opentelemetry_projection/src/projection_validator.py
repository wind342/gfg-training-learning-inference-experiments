from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .canonical_otel import canonicalize_trace
from .projection_errors import ProjectionError


@dataclass(frozen=True)
class TraceDiff:
    exact: bool
    span_false_positives: int
    span_false_negatives: int
    parent_edge_false_positives: int
    parent_edge_false_negatives: int
    link_edge_false_positives: int
    link_edge_false_negatives: int
    attribute_mismatches: int
    status_mismatches: int
    event_mismatches: int


def _span_map(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {span["span_semantic_key"]: span for span in trace["spans"]}


def trace_diff(expected: dict[str, Any], actual: dict[str, Any]) -> TraceDiff:
    expected = canonicalize_trace(expected)
    actual = canonicalize_trace(actual)
    expected_by_key = _span_map(expected)
    actual_by_key = _span_map(actual)
    expected_keys = set(expected_by_key)
    actual_keys = set(actual_by_key)
    parent_fp = parent_fn = link_fp = link_fn = 0
    attribute_mismatches = status_mismatches = event_mismatches = 0
    for key in expected_keys & actual_keys:
        expected_span = expected_by_key[key]
        actual_span = actual_by_key[key]
        expected_parent = expected_span["parent_semantic_key"]
        actual_parent = actual_span["parent_semantic_key"]
        if expected_parent != actual_parent:
            parent_fn += int(expected_parent is not None)
            parent_fp += int(actual_parent is not None)
        expected_links = list(expected_span["linked_semantic_keys"])
        actual_links = list(actual_span["linked_semantic_keys"])
        unmatched_actual = list(actual_links)
        for link in expected_links:
            if link in unmatched_actual:
                unmatched_actual.remove(link)
            else:
                link_fn += 1
        link_fp += len(unmatched_actual)
        if (
            expected_span["attributes"] != actual_span["attributes"]
            or expected_span["name"] != actual_span["name"]
        ):
            attribute_mismatches += 1
        if expected_span["status"] != actual_span["status"]:
            status_mismatches += 1
        if expected_span["events"] != actual_span["events"]:
            event_mismatches += 1
    exact = expected == actual
    return TraceDiff(
        exact=exact,
        span_false_positives=len(actual_keys - expected_keys),
        span_false_negatives=len(expected_keys - actual_keys),
        parent_edge_false_positives=parent_fp,
        parent_edge_false_negatives=parent_fn,
        link_edge_false_positives=link_fp,
        link_edge_false_negatives=link_fn,
        attribute_mismatches=attribute_mismatches,
        status_mismatches=status_mismatches,
        event_mismatches=event_mismatches,
    )


def assert_trace_equal(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    mismatch_reason: str | None = None,
) -> TraceDiff:
    try:
        expected_canonical = canonicalize_trace(expected)
        actual_canonical = canonicalize_trace(actual)
        actual_spans = actual_canonical["spans"]
        for span in actual_spans:
            if span["attributes"].get("occurrence.cardinality", 1) != 1:
                raise ProjectionError("OCCURRENCE_MERGED", span["span_semantic_key"])
        expected_by_key = _span_map(expected_canonical)
        actual_by_key = _span_map(actual_canonical)
        expected_keys = set(expected_by_key)
        actual_keys = set(actual_by_key)
        missing = expected_keys - actual_keys
        if missing:
            raise ProjectionError("MISSING_SPAN", sorted(missing)[0])
        extra = actual_keys - expected_keys
        if extra:
            extra_key = sorted(extra)[0]
            if extra_key.startswith("span:binding:"):
                raise ProjectionError("BINDING_DERIVED_SPAN", extra_key)
            if extra_key.startswith("span:occurrence:"):
                raise ProjectionError("UNKNOWN_OCCURRENCE", extra_key)
            raise ProjectionError("FABRICATED_SPAN", extra_key)
        for key in sorted(expected_keys):
            expected_span = expected_by_key[key]
            actual_span = actual_by_key[key]
            if (
                expected_span["parent_semantic_key"]
                != actual_span["parent_semantic_key"]
            ):
                raise ProjectionError("PARENT_EDGE_MISMATCH", key)
            if (
                expected_span["linked_semantic_keys"]
                != actual_span["linked_semantic_keys"]
            ):
                raise ProjectionError("LINK_EDGE_MISMATCH", key)
            if expected_span["status"] != actual_span["status"]:
                raise ProjectionError("STATUS_MISMATCH", key)
            operator_fields = {"operation.type", "transform.operator_type"}
            if any(
                expected_span["attributes"].get(field)
                != actual_span["attributes"].get(field)
                for field in operator_fields
            ):
                raise ProjectionError("OPERATION_TYPE_MISMATCH", key)
            if expected_span["name"] != actual_span["name"]:
                raise ProjectionError("OPERATION_TYPE_MISMATCH", key)
            if expected_span["attributes"] != actual_span["attributes"]:
                raise ProjectionError("ATTRIBUTE_MISMATCH", key)
            if expected_span["events"] != actual_span["events"]:
                raise ProjectionError("EVENT_MISMATCH", key)
        return trace_diff(expected_canonical, actual_canonical)
    except ProjectionError as exc:
        if mismatch_reason is not None and exc.reason_code != mismatch_reason:
            raise ProjectionError(mismatch_reason, exc.reason_code) from exc
        raise


def _occurrence_spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
        for span in trace["spans"]
        if span["attributes"]["span.kind"] == "occurrence"
    ]


def run_negative_controls(valid_trace: dict[str, Any]) -> list[dict[str, str]]:
    """Run all required mutations against immutable copies of a valid trace."""

    valid_trace = canonicalize_trace(valid_trace)
    occurrences = _occurrence_spans(valid_trace)
    if len(occurrences) < 3:
        raise ValueError("negative controls require at least three occurrence spans")
    root = next(
        span for span in valid_trace["spans"] if span["parent_semantic_key"] is None
    )

    controls: list[tuple[str, str, dict[str, Any], str | None]] = []

    missing = deepcopy(valid_trace)
    missing["spans"].remove(missing["spans"][-1])
    controls.append(("delete_expected_span", "MISSING_SPAN", missing, None))

    fabricated = deepcopy(valid_trace)
    fabricated_span = deepcopy(root)
    fabricated_span["span_semantic_key"] = "span:fabricated:diagnostic"
    fabricated_span["parent_semantic_key"] = root["span_semantic_key"]
    fabricated["spans"].append(fabricated_span)
    controls.append(("add_fabricated_span", "FABRICATED_SPAN", fabricated, None))

    parent = deepcopy(valid_trace)
    _occurrence_spans(parent)[0]["parent_semantic_key"] = occurrences[1][
        "span_semantic_key"
    ]
    controls.append(("modify_parent", "PARENT_EDGE_MISMATCH", parent, None))

    link = deepcopy(valid_trace)
    target = next(
        (span for span in _occurrence_spans(link) if span["linked_semantic_keys"]),
        _occurrence_spans(link)[1],
    )
    target["linked_semantic_keys"] = [root["span_semantic_key"]]
    controls.append(("modify_causal_link", "LINK_EDGE_MISMATCH", link, None))

    status = deepcopy(valid_trace)
    _occurrence_spans(status)[0]["status"] = "ERROR"
    controls.append(("modify_status", "STATUS_MISMATCH", status, None))

    operation = deepcopy(valid_trace)
    operation_span = _occurrence_spans(operation)[0]
    operation_span["attributes"]["operation.type"] = "fabricated_operator"
    operation_span["attributes"]["transform.operator_type"] = "fabricated_operator"
    operation_span["name"] = "operator.fabricated_operator"
    controls.append(
        ("modify_operation_type", "OPERATION_TYPE_MISMATCH", operation, None)
    )

    attribute = deepcopy(valid_trace)
    _occurrence_spans(attribute)[0]["attributes"]["transform.stage"] = "wrong_stage"
    controls.append(
        ("modify_selected_attribute", "ATTRIBUTE_MISMATCH", attribute, None)
    )

    duplicate = deepcopy(valid_trace)
    duplicate["spans"].append(deepcopy(_occurrence_spans(duplicate)[0]))
    controls.append(("duplicate_span", "DUPLICATE_SPAN", duplicate, None))

    merged = deepcopy(valid_trace)
    merged_occurrences = _occurrence_spans(merged)
    merged_occurrences[0]["attributes"]["occurrence.cardinality"] = 2
    merged["spans"].remove(merged_occurrences[-1])
    controls.append(("merge_occurrences", "OCCURRENCE_MERGED", merged, None))

    unknown = deepcopy(valid_trace)
    unknown_span = deepcopy(_occurrence_spans(unknown)[0])
    unknown_span["span_semantic_key"] = "span:occurrence:99999999:unknown:unknown"
    unknown_span["attributes"]["logical.order"] = 99999999
    unknown_span["attributes"]["occurrence.index"] = 99999998
    unknown_span["events"][0]["attributes"]["occurrence.index"] = 99999998
    unknown["spans"].append(unknown_span)
    controls.append(("unknown_occurrence", "UNKNOWN_OCCURRENCE", unknown, None))

    binding = deepcopy(valid_trace)
    binding_span = deepcopy(_occurrence_spans(binding)[0])
    binding_span["span_semantic_key"] = "span:binding:fabricated"
    binding_span["attributes"]["logical.order"] = 99999998
    binding["spans"].append(binding_span)
    controls.append(("binding_derived_span", "BINDING_DERIVED_SPAN", binding, None))

    hierarchical = deepcopy(valid_trace)
    _occurrence_spans(hierarchical)[0]["status"] = "ERROR"
    controls.append(
        (
            "break_hierarchical_path",
            "HIERARCHICAL_PROJECTION_MISMATCH",
            hierarchical,
            "HIERARCHICAL_PROJECTION_MISMATCH",
        )
    )

    results = []
    for name, expected_reason, candidate, override in controls:
        try:
            assert_trace_equal(valid_trace, candidate, mismatch_reason=override)
        except ProjectionError as exc:
            if exc.reason_code != expected_reason:
                raise AssertionError(
                    f"{name}: expected {expected_reason}, got {exc.reason_code}"
                ) from exc
            results.append(
                {
                    "control": name,
                    "reason_code": exc.reason_code,
                    "result": "FAIL_CLOSED",
                }
            )
        else:
            raise AssertionError(f"negative control unexpectedly passed: {name}")
    return results
