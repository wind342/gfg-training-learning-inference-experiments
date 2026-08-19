from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable
from pathlib import Path

from .events import (
    ActivityEvent,
    AgentEvent,
    BindingEvent,
    BridgeEvent,
    GeneratedOutput,
    GenerationSink,
    GeneratorVariant,
    OutcomeEvent,
    SourceEvent,
    TransformExecutionReceipt,
    TransformReceiptSink,
)


GENERATOR_CODE_IDENTITY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _emit(sinks: Iterable[GenerationSink], method: str, event: object) -> None:
    for sink in sinks:
        getattr(sink, method)(event)


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["item", "score"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _execute_left_associative(x: int, y: int, z: int) -> tuple[int, tuple[int, ...]]:
    first = x + y
    result = first + z
    return result, (first, result)


def _execute_right_associative(x: int, y: int, z: int) -> tuple[int, tuple[int, ...]]:
    first = y + z
    result = x + first
    return result, (first, result)


def _execute_transform(variant: str) -> tuple[int, dict[str, object], dict[str, object], TransformExecutionReceipt]:
    inputs = {"x": 5, "y": 6, "z": 0}
    if variant == "left_associative":
        output, intermediate = _execute_left_associative(**inputs)
        branch_id = "integer-sum-left-associative"
        code_path = "generator._execute_left_associative"
        evaluation_order = ["x+y", "intermediate+z"]
    elif variant == "right_associative":
        output, intermediate = _execute_right_associative(**inputs)
        branch_id = "integer-sum-right-associative"
        code_path = "generator._execute_right_associative"
        evaluation_order = ["y+z", "x+intermediate"]
    else:
        raise ValueError(f"unsupported transform variant: {variant}")
    plan = {
        "branch_id": branch_id,
        "code_path": code_path,
        "evaluation_order": evaluation_order,
        "input_values": inputs,
    }
    transform_reference = {
        "operation_type": "render-json-and-text",
        "transform_variant": variant,
        "transformation_plan_sha256": _canonical_sha256(plan),
    }
    transform_context = {
        "executed_branch_id": branch_id,
        "executed_function_or_code_path": code_path,
        "evaluation_order": evaluation_order,
        "input_values": inputs,
        "intermediate_state_sha256": _canonical_sha256(list(intermediate)),
        "intermediate_values": list(intermediate),
        "output_value": output,
    }
    occurrence_payload = {
        "operation_type": "render-json-and-text",
        "diagnostic_context": {
            "actual_transform_context": transform_context,
            "format_order": ["json", "text"],
            "variant_neutral": True,
        },
    }
    receipt = TransformExecutionReceipt(
        transform_variant=variant,
        executed_branch_id=branch_id,
        executed_function_or_code_path=code_path,
        input_values=inputs,
        intermediate_values=intermediate,
        output_value=output,
        transform_reference_sha256=_canonical_sha256(transform_reference),
        occurrence_payload_sha256=_canonical_sha256(occurrence_payload),
    )
    return output, transform_reference, occurrence_payload["diagnostic_context"], receipt


def run_generator(
    sinks: Iterable[GenerationSink] = (),
    *,
    variant: GeneratorVariant = GeneratorVariant(),
    receipt_sinks: Iterable[TransformReceiptSink] = (),
) -> GeneratedOutput:
    """Run a real deterministic selection, join, and final rendering process.

    Ordinary output depends only on source values. Capture variants affect only
    profile-external Core facts and never the generated files.
    """

    sinks = tuple(sinks)
    agent = AgentEvent("deterministic-tabular-generator", "1.0.0", GENERATOR_CODE_IDENTITY)
    _emit(sinks, "on_agent", agent)

    transform_output, render_transform_reference, render_diagnostic_context, transform_receipt = _execute_transform(
        variant.transform_variant
    )
    for receipt_sink in tuple(receipt_sinks):
        receipt_sink.on_transform_execution(transform_receipt)

    sources = [
        SourceEvent("left-a", "left-row-A", "csv-row", "left-table-row", "left/A", {"key": "K1", "value": 1}),
        SourceEvent("left-b", "left-row-B", "csv-row", "left-table-row", "left/B", {"key": "K1", "value": 1}),
        SourceEvent("right-1", "right-row-1", "csv-row", "right-table-row", "right/1", {"key": "K1", "weight": 2}),
        SourceEvent("right-2", "right-row-2", "csv-row", "right-table-row", "right/2", {"key": "K2", "weight": 9}),
    ]
    for source in sources:
        _emit(sinks, "on_source", source)

    selection = ActivityEvent(
        "select", "stage-1", "deterministic-selection", "fixture/select", 0,
        "select-matching-key", {"operation_type": "select-matching-key"},
        {"input_rows": 4, "variant_neutral": True},
    )
    join = ActivityEvent(
        "join", "stage-1", "deterministic-join", "fixture/join", 1,
        "join-and-derived-field", {"operation_type": "join-and-derived-field"},
        {"join_key": "K1", "variant_neutral": True},
    )
    render = ActivityEvent(
        "render", "stage-2", "deterministic-render", "fixture/render", 2,
        "render-json-and-text", render_transform_reference, render_diagnostic_context,
    )
    for activity in (selection, join, render):
        _emit(sinks, "on_activity", activity)

    selected = OutcomeEvent(
        "selected", "support", "selection-manifest", "selected/K1",
        {"selected_count": 3, "native_support_key": "selected/K1"},
    )
    unmatched = OutcomeEvent(
        "unmatched", "disposition", payload={"unmatched_key": "K2", "profile_external_note": "not joined"},
        disposition_category="non_rendered", reason_code="NO_MATCHING_LEFT_KEY",
    )
    intermediate_rows = [{"item": "alpha", "score": 5}, {"item": "beta", "score": 6}]
    intermediate_a = OutcomeEvent(
        "intermediate-a", "support", "intermediate-row", "intermediate/alpha",
        {"native_support_key": "intermediate/alpha", **intermediate_rows[0]},
    )
    intermediate_b = OutcomeEvent(
        "intermediate-b", "support", "intermediate-row", "intermediate/beta",
        {"native_support_key": "intermediate/beta", **intermediate_rows[1]},
    )
    final_a = OutcomeEvent(
        "final-a", "support", "final-result", "final/result-set",
        {"native_support_key": "final/result-set", "count": 2, "total": transform_output},
    )
    final_b = OutcomeEvent(
        "final-b", "support", "final-summary", "final/summary",
        {"native_support_key": "final/summary", "text": f"count=2 total={transform_output}"},
    )
    for outcome in (selected, unmatched, intermediate_a, intermediate_b, final_a, final_b):
        _emit(sinks, "on_outcome", outcome)

    for binding in (
        BindingEvent("source", "left-a", "select", "selected", "selected-input", 0),
        BindingEvent("source", "left-b", "select", "selected", "selected-input", 0),
        BindingEvent("source", "right-1", "select", "selected", "selected-input", 0),
        BindingEvent("source", "right-2", "select", "unmatched", "unmatched-input", 0),
        BindingEvent("source", "left-a", "join", "intermediate-a", "join-contributor", 0),
        BindingEvent("source", "left-a", "join", "intermediate-b", "join-contributor", 1),
        BindingEvent("source", "left-b", "join", "intermediate-a", "join-contributor", 0),
        BindingEvent("source", "left-b", "join", "intermediate-b", "join-contributor", 1),
        BindingEvent("source", "right-1", "join", "intermediate-a", "join-contributor", 0),
        BindingEvent("source", "right-1", "join", "intermediate-b", "join-contributor", 1),
    ):
        _emit(sinks, "on_binding", binding)

    bridge_a = BridgeEvent("bridge-a", "intermediate-a", variant.bridge_detail + ":a")
    bridge_b = BridgeEvent("bridge-b", "intermediate-b", variant.bridge_detail + ":b")
    _emit(sinks, "on_bridge", bridge_a)
    _emit(sinks, "on_bridge", bridge_b)
    for binding in (
        BindingEvent("generated", "bridge-a", "render", "final-a", "stage2-input", 0),
        BindingEvent("generated", "bridge-a", "render", "final-b", "stage2-input", 1),
        BindingEvent("generated", "bridge-b", "render", "final-a", "stage2-input", 0),
        BindingEvent("generated", "bridge-b", "render", "final-b", "stage2-input", 1),
    ):
        _emit(sinks, "on_binding", binding)

    csv_value = _csv_bytes(intermediate_rows)
    final_value = (json.dumps(
        {"results": intermediate_rows, "total": transform_output},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")
    summary_value = f"count=2 total={transform_output}\n".encode("utf-8")
    return GeneratedOutput(
        files={"stage1.csv": csv_value, "final.json": final_value, "summary.txt": summary_value},
        media_types={"stage1.csv": "text/csv", "final.json": "application/json", "summary.txt": "text/plain"},
    )
