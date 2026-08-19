from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Link, Status, StatusCode

from experiments.database_lineage.src.relational_executor import RelationTuple

from .canonical_otel import (
    OCCURRENCE_ATTRIBUTE_KEYS,
    ROOT_ATTRIBUTE_KEYS,
    SCHEMA_VERSION,
    canonicalize_trace,
    occurrence_span_key,
    root_span_key,
    trace_key,
)
from .projection_errors import ProjectionError


class NativeOtelCapture:
    """Official-SDK capture coupled to actual executor capture callbacks."""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(
            span_limits=SpanLimits(
                max_span_attributes=64,
                max_events=16,
                max_links=1_000_000,
                max_event_attributes=16,
                max_link_attributes=16,
            )
        )
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = self.provider.get_tracer(
            "source-information-continuity.opentelemetry-projection",
            "1.0.0",
        )
        self._occurrence_index = 0
        self._span_context_by_tuple_id: dict[str, Any] = {}
        self._semantic_key_by_span_id: dict[int, str] = {}
        self._finished = False
        self._root = self.tracer.start_span(
            "query.execute",
            context=Context(),
            attributes={
                "logical.order": 0,
                "span.kind": "query_root",
                "execution.kind": "deterministic_relational_query",
                "execution.run_id": run_id,
            },
        )
        self._root.add_event("query.execution", attributes={"execution.run_id": run_id})
        root_context = self._root.get_span_context()
        self._semantic_key_by_span_id[root_context.span_id] = root_span_key(run_id)

    def _record(
        self,
        *,
        stage: str,
        operator_type: str,
        stable_instance_key: str,
        outcome_kind: str,
        inputs: Sequence[RelationTuple],
        output_tuple_id: str | None,
    ) -> None:
        if self._finished:
            raise RuntimeError("native capture already finished")
        occurrence_index = self._occurrence_index
        self._occurrence_index += 1
        occurrence_type = f"relational_{operator_type}_execution"
        semantic_key = occurrence_span_key(
            occurrence_index=occurrence_index,
            occurrence_type=occurrence_type,
            stable_instance_key=stable_instance_key,
        )
        links = [
            Link(self._span_context_by_tuple_id[item.tuple_id])
            for item in inputs
            if item.tuple_id in self._span_context_by_tuple_id
        ]
        parent = trace.set_span_in_context(self._root)
        span = self.tracer.start_span(
            f"operator.{operator_type}",
            context=parent,
            links=links,
            attributes={
                "logical.order": occurrence_index + 1,
                "span.kind": "occurrence",
                "operation.type": operator_type,
                "operation.stage": stage,
                "occurrence.type": occurrence_type,
                "occurrence.stable_instance_key": stable_instance_key,
                "occurrence.index": occurrence_index,
                "outcome.kind": outcome_kind,
                "transform.operator_type": operator_type,
                "transform.stage": stage,
                "occurrence.cardinality": 1,
            },
        )
        span.add_event(
            "generation.occurrence",
            attributes={
                "occurrence.index": occurrence_index,
                "outcome.kind": outcome_kind,
            },
        )
        span.set_status(Status(StatusCode.OK))
        span_context = span.get_span_context()
        self._semantic_key_by_span_id[span_context.span_id] = semantic_key
        if output_tuple_id is not None:
            self._span_context_by_tuple_id[output_tuple_id] = span_context
        span.end()

    def record_output(
        self,
        *,
        stage: str,
        operator_type: str,
        output: RelationTuple,
        inputs: Sequence[RelationTuple],
    ) -> None:
        self._record(
            stage=stage,
            operator_type=operator_type,
            stable_instance_key=output.tuple_id,
            outcome_kind="support",
            inputs=inputs,
            output_tuple_id=output.tuple_id,
        )

    def record_disposition(
        self,
        *,
        stage: str,
        operator_type: str,
        input_tuple: RelationTuple,
    ) -> None:
        self._record(
            stage=stage,
            operator_type=operator_type,
            stable_instance_key=f"{stage}:disposition:{input_tuple.tuple_id}",
            outcome_kind="disposition",
            inputs=[input_tuple],
            output_tuple_id=None,
        )

    def finish(self) -> tuple[ReadableSpan, ...]:
        if not self._finished:
            self._root.set_status(Status(StatusCode.OK))
            self._root.end()
            self.provider.force_flush()
            self._finished = True
        return tuple(self.exporter.get_finished_spans())

    def clear_native_records(self) -> None:
        self.exporter.clear()


def canonicalize_native_trace(
    spans: Sequence[ReadableSpan], *, expected_run_id: str
) -> dict[str, Any]:
    """Normalize only random IDs and wall-clock timestamps from SDK spans."""

    if not spans:
        raise ProjectionError("MISSING_SPAN", "NATIVE_EMPTY")
    trace_ids = {span.context.trace_id for span in spans if span.context is not None}
    if len(trace_ids) != 1:
        raise ProjectionError("TRACE_COUNT_MISMATCH", str(len(trace_ids)))

    semantic_candidates: list[tuple[int, str]] = []
    for span in spans:
        if span.dropped_attributes:
            raise ProjectionError("ATTRIBUTE_MISMATCH", "DROPPED_ATTRIBUTES")
        if span.dropped_events:
            raise ProjectionError("EVENT_MISMATCH", "DROPPED_EVENTS")
        if span.dropped_links:
            raise ProjectionError("LINK_EDGE_MISMATCH", "DROPPED_LINKS")
        attributes = dict(span.attributes or {})
        kind = attributes.get("span.kind")
        expected_keys = (
            ROOT_ATTRIBUTE_KEYS if kind == "query_root" else OCCURRENCE_ATTRIBUTE_KEYS
        )
        if set(attributes) != expected_keys:
            raise ProjectionError("ATTRIBUTE_MISMATCH", span.name)
        if kind == "query_root":
            if attributes["execution.run_id"] != expected_run_id:
                raise ProjectionError("ATTRIBUTE_MISMATCH", "RUN_ID")
            semantic_key = root_span_key(expected_run_id)
        elif kind == "occurrence":
            semantic_key = occurrence_span_key(
                occurrence_index=attributes["occurrence.index"],
                occurrence_type=attributes["occurrence.type"],
                stable_instance_key=attributes["occurrence.stable_instance_key"],
            )
        else:
            raise ProjectionError("UNKNOWN_OCCURRENCE", str(kind))
        if span.context is None:
            raise ProjectionError("UNKNOWN_OCCURRENCE", "MISSING_SPAN_CONTEXT")
        semantic_candidates.append((span.context.span_id, semantic_key))
    semantic_keys = [key for _raw_id, key in semantic_candidates]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ProjectionError("DUPLICATE_SPAN")
    semantic_by_raw_id = dict(semantic_candidates)

    projected = []
    for span in spans:
        attributes = dict(span.attributes or {})
        raw_span_id = span.context.span_id  # type: ignore[union-attr]
        parent_key = None
        if span.parent is not None:
            parent_key = semantic_by_raw_id.get(span.parent.span_id)
            if parent_key is None:
                raise ProjectionError("PARENT_EDGE_MISMATCH", span.name)
        linked_keys = []
        for link in span.links:
            linked_key = semantic_by_raw_id.get(link.context.span_id)
            if linked_key is None:
                raise ProjectionError("LINK_EDGE_MISMATCH", span.name)
            linked_keys.append(linked_key)
        events = [
            {
                "name": event.name,
                "attributes": dict(event.attributes or {}),
            }
            for event in span.events
        ]
        projected.append(
            {
                "span_semantic_key": semantic_by_raw_id[raw_span_id],
                "name": span.name,
                "parent_semantic_key": parent_key,
                "linked_semantic_keys": linked_keys,
                "status": span.status.status_code.name,
                "attributes": attributes,
                "events": events,
            }
        )
    return canonicalize_trace(
        {
            "schema_version": SCHEMA_VERSION,
            "trace_semantic_key": trace_key(expected_run_id),
            "spans": projected,
        }
    )
