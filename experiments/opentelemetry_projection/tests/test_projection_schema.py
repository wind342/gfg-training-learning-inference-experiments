from copy import deepcopy

import pytest

from experiments.opentelemetry_projection.src.canonical_otel import canonicalize_trace
from experiments.opentelemetry_projection.src.core_to_otel_projection import (
    project_core_to_otel,
)


def test_schema_blocks_wider_core_fact_leakage(captured_business_run) -> None:
    run = captured_business_run
    assert run.snapshot and run.validation
    candidate = deepcopy(project_core_to_otel(run.snapshot, run.validation))
    candidate["spans"][1]["attributes"]["source_information_id"] = "forbidden"
    with pytest.raises(Exception) as exc_info:
        canonicalize_trace(candidate)
    assert getattr(exc_info.value, "reason_code", None) == "PROJECTION_SCHEMA_INVALID"


def test_projection_contains_one_span_per_occurrence(captured_business_run) -> None:
    run = captured_business_run
    assert run.snapshot and run.validation
    trace = project_core_to_otel(run.snapshot, run.validation)
    assert len(trace["spans"]) == len(run.snapshot.tables.generation_occurrences) + 1
    assert all(
        span["attributes"].get("occurrence.cardinality", 1) == 1
        for span in trace["spans"]
    )
