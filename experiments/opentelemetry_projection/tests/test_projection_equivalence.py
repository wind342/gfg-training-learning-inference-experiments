from experiments.opentelemetry_projection.src.core_to_otel_projection import (
    project_core_to_otel,
)
from experiments.opentelemetry_projection.src.experiment_fixtures import (
    run_captured,
    selection_fixture,
)
from experiments.opentelemetry_projection.src.independent_oracle import (
    BUSINESS_TOPOLOGY_ORACLE,
    SELECTION_TRACE_ORACLE,
)
from experiments.opentelemetry_projection.src.projection_validator import (
    assert_trace_equal,
    trace_diff,
)


def test_native_and_direct_projection_equal_frozen_selection_oracle() -> None:
    run = run_captured(
        selection_fixture,
        run_id="oracle-selection-run",
        core_enabled=True,
        otel_enabled=True,
    )
    assert run.snapshot and run.validation and run.native_trace
    direct = project_core_to_otel(run.snapshot, run.validation)
    assert_trace_equal(SELECTION_TRACE_ORACLE, direct)
    assert_trace_equal(SELECTION_TRACE_ORACLE, run.native_trace)


def test_native_and_direct_projection_exact_for_multistage_business(
    captured_business_run,
) -> None:
    run = captured_business_run
    assert run.snapshot and run.validation and run.native_trace
    direct = project_core_to_otel(run.snapshot, run.validation)
    diff = trace_diff(direct, run.native_trace)
    assert diff.exact
    assert len(direct["spans"]) == BUSINESS_TOPOLOGY_ORACLE["total_span_count"]
    occurrence_spans = [
        span
        for span in direct["spans"]
        if span["attributes"]["span.kind"] == "occurrence"
    ]
    stages = list(
        dict.fromkeys(
            span["attributes"]["operation.stage"] for span in occurrence_spans
        )
    )
    assert stages == BUSINESS_TOPOLOGY_ORACLE["stage_order"]
