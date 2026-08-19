import pytest

from experiments.opentelemetry_projection.src.core_to_otel_projection import (
    project_core_to_otel,
)
from experiments.opentelemetry_projection.src.isolation import (
    assert_injected_dependency_rejected,
)
from experiments.opentelemetry_projection.src.projection_validator import (
    run_negative_controls,
)


def test_all_thirteen_negative_controls_fail_closed(captured_business_run) -> None:
    run = captured_business_run
    assert run.snapshot and run.validation
    valid_trace = project_core_to_otel(run.snapshot, run.validation)
    results = run_negative_controls(valid_trace)
    results.append(
        {
            "control": "projection_reads_oracle_or_native",
            "reason_code": assert_injected_dependency_rejected(),
            "result": "FAIL_CLOSED",
        }
    )
    assert len(results) == 13
    assert all(result["result"] == "FAIL_CLOSED" for result in results)
    assert len({result["reason_code"] for result in results}) == 13


def test_unvalidated_snapshot_is_rejected(captured_business_run) -> None:
    from generation_relation_core.snapshots import SnapshotValidation

    run = captured_business_run
    assert run.snapshot
    with pytest.raises(Exception) as exc_info:
        project_core_to_otel(run.snapshot, SnapshotValidation("wrong", {}))
    assert (
        getattr(exc_info.value, "reason_code", None) == "SNAPSHOT_VALIDATION_REQUIRED"
    )
