from __future__ import annotations

from experiments.provenance_semiring_projection_v1.src.ordinary_execution import execute_ordinary
from experiments.provenance_semiring_projection_v1.src.workloads import load_workloads, workload_by_id


def test_exactly_twelve_frozen_workloads_load() -> None:
    workloads = load_workloads()
    assert [item["id"] for item in workloads] == [f"W{index}" for index in range(1, 13)]


def test_observation_does_not_change_ordinary_bytes() -> None:
    for workload in load_workloads():
        without_capture, _ = execute_ordinary(workload)
        events: list[dict[str, object]] = []
        with_capture, _ = execute_ordinary(workload, collector=lambda event: events.append(event))
        assert with_capture == without_capture
        assert events


def test_w11_plan_variants_have_exactly_equal_ordinary_rows() -> None:
    workload = workload_by_id("W11")
    plan_a, _ = execute_ordinary(workload, variant="plan_a")
    plan_b, _ = execute_ordinary(workload, variant="plan_b")
    assert plan_a.replace(b'"plan_a"', b'"plan"') == plan_b.replace(b'"plan_b"', b'"plan"')


def test_stress_workload_scale_and_identity_count() -> None:
    workload = workload_by_id("W12")
    output, measurements = execute_ordinary(workload)
    assert measurements["source_count"] == 100
    assert measurements["occurrence_count"] >= 100
    assert measurements["ordinary_row_count"] >= 100
    assert output.endswith(b"\n")
