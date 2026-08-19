from __future__ import annotations

import pytest

from experiments.provenance_semiring_projection_v1.src.runtime import run_relational_generation


def test_all_runtime_modes_execute_real_paths() -> None:
    output_only = run_relational_generation({"mode": "output_only", "workload_id": "W4"})
    native = run_relational_generation({"mode": "native_semiring_only", "workload_id": "W4"})
    core = run_relational_generation({"mode": "core_only", "workload_id": "W4"})
    dual = run_relational_generation({"mode": "dual", "workload_id": "W4"})
    assert output_only["ordinary_output"] == core["ordinary_output"] == dual["ordinary_output"]
    assert native["native"]["outputs"] == core["candidate"]["outputs"]
    assert dual["semantic_exact"] is True


def test_write_only_collector_cannot_control_output() -> None:
    events = []
    baseline = run_relational_generation({"mode": "output_only", "workload_id": "W1"})
    captured = run_relational_generation(
        {"mode": "output_only", "workload_id": "W1"},
        collector=lambda event: events.append(event) or {"attempted": "control"},
    )
    assert captured["ordinary_output"] == baseline["ordinary_output"]
    assert events


def test_unknown_runtime_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown runtime mode"):
        run_relational_generation({"mode": "repair", "workload_id": "W1"})
