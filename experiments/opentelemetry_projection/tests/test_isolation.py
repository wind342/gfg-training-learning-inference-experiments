from __future__ import annotations

import builtins
from pathlib import Path

from experiments.opentelemetry_projection.src.core_to_otel_projection import (
    project_core_to_otel,
)
from experiments.opentelemetry_projection.src.experiment_fixtures import (
    run_captured,
    selection_fixture,
)
from experiments.opentelemetry_projection.src.isolation import (
    assert_static_projection_isolation,
    count_otel_core_fields,
)
from experiments.opentelemetry_projection.src.projection_validator import (
    assert_trace_equal,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECTION_SOURCE = Path(__file__).resolve().parents[1] / "src"


def test_projection_modules_have_no_oracle_or_native_imports() -> None:
    assert_static_projection_isolation(
        [
            PROJECTION_SOURCE / "core_to_otel_projection.py",
            PROJECTION_SOURCE / "database_projection.py",
            PROJECTION_SOURCE / "database_to_otel_projection.py",
        ]
    )


def test_projection_survives_native_record_deletion(captured_business_run) -> None:
    run = captured_business_run
    assert run.snapshot and run.validation and run.native and run.native_trace
    expected = project_core_to_otel(run.snapshot, run.validation)
    run.native.clear_native_records()
    assert run.native.exporter.get_finished_spans() == ()
    assert_trace_equal(expected, project_core_to_otel(run.snapshot, run.validation))


def test_runtime_traps_prohibited_projection_dependencies(
    monkeypatch, captured_business_run
) -> None:
    original_import = builtins.__import__

    def trapped_import(name, *args, **kwargs):
        if "native_otel_capture" in name or "independent_oracle" in name:
            raise AssertionError(f"prohibited runtime import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", trapped_import)
    run = captured_business_run
    assert run.snapshot and run.validation
    project_core_to_otel(run.snapshot, run.validation)


def test_native_capture_runs_when_projection_import_is_trapped(monkeypatch) -> None:
    original_import = builtins.__import__

    def trapped_import(name, *args, **kwargs):
        if "core_to_otel_projection" in name or "database_to_otel_projection" in name:
            raise AssertionError(f"prohibited native dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", trapped_import)
    run = run_captured(
        selection_fixture,
        run_id="native-isolation",
        core_enabled=False,
        otel_enabled=True,
    )
    assert run.native_trace is not None


def test_core_has_no_otel_specific_fields() -> None:
    assert count_otel_core_fields(REPOSITORY_ROOT) == 0
