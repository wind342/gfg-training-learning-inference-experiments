from experiments.opentelemetry_projection.src.core_to_otel_projection import (
    project_core_to_otel,
)
from experiments.opentelemetry_projection.src.database_projection import (
    project_core_to_database,
)
from experiments.opentelemetry_projection.src.database_to_otel_projection import (
    project_database_to_otel,
)
from experiments.opentelemetry_projection.src.projection_validator import (
    assert_trace_equal,
)


def test_direct_and_hierarchical_projection_are_exact(captured_business_run) -> None:
    run = captured_business_run
    assert run.snapshot and run.validation
    direct = project_core_to_otel(run.snapshot, run.validation)
    database = project_core_to_database(run.snapshot, run.validation)
    hierarchical = project_database_to_otel(database)
    diff = assert_trace_equal(direct, hierarchical)
    assert diff.exact
    assert database.sources
    assert database.produced_tuples
    assert database.exclusions
    assert database.generated_bridges
    assert database.bindings
