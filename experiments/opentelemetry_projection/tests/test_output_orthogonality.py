from experiments.opentelemetry_projection.src.experiment_fixtures import (
    q6_like_small_fixture,
)
from experiments.opentelemetry_projection.src.output_orthogonality import (
    run_four_mode_orthogonality,
)


def test_core_and_otel_four_modes_are_output_orthogonal() -> None:
    report = run_four_mode_orthogonality(
        q6_like_small_fixture, run_id="orthogonality-q6-small"
    )
    assert report["passed"]
    assert report["forbidden_output_fields"] == []
