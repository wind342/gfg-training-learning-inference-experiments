from __future__ import annotations

import pytest

from experiments.opentelemetry_projection.src.experiment_fixtures import (
    business_fixture,
    run_captured,
)


@pytest.fixture(scope="session")
def captured_business_run():
    return run_captured(
        business_fixture,
        run_id="business-otel-equivalence",
        core_enabled=True,
        otel_enabled=True,
    )
