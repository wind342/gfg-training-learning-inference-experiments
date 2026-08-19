import pytest


@pytest.mark.skip(
    reason="NOT_EVALUATED: OTel P1 prerequisite is not established on the base commit"
)
def test_otel_p2_requires_completed_native_experiment() -> None:
    pass
