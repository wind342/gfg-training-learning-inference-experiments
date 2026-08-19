import pytest


@pytest.mark.skip(
    reason="NOT_EVALUATED: completed frozen OTel workload and official-SDK artifacts are absent on the base commit"
)
def test_otel_p1_requires_completed_native_experiment() -> None:
    pass
