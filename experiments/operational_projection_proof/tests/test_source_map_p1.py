import pytest


@pytest.mark.skip(
    reason="NOT_EVALUATED: no Source Map implementation or official reference artifact exists on the base commit"
)
def test_source_map_p1_requires_completed_native_experiment() -> None:
    pass
