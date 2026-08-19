import pytest


@pytest.mark.skip(
    reason="NOT_EVALUATED: native stage-1/stage-2 maps and standard composition result are absent"
)
def test_source_map_composition_requires_completed_two_stage_fixture() -> None:
    pass
