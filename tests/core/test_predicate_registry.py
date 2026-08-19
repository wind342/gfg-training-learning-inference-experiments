from __future__ import annotations

import pytest

from generation_relation_core.entities import query_request
from generation_relation_core.query_engine import QueryEngine


@pytest.mark.parametrize(
    ("profile_name", "payload"),
    [
        ("page_2d_rectangle", {"x": 110, "y": 210}),
        ("pixel_support_set", {"pixel": [11, 12]}),
        (
            "fragment_depth_3d",
            {"frame_id": "frame_heterogeneous", "x": 12, "y": 8, "depth": 0.42},
        ),
        (
            "field_time_cell",
            {"grid_cell": [4, 7], "time_step": 12, "field_component": "temperature"},
        ),
    ],
)
def test_support_space_profiles_dispatch(core_fixture, profile_name, payload) -> None:
    profile = core_fixture.bundle.profiles_by_name[profile_name]
    space = core_fixture.bundle.spaces_by_name[profile_name]
    request = query_request(
        domain_scope_id="core_v3_synthetic_generation",
        support_space_id=space["support_space_id"],
        predicate_profile_id=profile["predicate_profile_id"],
        predicate="membership",
        query_payload=payload,
    )
    result = QueryEngine(core_fixture.snapshot, core_fixture.registry).execute(request).result
    assert result["query_status"] == "valid_nonempty"


def test_unknown_predicate_profile_fails_closed(core_fixture) -> None:
    space = core_fixture.bundle.spaces_by_name["page_2d_rectangle"]
    request = query_request(
        domain_scope_id="core_v3_synthetic_generation",
        support_space_id=space["support_space_id"],
        predicate_profile_id="pp3_" + "0" * 64,
        predicate="membership",
        query_payload={"x": 1, "y": 1},
    )
    result = QueryEngine(core_fixture.snapshot, core_fixture.registry).execute_controlled(request).result
    assert result["query_status"] == "controlled_error"
    assert result["reason_code"] == "PREDICATE_PROFILE_UNKNOWN"
