from __future__ import annotations

from generation_relation_core.query_engine import QueryEngine


def test_non_support_outcomes_are_explicit_and_queryable(core_fixture) -> None:
    expected = {"occluded", "clipped", "culled", "suppressed", "depth_rejected"}
    assert {row["core_disposition_category"] for row in core_fixture.snapshot.tables.explicit_dispositions} == expected
    relations = QueryEngine(core_fixture.snapshot, core_fixture.registry).disposition_relations(
        "core_v3_synthetic_generation"
    )
    assert {row["status"] for row in relations} == expected
    assert all(row["outcome"]["kind"] == "disposition" for row in relations)
    assert all(row["origin"]["kind"] == "registered_source" for row in relations)
