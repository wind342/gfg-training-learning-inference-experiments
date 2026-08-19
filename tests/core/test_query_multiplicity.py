from __future__ import annotations

from generation_relation_core.entities import query_request
from generation_relation_core.query_engine import QueryEngine
from tests.fixtures.runtime import source_id


def test_set_projection_is_distinct_union_while_relations_keep_multiplicity(core_fixture) -> None:
    repeated_id = source_id(core_fixture, "repeated_source")
    bindings = [
        row
        for row in core_fixture.snapshot.tables.generation_bindings
        if row["origin_reference"].get("source_information_id") == repeated_id
    ]
    supports = {
        row["support_id"]: row for row in core_fixture.snapshot.tables.perceptual_support_records
    }
    first_support = supports[bindings[0]["outcome_reference"]["support_id"]]
    request = query_request(
        domain_scope_id=first_support["domain_scope_id"],
        support_space_id=first_support["support_space_id"],
        predicate_profile_id=first_support["predicate_profile_id"],
        predicate="containment",
        query_payload={
            "frame_id": "frame_1",
            "x_min": 0,
            "x_max": 100,
            "y_min": 0,
            "y_max": 100,
            "depth_min": 0,
            "depth_max": 1,
        },
    )
    result = QueryEngine(core_fixture.snapshot, core_fixture.registry).execute(request).result
    relations = [relation for hit in result["hits"] for relation in hit["generation_relations"]]
    assert len([row for row in relations if row["origin"].get("source_information_id") == repeated_id]) == 2
    assert result["source_information_ids"].count(repeated_id) == 1
    assert len(result["occurrence_ids"]) == len(set(result["occurrence_ids"]))
