from __future__ import annotations

from generation_relation_core.entities import query_request
from generation_relation_core.query_engine import QueryEngine
from tests.fixtures.runtime import occurrence_id, source_id


def test_query_returns_precise_declared_relations(core_fixture) -> None:
    binding = next(
        row
        for row in core_fixture.snapshot.tables.generation_bindings
        if row["origin_reference"].get("source_information_id")
        == source_id(core_fixture, "pair_source_1")
    )
    support = next(
        row
        for row in core_fixture.snapshot.tables.perceptual_support_records
        if row["support_id"] == binding["outcome_reference"]["support_id"]
    )
    request = query_request(
        domain_scope_id=support["domain_scope_id"],
        support_space_id=support["support_space_id"],
        predicate_profile_id=support["predicate_profile_id"],
        predicate="membership",
        query_payload={"x": 20, "y": 30},
    )
    hit = QueryEngine(core_fixture.snapshot, core_fixture.registry).execute(request).result["hits"][0]
    pairs = {
        (
            relation["origin"].get("source_information_id"),
            relation["generation_occurrence_id"],
        )
        for relation in hit["generation_relations"]
    }
    assert pairs == {
        (source_id(core_fixture, "pair_source_1"), occurrence_id(core_fixture, "pair_occurrence_1")),
        (source_id(core_fixture, "pair_source_2"), occurrence_id(core_fixture, "pair_occurrence_2")),
    }
