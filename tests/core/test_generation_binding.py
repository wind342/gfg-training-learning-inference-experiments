from __future__ import annotations

from tests.fixtures.runtime import bindings_for_sources, occurrence_id, source_id


def test_shared_support_preserves_only_true_pairings(core_fixture) -> None:
    bindings = bindings_for_sources(core_fixture, {"pair_source_1", "pair_source_2"})
    actual = {
        (row["origin_reference"]["source_information_id"], row["generation_occurrence_id"])
        for row in bindings
    }
    assert actual == {
        (source_id(core_fixture, "pair_source_1"), occurrence_id(core_fixture, "pair_occurrence_1")),
        (source_id(core_fixture, "pair_source_2"), occurrence_id(core_fixture, "pair_occurrence_2")),
    }
    assert len({row["outcome_reference"]["support_id"] for row in bindings}) == 1


def test_many_to_one_one_to_many_and_noncartesian_relations(core_fixture) -> None:
    shared_occurrence = bindings_for_sources(
        core_fixture, {"multi_source_1", "multi_source_2", "multi_source_3"}
    )
    assert len(shared_occurrence) == 3
    assert len({row["generation_occurrence_id"] for row in shared_occurrence}) == 1

    repeated = bindings_for_sources(core_fixture, {"repeated_source"})
    assert len(repeated) == 2
    assert len({row["generation_occurrence_id"] for row in repeated}) == 2

    split = bindings_for_sources(core_fixture, {"split_source"})
    assert len(split) == 2
    assert len({row["generation_occurrence_id"] for row in split}) == 1
    assert len({row["outcome_reference"]["support_id"] for row in split}) == 2

    selected = bindings_for_sources(
        core_fixture,
        {"noncartesian_source_1", "noncartesian_source_2", "noncartesian_source_3"},
    )
    assert len(selected) == 3
    assert len({row["generation_occurrence_id"] for row in selected}) == 3
    assert len(selected) != 9
