from __future__ import annotations


def test_generator_created_support_uses_generated_origin(core_fixture) -> None:
    binding = next(
        row
        for row in core_fixture.snapshot.tables.generation_bindings
        if row["relation_role"] == "generator_created_support"
    )
    assert binding["origin_reference"]["kind"] == "generated_origin"
    assert binding["origin_reference"]["generated_origin_id"] in {
        row["generated_origin_id"] for row in core_fixture.snapshot.tables.generated_origins
    }
    assert all(
        row["source_identity"] != "generated_grid_overlay"
        for row in core_fixture.snapshot.tables.source_information_records
    )
