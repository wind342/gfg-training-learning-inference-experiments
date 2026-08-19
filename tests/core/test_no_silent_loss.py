from __future__ import annotations

import copy

import pytest

from generation_relation_core.errors import CoreV3Error
from generation_relation_core.snapshots import validate_tables
from tests.fixtures.runtime import occurrence_id, source_id


def _remove_only_binding(tables, *, source=None, occurrence=None) -> None:
    tables.generation_bindings = [
        row
        for row in tables.generation_bindings
        if not (
            (source is not None and row["origin_reference"].get("source_information_id") == source)
            or (occurrence is not None and row["generation_occurrence_id"] == occurrence)
        )
    ]


def test_no_silent_source_loss(core_fixture) -> None:
    tables = copy.deepcopy(core_fixture.snapshot.tables)
    _remove_only_binding(tables, source=source_id(core_fixture, "pair_source_1"))
    with pytest.raises(CoreV3Error) as exc:
        validate_tables(tables, core_fixture.registry)
    assert exc.value.reason_code == "SOURCE_COVERAGE_FAILED"
    assert exc.value.detail == "REGISTERED_SOURCE"


def test_no_silent_occurrence_loss(core_fixture) -> None:
    tables = copy.deepcopy(core_fixture.snapshot.tables)
    _remove_only_binding(tables, occurrence=occurrence_id(core_fixture, "repeated_occurrence_1"))
    with pytest.raises(CoreV3Error) as exc:
        validate_tables(tables, core_fixture.registry)
    assert exc.value.reason_code == "SOURCE_COVERAGE_FAILED"
    assert exc.value.detail == "OCCURRENCE"
