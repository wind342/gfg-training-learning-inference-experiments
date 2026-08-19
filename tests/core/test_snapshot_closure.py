from __future__ import annotations

import copy

import pytest

from generation_relation_core.canonical import finalize_entity, projection
from generation_relation_core.errors import CoreV3Error
from generation_relation_core.snapshots import validate_snapshot


def test_snapshot_table_counts_and_hashes_close(core_fixture) -> None:
    validation = validate_snapshot(
        core_fixture.snapshot,
        core_fixture.registry,
        expected_implementation_hashes=core_fixture.implementation,
    )
    assert validation.snapshot_id == core_fixture.snapshot.snapshot_id


def test_declared_count_and_hash_drift_fail_closed(core_fixture) -> None:
    snapshot = copy.deepcopy(core_fixture.snapshot)
    record = projection("ValidatedSnapshot", snapshot.record)
    record["authoritative_table_counts"]["generation_bindings"] += 1
    snapshot.record = finalize_entity("ValidatedSnapshot", record)
    with pytest.raises(CoreV3Error) as exc:
        validate_snapshot(snapshot, core_fixture.registry, expected_implementation_hashes=core_fixture.implementation)
    assert exc.value.reason_code == "SNAPSHOT_TABLE_COUNT_MISMATCH"

    snapshot = copy.deepcopy(core_fixture.snapshot)
    record = projection("ValidatedSnapshot", snapshot.record)
    record["authoritative_table_hashes"]["generation_bindings"] = "0" * 64
    snapshot.record = finalize_entity("ValidatedSnapshot", record)
    with pytest.raises(CoreV3Error) as exc:
        validate_snapshot(snapshot, core_fixture.registry, expected_implementation_hashes=core_fixture.implementation)
    assert exc.value.reason_code == "SNAPSHOT_TABLE_HASH_MISMATCH"
