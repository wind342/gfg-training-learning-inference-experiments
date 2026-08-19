import copy

import pytest

from generation_relation_core.errors import CoreV3Error
from generation_relation_core.snapshots import validate_snapshot


def test_snapshot_counts_hashes_foreign_keys_and_coverage_pass(business_run) -> None:
    adapter, _rows, snapshot, _reader = business_run
    token = validate_snapshot(snapshot, adapter.registry)
    assert token.snapshot_id == snapshot.snapshot_id
    assert (
        snapshot.record["authoritative_table_counts"]
        == snapshot.tables.authoritative_counts()
    )


def test_snapshot_rejects_orphan_and_hash_drift(business_run) -> None:
    adapter, _rows, snapshot, _reader = business_run
    damaged = copy.deepcopy(snapshot)
    damaged.tables.generation_bindings.pop()
    with pytest.raises(CoreV3Error):
        validate_snapshot(damaged, adapter.registry)
