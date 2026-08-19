from generation_relation_core.snapshots import validate_snapshot

from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.core_lineage_reader_reference import (
    LegacyScanCoreLineageReader,
)
from experiments.database_lineage.src.synthetic_oracle import BUSINESS_FORWARD


def test_scan_and_indexed_readers_return_identical_paths(business_run) -> None:
    adapter, rows, snapshot, _reader = business_run
    token = validate_snapshot(snapshot, adapter.registry)
    legacy = LegacyScanCoreLineageReader(snapshot, adapter.registry, prevalidated=token)
    indexed = CoreLineageReader(snapshot, adapter.registry, prevalidated=token)
    final_ids = {row.tuple_id for row in rows}
    assert legacy.backward(rows[0].tuple_id) == indexed.backward(rows[0].tuple_id)
    for source_id in BUSINESS_FORWARD:
        assert legacy.forward(source_id, final_ids) == indexed.forward(
            source_id, final_ids
        )


def test_indexed_reader_can_be_discarded_and_rebuilt(business_run) -> None:
    adapter, rows, snapshot, _reader = business_run
    token = validate_snapshot(snapshot, adapter.registry)
    first = CoreLineageReader(snapshot, adapter.registry, prevalidated=token).backward(
        rows[0].tuple_id
    )
    second = CoreLineageReader(snapshot, adapter.registry, prevalidated=token).backward(
        rows[0].tuple_id
    )
    assert first == second
