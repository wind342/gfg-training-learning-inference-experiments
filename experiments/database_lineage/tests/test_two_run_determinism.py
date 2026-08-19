from generation_relation_core.canonical import canonical_bytes

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.result_serializer import csv_bytes, json_bytes
from experiments.database_lineage.src.synthetic_cases import execute_business_query


def _run():
    adapter = CoreAdapter(run_id="deterministic-run")
    rows, _executor = execute_business_query(adapter)
    snapshot = adapter.validated_snapshot()
    reader = CoreLineageReader(snapshot, adapter.registry)
    lineage = reader.backward(rows[0].tuple_id)
    return adapter, rows, snapshot, lineage


def test_two_complete_runs_are_semantically_identical() -> None:
    left_adapter, left_rows, left, left_lineage = _run()
    right_adapter, right_rows, right, right_lineage = _run()
    assert csv_bytes(left_rows) == csv_bytes(right_rows)
    assert json_bytes(left_rows) == json_bytes(right_rows)
    assert left.snapshot_id == right.snapshot_id
    assert canonical_bytes(left.record) == canonical_bytes(right.record)
    for field in left_adapter.tables.__dataclass_fields__:
        assert canonical_bytes(getattr(left.tables, field)) == canonical_bytes(
            getattr(right.tables, field)
        )
    assert left_lineage == right_lineage
    assert left_adapter.registry.profile_ids == right_adapter.registry.profile_ids
