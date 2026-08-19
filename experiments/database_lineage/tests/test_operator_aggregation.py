from decimal import Decimal

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.operators import Aggregate
from experiments.database_lineage.src.relational_executor import (
    RelationalExecutor,
    base_tuple,
)


def test_sum_count_avg_bind_every_duplicate_identity() -> None:
    rows = [
        base_tuple("agg:a", "Agg", {"group": "x", "amount": Decimal("1.25")}, 0),
        base_tuple("agg:b", "Agg", {"group": "x", "amount": Decimal("1.25")}, 1),
        base_tuple("agg:c", "Agg", {"group": "x", "amount": Decimal("-0.50")}, 2),
    ]
    adapter = CoreAdapter(run_id="aggregation")
    executor = RelationalExecutor(adapter)
    outputs = executor.group_by(
        rows,
        stage="aggregation",
        group_keys=["group"],
        aggregates=[
            Aggregate("sum", "SUM", "amount", lambda row: row["amount"]),
            Aggregate("count", "COUNT"),
            Aggregate("avg", "AVG", "amount", lambda row: row["amount"]),
        ],
    )
    assert outputs[0].values == {
        "group": "x",
        "sum": Decimal("2.00"),
        "count": 3,
        "avg": Decimal("0.6666666666666666666666666667"),
    }
    snapshot = adapter.validated_snapshot()
    reader = CoreLineageReader(snapshot, adapter.registry)
    assert set(reader.backward(outputs[0].tuple_id).tuple_ids) == {
        "agg:a",
        "agg:b",
        "agg:c",
    }
    assert len(reader.direct_input_tuple_ids(outputs[0].tuple_id)) == 3
