from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.operators import SortKey
from experiments.database_lineage.src.relational_executor import (
    RelationalExecutor,
    base_tuple,
)


def test_sort_has_identity_tie_break_and_limit_has_dispositions() -> None:
    rows = [
        base_tuple("sort:b", "Sort", {"score": 1}, 0),
        base_tuple("sort:a", "Sort", {"score": 1}, 1),
        base_tuple("sort:c", "Sort", {"score": 2}, 2),
    ]
    adapter = CoreAdapter(run_id="sort-limit")
    executor = RelationalExecutor(adapter)
    ordered = executor.sort(
        rows, stage="sort", sort_keys=[SortKey("score", descending=True)]
    )
    assert [adapter.tables.generated_origins for _ in ()] == []
    limited = executor.limit(ordered, stage="limit", count=2)
    assert [row.values["score"] for row in limited] == [2, 1]
    assert (
        adapter.tables.explicit_dispositions[0]["domain_reason_code"]
        == "limit_excluded"
    )
    assert (
        adapter.tables.explicit_dispositions[0]["disposition_payload"]["tuple_identity"]
        == "sort:00000002"
    )
    adapter.validated_snapshot()
