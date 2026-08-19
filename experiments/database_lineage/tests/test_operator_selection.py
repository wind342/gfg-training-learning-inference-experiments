from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.relational_executor import RelationalExecutor
from experiments.database_lineage.src.synthetic_cases import adversarial_tables


def test_selection_executes_predicate_and_records_each_exclusion() -> None:
    adapter = CoreAdapter(run_id="selection")
    executor = RelationalExecutor(adapter)
    rows = executor.selection(
        adversarial_tables()["Orders"],
        stage="selection",
        predicate=lambda row: row["status"] == "open",
        predicate_description="status = 'open'",
    )
    assert len(rows) == 4
    assert {row.values["order_id"] for row in rows} == {"O1", "O2", "O4", "O5"}
    reasons = [
        row["domain_reason_code"] for row in adapter.tables.explicit_dispositions
    ]
    assert reasons == ["selection_excluded"]
    adapter.validated_snapshot()
