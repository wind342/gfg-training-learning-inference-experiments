from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.relational_executor import RelationalExecutor
from experiments.database_lineage.src.synthetic_cases import (
    adversarial_tables,
    execute_many_to_many_case,
)


def test_one_to_many_join_uses_actual_pairs_and_marks_both_unmatched_sides() -> None:
    adapter = CoreAdapter(run_id="one-to-many")
    executor = RelationalExecutor(adapter)
    tables = adversarial_tables()
    outputs = executor.equi_join(
        tables["Orders"],
        tables["OrderItems"],
        stage="order_items",
        left_keys=["order_id"],
        right_keys=["order_id"],
        right_prefix="item_",
    )
    assert len(outputs) == 5
    assert sum(row.values["order_id"] == "O1" for row in outputs) == 2
    assert {
        row["domain_reason_code"] for row in adapter.tables.explicit_dispositions
    } == {
        "join_unmatched_left",
        "join_unmatched_right",
    }
    adapter.validated_snapshot()


def test_many_to_many_join_has_four_real_outputs_and_eight_direct_edges() -> None:
    adapter = CoreAdapter(run_id="many-to-many")
    outputs, _executor = execute_many_to_many_case(adapter)
    snapshot = adapter.validated_snapshot()
    reader = CoreLineageReader(snapshot, adapter.registry)
    support_edges = [
        row for row in reader.direct_relations() if row["outcome_kind"] == "support"
    ]
    assert len(outputs) == 4
    assert len(support_edges) == 8
    assert all(len(reader.direct_input_tuple_ids(row.tuple_id)) == 2 for row in outputs)
