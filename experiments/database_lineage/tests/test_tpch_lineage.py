from __future__ import annotations

from datetime import date
from decimal import Decimal

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.relational_executor import base_tuple
from experiments.database_lineage.src.tpch_plans import PLANS


def lineitem(tuple_id: str, *, qualifies: bool, ordinal: int):
    return base_tuple(
        tuple_id,
        "lineitem",
        {
            "l_orderkey": ordinal + 1,
            "l_linenumber": 1,
            "l_extendedprice": Decimal("100.00"),
            "l_discount": Decimal("0.06" if qualifies else "0.04"),
            "l_quantity": Decimal("10.00"),
            "l_shipdate": date(1994, 6, 1),
        },
        ordinal,
    )


def test_q6_plan_reads_lineage_only_from_core_direct_facts() -> None:
    adapter = CoreAdapter(run_id="tpch-q6-lineage-fixture")
    rows = PLANS[6](
        {
            "lineitem": [
                lineitem("lineitem:1:1", qualifies=True, ordinal=0),
                lineitem("lineitem:2:1", qualifies=False, ordinal=1),
            ]
        },
        adapter,
    )
    snapshot = adapter.validated_snapshot()
    reader = CoreLineageReader(snapshot, adapter.registry)
    lineage = reader.backward(rows[0].tuple_id)
    assert set(lineage.tuple_ids) == {"lineitem:1:1"}
    assert lineage.derivation_path_count == 1
    dispositions = {
        (edge["input_tuple_id"], edge["role"])
        for edge in reader.direct_relations()
        if edge["outcome_kind"] == "disposition"
    }
    assert dispositions == {("lineitem:2:1", "selection_excluded")}
    assert not any(
        edge["input_tuple_id"].startswith("lineitem:")
        and edge["output_tuple_id"] == rows[0].tuple_id
        for edge in reader.direct_relations()
    )
