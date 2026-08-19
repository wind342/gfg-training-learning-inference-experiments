from decimal import Decimal

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.operators import Projection
from experiments.database_lineage.src.relational_executor import RelationalExecutor
from experiments.database_lineage.src.synthetic_cases import adversarial_tables


def test_projection_renaming_and_derived_decimal_are_executed() -> None:
    adapter = CoreAdapter(run_id="projection")
    executor = RelationalExecutor(adapter)
    rows = executor.projection(
        adversarial_tables()["OrderItems"][:2],
        stage="projection",
        projections=[
            Projection("id", "product_id", lambda row: row["product_id"]),
            Projection(
                "line_total",
                "quantity * unit_price",
                lambda row: row["quantity"] * row["unit_price"],
            ),
        ],
    )
    assert [row.values for row in rows] == [
        {"id": "P1", "line_total": Decimal("20.00")},
        {"id": "P2", "line_total": Decimal("5.00")},
    ]
    payload = adapter.tables.generation_occurrences[0]["occurrence_payload"]
    assert payload["projections"][1]["expression"] == "quantity * unit_price"
    adapter.validated_snapshot()
