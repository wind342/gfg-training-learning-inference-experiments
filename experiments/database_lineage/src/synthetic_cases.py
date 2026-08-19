from __future__ import annotations

from decimal import Decimal

from .core_adapter import CoreAdapter
from .operators import Aggregate, Projection, SortKey
from .relational_executor import (
    RelationTuple,
    RelationalExecutor,
    base_tuple,
    ensure_unique_tuple_identities,
)


def _rows(table: str, specs: list[tuple[str, dict]]) -> list[RelationTuple]:
    rows = [
        base_tuple(tuple_id, table, values, index)
        for index, (tuple_id, values) in enumerate(specs)
    ]
    ensure_unique_tuple_identities(rows)
    return rows


def adversarial_tables() -> dict[str, list[RelationTuple]]:
    tables = {
        "Customers": _rows(
            "Customers",
            [
                (
                    "customers:c1",
                    {
                        "customer_id": "C1",
                        "customer_name": "Alice",
                        "segment": "BUILDING",
                    },
                ),
                (
                    "customers:c2",
                    {
                        "customer_id": "C2",
                        "customer_name": "Bob",
                        "segment": "AUTOMOBILE",
                    },
                ),
                (
                    "customers:c3",
                    {
                        "customer_id": "C3",
                        "customer_name": "Eve",
                        "segment": "HOUSEHOLD",
                    },
                ),
            ],
        ),
        "Products": _rows(
            "Products",
            [
                (
                    "products:p1a",
                    {
                        "product_id": "P1",
                        "product_name": "Widget",
                        "category": "tools",
                        "list_price": Decimal("10.00"),
                    },
                ),
                (
                    "products:p1b",
                    {
                        "product_id": "P1",
                        "product_name": "Widget",
                        "category": "tools",
                        "list_price": Decimal("10.00"),
                    },
                ),
                (
                    "products:p2",
                    {
                        "product_id": "P2",
                        "product_name": "Bolt",
                        "category": "parts",
                        "list_price": Decimal("5.00"),
                    },
                ),
                (
                    "products:p3",
                    {
                        "product_id": "P3",
                        "product_name": "Cable",
                        "category": "parts",
                        "list_price": Decimal("7.50"),
                    },
                ),
                (
                    "products:p9",
                    {
                        "product_id": "P9",
                        "product_name": "Unmatched",
                        "category": "edge",
                        "list_price": Decimal("-1.00"),
                    },
                ),
            ],
        ),
        "Orders": _rows(
            "Orders",
            [
                (
                    "orders:o1",
                    {
                        "order_id": "O1",
                        "customer_id": "C1",
                        "status": "open",
                        "priority": 1,
                    },
                ),
                (
                    "orders:o2",
                    {
                        "order_id": "O2",
                        "customer_id": "C1",
                        "status": "open",
                        "priority": 1,
                    },
                ),
                (
                    "orders:o3",
                    {
                        "order_id": "O3",
                        "customer_id": "C2",
                        "status": "cancelled",
                        "priority": 0,
                    },
                ),
                (
                    "orders:o4",
                    {
                        "order_id": "O4",
                        "customer_id": "C404",
                        "status": "open",
                        "priority": -1,
                    },
                ),
                (
                    "orders:o5",
                    {
                        "order_id": "O5",
                        "customer_id": "C2",
                        "status": "open",
                        "priority": 1,
                    },
                ),
            ],
        ),
        "OrderItems": _rows(
            "OrderItems",
            [
                (
                    "items:i1",
                    {
                        "order_id": "O1",
                        "product_id": "P1",
                        "quantity": Decimal("2"),
                        "unit_price": Decimal("10.00"),
                    },
                ),
                (
                    "items:i2",
                    {
                        "order_id": "O1",
                        "product_id": "P2",
                        "quantity": Decimal("1"),
                        "unit_price": Decimal("5.00"),
                    },
                ),
                (
                    "items:i3",
                    {
                        "order_id": "O2",
                        "product_id": "P1",
                        "quantity": Decimal("1"),
                        "unit_price": Decimal("10.00"),
                    },
                ),
                (
                    "items:i4",
                    {
                        "order_id": "O404",
                        "product_id": "P1",
                        "quantity": Decimal("0"),
                        "unit_price": Decimal("0.00"),
                    },
                ),
                (
                    "items:i5",
                    {
                        "order_id": "O5",
                        "product_id": "P3",
                        "quantity": Decimal("1"),
                        "unit_price": Decimal("7.50"),
                    },
                ),
                (
                    "items:i6",
                    {
                        "order_id": "O5",
                        "product_id": "PX",
                        "quantity": Decimal("0"),
                        "unit_price": Decimal("-1.00"),
                    },
                ),
            ],
        ),
        "Promotions": _rows(
            "Promotions",
            [
                (
                    "promotions:m1",
                    {
                        "product_id": "P1",
                        "promotion_code": "SAVE",
                        "discount": Decimal("0.10"),
                    },
                ),
                (
                    "promotions:m2",
                    {
                        "product_id": "P1",
                        "promotion_code": "SAVE",
                        "discount": Decimal("0.10"),
                    },
                ),
                (
                    "promotions:m9",
                    {
                        "product_id": "P404",
                        "promotion_code": "NONE",
                        "discount": Decimal("0"),
                    },
                ),
            ],
        ),
    }
    return tables


def execute_business_query(
    contract: CoreAdapter | None = None,
) -> tuple[list[RelationTuple], RelationalExecutor]:
    tables = adversarial_tables()
    executor = RelationalExecutor(contract)
    selected = executor.selection(
        tables["Orders"],
        stage="orders_selected",
        predicate=lambda row: row["status"] == "open",
        predicate_description="status = :status",
        parameters={"status": "open"},
    )
    with_customers = executor.equi_join(
        selected,
        tables["Customers"],
        stage="orders_customers",
        left_keys=["customer_id"],
        right_keys=["customer_id"],
        right_prefix="customer_",
    )
    with_items = executor.equi_join(
        with_customers,
        tables["OrderItems"],
        stage="orders_customers_items",
        left_keys=["order_id"],
        right_keys=["order_id"],
        right_prefix="item_",
    )
    with_products = executor.equi_join(
        with_items,
        tables["Products"],
        stage="orders_customers_items_products",
        left_keys=["product_id"],
        right_keys=["product_id"],
        right_prefix="product_",
    )
    derived = executor.projection(
        with_products,
        stage="line_totals",
        projections=[
            Projection("customer_id", "customer_id", lambda row: row["customer_id"]),
            Projection(
                "customer_name", "customer_name", lambda row: row["customer_name"]
            ),
            Projection(
                "line_total",
                "quantity * unit_price",
                lambda row: row["quantity"] * row["unit_price"],
            ),
        ],
    )
    grouped = executor.group_by(
        derived,
        stage="customer_aggregates",
        group_keys=["customer_id", "customer_name"],
        aggregates=[
            Aggregate("revenue", "SUM", "line_total", lambda row: row["line_total"]),
            Aggregate("line_count", "COUNT"),
        ],
    )
    ordered = executor.sort(
        grouped,
        stage="customer_rank",
        sort_keys=[SortKey("revenue", descending=True), SortKey("customer_id")],
    )
    limited = executor.limit(ordered, stage="customer_top_1", count=1)
    return limited, executor


def execute_many_to_many_case(
    contract: CoreAdapter | None = None,
) -> tuple[list[RelationTuple], RelationalExecutor]:
    tables = adversarial_tables()
    executor = RelationalExecutor(contract)
    left = [
        base_tuple(
            "m2m:left:1", "M2MLeft", {"product_id": "P1", "left_value": "same"}, 0
        ),
        base_tuple(
            "m2m:left:2", "M2MLeft", {"product_id": "P1", "left_value": "same"}, 1
        ),
    ]
    outputs = executor.equi_join(
        left,
        tables["Products"],
        stage="many_to_many",
        left_keys=["product_id"],
        right_keys=["product_id"],
        right_prefix="product_",
    )
    return outputs, executor
