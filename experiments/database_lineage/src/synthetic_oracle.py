"""Hand-authored expected facts; imports from the tested path are forbidden."""

from __future__ import annotations

from decimal import Decimal


BUSINESS_OUTPUT = [
    {
        "customer_id": "C1",
        "customer_name": "Alice",
        "revenue": Decimal("65.00"),
        "line_count": 5,
    },
]

BUSINESS_BACKWARD = {
    "customer_top_1:00000000": {
        "customers:c1",
        "orders:o1",
        "orders:o2",
        "items:i1",
        "items:i2",
        "items:i3",
        "products:p1a",
        "products:p1b",
        "products:p2",
    },
}

BUSINESS_FORWARD = {
    tuple_id: {"customer_top_1:00000000"}
    for tuple_id in BUSINESS_BACKWARD["customer_top_1:00000000"]
}
BUSINESS_FORWARD.update(
    {
        tuple_id: set()
        for tuple_id in {
            "customers:c2",
            "customers:c3",
            "orders:o3",
            "orders:o4",
            "orders:o5",
            "items:i4",
            "items:i5",
            "items:i6",
            "products:p3",
            "products:p9",
        }
    }
)

BUSINESS_AGGREGATE_CONTRIBUTORS = {
    "customer_aggregates:00000000": {
        "line_totals:00000000",
        "line_totals:00000001",
        "line_totals:00000002",
        "line_totals:00000003",
        "line_totals:00000004",
    },
    "customer_aggregates:00000001": {"line_totals:00000005"},
}

BUSINESS_DISPOSITIONS = {
    ("orders:o3", "selection_excluded"),
    ("orders_selected:00000002", "join_unmatched_left"),
    ("customers:c3", "join_unmatched_right"),
    ("items:i4", "join_unmatched_right"),
    ("orders_customers_items:00000004", "join_unmatched_left"),
    ("products:p9", "join_unmatched_right"),
    ("customer_rank:00000001", "limit_excluded"),
}

BUSINESS_DIRECT_PAIRS = {
    ("orders:o1", "orders_selected:00000000"),
    ("orders:o2", "orders_selected:00000001"),
    ("orders:o4", "orders_selected:00000002"),
    ("orders:o5", "orders_selected:00000003"),
    ("orders_selected:00000000", "orders_customers:00000000"),
    ("customers:c1", "orders_customers:00000000"),
    ("orders_selected:00000001", "orders_customers:00000001"),
    ("customers:c1", "orders_customers:00000001"),
    ("orders_selected:00000003", "orders_customers:00000002"),
    ("customers:c2", "orders_customers:00000002"),
    ("orders_customers:00000000", "orders_customers_items:00000000"),
    ("items:i1", "orders_customers_items:00000000"),
    ("orders_customers:00000000", "orders_customers_items:00000001"),
    ("items:i2", "orders_customers_items:00000001"),
    ("orders_customers:00000001", "orders_customers_items:00000002"),
    ("items:i3", "orders_customers_items:00000002"),
    ("orders_customers:00000002", "orders_customers_items:00000003"),
    ("items:i5", "orders_customers_items:00000003"),
    ("orders_customers:00000002", "orders_customers_items:00000004"),
    ("items:i6", "orders_customers_items:00000004"),
    ("orders_customers_items:00000000", "orders_customers_items_products:00000000"),
    ("products:p1a", "orders_customers_items_products:00000000"),
    ("orders_customers_items:00000000", "orders_customers_items_products:00000001"),
    ("products:p1b", "orders_customers_items_products:00000001"),
    ("orders_customers_items:00000001", "orders_customers_items_products:00000002"),
    ("products:p2", "orders_customers_items_products:00000002"),
    ("orders_customers_items:00000002", "orders_customers_items_products:00000003"),
    ("products:p1a", "orders_customers_items_products:00000003"),
    ("orders_customers_items:00000002", "orders_customers_items_products:00000004"),
    ("products:p1b", "orders_customers_items_products:00000004"),
    ("orders_customers_items:00000003", "orders_customers_items_products:00000005"),
    ("products:p3", "orders_customers_items_products:00000005"),
    *{
        (f"orders_customers_items_products:{index:08d}", f"line_totals:{index:08d}")
        for index in range(6)
    },
    *{
        (f"line_totals:{index:08d}", "customer_aggregates:00000000")
        for index in range(5)
    },
    ("line_totals:00000005", "customer_aggregates:00000001"),
    ("customer_aggregates:00000000", "customer_rank:00000000"),
    ("customer_aggregates:00000001", "customer_rank:00000001"),
    ("customer_rank:00000000", "customer_top_1:00000000"),
}

MANY_TO_MANY_DIRECT_PAIRS = {
    ("m2m:left:1", "many_to_many:00000000"),
    ("products:p1a", "many_to_many:00000000"),
    ("m2m:left:1", "many_to_many:00000001"),
    ("products:p1b", "many_to_many:00000001"),
    ("m2m:left:2", "many_to_many:00000002"),
    ("products:p1a", "many_to_many:00000002"),
    ("m2m:left:2", "many_to_many:00000003"),
    ("products:p1b", "many_to_many:00000003"),
}
