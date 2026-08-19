from __future__ import annotations

from datetime import date

from .core_adapter import CoreAdapter
from .operators import Aggregate, Projection, SortKey
from .relational_executor import RelationTuple, RelationalExecutor


PLAN_DESCRIPTIONS = {
    1: [
        "selection",
        "projection/derived expressions",
        "group-by SUM/AVG/COUNT",
        "projection of SQL AVG type",
        "sort",
    ],
    3: [
        "customer selection",
        "orders selection",
        "customer-orders join",
        "lineitem selection",
        "orders-lineitem join",
        "derived revenue",
        "group-by SUM",
        "sort",
        "limit",
    ],
    6: ["selection", "derived revenue", "scalar SUM"],
    10: [
        "orders selection",
        "lineitem selection",
        "customer-orders join",
        "orders-lineitem join",
        "nation join",
        "derived revenue",
        "group-by SUM",
        "sort",
        "limit",
    ],
}


def execute_q1(
    tables: dict[str, list[RelationTuple]], contract: CoreAdapter | None
) -> list[RelationTuple]:
    executor = RelationalExecutor(contract)
    selected = executor.selection(
        tables["lineitem"],
        stage="q1_shipdate_selection",
        predicate=lambda row: row["l_shipdate"] <= date(1998, 9, 2),
        predicate_description="l_shipdate <= DATE '1998-09-02'",
        parameters={"shipdate": "1998-09-02"},
    )
    projected = executor.projection(
        selected,
        stage="q1_derived",
        projections=[
            Projection("l_returnflag", "l_returnflag", lambda row: row["l_returnflag"]),
            Projection("l_linestatus", "l_linestatus", lambda row: row["l_linestatus"]),
            Projection("l_quantity", "l_quantity", lambda row: row["l_quantity"]),
            Projection(
                "l_extendedprice", "l_extendedprice", lambda row: row["l_extendedprice"]
            ),
            Projection("l_discount", "l_discount", lambda row: row["l_discount"]),
            Projection(
                "disc_price",
                "l_extendedprice * (1 - l_discount)",
                lambda row: row["l_extendedprice"] * (1 - row["l_discount"]),
            ),
            Projection(
                "charge",
                "l_extendedprice * (1 - l_discount) * (1 + l_tax)",
                lambda row: (
                    row["l_extendedprice"]
                    * (1 - row["l_discount"])
                    * (1 + row["l_tax"])
                ),
            ),
        ],
    )
    grouped = executor.group_by(
        projected,
        stage="q1_aggregation",
        group_keys=["l_returnflag", "l_linestatus"],
        aggregates=[
            Aggregate("sum_qty", "SUM", "l_quantity", lambda row: row["l_quantity"]),
            Aggregate(
                "sum_base_price",
                "SUM",
                "l_extendedprice",
                lambda row: row["l_extendedprice"],
            ),
            Aggregate(
                "sum_disc_price", "SUM", "disc_price", lambda row: row["disc_price"]
            ),
            Aggregate("sum_charge", "SUM", "charge", lambda row: row["charge"]),
            Aggregate("avg_qty", "AVG", "l_quantity", lambda row: row["l_quantity"]),
            Aggregate(
                "avg_price",
                "AVG",
                "l_extendedprice",
                lambda row: row["l_extendedprice"],
            ),
            Aggregate("avg_disc", "AVG", "l_discount", lambda row: row["l_discount"]),
            Aggregate("count_order", "COUNT"),
        ],
    )
    typed = executor.projection(
        grouped,
        stage="q1_sql_result_types",
        projections=[
            Projection("l_returnflag", "l_returnflag", lambda row: row["l_returnflag"]),
            Projection("l_linestatus", "l_linestatus", lambda row: row["l_linestatus"]),
            Projection("sum_qty", "sum_qty", lambda row: row["sum_qty"]),
            Projection(
                "sum_base_price", "sum_base_price", lambda row: row["sum_base_price"]
            ),
            Projection(
                "sum_disc_price", "sum_disc_price", lambda row: row["sum_disc_price"]
            ),
            Projection("sum_charge", "sum_charge", lambda row: row["sum_charge"]),
            Projection(
                "avg_qty",
                "CAST(exact_sum/count AS DOUBLE)",
                lambda row: float(row["avg_qty"]),
            ),
            Projection(
                "avg_price",
                "CAST(exact_sum/count AS DOUBLE)",
                lambda row: float(row["avg_price"]),
            ),
            Projection(
                "avg_disc",
                "CAST(exact_sum/count AS DOUBLE)",
                lambda row: float(row["avg_disc"]),
            ),
            Projection("count_order", "count_order", lambda row: row["count_order"]),
        ],
    )
    return executor.sort(
        typed,
        stage="q1_sort",
        sort_keys=[SortKey("l_returnflag"), SortKey("l_linestatus")],
    )


def execute_q3(
    tables: dict[str, list[RelationTuple]], contract: CoreAdapter | None
) -> list[RelationTuple]:
    executor = RelationalExecutor(contract)
    customers = executor.selection(
        tables["customer"],
        stage="q3_customer_selection",
        predicate=lambda row: row["c_mktsegment"] == "BUILDING",
        predicate_description="c_mktsegment = 'BUILDING'",
    )
    orders = executor.selection(
        tables["orders"],
        stage="q3_orders_selection",
        predicate=lambda row: row["o_orderdate"] < date(1995, 3, 15),
        predicate_description="o_orderdate < DATE '1995-03-15'",
    )
    customer_orders = executor.equi_join(
        customers,
        orders,
        stage="q3_customer_orders",
        left_keys=["c_custkey"],
        right_keys=["o_custkey"],
        right_prefix="orders_",
    )
    lineitem = executor.selection(
        tables["lineitem"],
        stage="q3_lineitem_selection",
        predicate=lambda row: row["l_shipdate"] > date(1995, 3, 15),
        predicate_description="l_shipdate > DATE '1995-03-15'",
    )
    joined = executor.equi_join(
        customer_orders,
        lineitem,
        stage="q3_orders_lineitem",
        left_keys=["o_orderkey"],
        right_keys=["l_orderkey"],
        right_prefix="lineitem_",
    )
    projected = executor.projection(
        joined,
        stage="q3_revenue",
        projections=[
            Projection("l_orderkey", "l_orderkey", lambda row: row["l_orderkey"]),
            Projection(
                "revenue_term",
                "l_extendedprice * (1 - l_discount)",
                lambda row: row["l_extendedprice"] * (1 - row["l_discount"]),
            ),
            Projection("o_orderdate", "o_orderdate", lambda row: row["o_orderdate"]),
            Projection(
                "o_shippriority", "o_shippriority", lambda row: row["o_shippriority"]
            ),
        ],
    )
    grouped = executor.group_by(
        projected,
        stage="q3_aggregation",
        group_keys=["l_orderkey", "o_orderdate", "o_shippriority"],
        aggregates=[
            Aggregate("revenue", "SUM", "revenue_term", lambda row: row["revenue_term"])
        ],
    )
    ordered = executor.sort(
        grouped,
        stage="q3_sort",
        sort_keys=[SortKey("revenue", descending=True), SortKey("o_orderdate")],
    )
    return executor.limit(ordered, stage="q3_limit", count=10)


def execute_q6(
    tables: dict[str, list[RelationTuple]], contract: CoreAdapter | None
) -> list[RelationTuple]:
    executor = RelationalExecutor(contract)
    selected = executor.selection(
        tables["lineitem"],
        stage="q6_selection",
        predicate=lambda row: (
            date(1994, 1, 1) <= row["l_shipdate"] < date(1995, 1, 1)
            and row["l_discount"] >= row["l_discount"].__class__("0.05")
            and row["l_discount"] <= row["l_discount"].__class__("0.07")
            and row["l_quantity"] < row["l_quantity"].__class__("24")
        ),
        predicate_description="l_shipdate in 1994 AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24",
    )
    projected = executor.projection(
        selected,
        stage="q6_revenue",
        projections=[
            Projection(
                "revenue_term",
                "l_extendedprice * l_discount",
                lambda row: row["l_extendedprice"] * row["l_discount"],
            )
        ],
    )
    return executor.group_by(
        projected,
        stage="q6_aggregation",
        group_keys=[],
        aggregates=[
            Aggregate("revenue", "SUM", "revenue_term", lambda row: row["revenue_term"])
        ],
    )


def execute_q10(
    tables: dict[str, list[RelationTuple]], contract: CoreAdapter | None
) -> list[RelationTuple]:
    executor = RelationalExecutor(contract)
    orders = executor.selection(
        tables["orders"],
        stage="q10_orders_selection",
        predicate=lambda row: (
            date(1993, 10, 1) <= row["o_orderdate"] < date(1994, 1, 1)
        ),
        predicate_description="o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'",
    )
    lineitem = executor.selection(
        tables["lineitem"],
        stage="q10_lineitem_selection",
        predicate=lambda row: row["l_returnflag"] == "R",
        predicate_description="l_returnflag = 'R'",
    )
    customer_orders = executor.equi_join(
        tables["customer"],
        orders,
        stage="q10_customer_orders",
        left_keys=["c_custkey"],
        right_keys=["o_custkey"],
        right_prefix="orders_",
    )
    with_lineitem = executor.equi_join(
        customer_orders,
        lineitem,
        stage="q10_orders_lineitem",
        left_keys=["o_orderkey"],
        right_keys=["l_orderkey"],
        right_prefix="lineitem_",
    )
    with_nation = executor.equi_join(
        with_lineitem,
        tables["nation"],
        stage="q10_nation",
        left_keys=["c_nationkey"],
        right_keys=["n_nationkey"],
        right_prefix="nation_",
    )
    projected = executor.projection(
        with_nation,
        stage="q10_revenue",
        projections=[
            Projection("c_custkey", "c_custkey", lambda row: row["c_custkey"]),
            Projection("c_name", "c_name", lambda row: row["c_name"]),
            Projection(
                "revenue_term",
                "l_extendedprice * (1 - l_discount)",
                lambda row: row["l_extendedprice"] * (1 - row["l_discount"]),
            ),
            Projection("c_acctbal", "c_acctbal", lambda row: row["c_acctbal"]),
            Projection("n_name", "n_name", lambda row: row["n_name"]),
            Projection("c_address", "c_address", lambda row: row["c_address"]),
            Projection("c_phone", "c_phone", lambda row: row["c_phone"]),
            Projection("c_comment", "c_comment", lambda row: row["c_comment"]),
        ],
    )
    grouped = executor.group_by(
        projected,
        stage="q10_aggregation",
        group_keys=[
            "c_custkey",
            "c_name",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        ],
        aggregates=[
            Aggregate("revenue", "SUM", "revenue_term", lambda row: row["revenue_term"])
        ],
    )
    # SQL projection order differs from GROUP BY key order.
    reordered = executor.projection(
        grouped,
        stage="q10_result_columns",
        projections=[
            Projection("c_custkey", "c_custkey", lambda row: row["c_custkey"]),
            Projection("c_name", "c_name", lambda row: row["c_name"]),
            Projection("revenue", "revenue", lambda row: row["revenue"]),
            Projection("c_acctbal", "c_acctbal", lambda row: row["c_acctbal"]),
            Projection("n_name", "n_name", lambda row: row["n_name"]),
            Projection("c_address", "c_address", lambda row: row["c_address"]),
            Projection("c_phone", "c_phone", lambda row: row["c_phone"]),
            Projection("c_comment", "c_comment", lambda row: row["c_comment"]),
        ],
    )
    ordered = executor.sort(
        reordered, stage="q10_sort", sort_keys=[SortKey("revenue", descending=True)]
    )
    return executor.limit(ordered, stage="q10_limit", count=20)


PLANS = {1: execute_q1, 3: execute_q3, 6: execute_q6, 10: execute_q10}
