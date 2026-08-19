from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from experiments.database_lineage.src.duckdb_reference import (
    compare_official_typed,
    compare_rows,
    execute_reference,
    parse_official_answer,
)
from experiments.database_lineage.src.result_serializer import ordinary_rows
from experiments.database_lineage.src.tpch_loader import (
    load_tables,
    official_sql_and_answers,
)
from experiments.database_lineage.src.tpch_plans import PLANS


TABLES = {
    1: ("lineitem",),
    3: ("customer", "orders", "lineitem"),
    6: ("lineitem",),
    10: ("customer", "orders", "lineitem", "nation"),
}
DATABASE = Path("experiments/database_lineage/runtime/tpch_sf_0_01.duckdb")


@pytest.mark.parametrize("query_number", [1, 3, 6, 10])
def test_fixed_tpch_operator_plan_matches_duckdb_and_official_answer(
    query_number: int,
) -> None:
    assert DATABASE.is_file(), "run generate_tpch.py before the experiment test suite"
    connection = duckdb.connect(str(DATABASE), read_only=True)
    connection.execute("LOAD tpch")
    official = official_sql_and_answers(connection)
    actual = PLANS[query_number](load_tables(connection, TABLES[query_number]), None)
    reference = execute_reference(connection, official["queries"][query_number])
    answer = parse_official_answer(official["answers"][f"0.01:{query_number}"])
    assert compare_rows(ordinary_rows(actual), reference["text_rows"])["exact"]
    assert compare_official_typed([row.values for row in actual], answer["text_rows"])[
        "exact_after_typed_parse"
    ]
    connection.close()
