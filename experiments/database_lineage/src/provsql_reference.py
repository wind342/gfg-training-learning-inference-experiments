"""Independent ProvSQL evaluator.

This module is intentionally not imported by the tested executor, adapter, or
Core lineage reader.  It consumes their already-produced evaluation artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg

from .canonical_lineage import compare_lineage


TABLE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "region": (("r_regionkey", "integer"), ("r_name", "text"), ("r_comment", "text")),
    "nation": (
        ("n_nationkey", "integer"),
        ("n_name", "text"),
        ("n_regionkey", "integer"),
        ("n_comment", "text"),
    ),
    "supplier": (
        ("s_suppkey", "bigint"),
        ("s_name", "text"),
        ("s_address", "text"),
        ("s_nationkey", "integer"),
        ("s_phone", "text"),
        ("s_acctbal", "numeric(15,2)"),
        ("s_comment", "text"),
    ),
    "customer": (
        ("c_custkey", "bigint"),
        ("c_name", "text"),
        ("c_address", "text"),
        ("c_nationkey", "integer"),
        ("c_phone", "text"),
        ("c_acctbal", "numeric(15,2)"),
        ("c_mktsegment", "text"),
        ("c_comment", "text"),
    ),
    "part": (
        ("p_partkey", "bigint"),
        ("p_name", "text"),
        ("p_mfgr", "text"),
        ("p_brand", "text"),
        ("p_type", "text"),
        ("p_size", "integer"),
        ("p_container", "text"),
        ("p_retailprice", "numeric(15,2)"),
        ("p_comment", "text"),
    ),
    "partsupp": (
        ("ps_partkey", "bigint"),
        ("ps_suppkey", "bigint"),
        ("ps_availqty", "bigint"),
        ("ps_supplycost", "numeric(15,2)"),
        ("ps_comment", "text"),
    ),
    "orders": (
        ("o_orderkey", "bigint"),
        ("o_custkey", "bigint"),
        ("o_orderstatus", "text"),
        ("o_totalprice", "numeric(15,2)"),
        ("o_orderdate", "date"),
        ("o_orderpriority", "text"),
        ("o_clerk", "text"),
        ("o_shippriority", "integer"),
        ("o_comment", "text"),
    ),
    "lineitem": (
        ("l_orderkey", "bigint"),
        ("l_partkey", "bigint"),
        ("l_suppkey", "bigint"),
        ("l_linenumber", "bigint"),
        ("l_quantity", "numeric(15,2)"),
        ("l_extendedprice", "numeric(15,2)"),
        ("l_discount", "numeric(15,2)"),
        ("l_tax", "numeric(15,2)"),
        ("l_returnflag", "text"),
        ("l_linestatus", "text"),
        ("l_shipdate", "date"),
        ("l_commitdate", "date"),
        ("l_receiptdate", "date"),
        ("l_shipinstruct", "text"),
        ("l_shipmode", "text"),
        ("l_comment", "text"),
    ),
}

LABEL_SQL = {
    "region": "'region:' || r_regionkey",
    "nation": "'nation:' || n_nationkey",
    "supplier": "'supplier:' || s_suppkey",
    "customer": "'customer:' || c_custkey",
    "part": "'part:' || p_partkey",
    "partsupp": "'partsupp:' || ps_partkey || ':' || ps_suppkey",
    "orders": "'orders:' || o_orderkey",
    "lineitem": "'lineitem:' || l_orderkey || ':' || l_linenumber",
}


def connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=True)


def server_capabilities(connection: psycopg.Connection) -> dict[str, Any]:
    with connection.cursor() as cursor:
        postgres_version = cursor.execute("SHOW server_version").fetchone()[0]
        installed = cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'provsql'"
        ).fetchone()
        functions = cursor.execute(
            """
            SELECT DISTINCT p.proname
            FROM pg_proc AS p
            JOIN pg_namespace AS n ON n.oid = p.pronamespace
            WHERE n.nspname = 'provsql'
              AND p.proname IN ('sr_which', 'sr_why', 'sr_how', 'sr_counting')
            ORDER BY p.proname
            """
        ).fetchall()
    names = {row[0] for row in functions}
    return {
        "postgresql_version": postgres_version,
        "provsql_extension_version": installed[0] if installed else None,
        "semiring_functions": sorted(names),
        "sr_which_available": "sr_which" in names,
        "sr_why_available": "sr_why" in names,
        "sr_how_available": "sr_how" in names,
        "sr_counting_available": "sr_counting" in names,
    }


def initialize_database(
    connection: psycopg.Connection, csv_dir: Path
) -> dict[str, int]:
    """Load the exact DuckDB exports into the dedicated evaluator database."""
    with connection.cursor() as cursor:
        cursor.execute("DROP SCHEMA IF EXISTS public CASCADE")
        cursor.execute("CREATE SCHEMA public")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS provsql CASCADE")
        cursor.execute("SET search_path TO public, provsql")
        counts: dict[str, int] = {}
        for table, columns in TABLE_COLUMNS.items():
            declaration = ", ".join(
                f"{name} {data_type}" for name, data_type in columns
            )
            cursor.execute(f"CREATE TABLE {table} ({declaration})")
            names = ", ".join(name for name, _type in columns)
            copy_sql = (
                f"COPY {table} ({names}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
            )
            with (
                (csv_dir / f"{table}.csv").open("rb") as source,
                cursor.copy(copy_sql) as copy,
            ):
                while chunk := source.read(1024 * 1024):
                    copy.write(chunk)
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN stable_tuple_label text")
            cursor.execute(
                f"UPDATE {table} SET stable_tuple_label = {LABEL_SQL[table]}"
            )
            cursor.execute(
                f"ALTER TABLE {table} ALTER COLUMN stable_tuple_label SET NOT NULL"
            )
            cursor.execute(
                f"CREATE UNIQUE INDEX {table}_stable_tuple_label_idx ON {table}(stable_tuple_label)"
            )
            counts[table] = cursor.execute(f"SELECT count(*) FROM {table}").fetchone()[
                0
            ]
            cursor.execute("SELECT provsql.add_provenance(%s)", (table,))
        cursor.execute(
            "CREATE TABLE provsql_tuple_mapping (token uuid PRIMARY KEY, value text NOT NULL)"
        )
        for table in TABLE_COLUMNS:
            cursor.execute(
                f"INSERT INTO provsql_tuple_mapping(token, value) "
                f"SELECT provenance(), stable_tuple_label FROM {table}"
            )
    return counts


def _inject_provenance(sql: str) -> str:
    """Add evaluator-only provenance columns without changing query semantics."""
    clean = sql.strip().removesuffix(";")
    marker = "\nFROM"
    position = clean.upper().find(marker)
    if position < 0:
        raise ValueError("official query has no FROM clause")
    return (
        clean[:position]
        + ",\n    provenance() AS __provsql_token,\n"
        + "    provsql.sr_which(provenance(), 'provsql_tuple_mapping') AS __which_labels"
        + clean[position:]
    )


def _alignment_key(query_number: int, row: dict[str, Any], ordinal: int) -> str:
    if query_number == 1:
        return f"{row['l_returnflag']}|{row['l_linestatus']}"
    if query_number == 3:
        return f"{row['l_orderkey']}|{row['o_orderdate'].isoformat()}|{row['o_shippriority']}"
    if query_number == 6:
        return "scalar:0"
    if query_number == 10:
        return f"{row['c_custkey']}|{row['c_name']}|{ordinal}"
    raise ValueError(query_number)


def execute_which(
    connection: psycopg.Connection, query_number: int, sql: str
) -> tuple[dict[str, list[str]], float]:
    start = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute("SET search_path TO public, provsql")
        cursor.execute(_inject_provenance(sql))
        names = [column.name for column in cursor.description]
        rows = [dict(zip(names, record, strict=True)) for record in cursor.fetchall()]
    elapsed = time.perf_counter() - start
    result = {}
    for ordinal, row in enumerate(rows):
        labels = row.pop("__which_labels")
        row.pop("__provsql_token")
        result[_alignment_key(query_number, row, ordinal)] = sorted(labels or [])
    return result, elapsed


def compare_core_and_provsql(
    core: dict[str, Iterable[str]],
    provsql: dict[str, Iterable[str]],
) -> dict[str, Any]:
    keys = sorted(set(core) | set(provsql))
    comparisons = {
        key: compare_lineage(core.get(key, ()), provsql.get(key, ())) for key in keys
    }
    return {
        "output_rows": len(keys),
        "exact_output_rows": sum(item["exact"] for item in comparisons.values()),
        "false_positives": sum(
            len(item["false_positives"]) for item in comparisons.values()
        ),
        "false_negatives": sum(
            len(item["false_negatives"]) for item in comparisons.values()
        ),
        "comparisons": comparisons,
    }


def read_core_lineage(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))
