from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import duckdb

from .relational_executor import RelationTuple, base_tuple


TABLES = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)
ORDER_BY = {
    "region": "r_regionkey",
    "nation": "n_nationkey",
    "supplier": "s_suppkey",
    "customer": "c_custkey",
    "part": "p_partkey",
    "partsupp": "ps_partkey, ps_suppkey",
    "orders": "o_orderkey",
    "lineitem": "l_orderkey, l_linenumber",
}


def scale_name(scale_factor: float) -> str:
    return str(scale_factor).replace(".", "_")


def connect_database(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute("LOAD tpch")
    return connection


def generate_database(
    scale_factor: float, database_path: Path, export_dir: Path
) -> dict:
    if database_path.exists():
        database_path.unlink()
    export_dir.mkdir(parents=True, exist_ok=True)
    connection = connect_database(database_path)
    connection.execute(f"CALL dbgen(sf = {scale_factor})")
    version = connection.execute("SELECT version()").fetchone()[0]
    extension = connection.execute(
        "SELECT installed_from, extension_version FROM duckdb_extensions() WHERE extension_name = 'tpch'"
    ).fetchone()
    row_counts: dict[str, int] = {}
    export_hashes: dict[str, str] = {}
    export_sizes: dict[str, int] = {}
    for table in TABLES:
        row_counts[table] = connection.execute(
            f"SELECT count(*) FROM {table}"
        ).fetchone()[0]
        target = export_dir / f"{table}.csv"
        escaped = str(target).replace("'", "''")
        connection.execute(
            f"COPY (SELECT * FROM {table} ORDER BY {ORDER_BY[table]}) "
            f"TO '{escaped}' (FORMAT CSV, HEADER true)"
        )
        payload = target.read_bytes()
        export_hashes[table] = hashlib.sha256(payload).hexdigest()
        export_sizes[table] = len(payload)
    connection.close()
    return {
        "scale_factor": scale_factor,
        "database_path": str(database_path),
        "duckdb_version": version,
        "tpch_extension_installed_from": extension[0],
        "tpch_extension_version": extension[1],
        "table_row_counts": row_counts,
        "export_sha256": export_hashes,
        "export_sizes": export_sizes,
    }


def tuple_label(table: str, values: dict) -> str:
    if table == "region":
        return f"region:{values['r_regionkey']}"
    if table == "nation":
        return f"nation:{values['n_nationkey']}"
    if table == "supplier":
        return f"supplier:{values['s_suppkey']}"
    if table == "customer":
        return f"customer:{values['c_custkey']}"
    if table == "part":
        return f"part:{values['p_partkey']}"
    if table == "partsupp":
        return f"partsupp:{values['ps_partkey']}:{values['ps_suppkey']}"
    if table == "orders":
        return f"orders:{values['o_orderkey']}"
    if table == "lineitem":
        return f"lineitem:{values['l_orderkey']}:{values['l_linenumber']}"
    raise ValueError(f"unknown TPC-H table: {table}")


def load_table(
    connection: duckdb.DuckDBPyConnection, table: str
) -> list[RelationTuple]:
    cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {ORDER_BY[table]}")
    columns = [item[0] for item in cursor.description]
    result = []
    for index, record in enumerate(cursor.fetchall()):
        values = dict(zip(columns, record, strict=True))
        result.append(base_tuple(tuple_label(table, values), table, values, index))
    return result


def load_tables(
    connection: duckdb.DuckDBPyConnection, names: Iterable[str]
) -> dict[str, list[RelationTuple]]:
    return {name: load_table(connection, name) for name in names}


def official_sql_and_answers(connection: duckdb.DuckDBPyConnection) -> dict:
    queries = {
        int(number): sql
        for number, sql in connection.execute(
            "SELECT query_nr, query FROM tpch_queries()"
        ).fetchall()
    }
    answers = {
        f"{scale}:{int(number)}": answer
        for number, scale, answer in connection.execute(
            "SELECT query_nr, scale_factor, answer FROM tpch_answers()"
        ).fetchall()
    }
    return {"queries": queries, "answers": answers}
