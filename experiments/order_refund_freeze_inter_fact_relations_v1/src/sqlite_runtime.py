from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..common import EXPERIMENT_ROOT, file_sha256


SCHEMA_PATH = EXPERIMENT_ROOT / "sql" / "schema.sql"
INITIAL_STATE_PATH = EXPERIMENT_ROOT / "sql" / "initial_state.sql"


def connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(db_path),
        timeout=15,
        isolation_level=None,
    )
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    return connection


def initialize_database(db_path: Path) -> sqlite3.Connection:
    connection = connect(db_path)
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    if mode.lower() != "wal":
        raise RuntimeError("SQLITE_WAL_MODE_REQUIRED")
    connection.executescript(INITIAL_STATE_PATH.read_text(encoding="utf-8"))
    return connection


def canonical_dump(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    connection = connect(db_path)
    try:
        tables: dict[str, list[dict[str, Any]]] = {}
        for table, columns, order_by in (
            (
                "orders",
                ("order_id", "amount_cents", "status", "version"),
                "order_id",
            ),
            (
                "refunds",
                (
                    "refund_id",
                    "order_id",
                    "amount_cents",
                    "status",
                    "idempotency_key",
                ),
                "refund_id",
            ),
            (
                "notifications",
                (
                    "notification_id",
                    "order_id",
                    "refund_id",
                    "status",
                    "notification_kind",
                ),
                "notification_id",
            ),
        ):
            cursor = connection.execute(
                f"SELECT {','.join(columns)} FROM {table} ORDER BY {order_by}"
            )
            tables[table] = [
                dict(zip(columns, row, strict=True)) for row in cursor.fetchall()
            ]
        return tables
    finally:
        connection.close()


def sqlite_binary_identity(db_path: Path) -> dict[str, Any]:
    wal_path = Path(f"{db_path}-wal")
    return {
        "sqlite_version": sqlite3.sqlite_version,
        "db_sha256": file_sha256(db_path),
        "wal_present": wal_path.exists(),
        "wal_sha256": file_sha256(wal_path) if wal_path.exists() else None,
        "schema_sha256": file_sha256(SCHEMA_PATH),
        "initial_state_sha256": file_sha256(INITIAL_STATE_PATH),
        "binary_hashes_excluded_from_scientific_hash": True,
    }
