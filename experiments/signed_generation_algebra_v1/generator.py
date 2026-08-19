"""Deterministic native executions with synchronous effect receipts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import sqlite3
from typing import Any, Callable


@dataclass(frozen=True)
class OperationReceipt:
    case_id: str
    execution_id: str
    sequence_index: int
    occurrence_key: str
    occurrence_type: str
    transform_operation: str
    relation_role: str
    source_identity: str
    effect_identity: str
    effect_multiplicity: int
    native_before: object
    native_after: object
    outcome_kind: str
    disposition_reason: str | None = None


@dataclass(frozen=True)
class NativeExecution:
    case_id: str
    execution_id: str
    final_output: object
    receipt_count: int


ReceiptCallback = Callable[[OperationReceipt], None]


def _receipt(
    execution: dict[str, Any],
    operation: dict[str, Any],
    *,
    native_before: object,
    native_after: object,
) -> OperationReceipt:
    return OperationReceipt(
        case_id=execution["case_id"],
        execution_id=execution["execution_id"],
        sequence_index=operation["sequence_index"],
        occurrence_key=operation["occurrence_key"],
        occurrence_type=operation["occurrence_type"],
        transform_operation=operation["kind"],
        relation_role=operation["relation_role"],
        source_identity=operation["source_identity"],
        effect_identity=operation["effect_identity"],
        effect_multiplicity=operation["effect_multiplicity"],
        native_before=deepcopy(native_before),
        native_after=deepcopy(native_after),
        outcome_kind=operation["outcome_kind"],
        disposition_reason=operation.get("disposition_reason"),
    )


def _execute_record_database(
    execution: dict[str, Any], callback: ReceiptCallback
) -> object:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE records (record_id TEXT PRIMARY KEY)"
        )
        for record_id in execution["initial_state"]["records"]:
            connection.execute(
                "INSERT INTO records(record_id) VALUES (?)", (record_id,)
            )
        connection.commit()
        for operation in execution["operations"]:
            before = [
                row[0]
                for row in connection.execute(
                    "SELECT record_id FROM records ORDER BY record_id"
                )
            ]
            if operation["kind"] == "insert_record":
                cursor = connection.execute(
                    "INSERT INTO records(record_id) VALUES (?)",
                    (operation["record_id"],),
                )
            elif operation["kind"] == "delete_record":
                cursor = connection.execute(
                    "DELETE FROM records WHERE record_id = ?",
                    (operation["record_id"],),
                )
            else:
                raise ValueError(
                    f"unsupported record operation: {operation['kind']}"
                )
            if cursor.rowcount != 1:
                raise RuntimeError("native record operation affected != 1 row")
            connection.commit()
            after = [
                row[0]
                for row in connection.execute(
                    "SELECT record_id FROM records ORDER BY record_id"
                )
            ]
            callback(
                _receipt(
                    execution,
                    operation,
                    native_before={"records": before},
                    native_after={"records": after},
                )
            )
        final_rows = [
            row[0]
            for row in connection.execute(
                "SELECT record_id FROM records ORDER BY record_id"
            )
        ]
        return {"records": final_rows}
    finally:
        connection.close()


def _execute_scalar_database(
    execution: dict[str, Any], callback: ReceiptCallback
) -> object:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE scalar_value (value INTEGER)")
        connection.execute(
            "INSERT INTO scalar_value(value) VALUES (?)",
            (execution["initial_state"]["value"],),
        )
        connection.commit()
        for operation in execution["operations"]:
            before = connection.execute(
                "SELECT value FROM scalar_value"
            ).fetchone()[0]
            cursor = connection.execute(
                "UPDATE scalar_value SET value = value + ?",
                (operation["delta"],),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("native scalar update affected != 1 row")
            connection.commit()
            after = connection.execute(
                "SELECT value FROM scalar_value"
            ).fetchone()[0]
            callback(
                _receipt(
                    execution,
                    operation,
                    native_before={"value": before},
                    native_after={"value": after},
                )
            )
        final_value = connection.execute(
            "SELECT value FROM scalar_value"
        ).fetchone()[0]
        return {"value": final_value}
    finally:
        connection.close()


def _execute_counter(
    execution: dict[str, Any], callback: ReceiptCallback
) -> object:
    counter = int(execution["initial_state"]["counter"])
    for operation in execution["operations"]:
        before = counter
        counter += int(operation["delta"])
        callback(
            _receipt(
                execution,
                operation,
                native_before={"counter": before},
                native_after={"counter": counter},
            )
        )
    return {"counter": counter}


def _execute_multiset(
    execution: dict[str, Any], callback: ReceiptCallback
) -> object:
    values = {
        str(key): int(value)
        for key, value in execution["initial_state"].items()
    }
    for operation in execution["operations"]:
        before = dict(sorted(values.items()))
        item = operation["item"]
        values[item] = values.get(item, 0) + int(operation["delta"])
        after = dict(sorted(values.items()))
        callback(
            _receipt(
                execution,
                operation,
                native_before=before,
                native_after=after,
            )
        )
    return dict(sorted(values.items()))


def _execute_filter(
    execution: dict[str, Any], callback: ReceiptCallback
) -> object:
    selected: list[str] = []
    for operation in execution["operations"]:
        item = operation["item"]
        before = {"selected": list(selected)}
        keep = item.startswith(operation["required_prefix"])
        if keep:
            selected.append(item)
        after = {"selected": list(selected)}
        if operation["outcome_kind"] != (
            "support" if keep else "disposition"
        ):
            raise RuntimeError("filter outcome contract disagrees with runtime")
        callback(
            _receipt(
                execution,
                operation,
                native_before=before,
                native_after=after,
            )
        )
    return {"selected": selected}


def execute_native(
    execution: dict[str, Any], callback: ReceiptCallback
) -> NativeExecution:
    engine = execution["engine"]
    if engine == "sqlite_records":
        final_output = _execute_record_database(execution, callback)
    elif engine == "sqlite_scalar":
        final_output = _execute_scalar_database(execution, callback)
    elif engine == "counter":
        final_output = _execute_counter(execution, callback)
    elif engine == "multiset":
        final_output = _execute_multiset(execution, callback)
    elif engine == "filter":
        final_output = _execute_filter(execution, callback)
    else:
        raise ValueError(f"unsupported native engine: {engine}")
    return NativeExecution(
        case_id=execution["case_id"],
        execution_id=execution["execution_id"],
        final_output=final_output,
        receipt_count=len(execution["operations"]),
    )
