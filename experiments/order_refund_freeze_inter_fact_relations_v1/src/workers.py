from __future__ import annotations

from typing import Any

from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from ..common import TIMEOUT_SECONDS, canonical_sha256, content_id
from .sqlite_runtime import connect


class _DeterministicIdGenerator(IdGenerator):
    def __init__(self, seed: str) -> None:
        self.seed = seed
        self.span_index = 0
        self.trace_index = 0

    def generate_span_id(self) -> int:
        self.span_index += 1
        value = int(
            canonical_sha256(
                {"seed": self.seed, "span_index": self.span_index}
            )[:16],
            16,
        )
        return value or 1

    def generate_trace_id(self) -> int:
        self.trace_index += 1
        value = int(
            canonical_sha256(
                {"seed": self.seed, "trace_index": self.trace_index}
            )[:32],
            16,
        )
        return value or 1


def _event(
    rows: list[dict[str, Any]],
    *,
    execution_run_id: str,
    actor_id: str,
    event_type: str,
    detail: dict[str, Any] | None = None,
) -> str:
    sequence_index = len(rows)
    material = {
        "execution_run_id": execution_run_id,
        "actor_id": actor_id,
        "sequence_index": sequence_index,
        "event_type": event_type,
    }
    occurrence_id = content_id("orocc1_", material)
    rows.append(
        {
            "occurrence_id": occurrence_id,
            **material,
            "detail": detail or {},
        }
    )
    return occurrence_id


def _span(
    *,
    name: str,
    execution_run_id: str,
    action_type: str,
    status: str,
    order_id: str,
) -> dict[str, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "order-workflow"}),
        id_generator=_DeterministicIdGenerator(
            f"{execution_run_id}:{name}:{action_type}:{status}"
        ),
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("order-refund-freeze-v1")
    with tracer.start_as_current_span(
        name,
        attributes={
            "workflow.run_id": execution_run_id,
            "order.id": order_id,
            "action.type": action_type,
            "action.status": status,
        },
    ):
        pass
    provider.shutdown()
    finished = exporter.get_finished_spans()
    span = finished[0]
    return {
        "name": span.name,
        "trace_id": f"{span.context.trace_id:032x}",
        "span_id": f"{span.context.span_id:016x}",
        "parent_span_id": None,
        "links": [],
        "status_code": span.status.status_code.name,
        "attributes": {
            key: span.attributes[key] for key in sorted(span.attributes)
        },
        "profile": "CONVENTIONAL_NATIVE_TRACE_V1",
    }


def _message(
    *,
    execution_run_id: str,
    action_id: str,
    message_type: str,
    outcome: str,
    result_id: str,
    order_id: str,
    refund_id: str,
) -> dict[str, Any]:
    payload = {
        "execution_run_id": execution_run_id,
        "action_id": action_id,
        "message_type": message_type,
        "outcome": outcome,
        "result_id": result_id,
        "order_id": order_id,
        "refund_id": refund_id,
    }
    return {
        "message_id": content_id("ormsg1_", payload),
        "payload": payload,
        "payload_digest": canonical_sha256(payload),
    }


def refund_worker(
    config: dict[str, Any],
    read_barrier: Any,
    read_start_event: Any,
    commit_gate: Any,
    done_event: Any,
    notification_queue: Any,
    result_queue: Any,
) -> None:
    actor_id = config["actor_id"]
    run_id = config["execution_run_id"]
    events: list[dict[str, Any]] = []
    sql_receipts: list[dict[str, Any]] = []
    queue_receipts: list[dict[str, Any]] = []
    connection = None
    try:
        if read_start_event is not None:
            if not read_start_event.wait(TIMEOUT_SECONDS):
                raise RuntimeError("REFUND_READ_START_TIMEOUT")
            _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type="event_wait_released",
                detail={"event_id": config["read_start_event_id"]},
            )
        connection = connect(config["db_path"])
        if config.get("duplicate_check_only"):
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT refund_id,order_id,status,idempotency_key "
                "FROM refunds WHERE idempotency_key=?",
                (config["idempotency_key"],),
            ).fetchone()
            connection.execute("COMMIT")
            read_occurrence = _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type="sqlite_idempotency_read",
                detail={"idempotency_key": config["idempotency_key"]},
            )
            sql_receipts.append(
                {
                    "receipt_id": content_id(
                        "orsql1_",
                        {
                            "run": run_id,
                            "action": config["action_id"],
                            "kind": "idempotency_read",
                        },
                    ),
                    "transaction_id": f"tx-{config['action_id']}-read",
                    "action_id": config["action_id"],
                    "operation": "SELECT",
                    "table": "refunds",
                    "row": {
                        "refund_id": row[0],
                        "order_id": row[1],
                        "status": row[2],
                        "idempotency_key": row[3],
                    }
                    if row
                    else None,
                    "rowcount": 1 if row else 0,
                    "transaction_outcome": "COMMIT",
                    "occurrence_id": read_occurrence,
                }
            )
            if row is None:
                raise RuntimeError("DUPLICATE_REFUND_PREDECESSOR_MISSING")
            outcome = "IDEMPOTENT_DUPLICATE_REFUND"
            result_kind = "ExplicitDisposition"
            result_id = content_id(
                "orresult1_",
                {
                    "run": run_id,
                    "action": config["action_id"],
                    "outcome": outcome,
                },
            )
            result_occurrence = _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type="explicit_disposition",
                detail={"outcome": outcome},
            )
            message_type = "RefundDisposed"
            read_order = None
            conditional_rowcount = None
            transaction_outcome = "COMMIT"
        else:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT order_id,amount_cents,status,version "
                "FROM orders WHERE order_id=?",
                (config["order_id"],),
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise RuntimeError("ORDER_NOT_FOUND")
            read_order = {
                "order_id": row[0],
                "amount_cents": row[1],
                "status": row[2],
                "version": row[3],
            }
            read_occurrence = _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type="sqlite_order_read",
                detail=read_order,
            )
            sql_receipts.append(
                {
                    "receipt_id": content_id(
                        "orsql1_",
                        {
                            "run": run_id,
                            "action": config["action_id"],
                            "kind": "order_read",
                        },
                    ),
                    "transaction_id": f"tx-{config['action_id']}-read",
                    "action_id": config["action_id"],
                    "operation": "SELECT",
                    "table": "orders",
                    "resource_id": config["order_id"],
                    "version_id": f"order-001-v{row[3]}",
                    "row": read_order,
                    "rowcount": 1,
                    "transaction_outcome": "COMMIT",
                    "occurrence_id": read_occurrence,
                }
            )
            if read_barrier is not None:
                read_barrier.wait(TIMEOUT_SECONDS)
                _event(
                    events,
                    execution_run_id=run_id,
                    actor_id=actor_id,
                    event_type="barrier_passed",
                    detail={"barrier_id": config["barrier_id"]},
                )
            if not commit_gate.wait(TIMEOUT_SECONDS):
                raise RuntimeError("REFUND_COMMIT_GATE_TIMEOUT")
            _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type="event_wait_released",
                detail={"event_id": config["commit_gate_id"]},
            )
            if read_order["status"] != "OPEN":
                connection.execute("BEGIN")
                connection.execute("ROLLBACK")
                conditional_rowcount = None
                transaction_outcome = "ROLLBACK"
                outcome = "REFUND_REJECTED_ORDER_ALREADY_FROZEN"
                result_kind = "ExplicitDisposition"
                rollback_occurrence = _event(
                    events,
                    execution_run_id=run_id,
                    actor_id=actor_id,
                    event_type="sqlite_business_rule_rollback",
                    detail={"observed_status": read_order["status"]},
                )
                sql_receipts.append(
                    {
                        "receipt_id": content_id(
                            "orsql1_",
                            {
                                "run": run_id,
                                "action": config["action_id"],
                                "kind": "business_rule_rollback",
                            },
                        ),
                        "transaction_id": f"tx-{config['action_id']}-write",
                        "action_id": config["action_id"],
                        "operation": "ROLLBACK",
                        "table": "orders",
                        "resource_id": config["order_id"],
                        "expected_version_id": (
                            f"order-001-v{read_order['version']}"
                        ),
                        "rowcount": 0,
                        "transaction_outcome": "ROLLBACK",
                        "occurrence_id": rollback_occurrence,
                    }
                )
            else:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "UPDATE orders SET status='REFUNDED',version=version+1 "
                    "WHERE order_id=? AND version=? AND status='OPEN'",
                    (config["order_id"], read_order["version"]),
                )
                conditional_rowcount = cursor.rowcount
                update_occurrence = _event(
                    events,
                    execution_run_id=run_id,
                    actor_id=actor_id,
                    event_type="sqlite_conditional_update",
                    detail={"rowcount": cursor.rowcount},
                )
                if cursor.rowcount == 1:
                    connection.execute(
                        "INSERT INTO refunds("
                        "refund_id,order_id,amount_cents,status,idempotency_key"
                        ") VALUES (?,?,?,?,?)",
                        (
                            config["refund_id"],
                            config["order_id"],
                            config["amount_cents"],
                            "COMMITTED",
                            config["idempotency_key"],
                        ),
                    )
                    connection.execute("COMMIT")
                    transaction_outcome = "COMMIT"
                    outcome = "RefundCommitted"
                    result_kind = "BusinessSupport"
                else:
                    connection.execute("ROLLBACK")
                    transaction_outcome = "ROLLBACK"
                    current = connection.execute(
                        "SELECT status,version FROM orders WHERE order_id=?",
                        (config["order_id"],),
                    ).fetchone()
                    if current and current[0] == "FROZEN":
                        outcome = "REFUND_VERSION_CONFLICT_AFTER_FREEZE"
                    else:
                        outcome = "REFUND_VERSION_CONFLICT"
                    result_kind = "ExplicitDisposition"
                sql_receipts.append(
                    {
                        "receipt_id": content_id(
                            "orsql1_",
                            {
                                "run": run_id,
                                "action": config["action_id"],
                                "kind": "conditional_update",
                            },
                        ),
                        "transaction_id": f"tx-{config['action_id']}-write",
                        "action_id": config["action_id"],
                        "operation": "UPDATE",
                        "table": "orders",
                        "resource_id": config["order_id"],
                        "expected_version_id": (
                            f"order-001-v{read_order['version']}"
                        ),
                        "rowcount": conditional_rowcount,
                        "transaction_outcome": transaction_outcome,
                        "occurrence_id": update_occurrence,
                    }
                )
            result_id = content_id(
                "orresult1_",
                {
                    "run": run_id,
                    "action": config["action_id"],
                    "outcome": outcome,
                },
            )
            result_occurrence = _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type=(
                    "business_result"
                    if result_kind == "BusinessSupport"
                    else "explicit_disposition"
                ),
                detail={"outcome": outcome},
            )
            message_type = (
                "RefundCommitted"
                if outcome == "RefundCommitted"
                else "RefundDisposed"
            )
        message = _message(
            execution_run_id=run_id,
            action_id=config["action_id"],
            message_type=message_type,
            outcome=outcome,
            result_id=result_id,
            order_id=config["order_id"],
            refund_id=config["refund_id"],
        )
        send_occurrence = _event(
            events,
            execution_run_id=run_id,
            actor_id=actor_id,
            event_type="queue_put",
            detail={
                "message_id": message["message_id"],
                "payload_digest": message["payload_digest"],
            },
        )
        notification_queue.put(message, timeout=TIMEOUT_SECONDS)
        queue_receipts.append(
            {
                "receipt_id": content_id(
                    "orqueue1_",
                    {"run": run_id, "message": message["message_id"], "op": "put"},
                ),
                "operation": "put",
                "message_id": message["message_id"],
                "payload_digest": message["payload_digest"],
                "occurrence_id": send_occurrence,
                "actor_id": actor_id,
            }
        )
        result_queue.put(
            {
                "worker": actor_id,
                "status": "PASS",
                "events": events,
                "sql_receipts": sql_receipts,
                "queue_receipts": queue_receipts,
                "action_results": [
                    {
                        "action_id": config["action_id"],
                        "action_type": "refund",
                        "result_id": result_id,
                        "result_kind": result_kind,
                        "outcome": outcome,
                        "occurrence_id": result_occurrence,
                        "read_order": read_order,
                        "conditional_rowcount": conditional_rowcount,
                        "transaction_outcome": transaction_outcome,
                        "refund_id": config["refund_id"],
                        "idempotency_key": config["idempotency_key"],
                    }
                ],
                "native_spans": [
                    _span(
                        name="refund",
                        execution_run_id=run_id,
                        action_type="refund",
                        status=outcome,
                        order_id=config["order_id"],
                    )
                ],
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "worker": actor_id,
                "status": "FAIL",
                "reason": f"{type(error).__name__}:{error}",
                "events": events,
                "sql_receipts": sql_receipts,
                "queue_receipts": queue_receipts,
                "action_results": [],
                "native_spans": [],
            }
        )
    finally:
        if connection is not None:
            connection.close()
        done_event.set()


def freeze_worker(
    config: dict[str, Any],
    read_barrier: Any,
    commit_gate: Any,
    done_event: Any,
    result_queue: Any,
) -> None:
    actor_id = config["actor_id"]
    run_id = config["execution_run_id"]
    events: list[dict[str, Any]] = []
    sql_receipts: list[dict[str, Any]] = []
    connection = None
    try:
        connection = connect(config["db_path"])
        connection.execute("BEGIN")
        row = connection.execute(
            "SELECT order_id,amount_cents,status,version "
            "FROM orders WHERE order_id=?",
            (config["order_id"],),
        ).fetchone()
        connection.execute("COMMIT")
        if row is None:
            raise RuntimeError("ORDER_NOT_FOUND")
        read_order = {
            "order_id": row[0],
            "amount_cents": row[1],
            "status": row[2],
            "version": row[3],
        }
        read_occurrence = _event(
            events,
            execution_run_id=run_id,
            actor_id=actor_id,
            event_type="sqlite_order_read",
            detail=read_order,
        )
        sql_receipts.append(
            {
                "receipt_id": content_id(
                    "orsql1_",
                    {
                        "run": run_id,
                        "action": config["action_id"],
                        "kind": "order_read",
                    },
                ),
                "transaction_id": f"tx-{config['action_id']}-read",
                "action_id": config["action_id"],
                "operation": "SELECT",
                "table": "orders",
                "resource_id": config["order_id"],
                "version_id": f"order-001-v{row[3]}",
                "row": read_order,
                "rowcount": 1,
                "transaction_outcome": "COMMIT",
                "occurrence_id": read_occurrence,
            }
        )
        read_barrier.wait(TIMEOUT_SECONDS)
        _event(
            events,
            execution_run_id=run_id,
            actor_id=actor_id,
            event_type="barrier_passed",
            detail={"barrier_id": config["barrier_id"]},
        )
        if not commit_gate.wait(TIMEOUT_SECONDS):
            raise RuntimeError("FREEZE_COMMIT_GATE_TIMEOUT")
        _event(
            events,
            execution_run_id=run_id,
            actor_id=actor_id,
            event_type="event_wait_released",
            detail={"event_id": config["commit_gate_id"]},
        )
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            "UPDATE orders SET status='FROZEN',version=version+1 "
            "WHERE order_id=? AND version=? AND status='OPEN'",
            (config["order_id"], read_order["version"]),
        )
        update_occurrence = _event(
            events,
            execution_run_id=run_id,
            actor_id=actor_id,
            event_type="sqlite_conditional_update",
            detail={"rowcount": cursor.rowcount},
        )
        if cursor.rowcount == 1:
            connection.execute("COMMIT")
            transaction_outcome = "COMMIT"
            outcome = "OrderFrozen"
            result_kind = "BusinessSupport"
        else:
            connection.execute("ROLLBACK")
            transaction_outcome = "ROLLBACK"
            outcome = "FREEZE_VERSION_CONFLICT_AFTER_REFUND"
            result_kind = "ExplicitDisposition"
        sql_receipts.append(
            {
                "receipt_id": content_id(
                    "orsql1_",
                    {
                        "run": run_id,
                        "action": config["action_id"],
                        "kind": "conditional_update",
                    },
                ),
                "transaction_id": f"tx-{config['action_id']}-write",
                "action_id": config["action_id"],
                "operation": "UPDATE",
                "table": "orders",
                "resource_id": config["order_id"],
                "expected_version_id": f"order-001-v{read_order['version']}",
                "rowcount": cursor.rowcount,
                "transaction_outcome": transaction_outcome,
                "occurrence_id": update_occurrence,
            }
        )
        result_id = content_id(
            "orresult1_",
            {
                "run": run_id,
                "action": config["action_id"],
                "outcome": outcome,
            },
        )
        result_occurrence = _event(
            events,
            execution_run_id=run_id,
            actor_id=actor_id,
            event_type=(
                "business_result"
                if result_kind == "BusinessSupport"
                else "explicit_disposition"
            ),
            detail={"outcome": outcome},
        )
        result_queue.put(
            {
                "worker": actor_id,
                "status": "PASS",
                "events": events,
                "sql_receipts": sql_receipts,
                "queue_receipts": [],
                "action_results": [
                    {
                        "action_id": config["action_id"],
                        "action_type": "freeze",
                        "result_id": result_id,
                        "result_kind": result_kind,
                        "outcome": outcome,
                        "occurrence_id": result_occurrence,
                        "read_order": read_order,
                        "conditional_rowcount": cursor.rowcount,
                        "transaction_outcome": transaction_outcome,
                    }
                ],
                "native_spans": [
                    _span(
                        name="freeze",
                        execution_run_id=run_id,
                        action_type="freeze",
                        status=outcome,
                        order_id=config["order_id"],
                    )
                ],
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "worker": actor_id,
                "status": "FAIL",
                "reason": f"{type(error).__name__}:{error}",
                "events": events,
                "sql_receipts": sql_receipts,
                "queue_receipts": [],
                "action_results": [],
                "native_spans": [],
            }
        )
    finally:
        if connection is not None:
            connection.close()
        done_event.set()


def notification_worker(
    config: dict[str, Any],
    notification_queue: Any,
    result_queue: Any,
) -> None:
    actor_id = config["actor_id"]
    run_id = config["execution_run_id"]
    events: list[dict[str, Any]] = []
    sql_receipts: list[dict[str, Any]] = []
    queue_receipts: list[dict[str, Any]] = []
    action_results: list[dict[str, Any]] = []
    native_spans: list[dict[str, Any]] = []
    connection = None
    try:
        connection = connect(config["db_path"])
        for message_index in range(config["expected_message_count"]):
            message = notification_queue.get(timeout=TIMEOUT_SECONDS)
            if canonical_sha256(message["payload"]) != message["payload_digest"]:
                raise RuntimeError("QUEUE_PAYLOAD_DIGEST_MISMATCH")
            receive_occurrence = _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type="queue_get",
                detail={
                    "message_id": message["message_id"],
                    "payload_digest": message["payload_digest"],
                },
            )
            queue_receipts.append(
                {
                    "receipt_id": content_id(
                        "orqueue1_",
                        {
                            "run": run_id,
                            "message": message["message_id"],
                            "op": "get",
                        },
                    ),
                    "operation": "get",
                    "message_id": message["message_id"],
                    "payload_digest": message["payload_digest"],
                    "occurrence_id": receive_occurrence,
                    "actor_id": actor_id,
                }
            )
            payload = message["payload"]
            action_id = f"notification-{message_index + 1}"
            if payload["message_type"] == "RefundCommitted":
                notification_id = config["notification_id"]
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO notifications("
                    "notification_id,order_id,refund_id,status,notification_kind"
                    ") VALUES (?,?,?,?,?)",
                    (
                        notification_id,
                        payload["order_id"],
                        payload["refund_id"],
                        "SENT",
                        "REFUND_COMMITTED",
                    ),
                )
                if cursor.rowcount == 1:
                    connection.execute("COMMIT")
                    outcome = "NotificationSent"
                    result_kind = "BusinessSupport"
                    transaction_outcome = "COMMIT"
                else:
                    connection.execute("ROLLBACK")
                    outcome = "NOTIFICATION_DUPLICATE_SUPPRESSED"
                    result_kind = "ExplicitDisposition"
                    transaction_outcome = "ROLLBACK"
                write_occurrence = _event(
                    events,
                    execution_run_id=run_id,
                    actor_id=actor_id,
                    event_type="sqlite_notification_insert",
                    detail={"rowcount": cursor.rowcount},
                )
                sql_receipts.append(
                    {
                        "receipt_id": content_id(
                            "orsql1_",
                            {
                                "run": run_id,
                                "action": action_id,
                                "kind": "notification_insert",
                            },
                        ),
                        "transaction_id": f"tx-{action_id}",
                        "action_id": action_id,
                        "operation": "INSERT",
                        "table": "notifications",
                        "rowcount": cursor.rowcount,
                        "transaction_outcome": transaction_outcome,
                        "occurrence_id": write_occurrence,
                    }
                )
            else:
                connection.execute("BEGIN")
                connection.execute("ROLLBACK")
                outcome = "NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND"
                result_kind = "ExplicitDisposition"
                notification_id = None
                transaction_outcome = "ROLLBACK"
                rollback_occurrence = _event(
                    events,
                    execution_run_id=run_id,
                    actor_id=actor_id,
                    event_type="sqlite_notification_rollback",
                    detail={"source_message_id": message["message_id"]},
                )
                sql_receipts.append(
                    {
                        "receipt_id": content_id(
                            "orsql1_",
                            {
                                "run": run_id,
                                "action": action_id,
                                "kind": "notification_rollback",
                            },
                        ),
                        "transaction_id": f"tx-{action_id}",
                        "action_id": action_id,
                        "operation": "ROLLBACK",
                        "table": "notifications",
                        "rowcount": 0,
                        "transaction_outcome": "ROLLBACK",
                        "occurrence_id": rollback_occurrence,
                    }
                )
            result_id = content_id(
                "orresult1_",
                {
                    "run": run_id,
                    "action": action_id,
                    "message": message["message_id"],
                    "outcome": outcome,
                },
            )
            result_occurrence = _event(
                events,
                execution_run_id=run_id,
                actor_id=actor_id,
                event_type=(
                    "business_result"
                    if result_kind == "BusinessSupport"
                    else "explicit_disposition"
                ),
                detail={"outcome": outcome},
            )
            action_results.append(
                {
                    "action_id": action_id,
                    "action_type": "notification",
                    "result_id": result_id,
                    "result_kind": result_kind,
                    "outcome": outcome,
                    "occurrence_id": result_occurrence,
                    "source_message_id": message["message_id"],
                    "source_result_id": payload["result_id"],
                    "notification_id": notification_id,
                    "transaction_outcome": transaction_outcome,
                }
            )
            native_spans.append(
                _span(
                    name="notification",
                    execution_run_id=run_id,
                    action_type="notification",
                    status=outcome,
                    order_id=payload["order_id"],
                )
            )
        result_queue.put(
            {
                "worker": actor_id,
                "status": "PASS",
                "events": events,
                "sql_receipts": sql_receipts,
                "queue_receipts": queue_receipts,
                "action_results": action_results,
                "native_spans": native_spans,
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "worker": actor_id,
                "status": "FAIL",
                "reason": f"{type(error).__name__}:{error}",
                "events": events,
                "sql_receipts": sql_receipts,
                "queue_receipts": queue_receipts,
                "action_results": action_results,
                "native_spans": native_spans,
            }
        )
    finally:
        if connection is not None:
            connection.close()
