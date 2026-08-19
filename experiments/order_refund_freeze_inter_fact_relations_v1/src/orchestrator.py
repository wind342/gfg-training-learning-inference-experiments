from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path
from typing import Any

from ..common import (
    SCENARIOS,
    TIMEOUT_SECONDS,
    ExperimentError,
    canonical_sha256,
)
from .sqlite_runtime import (
    canonical_dump,
    initialize_database,
    sqlite_binary_identity,
)
from .workers import freeze_worker, notification_worker, refund_worker


def _ids(scenario: str, repeat_index: int) -> dict[str, str]:
    slug = scenario.lower().replace("_", "-")
    pair_id = f"{slug}-{repeat_index:02d}"
    return {
        "pair_id": pair_id,
        "order_id": "order-001",
        "refund_id": f"refund-{pair_id}",
        "idempotency_key": f"idempotency-{pair_id}",
        "notification_id": f"notification-{pair_id}",
    }


def _config(
    *,
    db_path: Path,
    execution_run_id: str,
    actor_id: str,
    action_id: str,
    ids: dict[str, str],
    barrier_id: str,
    commit_gate_id: str,
) -> dict[str, Any]:
    return {
        "db_path": str(db_path),
        "execution_run_id": execution_run_id,
        "actor_id": actor_id,
        "action_id": action_id,
        "order_id": ids["order_id"],
        "refund_id": ids["refund_id"],
        "amount_cents": 5000,
        "idempotency_key": ids["idempotency_key"],
        "notification_id": ids["notification_id"],
        "barrier_id": barrier_id,
        "commit_gate_id": commit_gate_id,
        "read_start_event_id": f"event-{execution_run_id}-refund-read-start",
    }


def _join(processes: list[multiprocessing.Process]) -> None:
    for process in processes:
        process.join(TIMEOUT_SECONDS)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
            raise ExperimentError("WORKER_TIMEOUT")
        if process.exitcode != 0:
            raise ExperimentError(f"WORKER_EXIT_NONZERO:{process.name}")


def run_workflow(
    scenario: str,
    *,
    repeat_index: int,
    capture_enabled: bool,
) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ExperimentError("SCENARIO_UNKNOWN")
    mode = "capture-enabled" if capture_enabled else "capture-disabled"
    execution_run_id = (
        f"run-{scenario.lower()}-{repeat_index:02d}-{mode}"
    )
    ids = _ids(scenario, repeat_index)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    notification_queue = context.Queue()
    sync_receipts: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="order-refund-freeze-") as temp:
        db_path = Path(temp) / "workflow.sqlite3"
        keeper = initialize_database(db_path)
        barrier_id = f"barrier-{execution_run_id}-reads"
        refund_gate_id = f"event-{execution_run_id}-refund-commit"
        freeze_gate_id = f"event-{execution_run_id}-freeze-commit"
        refund_gate = context.Event()
        freeze_gate = context.Event()
        refund_done = context.Event()
        freeze_done = context.Event()
        refund_read_start = None
        processes: list[multiprocessing.Process] = []

        notification_config = {
            "db_path": str(db_path),
            "execution_run_id": execution_run_id,
            "actor_id": "NotificationWorker",
            "notification_id": ids["notification_id"],
            "expected_message_count": (
                2 if scenario == "IDEMPOTENT_DUPLICATE_REFUND" else 1
            ),
        }
        processes.append(
            context.Process(
                name="NotificationWorker",
                target=notification_worker,
                args=(notification_config, notification_queue, result_queue),
            )
        )

        refund_config = _config(
            db_path=db_path,
            execution_run_id=execution_run_id,
            actor_id="RefundWorker",
            action_id="refund-primary",
            ids=ids,
            barrier_id=barrier_id,
            commit_gate_id=refund_gate_id,
        )

        if scenario in {
            "CONCURRENT_REFUND_WINS",
            "CONCURRENT_FREEZE_WINS",
        }:
            read_barrier = context.Barrier(3)
            processes.extend(
                [
                    context.Process(
                        name="RefundWorker",
                        target=refund_worker,
                        args=(
                            refund_config,
                            read_barrier,
                            None,
                            refund_gate,
                            refund_done,
                            notification_queue,
                            result_queue,
                        ),
                    ),
                    context.Process(
                        name="FreezeWorker",
                        target=freeze_worker,
                        args=(
                            _config(
                                db_path=db_path,
                                execution_run_id=execution_run_id,
                                actor_id="FreezeWorker",
                                action_id="freeze-primary",
                                ids=ids,
                                barrier_id=barrier_id,
                                commit_gate_id=freeze_gate_id,
                            ),
                            read_barrier,
                            freeze_gate,
                            freeze_done,
                            result_queue,
                        ),
                    ),
                ]
            )
            for process in processes:
                process.start()
            read_barrier.wait(TIMEOUT_SECONDS)
            sync_receipts.append(
                {
                    "sync_type": "Barrier",
                    "barrier_id": barrier_id,
                    "generation": 0,
                    "participants": [
                        "Orchestrator",
                        "RefundWorker",
                        "FreezeWorker",
                    ],
                    "status": "RELEASED",
                }
            )
            if scenario == "CONCURRENT_REFUND_WINS":
                refund_gate.set()
                sync_receipts.append(
                    {
                        "sync_type": "Event",
                        "event_id": refund_gate_id,
                        "released_worker": "RefundWorker",
                    }
                )
                if not refund_done.wait(TIMEOUT_SECONDS):
                    raise ExperimentError("REFUND_DONE_TIMEOUT")
                freeze_gate.set()
                sync_receipts.append(
                    {
                        "sync_type": "Event",
                        "event_id": freeze_gate_id,
                        "released_worker": "FreezeWorker",
                    }
                )
            else:
                freeze_gate.set()
                sync_receipts.append(
                    {
                        "sync_type": "Event",
                        "event_id": freeze_gate_id,
                        "released_worker": "FreezeWorker",
                    }
                )
                if not freeze_done.wait(TIMEOUT_SECONDS):
                    raise ExperimentError("FREEZE_DONE_TIMEOUT")
                refund_gate.set()
                sync_receipts.append(
                    {
                        "sync_type": "Event",
                        "event_id": refund_gate_id,
                        "released_worker": "RefundWorker",
                    }
                )
        elif scenario == "LATE_REFUND_AFTER_FREEZE":
            read_barrier = context.Barrier(2)
            refund_read_start = context.Event()
            processes.extend(
                [
                    context.Process(
                        name="RefundWorker",
                        target=refund_worker,
                        args=(
                            refund_config,
                            None,
                            refund_read_start,
                            refund_gate,
                            refund_done,
                            notification_queue,
                            result_queue,
                        ),
                    ),
                    context.Process(
                        name="FreezeWorker",
                        target=freeze_worker,
                        args=(
                            _config(
                                db_path=db_path,
                                execution_run_id=execution_run_id,
                                actor_id="FreezeWorker",
                                action_id="freeze-primary",
                                ids=ids,
                                barrier_id=barrier_id,
                                commit_gate_id=freeze_gate_id,
                            ),
                            read_barrier,
                            freeze_gate,
                            freeze_done,
                            result_queue,
                        ),
                    ),
                ]
            )
            for process in processes:
                process.start()
            read_barrier.wait(TIMEOUT_SECONDS)
            sync_receipts.append(
                {
                    "sync_type": "Barrier",
                    "barrier_id": barrier_id,
                    "generation": 0,
                    "participants": ["Orchestrator", "FreezeWorker"],
                    "status": "RELEASED",
                }
            )
            freeze_gate.set()
            sync_receipts.append(
                {
                    "sync_type": "Event",
                    "event_id": freeze_gate_id,
                    "released_worker": "FreezeWorker",
                }
            )
            if not freeze_done.wait(TIMEOUT_SECONDS):
                raise ExperimentError("FREEZE_DONE_TIMEOUT")
            refund_gate.set()
            refund_read_start.set()
            sync_receipts.extend(
                [
                    {
                        "sync_type": "Event",
                        "event_id": refund_config["read_start_event_id"],
                        "released_worker": "RefundWorker",
                    },
                    {
                        "sync_type": "Event",
                        "event_id": refund_gate_id,
                        "released_worker": "RefundWorker",
                    },
                ]
            )
        else:
            read_barrier = context.Barrier(2)
            duplicate_start = context.Event()
            duplicate_gate = context.Event()
            duplicate_gate.set()
            duplicate_done = context.Event()
            duplicate_config = _config(
                db_path=db_path,
                execution_run_id=execution_run_id,
                actor_id="RefundWorkerDuplicate",
                action_id="refund-duplicate",
                ids=ids,
                barrier_id=barrier_id,
                commit_gate_id=f"event-{execution_run_id}-duplicate",
            )
            duplicate_config["duplicate_check_only"] = True
            processes.extend(
                [
                    context.Process(
                        name="RefundWorker",
                        target=refund_worker,
                        args=(
                            refund_config,
                            read_barrier,
                            None,
                            refund_gate,
                            refund_done,
                            notification_queue,
                            result_queue,
                        ),
                    ),
                    context.Process(
                        name="RefundWorkerDuplicate",
                        target=refund_worker,
                        args=(
                            duplicate_config,
                            None,
                            duplicate_start,
                            duplicate_gate,
                            duplicate_done,
                            notification_queue,
                            result_queue,
                        ),
                    ),
                ]
            )
            for process in processes:
                process.start()
            read_barrier.wait(TIMEOUT_SECONDS)
            sync_receipts.append(
                {
                    "sync_type": "Barrier",
                    "barrier_id": barrier_id,
                    "generation": 0,
                    "participants": ["Orchestrator", "RefundWorker"],
                    "status": "RELEASED",
                }
            )
            refund_gate.set()
            sync_receipts.append(
                {
                    "sync_type": "Event",
                    "event_id": refund_gate_id,
                    "released_worker": "RefundWorker",
                }
            )
            if not refund_done.wait(TIMEOUT_SECONDS):
                raise ExperimentError("REFUND_DONE_TIMEOUT")
            duplicate_start.set()
            sync_receipts.append(
                {
                    "sync_type": "Event",
                    "event_id": duplicate_config["read_start_event_id"],
                    "released_worker": "RefundWorkerDuplicate",
                }
            )

        _join(processes)
        worker_results = [
            result_queue.get(timeout=TIMEOUT_SECONDS)
            for _ in range(len(processes))
        ]
        worker_results.sort(key=lambda row: row["worker"])
        failures = [row for row in worker_results if row["status"] != "PASS"]
        if failures:
            raise ExperimentError(
                "WORKER_FAILURE:" + ";".join(row["reason"] for row in failures)
            )
        db_dump = canonical_dump(db_path)
        binary_identity = sqlite_binary_identity(db_path)
        keeper.close()

    events = sorted(
        (event for row in worker_results for event in row["events"]),
        key=lambda row: (row["actor_id"], row["sequence_index"]),
    )
    sql_receipts = sorted(
        (receipt for row in worker_results for receipt in row["sql_receipts"]),
        key=lambda row: row["receipt_id"],
    )
    queue_receipts = sorted(
        (
            receipt
            for row in worker_results
            for receipt in row["queue_receipts"]
        ),
        key=lambda row: row["receipt_id"],
    )
    action_results = sorted(
        (
            result
            for row in worker_results
            for result in row["action_results"]
        ),
        key=lambda row: row["action_id"],
    )
    native_spans = sorted(
        (span for row in worker_results for span in row["native_spans"]),
        key=lambda row: (row["name"], row["span_id"]),
    )
    ordinary_business_view = {
        "orders": db_dump["orders"],
        "refunds": db_dump["refunds"],
        "notifications": db_dump["notifications"],
    }
    business_output = {
        "scenario": scenario,
        "ordinary_business_view": ordinary_business_view,
        "action_results": [
            {
                key: row.get(key)
                for key in (
                    "action_id",
                    "action_type",
                    "outcome",
                    "result_kind",
                    "refund_id",
                    "notification_id",
                )
                if row.get(key) is not None
            }
            for row in action_results
        ],
    }
    return {
        "status": "PASS",
        "scenario": scenario,
        "repeat_index": repeat_index,
        "capture_enabled": capture_enabled,
        "execution_run_id": execution_run_id,
        "deterministic_business_ids": ids,
        "journal_mode": "wal",
        "process_count": len(processes) + 1,
        "worker_names": sorted(row["worker"] for row in worker_results),
        "events": events,
        "sql_receipts": sql_receipts,
        "queue_receipts": queue_receipts,
        "synchronization_receipts": sync_receipts,
        "action_results": action_results,
        "native_spans": native_spans,
        "canonical_db_dump": db_dump,
        "ordinary_business_view": ordinary_business_view,
        "business_output": business_output,
        "business_output_sha256": canonical_sha256(business_output),
        "canonical_db_dump_sha256": canonical_sha256(db_dump),
        "transaction_receipts_sha256": canonical_sha256(sql_receipts),
        "result_sha256": canonical_sha256(action_results),
        "sqlite_binary_identity": binary_identity,
        "diagnostics": {
            "temporary_path_excluded": True,
            "process_ids_excluded": True,
            "wall_clock_excluded": True,
        },
    }
