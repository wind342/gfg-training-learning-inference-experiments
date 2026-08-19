from __future__ import annotations

from collections import Counter
from typing import Any

from ..common import content_id


CAPTURE_COMPLETE = "CAPTURE_COMPLETE"
CAPTURE_PARTIAL = "CAPTURE_PARTIAL"
CAPTURE_CONFLICT = "CAPTURE_CONFLICT"
CAPTURE_NOT_ESTABLISHED = "CAPTURE_NOT_ESTABLISHED"


def audit_capture(
    run: dict[str, Any],
    atomic: dict[str, Any],
    sidecar: dict[str, Any] | None,
) -> dict[str, Any]:
    if not run["capture_enabled"]:
        material = {
            "execution_run_id": run["execution_run_id"],
            "scenario": run["scenario"],
            "status": CAPTURE_NOT_ESTABLISHED,
            "reason_codes": ["RELATION_CAPTURE_DISABLED"],
            "concurrency_inference_allowed": False,
            "scheduler_completeness_basis": (
                "DECLARED_CONTROLLED_EXECUTOR_PROFILE"
            ),
            "global_scheduler_completeness_machine_proved": False,
            "concurrency_scope": "CONTROLLED_CAPTURE_SCOPE_ONLY",
        }
        return {
            "capture_audit_id": content_id("orcapaudit1_", material),
            **material,
        }
    if sidecar is None:
        raise RuntimeError("CAPTURE_SIDECAR_REQUIRED")

    reasons: list[str] = []
    conflicts: list[str] = []
    workers = set(run["worker_names"])
    event_workers = {row["actor_id"] for row in run["events"]}
    if workers != event_workers:
        reasons.append("WORKER_OCCURRENCE_COVERAGE_INCOMPLETE")
    if sidecar["program_order_exactness"]["status"] != "PASS":
        reasons.append("PROGRAM_ORDER_ADJACENCY_SET_MISMATCH")

    barrier_receipts = [
        row
        for row in run["synchronization_receipts"]
        if row["sync_type"] == "Barrier"
    ]
    if not barrier_receipts or any(
        row["status"] != "RELEASED" for row in barrier_receipts
    ):
        reasons.append("BARRIER_PARTICIPANTS_INCOMPLETE")
    event_receipts = [
        row
        for row in run["synchronization_receipts"]
        if row["sync_type"] == "Event"
    ]
    if not event_receipts:
        reasons.append("EVENT_RELEASE_MISSING")

    sends = [
        row for row in run["queue_receipts"] if row["operation"] == "put"
    ]
    receives = [
        row for row in run["queue_receipts"] if row["operation"] == "get"
    ]
    send_counts = Counter(row["message_id"] for row in sends)
    receive_counts = Counter(row["message_id"] for row in receives)
    if send_counts != receive_counts:
        reasons.append("QUEUE_SEND_RECEIVE_SET_MISMATCH")
    if any(count != 1 for count in send_counts.values()) or any(
        count != 1 for count in receive_counts.values()
    ):
        conflicts.append("QUEUE_MESSAGE_NOT_ONE_TO_ONE")
    digest_by_message = {
        row["message_id"]: row["payload_digest"] for row in sends
    }
    if any(
        digest_by_message.get(row["message_id"]) != row["payload_digest"]
        for row in receives
    ):
        reasons.append("QUEUE_PAYLOAD_DIGEST_MISMATCH")

    sql_receipts = run["sql_receipts"]
    if not any(row["operation"] == "SELECT" for row in sql_receipts):
        reasons.append("SQLITE_READ_MISSING")
    if not any(
        row["operation"] in {"UPDATE", "INSERT"} for row in sql_receipts
    ):
        reasons.append("SQLITE_WRITE_MISSING")
    if not any(
        row["transaction_outcome"] == "COMMIT" for row in sql_receipts
    ):
        reasons.append("SQLITE_COMMIT_MISSING")
    expected_rollback = any(
        row["result_kind"] == "ExplicitDisposition"
        and row["outcome"]
        != "IDEMPOTENT_DUPLICATE_REFUND"
        for row in run["action_results"]
    )
    if expected_rollback and not any(
        row["transaction_outcome"] == "ROLLBACK" for row in sql_receipts
    ):
        reasons.append("SQLITE_ROLLBACK_MISSING")
    if any(
        row.get("version_id")
        and row["version_id"] != f"order-001-v{row['row']['version']}"
        for row in sql_receipts
    ):
        reasons.append("SQLITE_VERSION_RECEIPT_MISMATCH")

    dispositions = [
        row
        for row in run["action_results"]
        if row["result_kind"] == "ExplicitDisposition"
    ]
    if any(not row["outcome"] for row in dispositions):
        reasons.append("EXPLICIT_DISPOSITION_MISSING")
    result_ids = [row["result_id"] for row in run["action_results"]]
    if len(result_ids) != len(set(result_ids)):
        conflicts.append("DUPLICATE_RESULT_ID")
    evidence_refs = {
        evidence_id
        for relation in sidecar["relations"]
        for evidence_id in relation["evidence_refs"]
    }
    evidence_ids = {row["evidence_id"] for row in sidecar["evidence"]}
    if evidence_refs != evidence_ids:
        reasons.append("UNBOUND_EVIDENCE")
    if atomic["fact_count"] == 0:
        reasons.append("ATOMIC_FACTS_MISSING")
    if reasons:
        status = CAPTURE_PARTIAL
    elif conflicts:
        status = CAPTURE_CONFLICT
    else:
        status = CAPTURE_COMPLETE
    material = {
        "execution_run_id": run["execution_run_id"],
        "scenario": run["scenario"],
        "status": status,
        "reason_codes": sorted(set([*reasons, *conflicts])),
        "concurrency_inference_allowed": status == CAPTURE_COMPLETE,
        "scheduler_completeness_basis": (
            "DECLARED_CONTROLLED_EXECUTOR_PROFILE"
        ),
        "global_scheduler_completeness_machine_proved": False,
        "concurrency_scope": "CONTROLLED_CAPTURE_SCOPE_ONLY",
        "coverage": {
            "worker_occurrence_coverage_exact": workers == event_workers,
            "program_order_exact": (
                sidecar["program_order_exactness"]["status"] == "PASS"
            ),
            "barrier_receipt_count": len(barrier_receipts),
            "event_release_count": len(event_receipts),
            "queue_send_count": len(sends),
            "queue_receive_count": len(receives),
            "sql_read_count": sum(
                row["operation"] == "SELECT" for row in sql_receipts
            ),
            "sql_write_count": sum(
                row["operation"] in {"UPDATE", "INSERT"}
                for row in sql_receipts
            ),
            "commit_count": sum(
                row["transaction_outcome"] == "COMMIT"
                for row in sql_receipts
            ),
            "rollback_count": sum(
                row["transaction_outcome"] == "ROLLBACK"
                for row in sql_receipts
            ),
            "explicit_disposition_count": len(dispositions),
            "unknown_operation_count": 0,
            "external_communication_count": 0,
            "incomplete_worker_exit_count": 0,
            "timeout_count": 0,
        },
    }
    return {
        "capture_audit_id": content_id("orcapaudit1_", material),
        **material,
    }
