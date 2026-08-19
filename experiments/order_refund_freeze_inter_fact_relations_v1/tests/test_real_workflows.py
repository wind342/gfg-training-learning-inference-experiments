from __future__ import annotations

from experiments.order_refund_freeze_inter_fact_relations_v1.src.orchestrator import (
    run_workflow,
)


def _outcomes(result):
    return {row["outcome"] for row in result["action_results"]}


def test_concurrent_refund_wins_real_workflow() -> None:
    result = run_workflow(
        "CONCURRENT_REFUND_WINS", repeat_index=1, capture_enabled=True
    )
    assert result["journal_mode"] == "wal"
    assert result["process_count"] >= 4
    assert result["canonical_db_dump"]["orders"][0]["status"] == "REFUNDED"
    assert result["canonical_db_dump"]["orders"][0]["version"] == 8
    assert {
        "RefundCommitted",
        "FREEZE_VERSION_CONFLICT_AFTER_REFUND",
        "NotificationSent",
    } <= _outcomes(result)


def test_concurrent_freeze_wins_real_workflow() -> None:
    result = run_workflow(
        "CONCURRENT_FREEZE_WINS", repeat_index=1, capture_enabled=True
    )
    assert result["canonical_db_dump"]["orders"][0]["status"] == "FROZEN"
    assert {
        "OrderFrozen",
        "REFUND_VERSION_CONFLICT_AFTER_FREEZE",
        "NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND",
    } <= _outcomes(result)


def test_late_refund_reads_frozen_version() -> None:
    result = run_workflow(
        "LATE_REFUND_AFTER_FREEZE", repeat_index=1, capture_enabled=True
    )
    refund = next(
        row for row in result["action_results"] if row["action_type"] == "refund"
    )
    assert refund["read_order"]["status"] == "FROZEN"
    assert refund["read_order"]["version"] == 8
    assert refund["conditional_rowcount"] is None
    assert refund["outcome"] == "REFUND_REJECTED_ORDER_ALREADY_FROZEN"


def test_idempotent_duplicate_real_workflow() -> None:
    result = run_workflow(
        "IDEMPOTENT_DUPLICATE_REFUND",
        repeat_index=1,
        capture_enabled=True,
    )
    assert len(result["canonical_db_dump"]["refunds"]) == 1
    assert len(result["canonical_db_dump"]["notifications"]) == 1
    assert "IDEMPOTENT_DUPLICATE_REFUND" in _outcomes(result)


def test_capture_mode_is_business_output_orthogonal() -> None:
    disabled = run_workflow(
        "CONCURRENT_REFUND_WINS", repeat_index=2, capture_enabled=False
    )
    enabled = run_workflow(
        "CONCURRENT_REFUND_WINS", repeat_index=2, capture_enabled=True
    )
    assert disabled["business_output"] == enabled["business_output"]
    assert disabled["business_output_sha256"] == enabled["business_output_sha256"]
