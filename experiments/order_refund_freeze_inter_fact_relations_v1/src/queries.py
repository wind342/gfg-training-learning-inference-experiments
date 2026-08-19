from __future__ import annotations


QUERY_TARGETS = {
    "Q01": "actual order version read by each refund result or disposition",
    "Q02": "RefundWorker and FreezeWorker concurrency",
    "Q03": "RefundWorker and FreezeWorker conflict",
    "Q04": "action that committed order version 8",
    "Q05": "exact refund non-commit reason",
    "Q06": "exact freeze non-commit reason",
    "Q07": "NotificationSent source stage",
    "Q08": "ExplicitDisposition causing NotificationSuppressed",
    "Q09": "concrete downstream results of RefundCommitted",
    "Q10": "results reading or competing for order version 7",
    "Q11": "directly affected compensation targets for RefundCommitted",
    "Q12": "Scenario B/C equal ordinary state and different formation",
    "Q13": "conflict-affected and unaffected concrete results",
    "Q14": "second refund and second notification under idempotent retry",
}


def frozen_queries() -> list[dict[str, str]]:
    return [
        {"query_id": query_id, "exact_target": target}
        for query_id, target in QUERY_TARGETS.items()
    ]
