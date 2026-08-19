from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from ..common import load_json, write_json


FORBIDDEN_KEYS = {
    "validated_atomic_facts",
    "validated_relation_sidecar",
    "candidate_answers",
    "candidate_indexes",
}


def validate_reference_input(payload: dict[str, Any]) -> None:
    if FORBIDDEN_KEYS & set(payload):
        raise RuntimeError("REFERENCE_FORBIDDEN_INPUT")
    if set(payload) != {"runs", "queries", "schema_version"}:
        raise RuntimeError("REFERENCE_INPUT_SCHEMA_MISMATCH")


def _answer_run(
    run: dict[str, Any],
    queries: list[dict[str, str]],
    paired: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario = run["scenario"]
    actions = run["action_results"]
    refunds = [row for row in actions if row["action_type"] == "refund"]
    freezes = [row for row in actions if row["action_type"] == "freeze"]
    notifications = [
        row for row in actions if row["action_type"] == "notification"
    ]
    sql = run["sql_receipts"]
    queue = run["queue_receipts"]

    def row(
        query: dict[str, str],
        answer: Any,
        *,
        result_ids: list[str] | None = None,
        dispositions: list[str] | None = None,
        receipt_ids: list[str] | None = None,
        status: str = "ESTABLISHED",
    ) -> dict[str, Any]:
        return {
            "query_id": query["query_id"],
            "scenario": scenario,
            "exact_target": query["exact_target"],
            "answer": answer,
            "status": status,
            "evidence_path": sorted(set(receipt_ids or [])),
            "result_ids": sorted(set(result_ids or [])),
            "explicit_disposition_ids": sorted(set(dispositions or [])),
        }

    output = []
    for query in queries:
        query_id = query["query_id"]
        if query_id == "Q01":
            values = [
                {
                    "refund_result_id": refund["result_id"],
                    "version_id": (
                        f"order-001-v{refund['read_order']['version']}"
                        if refund.get("read_order")
                        else None
                    ),
                }
                for refund in refunds
            ]
            receipts = [
                item["receipt_id"]
                for item in sql
                if item["action_id"]
                in {refund["action_id"] for refund in refunds}
                and item["operation"] == "SELECT"
                and item["table"] == "orders"
            ]
            output.append(
                row(
                    query,
                    sorted(values, key=lambda item: item["refund_result_id"]),
                    result_ids=[item["result_id"] for item in refunds],
                    dispositions=[
                        item["result_id"]
                        for item in refunds
                        if item["result_kind"] == "ExplicitDisposition"
                    ],
                    receipt_ids=receipts,
                )
            )
        elif query_id == "Q02":
            if not freezes:
                output.append(
                    row(query, "NOT_APPLICABLE", status="NOT_APPLICABLE")
                )
            else:
                same_initial_read = (
                    refunds[0]["read_order"]["version"]
                    == freezes[0]["read_order"]["version"]
                    == 7
                )
                barrier = next(
                    item
                    for item in run["synchronization_receipts"]
                    if item["sync_type"] == "Barrier"
                )
                concurrent = same_initial_read and {
                    "RefundWorker",
                    "FreezeWorker",
                } <= set(barrier["participants"])
                output.append(
                    row(
                        query,
                        concurrent,
                        result_ids=[
                            refunds[0]["result_id"],
                            freezes[0]["result_id"],
                        ],
                        receipt_ids=[barrier["barrier_id"]],
                    )
                )
        elif query_id == "Q03":
            refund_update = next(
                (
                    item
                    for item in sql
                    if item["action_id"] == "refund-primary"
                    and item["operation"] == "UPDATE"
                ),
                None,
            )
            freeze_update = next(
                (
                    item
                    for item in sql
                    if item["action_id"] == "freeze-primary"
                    and item["operation"] == "UPDATE"
                ),
                None,
            )
            conflict = bool(
                refund_update
                and freeze_update
                and refund_update["resource_id"] == freeze_update["resource_id"]
                and refund_update["expected_version_id"]
                == freeze_update["expected_version_id"]
            )
            output.append(
                row(
                    query,
                    conflict,
                    result_ids=[
                        item["result_id"] for item in refunds + freezes
                    ],
                    receipt_ids=[
                        item["receipt_id"]
                        for item in (refund_update, freeze_update)
                        if item
                    ],
                )
            )
        elif query_id == "Q04":
            committed = [
                item
                for item in actions
                if item["outcome"] in {"RefundCommitted", "OrderFrozen"}
                and item["transaction_outcome"] == "COMMIT"
            ]
            receipts = [
                item["receipt_id"]
                for item in sql
                if item["action_id"]
                in {action["action_id"] for action in committed}
                and item["operation"] == "UPDATE"
                and item["rowcount"] == 1
            ]
            output.append(
                row(
                    query,
                    sorted(item["result_id"] for item in committed),
                    result_ids=[item["result_id"] for item in committed],
                    receipt_ids=receipts,
                )
            )
        elif query_id == "Q05":
            answer = [
                {
                    "result_id": item["result_id"],
                    "reason": (
                        item["outcome"]
                        if item["result_kind"] == "ExplicitDisposition"
                        else "COMMITTED"
                    ),
                }
                for item in refunds
            ]
            output.append(
                row(
                    query,
                    sorted(answer, key=lambda item: item["result_id"]),
                    result_ids=[item["result_id"] for item in refunds],
                    dispositions=[
                        item["result_id"]
                        for item in refunds
                        if item["result_kind"] == "ExplicitDisposition"
                    ],
                    receipt_ids=[
                        item["receipt_id"]
                        for item in sql
                        if item["action_id"]
                        in {refund["action_id"] for refund in refunds}
                    ],
                )
            )
        elif query_id == "Q06":
            if not freezes:
                output.append(
                    row(query, "NOT_APPLICABLE", status="NOT_APPLICABLE")
                )
            else:
                freeze = freezes[0]
                output.append(
                    row(
                        query,
                        (
                            freeze["outcome"]
                            if freeze["result_kind"] == "ExplicitDisposition"
                            else "COMMITTED"
                        ),
                        result_ids=[freeze["result_id"]],
                        dispositions=(
                            [freeze["result_id"]]
                            if freeze["result_kind"] == "ExplicitDisposition"
                            else []
                        ),
                    )
                )
        elif query_id == "Q07":
            sent = [
                item
                for item in notifications
                if item["outcome"] == "NotificationSent"
            ]
            if not sent:
                output.append(
                    row(
                        query,
                        "NOT_APPLICABLE_NO_NOTIFICATION_SENT",
                        status="NOT_APPLICABLE",
                    )
                )
            else:
                source = next(
                    item
                    for item in actions
                    if item["result_id"] == sent[0]["source_result_id"]
                )
                output.append(
                    row(
                        query,
                        source["outcome"],
                        result_ids=[source["result_id"], sent[0]["result_id"]],
                        receipt_ids=[
                            item["receipt_id"]
                            for item in queue
                            if item["message_id"]
                            == sent[0]["source_message_id"]
                        ],
                    )
                )
        elif query_id == "Q08":
            suppressed = [
                item
                for item in notifications
                if item["outcome"]
                == "NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND"
            ]
            causes = [
                next(
                    item
                    for item in actions
                    if item["result_id"] == notification["source_result_id"]
                )
                for notification in suppressed
            ]
            output.append(
                row(
                    query,
                    sorted(item["outcome"] for item in causes),
                    result_ids=[
                        item["result_id"] for item in causes + suppressed
                    ],
                    dispositions=[
                        item["result_id"] for item in causes + suppressed
                    ],
                    receipt_ids=[
                        receipt["receipt_id"]
                        for receipt in queue
                        if receipt["message_id"]
                        in {
                            item["source_message_id"] for item in suppressed
                        }
                    ],
                )
            )
        elif query_id in {"Q09", "Q11"}:
            commits = [
                item for item in refunds if item["outcome"] == "RefundCommitted"
            ]
            downstream = []
            for commit in commits:
                linked = [
                    item
                    for item in notifications
                    if item["source_result_id"] == commit["result_id"]
                ]
                for notification in linked:
                    if query_id == "Q09":
                        downstream.append(
                            f"{run['execution_run_id']}:"
                            f"{notification['source_message_id']}"
                        )
                    if notification["outcome"] == "NotificationSent":
                        downstream.append(notification["result_id"])
            output.append(
                row(
                    query,
                    sorted(set(downstream)),
                    result_ids=[item["result_id"] for item in commits]
                    + downstream,
                )
            )
        elif query_id == "Q10":
            action_ids = {
                item["action_id"]
                for item in sql
                if item.get("version_id") == "order-001-v7"
            }
            result_ids = [
                item["result_id"]
                for item in actions
                if item["action_id"] in action_ids
            ]
            output.append(
                row(
                    query,
                    sorted(set(result_ids)),
                    result_ids=result_ids,
                    receipt_ids=[
                        item["receipt_id"]
                        for item in sql
                        if item.get("version_id") == "order-001-v7"
                        or item.get("expected_version_id") == "order-001-v7"
                    ],
                )
            )
        elif query_id == "Q12":
            output.append(
                row(
                    query,
                    paired,
                    result_ids=paired["result_ids"],
                    dispositions=paired["disposition_ids"],
                )
            )
        elif query_id == "Q13":
            update_receipts = [
                item
                for item in sql
                if item["operation"] == "UPDATE"
                and item.get("expected_version_id") == "order-001-v7"
            ]
            conflict = len(update_receipts) == 2
            affected: set[str] = set()
            if conflict:
                loser = next(
                    item
                    for item in refunds + freezes
                    if item["result_kind"] == "ExplicitDisposition"
                )
                affected.add(loser["result_id"])
                affected.update(
                    item["result_id"]
                    for item in notifications
                    if item["source_result_id"] == loser["result_id"]
                )
            all_ids = {item["result_id"] for item in actions}
            output.append(
                row(
                    query,
                    {
                        "affected_result_ids": sorted(affected),
                        "unaffected_result_ids": sorted(all_ids - affected),
                    },
                    result_ids=list(all_ids),
                    dispositions=[
                        item_id
                        for item_id in affected
                        if next(
                            item for item in actions if item["result_id"] == item_id
                        )["result_kind"]
                        == "ExplicitDisposition"
                    ],
                    receipt_ids=[
                        item["receipt_id"] for item in update_receipts
                    ],
                )
            )
        elif query_id == "Q14":
            if scenario != "IDEMPOTENT_DUPLICATE_REFUND":
                output.append(
                    row(query, "NOT_APPLICABLE", status="NOT_APPLICABLE")
                )
            else:
                output.append(
                    row(
                        query,
                        {
                            "refund_row_count": len(
                                run["canonical_db_dump"]["refunds"]
                            ),
                            "notification_sent_count": len(
                                run["canonical_db_dump"]["notifications"]
                            ),
                            "second_refund_formed": (
                                len(run["canonical_db_dump"]["refunds"]) > 1
                            ),
                            "second_notification_formed": (
                                len(run["canonical_db_dump"]["notifications"])
                                > 1
                            ),
                        },
                        result_ids=[
                            item["result_id"]
                            for item in refunds + notifications
                        ],
                        dispositions=[
                            item["result_id"]
                            for item in refunds
                            if item["result_kind"] == "ExplicitDisposition"
                        ],
                    )
                )
    return output


def resolve_reference(payload: dict[str, Any]) -> dict[str, Any]:
    validate_reference_input(payload)
    runs = payload["runs"]
    by_scenario = {run["scenario"]: run for run in runs}
    b = by_scenario["CONCURRENT_FREEZE_WINS"]
    c = by_scenario["LATE_REFUND_AFTER_FREEZE"]
    b_refund = next(
        item for item in b["action_results"] if item["action_id"] == "refund-primary"
    )
    c_refund = next(
        item for item in c["action_results"] if item["action_id"] == "refund-primary"
    )
    paired = {
        "ordinary_business_view_equal": (
            b["ordinary_business_view"] == c["ordinary_business_view"]
        ),
        "formation_answer_equal": b_refund["outcome"] == c_refund["outcome"],
        "scenario_b_refund_reason": b_refund["outcome"],
        "scenario_c_refund_reason": c_refund["outcome"],
        "result_ids": [b_refund["result_id"], c_refund["result_id"]],
        "disposition_ids": [b_refund["result_id"], c_refund["result_id"]],
    }
    answers = [
        answer
        for run in runs
        for answer in _answer_run(run, payload["queries"], paired)
    ]
    return {
        "status": "PASS",
        "answers": sorted(
            answers, key=lambda item: (item["scenario"], item["query_id"])
        ),
        "answer_count": len(answers),
        "schema_version": "reference-answers-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = resolve_reference(load_json(args.input))
    except Exception as error:
        output = {
            "status": "FAIL",
            "reason_code": str(error),
            "schema_version": "reference-answers-v1",
        }
    write_json(args.output, output)
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
