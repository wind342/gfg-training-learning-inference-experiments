from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from ..common import load_json, write_json


def resolve_trace(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"native_trace_export", "queries", "schema_version"}:
        raise RuntimeError("TRACE_INPUT_SCHEMA_MISMATCH")
    exports = payload["native_trace_export"]
    answers = []
    exactly_answerable = {"Q04", "Q05", "Q06", "Q14"}
    for export in exports:
        scenario = export["scenario"]
        spans = export["spans"]
        statuses = [
            span["attributes"]["action.status"] for span in spans
        ]
        types = [span["attributes"]["action.type"] for span in spans]
        for query in payload["queries"]:
            query_id = query["query_id"]
            if query_id not in exactly_answerable:
                answer = "NOT_ESTABLISHED"
                status = "NOT_ESTABLISHED"
            elif query_id == "Q04":
                answer = sorted(
                    span["attributes"]["action.type"]
                    for span in spans
                    if span["attributes"]["action.status"]
                    in {"RefundCommitted", "OrderFrozen"}
                )
                status = "ESTABLISHED_AT_ACTION_TYPE_LEVEL"
            elif query_id == "Q05":
                answer = sorted(
                    value
                    for value, action_type in zip(statuses, types)
                    if action_type == "refund"
                )
                status = "ESTABLISHED_AT_SPAN_STATUS_LEVEL"
            elif query_id == "Q06":
                freeze_values = [
                    value
                    for value, action_type in zip(statuses, types)
                    if action_type == "freeze"
                ]
                answer = freeze_values or "NOT_APPLICABLE"
                status = (
                    "ESTABLISHED_AT_SPAN_STATUS_LEVEL"
                    if freeze_values
                    else "NOT_APPLICABLE"
                )
            else:
                counts = Counter(statuses)
                answer = {
                    "refund_committed_span_count": counts["RefundCommitted"],
                    "notification_sent_span_count": counts["NotificationSent"],
                }
                status = "ESTABLISHED_AT_SPAN_COUNT_LEVEL"
            answers.append(
                {
                    "query_id": query_id,
                    "scenario": scenario,
                    "exact_target": query["exact_target"],
                    "answer": answer,
                    "status": status,
                    "evidence_path": [
                        span["span_id"] for span in spans
                    ]
                    if status != "NOT_ESTABLISHED"
                    else [],
                }
            )
    return {
        "status": "PASS",
        "answers": sorted(
            answers, key=lambda item: (item["scenario"], item["query_id"])
        ),
        "schema_version": "native-trace-answerability-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = resolve_trace(load_json(args.input))
    except Exception as error:
        output = {
            "status": "FAIL",
            "reason_code": str(error),
            "schema_version": "native-trace-answerability-v1",
        }
    write_json(args.output, output)
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
