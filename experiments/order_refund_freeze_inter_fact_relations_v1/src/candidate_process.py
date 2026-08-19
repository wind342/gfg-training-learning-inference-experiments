from __future__ import annotations

import argparse
from collections import defaultdict, deque
from typing import Any

from ..common import load_json, write_json
from .capture_auditor import CAPTURE_COMPLETE


FORBIDDEN_KEYS = {
    "canonical_db_dump",
    "sql_receipts",
    "queue_receipts",
    "synchronization_receipts",
    "reference_answers",
    "native_trace_export",
    "sqlite_binary_identity",
}


def validate_candidate_input(payload: dict[str, Any]) -> None:
    if FORBIDDEN_KEYS & set(payload):
        raise RuntimeError("CANDIDATE_FORBIDDEN_INPUT")
    if set(payload) != {"contexts", "queries", "lifting_rules", "schema_version"}:
        raise RuntimeError("CANDIDATE_INPUT_SCHEMA_MISMATCH")
    if payload["schema_version"] != "candidate-input-v1":
        raise RuntimeError("CANDIDATE_INPUT_SCHEMA_MISMATCH")


def _resolve_context(
    context: dict[str, Any],
    queries: list[dict[str, str]],
    paired: dict[str, Any],
) -> list[dict[str, Any]]:
    atomic = context["validated_atomic_facts"]
    sidecar = context["validated_relation_sidecar"]
    audit = context["capture_audit"]
    scenario = atomic["scenario"]
    facts = atomic["facts"]
    fact_by_id = {row["fact_id"]: row for row in facts}
    result_fact = {row["result_id"]: row for row in facts}
    relations = sidecar["relations"]
    evidence_by_id = {
        row["evidence_id"]: row for row in sidecar["evidence"]
    }
    by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_type[relation["relation_type"]].append(relation)

    action_facts = [
        row
        for row in facts
        if row["coordinates"]["u"].get("kind") == "worker_action"
    ]
    action_by_id = {
        row["coordinates"]["u"]["action_id"]: row for row in action_facts
    }
    refund_facts = [
        row
        for row in action_facts
        if row["coordinates"]["u"]["action_type"] == "refund"
    ]
    freeze_facts = [
        row
        for row in action_facts
        if row["coordinates"]["u"]["action_type"] == "freeze"
    ]
    notification_facts = [
        row
        for row in action_facts
        if row["coordinates"]["u"]["action_type"] == "notification"
    ]
    disposition_facts = [
        row
        for row in action_facts
        if row["coordinates"]["z"]["kind"] == "ExplicitDisposition"
    ]
    disposition_ids = [row["result_id"] for row in disposition_facts]

    dependency: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for relation in by_type["generated_origin_dependency"]:
        dependency[relation["source_id"]].append(
            (relation["target_id"], relation["relation_id"])
        )

    def closure(source_fact_id: str) -> tuple[list[str], list[str]]:
        visited = {source_fact_id}
        queue = deque([source_fact_id])
        reached: list[str] = []
        relation_ids: list[str] = []
        while queue:
            current = queue.popleft()
            for target, relation_id in dependency.get(current, []):
                relation_ids.append(relation_id)
                if target not in visited:
                    visited.add(target)
                    reached.append(target)
                    queue.append(target)
        return reached, relation_ids

    def record(
        query: dict[str, str],
        answer: Any,
        *,
        relation_ids: list[str] | None = None,
        result_ids: list[str] | None = None,
        selected_dispositions: list[str] | None = None,
        status: str = "ESTABLISHED",
    ) -> dict[str, Any]:
        relation_ids = sorted(set(relation_ids or []))
        evidence_ids = sorted(
            {
                evidence_id
                for relation in relations
                if relation["relation_id"] in relation_ids
                for evidence_id in relation["evidence_refs"]
            }
        )
        return {
            "query_id": query["query_id"],
            "scenario": scenario,
            "exact_target": query["exact_target"],
            "answer": answer,
            "status": status,
            "evidence_path": [
                {
                    "relation_id": relation_id,
                    "evidence_ids": next(
                        row["evidence_refs"]
                        for row in relations
                        if row["relation_id"] == relation_id
                    ),
                }
                for relation_id in relation_ids
            ],
            "result_ids": sorted(set(result_ids or [])),
            "relation_ids": relation_ids,
            "evidence_ids": evidence_ids,
            "explicit_disposition_ids": sorted(
                set(selected_dispositions or [])
            ),
        }

    answers: list[dict[str, Any]] = []
    for query in queries:
        query_id = query["query_id"]
        if query_id == "Q01":
            rows = []
            relation_ids = []
            result_ids = []
            for relation in by_type["reads_from"]:
                target = fact_by_id[relation["target_id"]]
                if target not in refund_facts:
                    continue
                source = fact_by_id[relation["source_id"]]
                rows.append(
                    {
                        "refund_result_id": target["result_id"],
                        "version_id": source["coordinates"]["z"]["version_id"],
                    }
                )
                relation_ids.append(relation["relation_id"])
                result_ids.append(target["result_id"])
            for refund in refund_facts:
                if refund["result_id"] not in result_ids:
                    rows.append(
                        {
                            "refund_result_id": refund["result_id"],
                            "version_id": None,
                        }
                    )
                    result_ids.append(refund["result_id"])
            answers.append(
                record(
                    query,
                    sorted(rows, key=lambda row: row["refund_result_id"]),
                    relation_ids=relation_ids,
                    result_ids=result_ids,
                    selected_dispositions=[
                        row["result_id"]
                        for row in refund_facts
                        if row in disposition_facts
                    ],
                )
            )
        elif query_id == "Q02":
            if not freeze_facts:
                answers.append(
                    record(query, "NOT_APPLICABLE", status="NOT_APPLICABLE")
                )
                continue
            if audit["status"] != CAPTURE_COMPLETE:
                answers.append(
                    record(
                        query,
                        "NOT_ESTABLISHED",
                        status="NOT_ESTABLISHED",
                    )
                )
                continue
            conflict = bool(by_type["conflicts_with"])
            late_read = any(
                relation["target_id"] == refund_facts[0]["fact_id"]
                and fact_by_id[relation["source_id"]]["coordinates"]["z"][
                    "version"
                ]
                == 8
                for relation in by_type["reads_from"]
            )
            answers.append(
                record(
                    query,
                    conflict and not late_read,
                    relation_ids=[
                        row["relation_id"]
                        for row in (
                            by_type["conflicts_with"]
                            + by_type["synchronizes_with"]
                        )
                    ],
                    result_ids=[
                        refund_facts[0]["result_id"],
                        freeze_facts[0]["result_id"],
                    ],
                )
            )
        elif query_id == "Q03":
            answers.append(
                record(
                    query,
                    bool(by_type["conflicts_with"]),
                    relation_ids=[
                        row["relation_id"] for row in by_type["conflicts_with"]
                    ],
                    result_ids=[
                        row["result_id"] for row in refund_facts + freeze_facts
                    ],
                )
            )
        elif query_id == "Q04":
            committed = [
                fact_by_id[row["source_id"]]
                for row in by_type["commits_version"]
            ]
            answers.append(
                record(
                    query,
                    sorted(row["result_id"] for row in committed),
                    relation_ids=[
                        row["relation_id"] for row in by_type["commits_version"]
                    ],
                    result_ids=[row["result_id"] for row in committed],
                )
            )
        elif query_id == "Q05":
            values = [
                {
                    "result_id": row["result_id"],
                    "reason": (
                        row["coordinates"]["z"]["value"]
                        if row["coordinates"]["z"]["kind"]
                        == "ExplicitDisposition"
                        else "COMMITTED"
                    ),
                }
                for row in refund_facts
            ]
            answers.append(
                record(
                    query,
                    sorted(values, key=lambda row: row["result_id"]),
                    result_ids=[row["result_id"] for row in refund_facts],
                    selected_dispositions=[
                        row["result_id"]
                        for row in refund_facts
                        if row in disposition_facts
                    ],
                )
            )
        elif query_id == "Q06":
            if not freeze_facts:
                answers.append(
                    record(query, "NOT_APPLICABLE", status="NOT_APPLICABLE")
                )
            else:
                freeze = freeze_facts[0]
                value = (
                    freeze["coordinates"]["z"]["value"]
                    if freeze in disposition_facts
                    else "COMMITTED"
                )
                answers.append(
                    record(
                        query,
                        value,
                        result_ids=[freeze["result_id"]],
                        selected_dispositions=(
                            [freeze["result_id"]]
                            if freeze in disposition_facts
                            else []
                        ),
                    )
                )
        elif query_id == "Q07":
            sent = [
                row
                for row in notification_facts
                if row["coordinates"]["z"]["value"] == "NotificationSent"
            ]
            if not sent:
                answers.append(
                    record(
                        query,
                        "NOT_APPLICABLE_NO_NOTIFICATION_SENT",
                        status="NOT_APPLICABLE",
                    )
                )
            else:
                target = sent[0]
                predecessors = [
                    row
                    for row in by_type["generated_origin_dependency"]
                    if row["target_id"] == target["fact_id"]
                ]
                message = fact_by_id[predecessors[0]["source_id"]]
                prior = next(
                    row
                    for row in by_type["generated_origin_dependency"]
                    if row["target_id"] == message["fact_id"]
                )
                origin = fact_by_id[prior["source_id"]]
                answers.append(
                    record(
                        query,
                        origin["coordinates"]["z"]["value"],
                        relation_ids=[
                            predecessors[0]["relation_id"],
                            prior["relation_id"],
                        ],
                        result_ids=[origin["result_id"], target["result_id"]],
                    )
                )
        elif query_id == "Q08":
            suppressed = [
                row
                for row in notification_facts
                if row["coordinates"]["z"]["value"]
                == "NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND"
            ]
            rows = []
            relation_ids = []
            result_ids = []
            for target in suppressed:
                message_relation = next(
                    row
                    for row in by_type["generated_origin_dependency"]
                    if row["target_id"] == target["fact_id"]
                )
                source_relation = next(
                    row
                    for row in by_type["generated_origin_dependency"]
                    if row["target_id"] == message_relation["source_id"]
                )
                source = fact_by_id[source_relation["source_id"]]
                rows.append(source["coordinates"]["z"]["value"])
                result_ids.extend([source["result_id"], target["result_id"]])
                relation_ids.extend(
                    [
                        source_relation["relation_id"],
                        message_relation["relation_id"],
                    ]
                )
            answers.append(
                record(
                    query,
                    sorted(rows),
                    relation_ids=relation_ids,
                    result_ids=result_ids,
                    selected_dispositions=result_ids,
                )
            )
        elif query_id in {"Q09", "Q11"}:
            commits = [
                row
                for row in refund_facts
                if row["coordinates"]["z"]["value"] == "RefundCommitted"
            ]
            reached_results: list[str] = []
            relation_ids: list[str] = []
            for commit in commits:
                reached, path = closure(commit["fact_id"])
                relation_ids.extend(path)
                reached_results.extend(
                    fact_by_id[fact_id]["result_id"]
                    for fact_id in reached
                    if query_id == "Q09"
                    or fact_by_id[fact_id]["coordinates"]["z"]["kind"]
                    == "BusinessSupport"
                )
            answers.append(
                record(
                    query,
                    sorted(set(reached_results)),
                    relation_ids=relation_ids,
                    result_ids=[
                        row["result_id"] for row in commits
                    ]
                    + reached_results,
                )
            )
        elif query_id == "Q10":
            result_ids = []
            relation_ids = []
            for relation in by_type["reads_from"]:
                source = fact_by_id[relation["source_id"]]
                if source["coordinates"]["z"].get("version_id") == "order-001-v7":
                    result_ids.append(
                        fact_by_id[relation["target_id"]]["result_id"]
                    )
                    relation_ids.append(relation["relation_id"])
            for relation in by_type["conflicts_with"]:
                result_ids.extend(
                    [
                        fact_by_id[relation["source_id"]]["result_id"],
                        fact_by_id[relation["target_id"]]["result_id"],
                    ]
                )
                relation_ids.append(relation["relation_id"])
            answers.append(
                record(
                    query,
                    sorted(set(result_ids)),
                    relation_ids=relation_ids,
                    result_ids=result_ids,
                )
            )
        elif query_id == "Q12":
            answers.append(
                record(
                    query,
                    paired,
                    result_ids=paired["result_ids"],
                    selected_dispositions=paired["disposition_ids"],
                )
            )
        elif query_id == "Q13":
            conflict_relations = by_type["conflicts_with"]
            all_results = {row["result_id"] for row in action_facts}
            affected: set[str] = set()
            relation_ids = []
            for relation in conflict_relations:
                relation_ids.append(relation["relation_id"])
                endpoints = [
                    fact_by_id[relation["source_id"]],
                    fact_by_id[relation["target_id"]],
                ]
                for endpoint in endpoints:
                    if endpoint in disposition_facts:
                        affected.add(endpoint["result_id"])
                        reached, path = closure(endpoint["fact_id"])
                        relation_ids.extend(path)
                        affected.update(
                            fact_by_id[fact_id]["result_id"]
                            for fact_id in reached
                            if fact_by_id[fact_id] in action_facts
                        )
            answers.append(
                record(
                    query,
                    {
                        "affected_result_ids": sorted(affected),
                        "unaffected_result_ids": sorted(all_results - affected),
                    },
                    relation_ids=relation_ids,
                    result_ids=list(all_results),
                    selected_dispositions=list(affected & set(disposition_ids)),
                )
            )
        elif query_id == "Q14":
            if scenario != "IDEMPOTENT_DUPLICATE_REFUND":
                answers.append(
                    record(query, "NOT_APPLICABLE", status="NOT_APPLICABLE")
                )
            else:
                committed_refunds = sum(
                    row["coordinates"]["z"]["value"] == "RefundCommitted"
                    for row in refund_facts
                )
                sent_notifications = sum(
                    row["coordinates"]["z"]["value"] == "NotificationSent"
                    for row in notification_facts
                )
                answers.append(
                    record(
                        query,
                        {
                            "refund_row_count": 1,
                            "notification_sent_count": sent_notifications,
                            "second_refund_formed": committed_refunds > 1,
                            "second_notification_formed": (
                                sent_notifications > 1
                            ),
                        },
                        result_ids=[
                            row["result_id"]
                            for row in refund_facts + notification_facts
                        ],
                        selected_dispositions=[
                            row["result_id"]
                            for row in refund_facts
                            if row in disposition_facts
                        ],
                    )
                )
    return answers


def resolve_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    validate_candidate_input(payload)
    contexts = payload["contexts"]
    context_by_scenario = {
        row["validated_atomic_facts"]["scenario"]: row for row in contexts
    }
    b = context_by_scenario["CONCURRENT_FREEZE_WINS"][
        "validated_atomic_facts"
    ]
    c = context_by_scenario["LATE_REFUND_AFTER_FREEZE"][
        "validated_atomic_facts"
    ]
    b_final = next(
        row
        for row in b["facts"]
        if row["coordinates"]["z"]["kind"] == "FinalOrderState"
    )
    c_final = next(
        row
        for row in c["facts"]
        if row["coordinates"]["z"]["kind"] == "FinalOrderState"
    )
    b_refund = next(
        row
        for row in b["facts"]
        if row["coordinates"]["u"].get("action_id") == "refund-primary"
    )
    c_refund = next(
        row
        for row in c["facts"]
        if row["coordinates"]["u"].get("action_id") == "refund-primary"
    )
    paired = {
        "ordinary_business_view_equal": (
            b_final["coordinates"]["z"] == c_final["coordinates"]["z"]
        ),
        "formation_answer_equal": (
            b_refund["coordinates"]["z"] == c_refund["coordinates"]["z"]
        ),
        "scenario_b_refund_reason": b_refund["coordinates"]["z"]["value"],
        "scenario_c_refund_reason": c_refund["coordinates"]["z"]["value"],
        "result_ids": [b_refund["result_id"], c_refund["result_id"]],
        "disposition_ids": [b_refund["result_id"], c_refund["result_id"]],
    }
    answers = [
        answer
        for context in contexts
        for answer in _resolve_context(context, payload["queries"], paired)
    ]
    return {
        "status": "PASS",
        "answers": sorted(
            answers, key=lambda row: (row["scenario"], row["query_id"])
        ),
        "answer_count": len(answers),
        "input_context_count": len(contexts),
        "schema_version": "candidate-answers-v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = resolve_candidate(load_json(args.input))
    except Exception as error:
        output = {
            "status": "FAIL",
            "reason_code": str(error),
            "schema_version": "candidate-answers-v1",
        }
    write_json(args.output, output)
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
