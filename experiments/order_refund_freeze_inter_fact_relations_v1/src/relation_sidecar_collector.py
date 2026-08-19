from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from experiments.inter_fact_relations_v0_hardening_scale_v1.src.capture_auditor import (
    _program_order_exactness,
)

from ..common import canonical_sha256, content_id


AUTHORITY = "controlled-order-workflow-executor-v1"
ESTABLISHMENT = "wrapper_established"


def _evidence(
    *,
    run_id: str,
    kind: str,
    receipt_ref: str,
    endpoint_ids: list[str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    material = {
        "execution_run_id": run_id,
        "evidence_kind": kind,
        "receipt_ref": receipt_ref,
        "endpoint_ids": sorted(endpoint_ids),
        "authority_id": AUTHORITY,
        "establishment_source": ESTABLISHMENT,
        "payload": payload,
    }
    return {"evidence_id": content_id("orev1_", material), **material}


def _relation(
    *,
    run_id: str,
    relation_type: str,
    source_id: str,
    target_id: str,
    endpoint_level: str,
    evidence_id: str,
) -> dict[str, Any]:
    material = {
        "execution_run_id": run_id,
        "relation_type": relation_type,
        "source_id": source_id,
        "target_id": target_id,
        "endpoint_level": endpoint_level,
        "authority_id": AUTHORITY,
        "establishment_source": ESTABLISHMENT,
        "evidence_refs": [evidence_id],
    }
    return {"relation_id": content_id("orrel1_", material), **material}


def collect_relation_sidecar(
    run: dict[str, Any], atomic: dict[str, Any]
) -> dict[str, Any]:
    run_id = run["execution_run_id"]
    facts_by_result = {row["result_id"]: row for row in atomic["facts"]}
    action_by_id = {row["action_id"]: row for row in run["action_results"]}
    relations: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    program_receipts: list[dict[str, Any]] = []

    events_by_actor: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in run["events"]:
        events_by_actor[event["actor_id"]].append(event)
    for actor_id, events in sorted(events_by_actor.items()):
        ordered = sorted(events, key=lambda row: row["sequence_index"])
        for source, target in zip(ordered, ordered[1:]):
            receipt = {
                "receipt_id": content_id(
                    "orporeceipt1_",
                    {
                        "run": run_id,
                        "source": source["occurrence_id"],
                        "target": target["occurrence_id"],
                    },
                ),
                "execution_run_id": run_id,
                "actor_id": actor_id,
                "source_occurrence_id": source["occurrence_id"],
                "target_occurrence_id": target["occurrence_id"],
                "source_sequence_index": source["sequence_index"],
                "target_sequence_index": target["sequence_index"],
                "recorded_by": "executor_wrapper",
                "authority_id": AUTHORITY,
                "establishment_source": ESTABLISHMENT,
            }
            program_receipts.append(receipt)
            item = _evidence(
                run_id=run_id,
                kind="program_order_log",
                receipt_ref=receipt["receipt_id"],
                endpoint_ids=[
                    source["occurrence_id"],
                    target["occurrence_id"],
                ],
                payload={
                    "actor_id": actor_id,
                    "source_sequence_index": source["sequence_index"],
                    "target_sequence_index": target["sequence_index"],
                    "recorded_by": "executor_wrapper",
                },
            )
            item["occurrence_ids"] = item.pop("endpoint_ids")
            item["fact_ids"] = []
            evidence.append(item)
            relations.append(
                _relation(
                    run_id=run_id,
                    relation_type="program_order",
                    source_id=source["occurrence_id"],
                    target_id=target["occurrence_id"],
                    endpoint_level="occurrence",
                    evidence_id=item["evidence_id"],
                )
            )

    for receipt in run["sql_receipts"]:
        if (
            receipt["operation"] == "SELECT"
            and receipt["table"] == "orders"
            and receipt.get("version_id")
        ):
            action = action_by_id[receipt["action_id"]]
            version_result_id = f"{run_id}:{receipt['version_id']}"
            item = _evidence(
                run_id=run_id,
                kind="sqlite_read_receipt",
                receipt_ref=receipt["receipt_id"],
                endpoint_ids=[
                    facts_by_result[version_result_id]["fact_id"],
                    facts_by_result[action["result_id"]]["fact_id"],
                ],
                payload={
                    "resource_id": receipt["resource_id"],
                    "version_id": receipt["version_id"],
                    "consumer_action_id": receipt["action_id"],
                    "row": receipt["row"],
                },
            )
            evidence.append(item)
            relations.append(
                _relation(
                    run_id=run_id,
                    relation_type="reads_from",
                    source_id=facts_by_result[version_result_id]["fact_id"],
                    target_id=facts_by_result[action["result_id"]]["fact_id"],
                    endpoint_level="fact",
                    evidence_id=item["evidence_id"],
                )
            )

    refund = next(
        (
            row
            for row in run["action_results"]
            if row["action_id"] == "refund-primary"
        ),
        None,
    )
    freeze = next(
        (row for row in run["action_results"] if row["action_type"] == "freeze"),
        None,
    )
    updates = {
        row["action_id"]: row
        for row in run["sql_receipts"]
        if row["operation"] == "UPDATE" and row["table"] == "orders"
    }
    if (
        refund
        and freeze
        and refund["action_id"] in updates
        and freeze["action_id"] in updates
        and refund["read_order"]["version"] == freeze["read_order"]["version"]
        and refund["read_order"]["order_id"] == freeze["read_order"]["order_id"]
    ):
        receipt_ref = content_id(
            "orconflictreceipt1_",
            {
                "run": run_id,
                "left": updates[refund["action_id"]]["receipt_id"],
                "right": updates[freeze["action_id"]]["receipt_id"],
            },
        )
        item = _evidence(
            run_id=run_id,
            kind="sqlite_conflict_receipt",
            receipt_ref=receipt_ref,
            endpoint_ids=[
                facts_by_result[refund["result_id"]]["fact_id"],
                facts_by_result[freeze["result_id"]]["fact_id"],
            ],
            payload={
                "resource_id": "order-001",
                "version_id": (
                    f"order-001-v{refund['read_order']['version']}"
                ),
                "left_access_mode": "write",
                "right_access_mode": "write",
                "left_update_receipt": updates[refund["action_id"]][
                    "receipt_id"
                ],
                "right_update_receipt": updates[freeze["action_id"]][
                    "receipt_id"
                ],
            },
        )
        evidence.append(item)
        relations.append(
            _relation(
                run_id=run_id,
                relation_type="conflicts_with",
                source_id=facts_by_result[refund["result_id"]]["fact_id"],
                target_id=facts_by_result[freeze["result_id"]]["fact_id"],
                endpoint_level="fact",
                evidence_id=item["evidence_id"],
            )
        )

    puts = {
        row["message_id"]: row
        for row in run["queue_receipts"]
        if row["operation"] == "put"
    }
    gets = {
        row["message_id"]: row
        for row in run["queue_receipts"]
        if row["operation"] == "get"
    }
    for message_id in sorted(set(puts) & set(gets)):
        send = puts[message_id]
        receive = gets[message_id]
        receipt_ref = content_id(
            "ormessagereceipt1_",
            {"run": run_id, "message_id": message_id},
        )
        item = _evidence(
            run_id=run_id,
            kind="multiprocessing_queue_receipt",
            receipt_ref=receipt_ref,
            endpoint_ids=[send["occurrence_id"], receive["occurrence_id"]],
            payload={
                "message_id": message_id,
                "payload_digest": send["payload_digest"],
                "send_receipt_id": send["receipt_id"],
                "receive_receipt_id": receive["receipt_id"],
            },
        )
        evidence.append(item)
        relations.append(
            _relation(
                run_id=run_id,
                relation_type="message_send_receive",
                source_id=send["occurrence_id"],
                target_id=receive["occurrence_id"],
                endpoint_level="occurrence",
                evidence_id=item["evidence_id"],
            )
        )

    event_index = {
        (
            event["actor_id"],
            event["event_type"],
            event["detail"].get("event_id"),
        ): event
        for event in run["events"]
    }
    for receipt in run["synchronization_receipts"]:
        if receipt["sync_type"] != "Event":
            continue
        worker = receipt["released_worker"]
        worker_event = event_index.get(
            (worker, "event_wait_released", receipt["event_id"])
        )
        if worker_event is None:
            continue
        release_occurrence = content_id(
            "orocc1_",
            {
                "run": run_id,
                "actor": "Orchestrator",
                "event": receipt["event_id"],
            },
        )
        receipt_ref = content_id(
            "orsyncreceipt1_",
            {"run": run_id, "event_id": receipt["event_id"]},
        )
        item = _evidence(
            run_id=run_id,
            kind="multiprocessing_event_receipt",
            receipt_ref=receipt_ref,
            endpoint_ids=[release_occurrence, worker_event["occurrence_id"]],
            payload=receipt,
        )
        evidence.append(item)
        relations.append(
            _relation(
                run_id=run_id,
                relation_type="synchronizes_with",
                source_id=release_occurrence,
                target_id=worker_event["occurrence_id"],
                endpoint_level="occurrence",
                evidence_id=item["evidence_id"],
            )
        )

    message_fact_by_id = {
        row["coordinates"]["z"]["message_id"]: row
        for row in atomic["facts"]
        if row["coordinates"]["z"]["kind"]
        in {"RefundCommittedMessage", "RefundDisposedMessage"}
    }
    for notification in (
        row
        for row in run["action_results"]
        if row["action_type"] == "notification"
    ):
        source_result_fact = facts_by_result[notification["source_result_id"]]
        message_fact = message_fact_by_id[notification["source_message_id"]]
        target_fact = facts_by_result[notification["result_id"]]
        for source_fact, next_fact, bridge in (
            (source_result_fact, message_fact, "result_to_message"),
            (message_fact, target_fact, "message_to_notification"),
        ):
            receipt_ref = content_id(
                "ororiginreceipt1_",
                {
                    "run": run_id,
                    "source": source_fact["fact_id"],
                    "target": next_fact["fact_id"],
                },
            )
            item = _evidence(
                run_id=run_id,
                kind="generated_origin_record",
                receipt_ref=receipt_ref,
                endpoint_ids=[source_fact["fact_id"], next_fact["fact_id"]],
                payload={
                    "bridge": bridge,
                    "message_id": notification["source_message_id"],
                },
            )
            evidence.append(item)
            relations.append(
                _relation(
                    run_id=run_id,
                    relation_type="generated_origin_dependency",
                    source_id=source_fact["fact_id"],
                    target_id=next_fact["fact_id"],
                    endpoint_level="fact",
                    evidence_id=item["evidence_id"],
                )
            )

    final_version_fact = next(
        row
        for row in atomic["facts"]
        if row["coordinates"]["z"]["kind"] == "FinalOrderState"
    )
    for result in run["action_results"]:
        if (
            result["transaction_outcome"] == "COMMIT"
            and result["outcome"] in {"RefundCommitted", "OrderFrozen"}
        ):
            source_fact = facts_by_result[result["result_id"]]
            receipt = next(
                row
                for row in run["sql_receipts"]
                if row["action_id"] == result["action_id"]
                and row["operation"] == "UPDATE"
            )
            item = _evidence(
                run_id=run_id,
                kind="sqlite_commit_receipt",
                receipt_ref=receipt["receipt_id"],
                endpoint_ids=[
                    source_fact["fact_id"],
                    final_version_fact["fact_id"],
                ],
                payload={
                    "committed_version_id": "order-001-v8",
                    "transaction_outcome": "COMMIT",
                    "rowcount": receipt["rowcount"],
                },
            )
            evidence.append(item)
            relations.append(
                _relation(
                    run_id=run_id,
                    relation_type="commits_version",
                    source_id=source_fact["fact_id"],
                    target_id=final_version_fact["fact_id"],
                    endpoint_level="fact",
                    evidence_id=item["evidence_id"],
                )
            )

    occurrences = [
        {
            "concrete_occurrence_instance_id": row["occurrence_id"],
            "actor_id": row["actor_id"],
            "sequence_index": row["sequence_index"],
        }
        for row in run["events"]
    ]
    program_relations = [
        row for row in relations if row["relation_type"] == "program_order"
    ]
    program_evidence = [
        {
            **row,
            "occurrence_ids": row.get("occurrence_ids", []),
        }
        for row in evidence
        if row["evidence_kind"] == "program_order_log"
    ]
    exactness = _program_order_exactness(
        run_id=run_id,
        scope_occurrences=occurrences,
        program_receipts=program_receipts,
        scope_relations=program_relations,
        evidence=program_evidence,
    )
    if not exactness["exact"]:
        raise RuntimeError(
            "PROGRAM_ORDER_AUDIT_FAILED:"
            + ",".join(exactness["reason_codes"])
        )

    relation_ids = [row["relation_id"] for row in relations]
    evidence_ids = [row["evidence_id"] for row in evidence]
    if len(relation_ids) != len(set(relation_ids)):
        raise RuntimeError("DUPLICATE_RELATION_ID")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise RuntimeError("DUPLICATE_EVIDENCE_ID")
    return {
        "status": "PASS",
        "execution_run_id": run_id,
        "scenario": run["scenario"],
        "relations": sorted(relations, key=lambda row: row["relation_id"]),
        "evidence": sorted(evidence, key=lambda row: row["evidence_id"]),
        "program_order_receipts": sorted(
            program_receipts, key=lambda row: row["receipt_id"]
        ),
        "program_order_exactness": {
            "status": "PASS",
            "expected_edge_count": len(exactness["expected_edges"]),
            "receipt_edge_count": len(exactness["receipt_edges"]),
            "relation_edge_count": len(exactness["relation_edges"]),
            "evidence_edge_count": len(exactness["evidence_edges"]),
            "edge_set_sha256": canonical_sha256(
                sorted(exactness["expected_edges"])
            ),
            "missing_edges": [],
            "extra_edges": [],
            "duplicate_sequence_indexes": [],
            "binding_issues": [],
        },
        "relation_type_counts": dict(
            sorted(Counter(row["relation_type"] for row in relations).items())
        ),
        "schema_version": "order-refund-freeze-relation-sidecar-v1",
    }
