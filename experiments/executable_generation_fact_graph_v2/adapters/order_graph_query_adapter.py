from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from ..graph_model import FactNode, ValidatedGenerationFactGraphV2
from ..graph_query import ExecutableGenerationFactGraphQueryEngineV2


CAPTURE_COMPLETE = "CAPTURE_COMPLETE"
ORDER_COMPENSATION_POLICY = {
    "policy_id": "order-compensation-target-policy-v1",
    "query_id": "Q11",
    "source_outcome_value": "RefundCommitted",
    "traversal_relation_type": "generated_origin_dependency",
    "traversal_direction": "out",
    "closure": "transitive",
    "target_native_z_kind": "BusinessSupport",
}


def _native_u(node: FactNode) -> dict[str, Any]:
    return node.u["entity"]["source_payload"]["native_u"]


def _outcome_payload(node: FactNode) -> dict[str, Any]:
    entity = node.z["entity"]
    if node.z["reference"]["kind"] == "support":
        return entity["support_payload"]
    return entity["disposition_payload"]


def _native_z(node: FactNode) -> dict[str, Any]:
    return _outcome_payload(node)["native_z"]


def _result_id(node: FactNode) -> str:
    return _outcome_payload(node)["native_result_id"]


def _is_disposition(node: FactNode) -> bool:
    return node.z["reference"]["kind"] == "disposition"


def _scenario(query: ExecutableGenerationFactGraphQueryEngineV2) -> str:
    run_id = query.graph.metadata.execution_run_id
    prefix = "run-"
    suffix = "-01-capture-enabled"
    if not run_id.startswith(prefix) or not run_id.endswith(suffix):
        raise ValueError("ORDER_GRAPH_EXECUTION_RUN_ID_INVALID")
    return run_id[len(prefix) : -len(suffix)].upper()


def _relations_by_type(
    query: ExecutableGenerationFactGraphQueryEngineV2,
) -> dict[str, list[dict[str, Any]]]:
    by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in query.all_relation_edges():
        by_type[relation["relation_type"]].append(relation)
    return by_type


def _directed_closure(
    query: ExecutableGenerationFactGraphQueryEngineV2,
    source_node_id: str,
    relation_type: str,
) -> tuple[list[str], list[str]]:
    visited = {source_node_id}
    queue = deque([source_node_id])
    reached = []
    relation_ids = []
    while queue:
        current = queue.popleft()
        for relation in query.relation_edges(
            current, "out", {relation_type}
        ):
            if relation["source_node_id"] != current:
                continue
            relation_ids.append(relation["original_relation_id"])
            target = relation["target_node_id"]
            if target not in visited:
                visited.add(target)
                reached.append(target)
                queue.append(target)
    return reached, relation_ids


def resolve_order_compensation_targets(
    query: ExecutableGenerationFactGraphQueryEngineV2,
    source_fact_node_ids: list[str],
    policy: dict[str, str],
) -> dict[str, Any]:
    if policy != ORDER_COMPENSATION_POLICY:
        raise ValueError("ORDER_COMPENSATION_POLICY_MISMATCH")
    if any(
        _native_z(query.fact_nodes[node_id]).get("value")
        != policy["source_outcome_value"]
        for node_id in source_fact_node_ids
    ):
        raise ValueError("ORDER_COMPENSATION_SOURCE_MISMATCH")
    target_node_ids = []
    relation_ids = []
    for source_node_id in source_fact_node_ids:
        reached, path_relations = _directed_closure(
            query,
            source_node_id,
            policy["traversal_relation_type"],
        )
        relation_ids.extend(path_relations)
        target_node_ids.extend(
            node_id
            for node_id in reached
            if _native_z(query.fact_nodes[node_id]).get("kind")
            == policy["target_native_z_kind"]
        )
    return {
        "target_node_ids": sorted(set(target_node_ids)),
        "relation_ids": sorted(set(relation_ids)),
        "policy_id": policy["policy_id"],
    }


def _record(
    query: ExecutableGenerationFactGraphQueryEngineV2,
    query_spec: dict[str, str],
    scenario: str,
    answer: Any,
    *,
    relation_ids: list[str] | None = None,
    result_ids: list[str] | None = None,
    disposition_ids: list[str] | None = None,
    status: str = "ESTABLISHED",
) -> dict[str, Any]:
    relation_ids = sorted(set(relation_ids or []))
    relation_by_id = {
        relation["original_relation_id"]: relation
        for relation in query.all_relation_edges()
    }
    return {
        "query_id": query_spec["query_id"],
        "scenario": scenario,
        "exact_target": query_spec["exact_target"],
        "answer": answer,
        "status": status,
        "evidence_path": [
            {
                "relation_id": relation_id,
                "evidence_ids": relation_by_id[relation_id][
                    "evidence_refs"
                ],
            }
            for relation_id in relation_ids
        ],
        "result_ids": sorted(set(result_ids or [])),
        "relation_ids": relation_ids,
        "evidence_ids": sorted(
            {
                evidence_id
                for relation_id in relation_ids
                for evidence_id in relation_by_id[relation_id][
                    "evidence_refs"
                ]
            }
        ),
        "explicit_disposition_ids": sorted(
            set(disposition_ids or [])
        ),
    }


def _resolve_context(
    query: ExecutableGenerationFactGraphQueryEngineV2,
    query_specs: list[dict[str, str]],
    paired: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario = _scenario(query)
    facts = list(query.fact_nodes.values())
    relations = _relations_by_type(query)
    action_facts = [
        node
        for node in facts
        if _native_u(node).get("kind") == "worker_action"
    ]
    refund_facts = [
        node
        for node in action_facts
        if _native_u(node).get("action_type") == "refund"
    ]
    freeze_facts = [
        node
        for node in action_facts
        if _native_u(node).get("action_type") == "freeze"
    ]
    notification_facts = [
        node
        for node in action_facts
        if _native_u(node).get("action_type") == "notification"
    ]
    disposition_facts = [
        node for node in action_facts if _is_disposition(node)
    ]
    disposition_node_ids = {
        node.graph_node_id for node in disposition_facts
    }
    answers = []

    for query_spec in query_specs:
        query_id = query_spec["query_id"]
        if query_id == "Q01":
            rows = []
            relation_ids = []
            result_ids = []
            for relation in relations["reads_from"]:
                target = query.fact_nodes[relation["target_node_id"]]
                if target.graph_node_id not in {
                    node.graph_node_id for node in refund_facts
                }:
                    continue
                source = query.fact_nodes[relation["source_node_id"]]
                rows.append(
                    {
                        "refund_result_id": _result_id(target),
                        "version_id": _native_z(source).get(
                            "version_id"
                        ),
                    }
                )
                relation_ids.append(relation["original_relation_id"])
                result_ids.append(_result_id(target))
            for refund in refund_facts:
                if _result_id(refund) not in result_ids:
                    rows.append(
                        {
                            "refund_result_id": _result_id(refund),
                            "version_id": None,
                        }
                    )
                    result_ids.append(_result_id(refund))
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    sorted(
                        rows,
                        key=lambda row: row["refund_result_id"],
                    ),
                    relation_ids=relation_ids,
                    result_ids=result_ids,
                    disposition_ids=[
                        _result_id(node)
                        for node in refund_facts
                        if node.graph_node_id in disposition_node_ids
                    ],
                )
            )
        elif query_id == "Q02":
            if not freeze_facts:
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        "NOT_APPLICABLE",
                        status="NOT_APPLICABLE",
                    )
                )
                continue
            audit = query.validated_graph.capture_audit
            if audit["status"] != CAPTURE_COMPLETE:
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        "NOT_ESTABLISHED",
                        status="NOT_ESTABLISHED",
                    )
                )
                continue
            conflict = bool(relations["conflicts_with"])
            late_read = any(
                relation["target_node_id"]
                == refund_facts[0].graph_node_id
                and _native_z(
                    query.fact_nodes[relation["source_node_id"]]
                ).get("version")
                == 8
                for relation in relations["reads_from"]
            )
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    conflict and not late_read,
                    relation_ids=[
                        relation["original_relation_id"]
                        for relation in (
                            relations["conflicts_with"]
                            + relations["synchronizes_with"]
                        )
                    ],
                    result_ids=[
                        _result_id(refund_facts[0]),
                        _result_id(freeze_facts[0]),
                    ],
                )
            )
        elif query_id == "Q03":
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    bool(relations["conflicts_with"]),
                    relation_ids=[
                        relation["original_relation_id"]
                        for relation in relations["conflicts_with"]
                    ],
                    result_ids=[
                        _result_id(node)
                        for node in refund_facts + freeze_facts
                    ],
                )
            )
        elif query_id == "Q04":
            committed = [
                query.fact_nodes[relation["source_node_id"]]
                for relation in relations["commits_version"]
            ]
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    sorted(_result_id(node) for node in committed),
                    relation_ids=[
                        relation["original_relation_id"]
                        for relation in relations["commits_version"]
                    ],
                    result_ids=[
                        _result_id(node) for node in committed
                    ],
                )
            )
        elif query_id == "Q05":
            values = [
                {
                    "result_id": _result_id(node),
                    "reason": (
                        _native_z(node)["value"]
                        if _is_disposition(node)
                        else "COMMITTED"
                    ),
                }
                for node in refund_facts
            ]
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    sorted(
                        values, key=lambda row: row["result_id"]
                    ),
                    result_ids=[
                        _result_id(node) for node in refund_facts
                    ],
                    disposition_ids=[
                        _result_id(node)
                        for node in refund_facts
                        if _is_disposition(node)
                    ],
                )
            )
        elif query_id == "Q06":
            if not freeze_facts:
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        "NOT_APPLICABLE",
                        status="NOT_APPLICABLE",
                    )
                )
            else:
                freeze = freeze_facts[0]
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        (
                            _native_z(freeze)["value"]
                            if _is_disposition(freeze)
                            else "COMMITTED"
                        ),
                        result_ids=[_result_id(freeze)],
                        disposition_ids=(
                            [_result_id(freeze)]
                            if _is_disposition(freeze)
                            else []
                        ),
                    )
                )
        elif query_id == "Q07":
            sent = [
                node
                for node in notification_facts
                if _native_z(node).get("value")
                == "NotificationSent"
            ]
            if not sent:
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        "NOT_APPLICABLE_NO_NOTIFICATION_SENT",
                        status="NOT_APPLICABLE",
                    )
                )
            else:
                target = sent[0]
                incoming = query.relation_edges(
                    target.graph_node_id,
                    "in",
                    {"generated_origin_dependency"},
                )
                message_relation = incoming[0]
                message_id = message_relation["source_node_id"]
                prior = query.relation_edges(
                    message_id,
                    "in",
                    {"generated_origin_dependency"},
                )[0]
                origin = query.fact_nodes[prior["source_node_id"]]
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        _native_z(origin)["value"],
                        relation_ids=[
                            message_relation["original_relation_id"],
                            prior["original_relation_id"],
                        ],
                        result_ids=[
                            _result_id(origin),
                            _result_id(target),
                        ],
                    )
                )
        elif query_id == "Q08":
            suppressed = [
                node
                for node in notification_facts
                if _native_z(node).get("value")
                == "NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND"
            ]
            values = []
            relation_ids = []
            result_ids = []
            for target in suppressed:
                message_relation = query.relation_edges(
                    target.graph_node_id,
                    "in",
                    {"generated_origin_dependency"},
                )[0]
                source_relation = query.relation_edges(
                    message_relation["source_node_id"],
                    "in",
                    {"generated_origin_dependency"},
                )[0]
                source = query.fact_nodes[
                    source_relation["source_node_id"]
                ]
                values.append(_native_z(source)["value"])
                result_ids.extend(
                    [_result_id(source), _result_id(target)]
                )
                relation_ids.extend(
                    [
                        source_relation["original_relation_id"],
                        message_relation["original_relation_id"],
                    ]
                )
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    sorted(values),
                    relation_ids=relation_ids,
                    result_ids=result_ids,
                    disposition_ids=result_ids,
                )
            )
        elif query_id == "Q09":
            commits = [
                node
                for node in refund_facts
                if _native_z(node).get("value") == "RefundCommitted"
            ]
            reached_node_ids = []
            relation_ids = []
            for commit in commits:
                reached, path = _directed_closure(
                    query,
                    commit.graph_node_id,
                    "generated_origin_dependency",
                )
                reached_node_ids.extend(reached)
                relation_ids.extend(path)
            reached_result_ids = [
                _result_id(query.fact_nodes[node_id])
                for node_id in reached_node_ids
            ]
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    sorted(set(reached_result_ids)),
                    relation_ids=relation_ids,
                    result_ids=[
                        _result_id(node) for node in commits
                    ]
                    + reached_result_ids,
                )
            )
        elif query_id == "Q10":
            result_ids = []
            relation_ids = []
            for relation in relations["reads_from"]:
                source = query.fact_nodes[relation["source_node_id"]]
                if (
                    _native_z(source).get("version_id")
                    == "order-001-v7"
                ):
                    result_ids.append(
                        _result_id(
                            query.fact_nodes[
                                relation["target_node_id"]
                            ]
                        )
                    )
                    relation_ids.append(
                        relation["original_relation_id"]
                    )
            for relation in relations["conflicts_with"]:
                result_ids.extend(
                    [
                        _result_id(
                            query.fact_nodes[
                                relation["source_node_id"]
                            ]
                        ),
                        _result_id(
                            query.fact_nodes[
                                relation["target_node_id"]
                            ]
                        ),
                    ]
                )
                relation_ids.append(relation["original_relation_id"])
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    sorted(set(result_ids)),
                    relation_ids=relation_ids,
                    result_ids=result_ids,
                )
            )
        elif query_id == "Q11":
            commits = [
                node
                for node in refund_facts
                if _native_z(node).get("value") == "RefundCommitted"
            ]
            resolved = resolve_order_compensation_targets(
                query,
                [node.graph_node_id for node in commits],
                ORDER_COMPENSATION_POLICY,
            )
            target_result_ids = [
                _result_id(query.fact_nodes[node_id])
                for node_id in resolved["target_node_ids"]
            ]
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    sorted(set(target_result_ids)),
                    relation_ids=resolved["relation_ids"],
                    result_ids=[
                        _result_id(node) for node in commits
                    ]
                    + target_result_ids,
                )
            )
        elif query_id == "Q12":
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    paired,
                    result_ids=paired["result_ids"],
                    disposition_ids=paired["disposition_ids"],
                )
            )
        elif query_id == "Q13":
            all_result_ids = {
                _result_id(node) for node in action_facts
            }
            affected = set()
            relation_ids = []
            for relation in relations["conflicts_with"]:
                relation_ids.append(relation["original_relation_id"])
                endpoint_ids = [
                    relation["source_node_id"],
                    relation["target_node_id"],
                ]
                for endpoint_id in endpoint_ids:
                    if endpoint_id not in disposition_node_ids:
                        continue
                    endpoint = query.fact_nodes[endpoint_id]
                    affected.add(_result_id(endpoint))
                    reached, path = _directed_closure(
                        query,
                        endpoint_id,
                        "generated_origin_dependency",
                    )
                    relation_ids.extend(path)
                    affected.update(
                        _result_id(query.fact_nodes[node_id])
                        for node_id in reached
                        if _native_u(
                            query.fact_nodes[node_id]
                        ).get("kind")
                        == "worker_action"
                    )
            answers.append(
                _record(
                    query,
                    query_spec,
                    scenario,
                    {
                        "affected_result_ids": sorted(affected),
                        "unaffected_result_ids": sorted(
                            all_result_ids - affected
                        ),
                    },
                    relation_ids=relation_ids,
                    result_ids=list(all_result_ids),
                    disposition_ids=list(
                        affected
                        & {
                            _result_id(node)
                            for node in disposition_facts
                        }
                    ),
                )
            )
        elif query_id == "Q14":
            if scenario != "IDEMPOTENT_DUPLICATE_REFUND":
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        "NOT_APPLICABLE",
                        status="NOT_APPLICABLE",
                    )
                )
            else:
                committed_refunds = sum(
                    _native_z(node).get("value") == "RefundCommitted"
                    for node in refund_facts
                )
                sent_notifications = sum(
                    _native_z(node).get("value") == "NotificationSent"
                    for node in notification_facts
                )
                answers.append(
                    _record(
                        query,
                        query_spec,
                        scenario,
                        {
                            "refund_row_count": committed_refunds,
                            "notification_sent_count": sent_notifications,
                            "second_refund_formed": (
                                committed_refunds > 1
                            ),
                            "second_notification_formed": (
                                sent_notifications > 1
                            ),
                        },
                        result_ids=[
                            _result_id(node)
                            for node in (
                                refund_facts + notification_facts
                            )
                        ],
                        disposition_ids=[
                            _result_id(node)
                            for node in refund_facts
                            if _is_disposition(node)
                        ],
                    )
                )
        else:
            raise ValueError("ORDER_GRAPH_QUERY_ID_UNKNOWN:" + query_id)
    return answers


def resolve_order_graph_queries(
    validated_graphs: list[ValidatedGenerationFactGraphV2],
    query_specs: list[dict[str, str]],
) -> dict[str, Any]:
    engines = [
        ExecutableGenerationFactGraphQueryEngineV2(validated)
        for validated in validated_graphs
    ]
    by_scenario = {_scenario(engine): engine for engine in engines}
    required = {
        "CONCURRENT_REFUND_WINS",
        "CONCURRENT_FREEZE_WINS",
        "LATE_REFUND_AFTER_FREEZE",
        "IDEMPOTENT_DUPLICATE_REFUND",
    }
    if set(by_scenario) != required:
        raise ValueError("ORDER_GRAPH_SCENARIO_SET_MISMATCH")
    b = by_scenario["CONCURRENT_FREEZE_WINS"]
    c = by_scenario["LATE_REFUND_AFTER_FREEZE"]
    b_final = next(
        node
        for node in b.fact_nodes.values()
        if _native_z(node).get("kind") == "FinalOrderState"
    )
    c_final = next(
        node
        for node in c.fact_nodes.values()
        if _native_z(node).get("kind") == "FinalOrderState"
    )
    b_refund = next(
        node
        for node in b.fact_nodes.values()
        if _native_u(node).get("action_id") == "refund-primary"
    )
    c_refund = next(
        node
        for node in c.fact_nodes.values()
        if _native_u(node).get("action_id") == "refund-primary"
    )
    paired = {
        "ordinary_business_view_equal": (
            _native_z(b_final) == _native_z(c_final)
        ),
        "formation_answer_equal": (
            _native_z(b_refund) == _native_z(c_refund)
        ),
        "scenario_b_refund_reason": _native_z(b_refund)["value"],
        "scenario_c_refund_reason": _native_z(c_refund)["value"],
        "result_ids": [
            _result_id(b_refund),
            _result_id(c_refund),
        ],
        "disposition_ids": [
            _result_id(b_refund),
            _result_id(c_refund),
        ],
    }
    answers = [
        answer
        for engine in engines
        for answer in _resolve_context(engine, query_specs, paired)
    ]
    return {
        "status": "PASS",
        "answers": sorted(
            answers,
            key=lambda row: (row["scenario"], row["query_id"]),
        ),
        "answer_count": len(answers),
        "input_context_count": len(engines),
        "query_execution": "DIRECT_VALIDATED_GRAPH_V2",
        "compensation_policy_id": ORDER_COMPENSATION_POLICY[
            "policy_id"
        ],
        "schema_version": "order-graph-candidate-answers-v2",
    }
