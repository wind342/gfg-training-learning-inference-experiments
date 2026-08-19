from __future__ import annotations

from typing import Any

from ..common import content_id


def _fact(
    *,
    execution_run_id: str,
    result_id: str,
    source: dict[str, Any],
    transformation: str,
    occurrence_id: str,
    outcome: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    coordinates = {
        "u": source,
        "tau": {"realized_transformation": transformation},
        "omega_bar": {"concrete_occurrence_id": occurrence_id},
        "z": outcome,
        "rho": {"role": role},
    }
    return {
        "fact_id": content_id(
            "orfact1_",
            {
                "execution_run_id": execution_run_id,
                "result_id": result_id,
                "coordinates": coordinates,
            },
        ),
        "execution_run_id": execution_run_id,
        "result_id": result_id,
        "coordinates": coordinates,
        "schema_version": "order-refund-freeze-atomic-fact-v1",
    }


def collect_atomic_facts(run: dict[str, Any]) -> dict[str, Any]:
    execution_run_id = run["execution_run_id"]
    facts: list[dict[str, Any]] = []
    for result in run["action_results"]:
        facts.append(
            _fact(
                execution_run_id=execution_run_id,
                result_id=result["result_id"],
                source={
                    "kind": "worker_action",
                    "action_id": result["action_id"],
                    "action_type": result["action_type"],
                },
                transformation=result["action_type"],
                occurrence_id=result["occurrence_id"],
                outcome={
                    "kind": result["result_kind"],
                    "value": result["outcome"],
                },
                role=(
                    "explicit_disposition"
                    if result["result_kind"] == "ExplicitDisposition"
                    else "business_support"
                ),
            )
        )

    read_versions: dict[str, dict[str, Any]] = {}
    for receipt in run["sql_receipts"]:
        version_id = receipt.get("version_id")
        if version_id and receipt.get("row"):
            read_versions.setdefault(
                version_id,
                {
                    "status": receipt["row"]["status"],
                    "version": receipt["row"]["version"],
                    "occurrence_id": receipt["occurrence_id"],
                },
            )
    for version_id, observed in sorted(read_versions.items()):
        facts.append(
            _fact(
                execution_run_id=execution_run_id,
                result_id=f"{execution_run_id}:{version_id}",
                source={"kind": "sqlite_order_resource", "order_id": "order-001"},
                transformation="observe_order_version",
                occurrence_id=observed["occurrence_id"],
                outcome={
                    "kind": "OrderVersionObserved",
                    "version_id": version_id,
                    "status": observed["status"],
                    "version": observed["version"],
                },
                role="version_support",
            )
        )

    final_order = run["canonical_db_dump"]["orders"][0]
    final_result_id = f"{execution_run_id}:final-order-state"
    facts.append(
        _fact(
            execution_run_id=execution_run_id,
            result_id=final_result_id,
            source={"kind": "sqlite_canonical_dump", "order_id": "order-001"},
            transformation="materialize_final_order_state",
            occurrence_id=content_id(
                "orocc1_",
                {"run": execution_run_id, "event": "final-order-state"},
            ),
            outcome={"kind": "FinalOrderState", **final_order},
            role="business_support",
        )
    )

    put_receipts = [
        row for row in run["queue_receipts"] if row["operation"] == "put"
    ]
    notification_sources = {
        row["source_message_id"]: row["source_result_id"]
        for row in run["action_results"]
        if row["action_type"] == "notification"
    }
    result_outcomes = {
        row["result_id"]: row["outcome"] for row in run["action_results"]
    }
    for receipt in put_receipts:
        source_result_id = notification_sources[receipt["message_id"]]
        message_result_id = f"{execution_run_id}:{receipt['message_id']}"
        message_kind = (
            "RefundCommittedMessage"
            if result_outcomes[source_result_id] == "RefundCommitted"
            else "RefundDisposedMessage"
        )
        facts.append(
            _fact(
                execution_run_id=execution_run_id,
                result_id=message_result_id,
                source={
                    "kind": "multiprocessing_queue",
                    "message_id": receipt["message_id"],
                },
                transformation="queue_put",
                occurrence_id=receipt["occurrence_id"],
                outcome={
                    "kind": message_kind,
                    "message_id": receipt["message_id"],
                    "payload_digest": receipt["payload_digest"],
                },
                role="message_support",
            )
        )

    fact_ids = [row["fact_id"] for row in facts]
    result_ids = [row["result_id"] for row in facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise RuntimeError("DUPLICATE_ATOMIC_FACT_ID")
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("DUPLICATE_RESULT_ID")
    return {
        "status": "PASS",
        "execution_run_id": execution_run_id,
        "scenario": run["scenario"],
        "facts": sorted(facts, key=lambda row: row["fact_id"]),
        "fact_count": len(facts),
        "coordinate_names": ["u", "tau", "omega_bar", "z", "rho"],
        "sixth_coordinate_present": False,
        "schema_version": "order-refund-freeze-atomic-facts-v1",
    }
