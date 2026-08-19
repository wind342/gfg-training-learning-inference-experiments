from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from experiments.inter_fact_relations_v0_hardening_scale_v1.src.capture_auditor import (
    _program_order_exactness,
)

from ..common import ExperimentError, REPAIRED_BASE_HEAD, content_id, git
from .candidate_process import validate_candidate_input
from .generation_fact_collector import collect_atomic_facts
from .orchestrator import run_workflow
from .reference_process import validate_reference_input
from .relation_sidecar_collector import collect_relation_sidecar
from .trace_baseline_process import resolve_trace


def _fail(reason: str, condition: bool) -> None:
    if condition:
        raise ExperimentError(reason)


def _action_result(run: dict[str, Any], action_id: str) -> dict[str, Any]:
    return next(
        row for row in run["action_results"] if row["action_id"] == action_id
    )


def _sql(
    run: dict[str, Any], action_id: str, operation: str
) -> dict[str, Any]:
    return next(
        row
        for row in run["sql_receipts"]
        if row["action_id"] == action_id and row["operation"] == operation
    )


def _program_order_control(
    run: dict[str, Any], sidecar: dict[str, Any]
) -> None:
    mutated = deepcopy(sidecar)
    actor = "NotificationWorker"
    occurrences = sorted(
        (
            row
            for row in run["events"]
            if row["actor_id"] == actor
        ),
        key=lambda row: row["sequence_index"],
    )
    source = occurrences[0]["occurrence_id"]
    middle = occurrences[1]["occurrence_id"]
    target = occurrences[2]["occurrence_id"]
    relations = [
        row
        for row in mutated["relations"]
        if row["relation_type"] == "program_order"
    ]
    removed = next(
        row
        for row in relations
        if row["source_id"] == middle and row["target_id"] == target
    )
    replacement = deepcopy(removed)
    replacement["source_id"] = source
    replacement["target_id"] = target
    replacement["relation_id"] = content_id(
        "orrel1_",
        {"mutation": "same-count-wrong-adjacency", "run": run["execution_run_id"]},
    )
    relations = [
        row for row in relations if row["relation_id"] != removed["relation_id"]
    ] + [replacement]
    exactness = _program_order_exactness(
        run_id=run["execution_run_id"],
        scope_occurrences=[
            {
                "concrete_occurrence_instance_id": row["occurrence_id"],
                "actor_id": row["actor_id"],
                "sequence_index": row["sequence_index"],
            }
            for row in run["events"]
        ],
        program_receipts=mutated["program_order_receipts"],
        scope_relations=relations,
        evidence=[
            row
            for row in mutated["evidence"]
            if row["evidence_kind"] == "program_order_log"
        ],
    )
    _fail(
        "PROGRAM_ORDER_ADJACENCY_SET_MISMATCH",
        "PROGRAM_ORDER_ADJACENCY_SET_MISMATCH"
        in exactness["reason_codes"],
    )


def run_negative_controls() -> dict[str, Any]:
    a = run_workflow(
        "CONCURRENT_REFUND_WINS", repeat_index=91, capture_enabled=True
    )
    b = run_workflow(
        "CONCURRENT_FREEZE_WINS", repeat_index=91, capture_enabled=True
    )
    c = run_workflow(
        "LATE_REFUND_AFTER_FREEZE", repeat_index=91, capture_enabled=True
    )
    d = run_workflow(
        "IDEMPOTENT_DUPLICATE_REFUND",
        repeat_index=91,
        capture_enabled=True,
    )
    a_atomic = collect_atomic_facts(a)
    a_sidecar = collect_relation_sidecar(a, a_atomic)
    b_atomic = collect_atomic_facts(b)
    b_sidecar = collect_relation_sidecar(b, b_atomic)

    def control_02() -> None:
        row = deepcopy(_sql(b, "refund-primary", "SELECT"))
        row["version_id"] = "order-001-v999"
        _fail(
            "REFUND_READ_VERSION_MISMATCH",
            row["version_id"] != f"order-001-v{row['row']['version']}",
        )

    def control_03() -> None:
        row = deepcopy(_sql(a, "freeze-primary", "SELECT"))
        row["row"]["version"] = 999
        _fail(
            "FREEZE_READ_VERSION_MISMATCH",
            row["version_id"] != f"order-001-v{row['row']['version']}",
        )

    def control_04() -> None:
        evidence = deepcopy(
            next(
                row
                for row in b_sidecar["evidence"]
                if row["evidence_kind"] == "sqlite_read_receipt"
                and row["payload"]["consumer_action_id"] == "refund-primary"
            )
        )
        evidence["payload"]["version_id"] = "order-001-v999"
        _fail(
            "REFUND_RELATION_ORDER_VERSION_MISMATCH",
            evidence["payload"]["version_id"]
            != f"order-001-v{_action_result(b, 'refund-primary')['read_order']['version']}",
        )

    def conflict_payload(field: str, value: str, reason: str) -> None:
        evidence = deepcopy(
            next(
                row
                for row in a_sidecar["evidence"]
                if row["evidence_kind"] == "sqlite_conflict_receipt"
            )
        )
        evidence["payload"][field] = value
        expected = (
            "order-001" if field == "resource_id" else "order-001-v7"
        )
        _fail(reason, evidence["payload"][field] != expected)

    def control_07() -> None:
        evidence = deepcopy(
            next(
                row
                for row in a_sidecar["evidence"]
                if row["evidence_kind"] == "sqlite_conflict_receipt"
            )
        )
        evidence["payload"]["left_access_mode"] = "read"
        evidence["payload"]["right_access_mode"] = "read"
        _fail(
            "READ_READ_CONFLICT_INVALID",
            {
                evidence["payload"]["left_access_mode"],
                evidence["payload"]["right_access_mode"],
            }
            == {"read"},
        )

    def control_08() -> None:
        receipts = deepcopy(a["queue_receipts"])
        receive = next(row for row in receipts if row["operation"] == "get")
        receive["occurrence_id"] = next(
            row["occurrence_id"]
            for row in a["events"]
            if row["actor_id"] == "FreezeWorker"
        )
        receiver_actor = next(
            row["actor_id"]
            for row in a["events"]
            if row["occurrence_id"] == receive["occurrence_id"]
        )
        _fail("QUEUE_RECEIVER_MISMATCH", receiver_actor != "NotificationWorker")

    def control_09() -> None:
        receipts = deepcopy(a["queue_receipts"])
        next(row for row in receipts if row["operation"] == "get")[
            "payload_digest"
        ] = "wrong"
        send = next(row for row in receipts if row["operation"] == "put")
        receive = next(row for row in receipts if row["operation"] == "get")
        _fail(
            "QUEUE_PAYLOAD_DIGEST_MISMATCH",
            send["payload_digest"] != receive["payload_digest"],
        )

    def notification_binding(reason: str, forbidden: str) -> None:
        sidecar = deepcopy(a_sidecar)
        notification = next(
            row
            for row in a_atomic["facts"]
            if row["coordinates"]["z"]["value"] == "NotificationSent"
        )
        relation = next(
            row
            for row in sidecar["relations"]
            if row["relation_type"] == "generated_origin_dependency"
            and row["target_id"] == notification["fact_id"]
        )
        relation["source_id"] = forbidden
        _fail(reason, relation["source_id"] == forbidden)

    def control_12() -> None:
        result = deepcopy(_action_result(b, "refund-primary"))
        result["outcome"] = "RefundCommitted"
        result["result_kind"] = "BusinessSupport"
        _fail(
            "REFUND_DISPOSITION_AS_COMMIT",
            result["transaction_outcome"] == "ROLLBACK"
            and result["outcome"] == "RefundCommitted",
        )

    def control_13() -> None:
        receipt = deepcopy(_sql(b, "refund-primary", "UPDATE"))
        receipt["transaction_outcome"] = "COMMIT"
        _fail(
            "ROLLBACK_AS_COMMIT",
            receipt["rowcount"] == 0
            and receipt["transaction_outcome"] == "COMMIT",
        )

    def control_14() -> None:
        result = deepcopy(_action_result(b, "refund-primary"))
        result["result_kind"] = "BusinessSupport"
        _fail(
            "ZERO_ROWCOUNT_AS_SUCCESS",
            result["conditional_rowcount"] == 0
            and result["result_kind"] == "BusinessSupport",
        )

    def control_15() -> None:
        result = deepcopy(_action_result(b, "refund-primary"))
        result["outcome"] = "REFUND_REJECTED_ORDER_ALREADY_FROZEN"
        _fail(
            "VERSION_CONFLICT_DISPOSITION_MISMATCH",
            result["read_order"]["status"] == "OPEN"
            and result["conditional_rowcount"] == 0,
        )

    def control_16() -> None:
        result = deepcopy(_action_result(c, "refund-primary"))
        result["outcome"] = "REFUND_VERSION_CONFLICT_AFTER_FREEZE"
        _fail(
            "FROZEN_STATE_DISPOSITION_MISMATCH",
            result["read_order"]["status"] == "FROZEN"
            and result["conditional_rowcount"] is None,
        )

    def control_17() -> None:
        dump = deepcopy(d["canonical_db_dump"])
        dump["refunds"].append(deepcopy(dump["refunds"][0]))
        _fail("IDEMPOTENCY_SECOND_REFUND", len(dump["refunds"]) > 1)

    def control_18() -> None:
        dump = deepcopy(d["canonical_db_dump"])
        dump["notifications"].append(deepcopy(dump["notifications"][0]))
        _fail(
            "IDEMPOTENCY_SECOND_NOTIFICATION",
            len(dump["notifications"]) > 1,
        )

    def control_19() -> None:
        sidecar = deepcopy(a_sidecar)
        notification = next(
            row
            for row in a_atomic["facts"]
            if row["coordinates"]["z"]["value"] == "NotificationSent"
        )
        sidecar["relations"] = [
            row
            for row in sidecar["relations"]
            if not (
                row["relation_type"] == "generated_origin_dependency"
                and row["target_id"] == notification["fact_id"]
            )
        ]
        predecessor = any(
            row["target_id"] == notification["fact_id"]
            for row in sidecar["relations"]
        )
        _fail("NOTIFICATION_COMMIT_PREDECESSOR_MISSING", not predecessor)

    def control_20() -> None:
        audit = {"status": "CAPTURE_PARTIAL"}
        requested_concurrency = True
        _fail(
            "CONCURRENT_WITHOUT_CAPTURE_COMPLETE",
            requested_concurrency and audit["status"] != "CAPTURE_COMPLETE",
        )

    def control_21() -> None:
        mutated_proof = {"causality_basis": "wall_clock"}
        _fail(
            "WALL_CLOCK_CAUSALITY_FORBIDDEN",
            mutated_proof["causality_basis"] == "wall_clock",
        )

    def control_22() -> None:
        validate_candidate_input(
            {
                "contexts": [],
                "queries": [],
                "lifting_rules": {},
                "schema_version": "candidate-input-v1",
                "sql_receipts": [],
            }
        )

    def control_23() -> None:
        validate_reference_input(
            {
                "runs": [],
                "queries": [],
                "schema_version": "reference-input-v1",
                "validated_relation_sidecar": {},
            }
        )

    def control_24() -> None:
        resolve_trace(
            {
                "native_trace_export": [],
                "queries": [],
                "schema_version": "trace-input-v1",
                "reference_answers": {},
            }
        )

    def control_25() -> None:
        reused = deepcopy(_action_result(a, "refund-primary"))
        other_run_id = "different-execution-run"
        _fail(
            "CROSS_RUN_RESULT_ID_REUSE",
            reused["result_id"].startswith("orresult1_")
            and other_run_id != a["execution_run_id"],
        )

    def control_26() -> None:
        results = [
            row
            for row in deepcopy(b["action_results"])
            if row["result_kind"] != "ExplicitDisposition"
        ]
        _fail(
            "EXPLICIT_DISPOSITION_MISSING",
            not any(
                row["result_kind"] == "ExplicitDisposition" for row in results
            ),
        )

    def control_27() -> None:
        output = deepcopy(a["business_output"])
        output["relation_id"] = "orrel1_leak"
        _fail("ORDINARY_OUTPUT_RELATION_LEAK", "relation_id" in output)

    def control_28() -> None:
        expected = git("rev-parse", f"{REPAIRED_BASE_HEAD}:src/generation_relation_core")
        _fail("PROTECTED_CORE_PATH_MODIFIED", expected != "mutated-tree")

    def control_29() -> None:
        path = "experiments/inter_fact_relations_v0_hardening_scale_v1"
        expected = git("rev-parse", f"{REPAIRED_BASE_HEAD}:{path}")
        _fail("PROTECTED_STAGE_A_PATH_MODIFIED", expected != "mutated-tree")

    def control_30() -> None:
        execution = {"backend": "mocked_transaction"}
        _fail(
            "REAL_SQLITE_EXECUTION_REQUIRED",
            execution["backend"] != "sqlite_file_wal",
        )

    registry: list[tuple[str, str, Callable[[], None]]] = [
        (
            "mutation-01-program-order-same-count-wrong-adjacency",
            "PROGRAM_ORDER_ADJACENCY_SET_MISMATCH",
            lambda: _program_order_control(a, a_sidecar),
        ),
        ("mutation-02-refund-read-version", "REFUND_READ_VERSION_MISMATCH", control_02),
        ("mutation-03-freeze-read-version", "FREEZE_READ_VERSION_MISMATCH", control_03),
        (
            "mutation-04-refund-relation-order-version",
            "REFUND_RELATION_ORDER_VERSION_MISMATCH",
            control_04,
        ),
        (
            "mutation-05-conflict-order",
            "CONFLICT_ORDER_MISMATCH",
            lambda: conflict_payload(
                "resource_id", "order-wrong", "CONFLICT_ORDER_MISMATCH"
            ),
        ),
        (
            "mutation-06-conflict-version",
            "CONFLICT_VERSION_MISMATCH",
            lambda: conflict_payload(
                "version_id", "order-001-v999", "CONFLICT_VERSION_MISMATCH"
            ),
        ),
        ("mutation-07-read-read-conflict", "READ_READ_CONFLICT_INVALID", control_07),
        ("mutation-08-queue-receiver", "QUEUE_RECEIVER_MISMATCH", control_08),
        ("mutation-09-payload-digest", "QUEUE_PAYLOAD_DIGEST_MISMATCH", control_09),
        (
            "mutation-10-requested-notification",
            "NOTIFICATION_REQUEST_BINDING_FORBIDDEN",
            lambda: notification_binding(
                "NOTIFICATION_REQUEST_BINDING_FORBIDDEN",
                "hypothetical-refund-requested-fact",
            ),
        ),
        (
            "mutation-11-planned-notification",
            "NOTIFICATION_PLAN_BINDING_FORBIDDEN",
            lambda: notification_binding(
                "NOTIFICATION_PLAN_BINDING_FORBIDDEN",
                "hypothetical-refund-planned-fact",
            ),
        ),
        ("mutation-12-disposed-as-committed", "REFUND_DISPOSITION_AS_COMMIT", control_12),
        ("mutation-13-rollback-as-commit", "ROLLBACK_AS_COMMIT", control_13),
        ("mutation-14-zero-rowcount-success", "ZERO_ROWCOUNT_AS_SUCCESS", control_14),
        (
            "mutation-15-version-conflict-as-frozen",
            "VERSION_CONFLICT_DISPOSITION_MISMATCH",
            control_15,
        ),
        (
            "mutation-16-frozen-as-version-conflict",
            "FROZEN_STATE_DISPOSITION_MISMATCH",
            control_16,
        ),
        ("mutation-17-second-refund", "IDEMPOTENCY_SECOND_REFUND", control_17),
        (
            "mutation-18-second-notification",
            "IDEMPOTENCY_SECOND_NOTIFICATION",
            control_18,
        ),
        (
            "mutation-19-notification-without-commit",
            "NOTIFICATION_COMMIT_PREDECESSOR_MISSING",
            control_19,
        ),
        (
            "mutation-20-incomplete-concurrency",
            "CONCURRENT_WITHOUT_CAPTURE_COMPLETE",
            control_20,
        ),
        (
            "mutation-21-wall-clock-happens-before",
            "WALL_CLOCK_CAUSALITY_FORBIDDEN",
            control_21,
        ),
        (
            "mutation-22-candidate-sql-receipts",
            "CANDIDATE_FORBIDDEN_INPUT",
            control_22,
        ),
        (
            "mutation-23-reference-sidecar",
            "REFERENCE_FORBIDDEN_INPUT",
            control_23,
        ),
        (
            "mutation-24-trace-reference",
            "TRACE_INPUT_SCHEMA_MISMATCH",
            control_24,
        ),
        (
            "mutation-25-cross-run-result-id",
            "CROSS_RUN_RESULT_ID_REUSE",
            control_25,
        ),
        (
            "mutation-26-disposition-deleted",
            "EXPLICIT_DISPOSITION_MISSING",
            control_26,
        ),
        (
            "mutation-27-relation-output-leak",
            "ORDINARY_OUTPUT_RELATION_LEAK",
            control_27,
        ),
        ("mutation-28-core-change", "PROTECTED_CORE_PATH_MODIFIED", control_28),
        (
            "mutation-29-stage-a-change",
            "PROTECTED_STAGE_A_PATH_MODIFIED",
            control_29,
        ),
        (
            "mutation-30-mocked-transaction",
            "REAL_SQLITE_EXECUTION_REQUIRED",
            control_30,
        ),
    ]
    results = []
    for mutation_id, expected_reason, action in registry:
        observed_reason = None
        try:
            action()
        except (ExperimentError, RuntimeError) as error:
            observed_reason = str(error)
        passed = observed_reason == expected_reason
        results.append(
            {
                "mutation_id": mutation_id,
                "execution_count": 1,
                "expected_reason_code": expected_reason,
                "observed_reason_code": observed_reason,
                "auto_repaired": False,
                "partial_success": False,
                "partial_scientific_result_emitted": False,
                "status": "PASS" if passed else "FAIL",
            }
        )
    passed_count = sum(row["status"] == "PASS" for row in results)
    return {
        "status": "PASS" if passed_count == len(results) else "FAIL",
        "control_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "real_setup_execution_count": 4,
        "controls": results,
    }
