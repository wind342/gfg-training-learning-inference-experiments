from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..common import ExperimentError, content_id


CAPTURE_COMPLETE = "CAPTURE_COMPLETE"
CAPTURE_PARTIAL = "CAPTURE_PARTIAL"
CAPTURE_CONFLICT = "CAPTURE_CONFLICT"
CAPTURE_NOT_ESTABLISHED = "CAPTURE_NOT_ESTABLISHED"

REQUIRED_CAPTURE_DECLARATIONS = (
    "program_order",
    "messages",
    "synchronization",
    "generated_origin",
    "reads_from",
    "resource_access",
)


def _edge_rows(edges: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"source_occurrence_id": source, "target_occurrence_id": target}
        for source, target in sorted(edges)
    ]


def _program_order_exactness(
    *,
    run_id: str,
    scope_occurrences: list[dict[str, Any]],
    program_receipts: list[dict[str, Any]],
    scope_relations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    occurrence_by_id = {
        row["concrete_occurrence_instance_id"]: row
        for row in scope_occurrences
    }
    by_actor: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in scope_occurrences:
        by_actor[occurrence["actor_id"]].append(occurrence)

    duplicate_sequence_indexes: list[dict[str, Any]] = []
    expected_edges: list[tuple[str, str]] = []
    for actor_id, actor_occurrences in sorted(by_actor.items()):
        sequence_counts = Counter(
            row["sequence_index"] for row in actor_occurrences
        )
        duplicate_sequence_indexes.extend(
            {
                "actor_id": actor_id,
                "sequence_index": sequence_index,
                "occurrence_count": count,
            }
            for sequence_index, count in sorted(sequence_counts.items())
            if count > 1
        )
        ordered = sorted(
            actor_occurrences,
            key=lambda row: (
                row["sequence_index"],
                row["concrete_occurrence_instance_id"],
            ),
        )
        expected_edges.extend(
            (
                source["concrete_occurrence_instance_id"],
                target["concrete_occurrence_instance_id"],
            )
            for source, target in zip(ordered, ordered[1:])
        )

    program_relations = [
        row
        for row in scope_relations
        if row["relation_type"] == "program_order"
    ]
    receipt_edges = [
        (row["source_occurrence_id"], row["target_occurrence_id"])
        for row in program_receipts
    ]
    relation_edges = [
        (row["source_id"], row["target_id"]) for row in program_relations
    ]

    evidence_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        evidence_groups[row["evidence_id"]].append(row)
    receipt_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in program_receipts:
        receipt_groups[row["receipt_id"]].append(row)

    evidence_edges: list[tuple[str, str]] = []
    evidence_reference_counts: Counter[str] = Counter()
    receipt_reference_counts: Counter[str] = Counter()
    binding_issues: list[dict[str, Any]] = []
    for relation in program_relations:
        edge = (relation["source_id"], relation["target_id"])
        refs = relation.get("evidence_refs", [])
        if len(refs) != 1:
            binding_issues.append(
                {
                    "relation_id": relation["relation_id"],
                    "issue": "RELATION_EVIDENCE_CARDINALITY",
                }
            )
            continue
        evidence_id = refs[0]
        evidence_reference_counts[evidence_id] += 1
        matching_evidence = evidence_groups.get(evidence_id, [])
        if len(matching_evidence) != 1:
            binding_issues.append(
                {
                    "relation_id": relation["relation_id"],
                    "evidence_id": evidence_id,
                    "issue": "EVIDENCE_ID_CARDINALITY",
                }
            )
            continue
        evidence_row = matching_evidence[0]
        evidence_edges.append(edge)
        receipt_ref = evidence_row.get("receipt_ref")
        receipt_reference_counts[receipt_ref] += 1
        matching_receipts = receipt_groups.get(receipt_ref, [])
        if len(matching_receipts) != 1:
            binding_issues.append(
                {
                    "relation_id": relation["relation_id"],
                    "evidence_id": evidence_id,
                    "receipt_ref": receipt_ref,
                    "issue": "RECEIPT_ID_CARDINALITY",
                }
            )
            continue
        receipt = matching_receipts[0]
        source = occurrence_by_id.get(edge[0])
        target = occurrence_by_id.get(edge[1])
        if source is None or target is None:
            binding_issues.append(
                {
                    "relation_id": relation["relation_id"],
                    "issue": "OCCURRENCE_ENDPOINT_UNKNOWN",
                }
            )
            continue
        field_checks = {
            "endpoint_set": (
                receipt["source_occurrence_id"],
                receipt["target_occurrence_id"],
            )
            == edge
            and sorted(evidence_row.get("occurrence_ids", []))
            == sorted(edge),
            "actor": (
                source["actor_id"]
                == target["actor_id"]
                == receipt.get("actor_id")
            ),
            "sequence": (
                source["sequence_index"]
                == receipt.get("source_sequence_index")
                and target["sequence_index"]
                == receipt.get("target_sequence_index")
            ),
            "run_id": (
                relation.get("execution_run_id")
                == evidence_row.get("execution_run_id")
                == receipt.get("execution_run_id")
                == run_id
            ),
            "authority": (
                relation.get("authority_id")
                == evidence_row.get("authority_id")
                == receipt.get("authority_id")
            ),
            "establishment_source": (
                relation.get("establishment_source")
                == evidence_row.get("establishment_source")
                == receipt.get("establishment_source")
            ),
        }
        failed_fields = sorted(
            name for name, passed in field_checks.items() if not passed
        )
        if failed_fields:
            binding_issues.append(
                {
                    "relation_id": relation["relation_id"],
                    "evidence_id": evidence_id,
                    "receipt_ref": receipt_ref,
                    "issue": "FIELD_BINDING_MISMATCH",
                    "failed_fields": failed_fields,
                }
            )

    binding_issues.extend(
        {
            "evidence_id": evidence_id,
            "issue": "EVIDENCE_REFERENCED_NOT_EXACTLY_ONCE",
            "reference_count": count,
        }
        for evidence_id, count in sorted(evidence_reference_counts.items())
        if count != 1
    )
    binding_issues.extend(
        {
            "receipt_ref": receipt_ref,
            "issue": "RECEIPT_REFERENCED_NOT_EXACTLY_ONCE",
            "reference_count": count,
        }
        for receipt_ref, count in sorted(receipt_reference_counts.items())
        if count != 1
    )
    unreferenced_program_evidence = sorted(
        row["evidence_id"]
        for row in evidence
        if row.get("evidence_kind") == "program_order_log"
        and set(row.get("occurrence_ids", [])) <= set(occurrence_by_id)
        and evidence_reference_counts[row["evidence_id"]] == 0
    )
    unreferenced_program_receipts = sorted(
        row["receipt_id"]
        for row in program_receipts
        if receipt_reference_counts[row["receipt_id"]] == 0
    )
    if unreferenced_program_evidence:
        binding_issues.append(
            {
                "issue": "UNREFERENCED_PROGRAM_ORDER_EVIDENCE",
                "evidence_ids": unreferenced_program_evidence,
            }
        )
    if unreferenced_program_receipts:
        binding_issues.append(
            {
                "issue": "UNREFERENCED_PROGRAM_ORDER_RECEIPT",
                "receipt_ids": unreferenced_program_receipts,
            }
        )

    expected_counter = Counter(expected_edges)
    receipt_counter = Counter(receipt_edges)
    relation_counter = Counter(relation_edges)
    evidence_counter = Counter(evidence_edges)
    expected_set = set(expected_edges)
    relation_set = set(relation_edges)
    missing_edges = expected_set - relation_set
    extra_edges = relation_set - expected_set

    reasons: list[str] = []
    if duplicate_sequence_indexes:
        reasons.append("PROGRAM_ORDER_SEQUENCE_INDEX_DUPLICATE")
    if relation_counter != expected_counter:
        reasons.append("PROGRAM_ORDER_ADJACENCY_SET_MISMATCH")
    if missing_edges:
        reasons.append("PROGRAM_ORDER_MISSING_EDGE")
    if extra_edges:
        reasons.append("PROGRAM_ORDER_EXTRA_EDGE")
    if (
        receipt_counter != relation_counter
        or evidence_counter != relation_counter
    ):
        reasons.append("PROGRAM_ORDER_RECEIPT_RELATION_SET_MISMATCH")
    if binding_issues:
        reasons.append("PROGRAM_ORDER_BINDING_NOT_ONE_TO_ONE")

    return {
        "reason_codes": reasons,
        "expected_edges": expected_edges,
        "receipt_edges": receipt_edges,
        "relation_edges": relation_edges,
        "evidence_edges": evidence_edges,
        "missing_edges": missing_edges,
        "extra_edges": extra_edges,
        "duplicate_sequence_indexes": duplicate_sequence_indexes,
        "binding_issues": binding_issues,
        "exact": not reasons,
    }


def _rows_for_scope(
    rows: list[dict[str, Any]],
    occurrence_scope: dict[str, str],
    occurrence_fields: tuple[str, ...],
    scope_id: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        endpoint_scopes = {
            occurrence_scope.get(row[field])
            for field in occurrence_fields
            if field in row
        }
        if endpoint_scopes == {scope_id}:
            selected.append(row)
    return selected


def audit_capture(
    declared_contract: dict[str, Any],
    runtime_receipts: dict[str, Any],
    validated_store: dict[str, Any],
) -> dict[str, Any]:
    if "allow_concurrency" in declared_contract:
        raise ExperimentError("CALLER_CONCURRENCY_OVERRIDE_FORBIDDEN")
    run_id = runtime_receipts["execution_run_id"]
    if declared_contract.get("execution_run_id") != run_id:
        raise ExperimentError("CAPTURE_CONTRACT_RUN_ID_MISMATCH")
    if validated_store.get("execution_run_id") != run_id:
        raise ExperimentError("CAPTURE_STORE_RUN_ID_MISMATCH")

    occurrences = runtime_receipts["occurrences"]
    occurrence_scope = {
        row["concrete_occurrence_instance_id"]: row["scope_id"]
        for row in occurrences
    }
    facts = runtime_receipts["facts"]
    fact_occurrence = {row["fact_id"]: row["occurrence_id"] for row in facts}
    relations = validated_store["primitive_relations"]
    evidence = validated_store["evidence"]
    evidence_index = {row["evidence_id"]: row for row in evidence}
    contract_scopes = {
        row["scope_id"]: row for row in declared_contract.get("scopes", [])
    }
    actual_scope_ids = sorted(set(occurrence_scope.values()))
    scope_audits: list[dict[str, Any]] = []

    for scope_id in sorted(set(actual_scope_ids) | set(contract_scopes)):
        contract = contract_scopes.get(scope_id)
        if contract is None:
            scope_audits.append(
                {
                    "scope_id": scope_id,
                    "status": CAPTURE_NOT_ESTABLISHED,
                    "reason_codes": ["CAPTURE_CONTRACT_SCOPE_MISSING"],
                    "concurrency_inference_allowed": False,
                    "scheduler_completeness_basis": (
                        "DECLARED_CONTROLLED_EXECUTOR_PROFILE"
                    ),
                    "global_scheduler_completeness_machine_proved": False,
                    "concurrency_scope": "CONTROLLED_CAPTURE_SCOPE_ONLY",
                    "program_order_exactness": {
                        "status": "FAIL",
                        "expected_adjacent_edges": [],
                        "receipt_edges": [],
                        "relation_edges": [],
                        "evidence_edges": [],
                        "missing_edges": [],
                        "extra_edges": [],
                        "duplicate_sequence_indexes": [],
                        "binding_issues": [
                            {"issue": "CAPTURE_CONTRACT_SCOPE_MISSING"}
                        ],
                        "receipt_relation_evidence_one_to_one": False,
                    },
                }
            )
            continue

        scope_occurrences = [
            row for row in occurrences if row["scope_id"] == scope_id
        ]
        occurrence_ids = {
            row["concrete_occurrence_instance_id"] for row in scope_occurrences
        }
        actor_counts = Counter(row["actor_id"] for row in scope_occurrences)
        scope_facts = [
            row for row in facts if row["occurrence_id"] in occurrence_ids
        ]
        scope_fact_ids = {row["fact_id"] for row in scope_facts}

        relation_counts: Counter[str] = Counter()
        scope_relations: list[dict[str, Any]] = []
        for relation in relations:
            if relation["endpoint_level"] == "occurrence":
                relation_occurrences = {
                    relation["source_id"],
                    relation["target_id"],
                }
            else:
                relation_occurrences = {
                    fact_occurrence[relation["source_id"]],
                    fact_occurrence[relation["target_id"]],
                }
            if relation_occurrences <= occurrence_ids:
                scope_relations.append(relation)
                relation_counts[relation["relation_type"]] += 1

        program_receipts = _rows_for_scope(
            runtime_receipts.get("program_order_receipts", []),
            occurrence_scope,
            ("source_occurrence_id", "target_occurrence_id"),
            scope_id,
        )
        message_receipts = _rows_for_scope(
            runtime_receipts.get("message_receipts", []),
            occurrence_scope,
            ("send_occurrence_id", "receive_occurrence_id"),
            scope_id,
        )
        synchronization_receipts = [
            row
            for row in runtime_receipts.get("synchronization_receipts", [])
            if row["release_occurrence_id"] in occurrence_ids
            and set(row["pre_occurrence_ids"]) <= occurrence_ids
        ]
        generated_origin_receipts = [
            row
            for row in runtime_receipts.get("generated_origin_receipts", [])
            if row["producer_fact_id"] in scope_fact_ids
            and row["consumer_fact_id"] in scope_fact_ids
        ]
        access_receipts = [
            row
            for row in runtime_receipts.get("resource_access_receipts", [])
            if row["occurrence_id"] in occurrence_ids
        ]
        reads_from_receipts = [
            row
            for row in runtime_receipts.get("reads_from_receipts", [])
            if row["source_fact_id"] in scope_fact_ids
            and row["target_fact_id"] in scope_fact_ids
        ]
        conflict_receipts = [
            row
            for row in runtime_receipts.get("conflict_receipts", [])
            if row["left_fact_id"] in scope_fact_ids
            and row["right_fact_id"] in scope_fact_ids
        ]
        program_order_exactness = _program_order_exactness(
            run_id=run_id,
            scope_occurrences=scope_occurrences,
            program_receipts=program_receipts,
            scope_relations=scope_relations,
            evidence=evidence,
        )
        expected_program_order = len(
            program_order_exactness["expected_edges"]
        )

        unknown_edges = [
            row
            for row in runtime_receipts.get("unknown_edges", [])
            if row.get("scope_id") == scope_id
        ]
        unclassified_messages = [
            row
            for row in runtime_receipts.get("unclassified_messages", [])
            if row.get("scope_id") == scope_id
        ]
        unclassified_operations = [
            row
            for row in runtime_receipts.get("unclassified_operations", [])
            if row.get("scope_id") == scope_id
        ]
        external_communication = [
            row
            for row in runtime_receipts.get("external_communications", [])
            if row.get("scope_id") == scope_id
        ]
        unclassified_sync = [
            row
            for row in runtime_receipts.get(
                "unclassified_synchronization_operations", []
            )
            if row.get("scope_id") == scope_id
        ]
        unclassified_access = [
            row
            for row in runtime_receipts.get("unclassified_resource_accesses", [])
            if row.get("scope_id") == scope_id
        ]

        bound_evidence_ids = {
            evidence_id
            for relation in scope_relations
            for evidence_id in relation["evidence_refs"]
        }
        unbound_evidence = [
            row
            for row in evidence
            if (
                (
                    bool(row["occurrence_ids"])
                    and set(row["occurrence_ids"]) <= occurrence_ids
                )
                or (
                    not row["occurrence_ids"]
                    and bool(row["fact_ids"])
                    and set(row["fact_ids"]) <= scope_fact_ids
                )
            )
            and row["evidence_id"] not in bound_evidence_ids
        ]
        covered_by_evidence = {
            occurrence_id
            for evidence_id in bound_evidence_ids
            for occurrence_id in evidence_index[evidence_id]["occurrence_ids"]
        }
        uncovered_occurrences = sorted(occurrence_ids - covered_by_evidence)
        # Isolated occurrences are explicitly covered by executor receipts.
        executor_covered = {
            row["occurrence_id"]
            for row in runtime_receipts.get("executor_coverage_receipts", [])
            if row["occurrence_id"] in occurrence_ids
        }
        uncovered_occurrences = sorted(
            set(uncovered_occurrences) - executor_covered
        )

        declared_set = set(contract.get("covered_occurrence_ids", []))
        coverage_exact = declared_set == occurrence_ids
        planned = contract.get("planned_capture", {})
        contract_satisfied = all(
            planned.get(name) is True for name in REQUIRED_CAPTURE_DECLARATIONS
        )
        expected_sync_edges = sum(
            len(row["pre_occurrence_ids"]) for row in synchronization_receipts
        )
        expected_counts = {
            "program_order": expected_program_order,
            "message_send_receive": len(message_receipts),
            "synchronizes_with": expected_sync_edges,
            "generated_origin_dependency": len(generated_origin_receipts),
            "reads_from": len(reads_from_receipts),
            "conflicts_with": len(conflict_receipts),
        }
        count_mismatches = {
            key: {
                "expected": expected,
                "observed": relation_counts.get(key, 0),
            }
            for key, expected in expected_counts.items()
            if relation_counts.get(key, 0) != expected
        }
        duplicate_relation_ids = len(scope_relations) - len(
            {row["relation_id"] for row in scope_relations}
        )
        semantic_pair_counts = Counter(
            (
                row["relation_type"],
                row["source_id"],
                row["target_id"],
                tuple(row["evidence_refs"]),
            )
            for row in scope_relations
        )
        duplicate_edges = sum(
            count - 1 for count in semantic_pair_counts.values() if count > 1
        )
        ambiguous_message_ids = sum(
            count - 1
            for count in Counter(
                row["message_id"] for row in message_receipts
            ).values()
            if count > 1
        )

        reasons: list[str] = []
        conflict_reasons: list[str] = []
        if not contract_satisfied:
            reasons.append("CAPTURE_DECLARED_CONTRACT_UNSATISFIED")
        if not coverage_exact:
            reasons.append("CAPTURE_OCCURRENCE_COVERAGE_INCOMPLETE")
        if count_mismatches:
            reasons.append("CAPTURE_EXPECTED_EDGE_COUNT_MISMATCH")
        reasons.extend(program_order_exactness["reason_codes"])
        if unknown_edges:
            reasons.append("CAPTURE_UNKNOWN_EDGE_PRESENT")
        if unclassified_messages:
            reasons.append("CAPTURE_UNCLASSIFIED_MESSAGE")
        if unclassified_operations:
            reasons.append("CAPTURE_UNCLASSIFIED_OPERATION")
        if unclassified_sync:
            reasons.append("CAPTURE_UNCLASSIFIED_SYNCHRONIZATION")
        if unclassified_access:
            reasons.append("CAPTURE_UNCLASSIFIED_RESOURCE_ACCESS")
        if external_communication:
            reasons.append("CAPTURE_EXTERNAL_COMMUNICATION_PRESENT")
        if unbound_evidence:
            reasons.append("CAPTURE_UNBOUND_EVIDENCE")
        if uncovered_occurrences:
            reasons.append("CAPTURE_UNCOVERED_OCCURRENCE")
        if not contract.get("external_communication_absent", False):
            reasons.append("CAPTURE_EXTERNAL_ABSENCE_NOT_DECLARED")
        if not contract.get(
            "unobserved_scheduler_relation_ruled_out", False
        ):
            reasons.append("CAPTURE_SCHEDULER_RELATION_NOT_RULED_OUT")
        if duplicate_relation_ids or duplicate_edges:
            conflict_reasons.append("CAPTURE_DUPLICATE_EDGE")
        if ambiguous_message_ids:
            conflict_reasons.append("CAPTURE_AMBIGUOUS_MESSAGE_PAIRING")

        if conflict_reasons:
            status = CAPTURE_CONFLICT
        elif reasons:
            status = CAPTURE_PARTIAL
        else:
            status = CAPTURE_COMPLETE
        reason_codes = sorted(set([*reasons, *conflict_reasons]))
        scope_audits.append(
            {
                "scope_id": scope_id,
                "status": status,
                "reason_codes": reason_codes,
                "declared_contract_satisfied": contract_satisfied,
                "measured_audit_complete": status == CAPTURE_COMPLETE,
                "concurrency_inference_allowed": status == CAPTURE_COMPLETE,
                "scheduler_completeness_basis": (
                    "DECLARED_CONTROLLED_EXECUTOR_PROFILE"
                ),
                "global_scheduler_completeness_machine_proved": False,
                "concurrency_scope": "CONTROLLED_CAPTURE_SCOPE_ONLY",
                "covered_occurrence_set_exact": coverage_exact,
                "program_order_exactness": {
                    "status": (
                        "PASS"
                        if program_order_exactness["exact"]
                        else "FAIL"
                    ),
                    "expected_adjacent_edges": _edge_rows(
                        set(program_order_exactness["expected_edges"])
                    ),
                    "receipt_edges": _edge_rows(
                        set(program_order_exactness["receipt_edges"])
                    ),
                    "relation_edges": _edge_rows(
                        set(program_order_exactness["relation_edges"])
                    ),
                    "evidence_edges": _edge_rows(
                        set(program_order_exactness["evidence_edges"])
                    ),
                    "missing_edges": _edge_rows(
                        program_order_exactness["missing_edges"]
                    ),
                    "extra_edges": _edge_rows(
                        program_order_exactness["extra_edges"]
                    ),
                    "duplicate_sequence_indexes": (
                        program_order_exactness[
                            "duplicate_sequence_indexes"
                        ]
                    ),
                    "binding_issues": program_order_exactness[
                        "binding_issues"
                    ],
                    "receipt_relation_evidence_one_to_one": not (
                        program_order_exactness["binding_issues"]
                    ),
                },
                "counts": {
                    "actor_count": len(actor_counts),
                    "events_by_actor": dict(sorted(actor_counts.items())),
                    "occurrence_count": len(scope_occurrences),
                    "fact_count": len(scope_facts),
                    "program_order_adjacent_expected": expected_program_order,
                    "program_order_receipt_count": len(program_receipts),
                    "message_send_count": len(message_receipts),
                    "message_receive_count": len(message_receipts),
                    "message_matched_count": relation_counts.get(
                        "message_send_receive", 0
                    ),
                    "synchronization_operation_count": len(
                        synchronization_receipts
                    ),
                    "generated_origin_entity_count": len(
                        generated_origin_receipts
                    ),
                    "generated_origin_edge_count": relation_counts.get(
                        "generated_origin_dependency", 0
                    ),
                    "write_count": sum(
                        row["access_mode"] == "write" for row in access_receipts
                    ),
                    "read_count": sum(
                        row["access_mode"] == "read" for row in access_receipts
                    ),
                    "reads_from_count": relation_counts.get("reads_from", 0),
                    "conflict_count": relation_counts.get("conflicts_with", 0),
                    "external_communication_count": len(external_communication),
                    "unknown_edge_count": len(unknown_edges),
                    "unclassified_message_count": len(unclassified_messages),
                    "unclassified_operation_count": len(
                        unclassified_operations
                    ),
                    "unclassified_synchronization_count": len(unclassified_sync),
                    "unclassified_resource_access_count": len(
                        unclassified_access
                    ),
                    "unbound_evidence_count": len(unbound_evidence),
                    "uncovered_occurrence_count": len(uncovered_occurrences),
                    "duplicate_edge_count": duplicate_edges,
                    "ambiguous_edge_count": ambiguous_message_ids,
                },
                "expected_observed_mismatches": count_mismatches,
            }
        )

    statuses = {row["status"] for row in scope_audits}
    if CAPTURE_CONFLICT in statuses:
        overall = CAPTURE_CONFLICT
    elif statuses == {CAPTURE_COMPLETE}:
        overall = CAPTURE_COMPLETE
    elif CAPTURE_NOT_ESTABLISHED in statuses and len(statuses) == 1:
        overall = CAPTURE_NOT_ESTABLISHED
    else:
        overall = CAPTURE_PARTIAL
    material = {
        "execution_run_id": run_id,
        "declared_contract_id": declared_contract["contract_id"],
        "overall_status": overall,
        "scopes": scope_audits,
        "schema_version": "capture-completeness-audit-v1",
    }
    return {
        "capture_audit_id": content_id("capaudit1_", material),
        **material,
    }
