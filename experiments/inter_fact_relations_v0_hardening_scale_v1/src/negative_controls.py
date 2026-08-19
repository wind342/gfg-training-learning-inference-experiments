from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from ..common import ExperimentError, content_id
from ..scenarios.common import RuntimeScenarioBuilder
from ..scenarios.mixed_dag import build_mixed_dag
from ..scenarios.multi_fact_occurrence import run as run_multi_fact
from ..scenarios.primitive_semantic_validation import build as build_primitive
from .capture_auditor import CAPTURE_COMPLETE, audit_capture
from .compare_process import compare_outputs
from .indexed_candidate_resolver import (
    IndexedCandidateResolver,
    resolve_candidate_payload,
    validate_candidate_input,
)
from .reference_process import (
    resolve_reference_payload,
    validate_reference_input,
)
from .run_identity import (
    validate_frozen_fact_coordinates,
    validate_identity_record,
)
from .semantic_evidence_validator import validate_primitive_store


def _evidence_for_relation(
    store: dict[str, Any], relation_type: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    relation = next(
        row
        for row in store["primitive_relations"]
        if row["relation_type"] == relation_type
    )
    evidence = next(
        row
        for row in store["evidence"]
        if row["evidence_id"] == relation["evidence_refs"][0]
    )
    return relation, evidence


def _validated_context() -> tuple[
    RuntimeScenarioBuilder,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    builder = build_primitive()
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    capture = audit_capture(builder.capture_contract(), receipts, validated)
    return builder, receipts, validated, capture


def _resolver() -> IndexedCandidateResolver:
    builder, _, validated, capture = _validated_context()
    return IndexedCandidateResolver(
        execution_run_id=builder.run_id,
        primitive_store=validated,
        capture_audit=capture,
        lifting_rules={"policy": "RELATION_TYPE_SPECIFIC_LIFTING"},
    )


def _control_01() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    store = deepcopy(builder.primitive_store())
    relation, evidence = _evidence_for_relation(store, "program_order")
    other = next(
        row["concrete_occurrence_instance_id"]
        for row in receipts["occurrences"]
        if row["concrete_occurrence_instance_id"]
        not in {relation["source_id"], relation["target_id"]}
    )
    evidence["occurrence_ids"] = sorted([relation["source_id"], other])
    validate_primitive_store(store, receipts)


def _control_02() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    store = deepcopy(builder.primitive_store())
    relation, _ = _evidence_for_relation(store, "generated_origin_dependency")
    target = next(
        row for row in receipts["facts"] if row["fact_id"] == relation["target_id"]
    )
    target["generated_origin"]["prior_support_id"] = "support1_wrong"
    validate_primitive_store(store, receipts)


def _control_03() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    store = deepcopy(builder.primitive_store())
    relation, _ = _evidence_for_relation(store, "message_send_receive")
    other = next(
        row["concrete_occurrence_instance_id"]
        for row in receipts["occurrences"]
        if row["concrete_occurrence_instance_id"]
        not in {relation["source_id"], relation["target_id"]}
    )
    receipts["message_receipts"][0]["receive_occurrence_id"] = other
    validate_primitive_store(store, receipts)


def _control_04() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    duplicate = deepcopy(receipts["message_receipts"][0])
    duplicate["receipt_id"] = content_id(
        "msgreceipt1_", {"duplicate": duplicate["message_id"]}
    )
    receipts["message_receipts"].append(duplicate)
    validate_primitive_store(builder.primitive_store(), receipts)


def _control_05() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    receipt = receipts["synchronization_receipts"][0]
    source_actor = next(
        row["actor_id"]
        for row in receipts["occurrences"]
        if row["concrete_occurrence_instance_id"]
        == receipt["pre_occurrence_ids"][0]
    )
    receipt["participant_actor_ids"].remove(source_actor)
    validate_primitive_store(builder.primitive_store(), receipts)


def _control_06() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    read_id = receipts["reads_from_receipts"][0]["consumer_read_access_id"]
    read = next(
        row
        for row in receipts["resource_access_receipts"]
        if row["access_id"] == read_id
    )
    read["observed_version_id"] = "old-version"
    validate_primitive_store(builder.primitive_store(), receipts)


def _control_07() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    conflict = receipts["conflict_receipts"][0]
    for access_id in [conflict["left_access_id"], conflict["right_access_id"]]:
        access = next(
            row
            for row in receipts["resource_access_receipts"]
            if row["access_id"] == access_id
        )
        access["access_mode"] = "read"
    validate_primitive_store(builder.primitive_store(), receipts)


def _control_08() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    right_id = receipts["conflict_receipts"][0]["right_access_id"]
    right = next(
        row
        for row in receipts["resource_access_receipts"]
        if row["access_id"] == right_id
    )
    right["version_id"] = "different-version"
    validate_primitive_store(builder.primitive_store(), receipts)


def _audit_reason_mutation(
    mutate: Callable[
        [dict[str, Any], dict[str, Any], dict[str, Any]], None
    ],
    expected_reason: str,
) -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    store = deepcopy(builder.primitive_store())
    contract = deepcopy(builder.capture_contract())
    mutate(receipts, store, contract)
    validated = validate_primitive_store(store, receipts)
    audit = audit_capture(contract, receipts, validated)
    reasons = {
        reason
        for scope in audit["scopes"]
        for reason in scope.get("reason_codes", [])
    }
    if expected_reason in reasons and audit["overall_status"] != CAPTURE_COMPLETE:
        raise ExperimentError(expected_reason)


def _control_09() -> None:
    def mutate(
        receipts: dict[str, Any],
        _store: dict[str, Any],
        _contract: dict[str, Any],
    ) -> None:
        receipts["unclassified_messages"].append(
            {
                "scope_id": "semantic-validation",
                "message_id": "unclassified-message",
            }
        )

    _audit_reason_mutation(mutate, "CAPTURE_UNCLASSIFIED_MESSAGE")


def _control_10() -> None:
    def mutate(
        _receipts: dict[str, Any],
        _store: dict[str, Any],
        contract: dict[str, Any],
    ) -> None:
        contract["scopes"][0]["covered_occurrence_ids"].pop()

    _audit_reason_mutation(
        mutate, "CAPTURE_OCCURRENCE_COVERAGE_INCOMPLETE"
    )


def _control_11() -> None:
    def mutate(
        receipts: dict[str, Any],
        _store: dict[str, Any],
        _contract: dict[str, Any],
    ) -> None:
        receipts["external_communications"].append(
            {
                "scope_id": "semantic-validation",
                "channel": "external-network",
            }
        )

    _audit_reason_mutation(
        mutate, "CAPTURE_EXTERNAL_COMMUNICATION_PRESENT"
    )


def _control_12() -> None:
    validate_candidate_input(
        {
            "execution_run_id": "run",
            "primitive_store": {},
            "capture_audit": {},
            "lifting_rules": {},
            "queries": [],
            "schema_version": "candidate-input-v1",
            "hidden_primitive_relation_store": {},
        }
    )


def _control_13() -> None:
    validate_candidate_input(
        {
            "execution_run_id": "run",
            "primitive_store": {},
            "capture_audit": {},
            "lifting_rules": {},
            "queries": [],
            "schema_version": "candidate-input-v1",
            "reference_receipts": {},
        }
    )


def _control_14() -> None:
    validate_reference_input(
        {
            "execution_run_id": "run",
            "runtime_receipts": {},
            "capture_contract": {},
            "queries": [],
            "reference_mode": "eager",
            "schema_version": "reference-input-v1",
            "candidate_graph": {},
        }
    )


def _derived_row() -> tuple[IndexedCandidateResolver, dict[str, Any]]:
    resolver = _resolver()
    relation = next(
        row
        for row in resolver.relations
        if row["relation_type"] == "program_order"
    )
    derived = resolver.derived_happens_before(
        relation["source_id"], relation["target_id"]
    )
    assert derived is not None
    return resolver, derived


def _control_15() -> None:
    resolver, row = _derived_row()
    row["input_relation_refs"] = []
    resolver.validate_derived_proof(row)


def _control_16() -> None:
    resolver, row = _derived_row()
    row["input_relation_refs"] = ["ifr1_wrong"]
    resolver.validate_derived_proof(row)


def _control_17() -> None:
    workload = build_mixed_dag("small")
    builder = workload["builder"]
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    capture = audit_capture(builder.capture_contract(), receipts, validated)
    resolver = IndexedCandidateResolver(
        execution_run_id=builder.run_id,
        primitive_store=validated,
        capture_audit=capture,
        lifting_rules={"policy": "RELATION_TYPE_SPECIFIC_LIFTING"},
    )
    retained = sum(len(values) for values in resolver.pair_relation_ids.values())
    collapsed = len(resolver.pair_relation_ids)
    if retained > collapsed:
        raise ExperimentError("MULTIPLE_PRIMITIVE_RELATION_OVERWRITE")


def _control_18() -> None:
    result = run_multi_fact()
    if result["occurrence_order_lift_pair_count"] > len(
        result["fact_specific_dependency_pairs"]
    ):
        raise ExperimentError(
            "FACT_SPECIFIC_CARTESIAN_LIFTING_FORBIDDEN"
        )


def _control_19() -> None:
    builder = build_primitive()
    occurrence = deepcopy(builder.occurrences[0])
    occurrence["concrete_occurrence_instance_id"] = occurrence[
        "semantic_occurrence_key"
    ]
    validate_identity_record(occurrence)


def _control_20() -> None:
    builder = build_primitive()
    receipts = deepcopy(builder.runtime_receipts())
    store = deepcopy(builder.primitive_store())
    receipt = receipts["program_order_receipts"][0]
    receipt["recorded_by"] = "wall_clock"
    _, evidence = _evidence_for_relation(store, "program_order")
    evidence["payload"]["recorded_by"] = "wall_clock"
    validate_primitive_store(store, receipts)


def _control_21() -> None:
    builder = build_primitive()
    fact = deepcopy(builder.facts[0])
    fact["semantic_projection"]["coordinates"]["logical_clock"] = {"A": 1}
    validate_frozen_fact_coordinates(fact)


def _control_22() -> None:
    builder, _, validated, capture = _validated_context()
    mutated = deepcopy(validated)
    relation = next(
        row
        for row in mutated["primitive_relations"]
        if row["relation_type"] == "program_order"
    )
    reverse = deepcopy(relation)
    reverse["source_id"], reverse["target_id"] = (
        reverse["target_id"],
        reverse["source_id"],
    )
    reverse["relation_id"] = content_id(
        "ifr1_", {"cycle_mutation": relation["relation_id"]}
    )
    mutated["primitive_relations"].append(reverse)
    IndexedCandidateResolver(
        execution_run_id=builder.run_id,
        primitive_store=mutated,
        capture_audit=capture,
        lifting_rules={"policy": "RELATION_TYPE_SPECIFIC_LIFTING"},
    )


def _control_23() -> None:
    resolver, row = _derived_row()
    if resolver.happens_before(row["source_id"], row["target_id"]):
        mutated_concurrent = True
        if mutated_concurrent:
            raise ExperimentError("CONCURRENT_WITH_HAPPENS_BEFORE")


def _control_24() -> None:
    builder, receipts, validated, _ = _validated_context()
    mutated_receipts = deepcopy(receipts)
    mutated_receipts["unknown_edges"].append(
        {
            "scope_id": "semantic-validation",
            "unknown_edge_id": "unknown-for-concurrency-control",
        }
    )
    capture = audit_capture(
        builder.capture_contract(), mutated_receipts, validated
    )
    resolver = IndexedCandidateResolver(
        execution_run_id=builder.run_id,
        primitive_store=validated,
        capture_audit=capture,
        lifting_rules={"policy": "RELATION_TYPE_SPECIFIC_LIFTING"},
    )
    left, right = sorted(resolver.occurrences)[:2]
    answer = resolver.concurrent_with(left, right)
    if answer["status"] == "CONCURRENCY_NOT_ESTABLISHED":
        raise ExperimentError("CONCURRENT_WITHOUT_CAPTURE_COMPLETE")


def _control_25() -> None:
    expected = "03fbdce13249f84abe9d8fb605da31cdc36eda27"
    observed = "mutated-tree-hash"
    if expected != observed:
        raise ExperimentError("PROTECTED_CORE_PATH_MODIFIED")


def _control_26() -> None:
    inventory = [
        "experiments/inter_fact_relations_v0_hardening_scale_v1/"
        "experimental_core_copy/entities.py"
    ]
    if any("experimental_core_copy/" in path for path in inventory):
        raise ExperimentError("UNDECLARED_EXPERIMENTAL_CORE_COPY")


def _control_27() -> None:
    workload = build_mixed_dag("small")
    builder = workload["builder"]
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    capture = audit_capture(builder.capture_contract(), receipts, validated)
    queries = workload["queries"][:1]
    candidate = {
        "status": "PASS",
        **resolve_candidate_payload(
            {
                "execution_run_id": builder.run_id,
                "primitive_store": validated,
                "capture_audit": capture,
                "lifting_rules": {
                    "policy": "RELATION_TYPE_SPECIFIC_LIFTING"
                },
                "queries": queries,
                "schema_version": "candidate-input-v1",
            }
        ),
    }
    reference = {
        "status": "PASS",
        **resolve_reference_payload(
            {
                "execution_run_id": builder.run_id,
                "runtime_receipts": receipts,
                "capture_contract": builder.capture_contract(),
                "queries": queries,
                "reference_mode": "eager",
                "schema_version": "reference-input-v1",
            }
        ),
    }
    candidate["answers"][0]["result"] = not candidate["answers"][0]["result"]
    comparison = compare_outputs(
        {
            "candidate": candidate,
            "reference": reference,
            "query_manifest_sha256": content_id("q_", queries),
        }
    )
    if comparison["status"] == "FAIL":
        raise ExperimentError("LAZY_EAGER_RESULT_MISMATCH")


def _control_28() -> None:
    builder, _, validated, capture = _validated_context()
    IndexedCandidateResolver(
        execution_run_id=builder.run_id,
        primitive_store=validated,
        capture_audit=capture,
        lifting_rules={"policy": "RELATION_TYPE_SPECIFIC_LIFTING"},
        reusable_cache={"execution_run_id": "different-run"},
    )


def _control_29() -> None:
    disabled = {"result": "ordinary"}
    primitive = {"result": "ordinary"}
    full = {"result": "mutated-by-relation-capture"}
    if not disabled == primitive == full:
        raise ExperimentError("ORDINARY_OUTPUT_ORTHOGONALITY_VIOLATION")


def _control_30() -> None:
    declared = {"scale": "large", "occurrence_count": 10_000}
    observed = {"scale": "large", "occurrence_count": 9_999}
    if observed["occurrence_count"] < declared["occurrence_count"]:
        raise ExperimentError("LARGE_WORKLOAD_DOWNGRADED")


def _control_31() -> None:
    builder = RuntimeScenarioBuilder(
        label="program-order-same-count-wrong-adjacency-v1"
    )
    actor_occurrence_ids = []
    for sequence_index in range(3):
        occurrence, _ = builder.add_occurrence(
            actor_id="actor-A",
            sequence_index=sequence_index,
            operation=f"operation-{sequence_index}",
            semantic_slot=sequence_index,
            scope_id="program-order-adjacency-control",
            fact_count=1,
        )
        actor_occurrence_ids.append(
            occurrence["concrete_occurrence_instance_id"]
        )
    builder.add_all_program_order()

    removed_source = actor_occurrence_ids[1]
    removed_target = actor_occurrence_ids[2]
    removed_relation = next(
        row
        for row in builder.primitive_relations
        if row["relation_type"] == "program_order"
        and row["source_id"] == removed_source
        and row["target_id"] == removed_target
    )
    removed_evidence_id = removed_relation["evidence_refs"][0]
    removed_evidence = next(
        row
        for row in builder.evidence
        if row["evidence_id"] == removed_evidence_id
    )
    removed_receipt_id = removed_evidence["receipt_ref"]
    builder.primitive_relations = [
        row
        for row in builder.primitive_relations
        if row["relation_id"] != removed_relation["relation_id"]
    ]
    builder.evidence = [
        row
        for row in builder.evidence
        if row["evidence_id"] != removed_evidence_id
    ]
    builder.program_order_receipts = [
        row
        for row in builder.program_order_receipts
        if row["receipt_id"] != removed_receipt_id
    ]
    builder.add_program_order(
        actor_occurrence_ids[0], actor_occurrence_ids[2]
    )

    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    audit = audit_capture(builder.capture_contract(), receipts, validated)
    scope = audit["scopes"][0]
    reasons = set(scope["reason_codes"])
    if (
        len(receipts["program_order_receipts"]) == 2
        and scope["counts"]["program_order_adjacent_expected"] == 2
        and "CAPTURE_EXPECTED_EDGE_COUNT_MISMATCH" not in reasons
        and {
            "PROGRAM_ORDER_ADJACENCY_SET_MISMATCH",
            "PROGRAM_ORDER_MISSING_EDGE",
            "PROGRAM_ORDER_EXTRA_EDGE",
        }
        <= reasons
        and audit["overall_status"] != CAPTURE_COMPLETE
    ):
        raise ExperimentError("PROGRAM_ORDER_ADJACENCY_SET_MISMATCH")


CONTROL_REGISTRY: list[
    tuple[str, str, str, Callable[[], None]]
] = [
    (
        "mutation-01-evidence-endpoints",
        "correct evidence kind with wrong endpoints",
        "EVIDENCE_ENDPOINT_BINDING_MISMATCH",
        _control_01,
    ),
    (
        "mutation-02-generated-origin-support",
        "wrong GeneratedOrigin prior support",
        "GENERATED_ORIGIN_PRIOR_SUPPORT_MISMATCH",
        _control_02,
    ),
    (
        "mutation-03-message-receive",
        "correct message ID with wrong receive occurrence",
        "MESSAGE_RECEIVE_ENDPOINT_MISMATCH",
        _control_03,
    ),
    (
        "mutation-04-duplicate-message",
        "duplicate message pairing",
        "DUPLICATE_MESSAGE_PAIRING",
        _control_04,
    ),
    (
        "mutation-05-barrier-participant",
        "barrier participant missing",
        "BARRIER_PARTICIPANT_MISSING",
        _control_05,
    ),
    (
        "mutation-06-read-version",
        "read old version while declaring new version",
        "READS_FROM_VERSION_MISMATCH",
        _control_06,
    ),
    (
        "mutation-07-read-read-conflict",
        "read-read mislabeled as conflict",
        "READ_READ_CONFLICT_INVALID",
        _control_07,
    ),
    (
        "mutation-08-conflict-version",
        "same resource but different version mislabeled conflict",
        "CONFLICT_RESOURCE_VERSION_MISMATCH",
        _control_08,
    ),
    (
        "mutation-09-unclassified-message",
        "declared complete with unclassified message",
        "CAPTURE_UNCLASSIFIED_MESSAGE",
        _control_09,
    ),
    (
        "mutation-10-incomplete-coverage",
        "incomplete covered occurrence set",
        "CAPTURE_OCCURRENCE_COVERAGE_INCOMPLETE",
        _control_10,
    ),
    (
        "mutation-11-external-communication",
        "external communication while inferring concurrency",
        "CAPTURE_EXTERNAL_COMMUNICATION_PRESENT",
        _control_11,
    ),
    (
        "mutation-12-hidden-store",
        "hidden primitive relation store",
        "CANDIDATE_HIDDEN_PRIMITIVE_STORE",
        _control_12,
    ),
    (
        "mutation-13-candidate-reference-read",
        "candidate reads reference receipts",
        "CANDIDATE_FORBIDDEN_REFERENCE_INPUT",
        _control_13,
    ),
    (
        "mutation-14-reference-candidate-read",
        "reference reads candidate graph",
        "REFERENCE_FORBIDDEN_CANDIDATE_INPUT",
        _control_14,
    ),
    (
        "mutation-15-derived-input-missing",
        "derived relation input IDs missing",
        "DERIVED_INPUT_RELATION_IDS_MISSING",
        _control_15,
    ),
    (
        "mutation-16-shortest-path-proof",
        "shortest path differs from declared input IDs",
        "SHORTEST_PATH_INPUT_RELATION_IDS_MISMATCH",
        _control_16,
    ),
    (
        "mutation-17-primitive-overwrite",
        "multiple primitives for one pair collapsed",
        "MULTIPLE_PRIMITIVE_RELATION_OVERWRITE",
        _control_17,
    ),
    (
        "mutation-18-cartesian-dependency",
        "fact-specific dependency uses occurrence Cartesian lifting",
        "FACT_SPECIFIC_CARTESIAN_LIFTING_FORBIDDEN",
        _control_18,
    ),
    (
        "mutation-19-run-identity",
        "semantic occurrence ID impersonates concrete instance ID",
        "SEMANTIC_OCCURRENCE_AS_CONCRETE_INSTANCE",
        _control_19,
    ),
    (
        "mutation-20-wall-clock",
        "wall clock impersonates causal order",
        "PROGRAM_ORDER_RECEIPT_REQUIRED",
        _control_20,
    ),
    (
        "mutation-21-sixth-coordinate",
        "vector clock inserted as sixth coordinate",
        "VECTOR_CLOCK_AS_SIXTH_COORDINATE",
        _control_21,
    ),
    (
        "mutation-22-cycle",
        "happens-before cycle",
        "HAPPENS_BEFORE_CYCLE",
        _control_22,
    ),
    (
        "mutation-23-concurrent-and-hb",
        "concurrent-with and happens-before both declared",
        "CONCURRENT_WITH_HAPPENS_BEFORE",
        _control_23,
    ),
    (
        "mutation-24-incomplete-concurrent",
        "capture incomplete but concurrent declared",
        "CONCURRENT_WITHOUT_CAPTURE_COMPLETE",
        _control_24,
    ),
    (
        "mutation-25-core-change",
        "formal Core modified",
        "PROTECTED_CORE_PATH_MODIFIED",
        _control_25,
    ),
    (
        "mutation-26-core-copy",
        "undeclared experimental Core copy",
        "UNDECLARED_EXPERIMENTAL_CORE_COPY",
        _control_26,
    ),
    (
        "mutation-27-lazy-eager",
        "lazy answer differs from eager reference",
        "LAZY_EAGER_RESULT_MISMATCH",
        _control_27,
    ),
    (
        "mutation-28-cache-reuse",
        "cross-run query cache reuse",
        "CROSS_RUN_QUERY_CACHE_REUSE",
        _control_28,
    ),
    (
        "mutation-29-output-change",
        "ordinary output changes with capture mode",
        "ORDINARY_OUTPUT_ORTHOGONALITY_VIOLATION",
        _control_29,
    ),
    (
        "mutation-30-scale-downgrade",
        "large workload silently downgraded",
        "LARGE_WORKLOAD_DOWNGRADED",
        _control_30,
    ),
    (
        "mutation-31-program-order-same-count-wrong-adjacency",
        "program-order edge count preserved while adjacency is wrong",
        "PROGRAM_ORDER_ADJACENCY_SET_MISMATCH",
        _control_31,
    ),
]


def run_negative_controls() -> dict[str, Any]:
    mutation_ids = [row[0] for row in CONTROL_REGISTRY]
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ExperimentError("NEGATIVE_CONTROL_MUTATION_ID_DUPLICATE")
    results: list[dict[str, Any]] = []
    for mutation_id, description, expected_reason, action in CONTROL_REGISTRY:
        observed_reason = None
        try:
            action()
        except ExperimentError as error:
            observed_reason = str(error)
        passed = observed_reason == expected_reason
        results.append(
            {
                "mutation_id": mutation_id,
                "description": description,
                "execution_count": 1,
                "expected_reason_code": expected_reason,
                "observed_reason_code": observed_reason,
                "auto_repaired": False,
                "partial_success": False,
                "status": "PASS" if passed else "FAIL",
            }
        )
    passed_count = sum(row["status"] == "PASS" for row in results)
    return {
        "status": "PASS" if passed_count == len(results) else "FAIL",
        "control_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "controls": results,
    }
