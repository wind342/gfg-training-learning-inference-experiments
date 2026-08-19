from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.inter_fact_relations_v0_hardening_scale_v1.common import (
    ExperimentError,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.primitive_semantic_validation import (
    build,
    run,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.reads_from_versions import (
    run as run_reads_from,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.capture_auditor import (
    CAPTURE_COMPLETE,
    CAPTURE_PARTIAL,
    audit_capture,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.semantic_evidence_validator import (
    validate_primitive_store,
)


def test_all_primitive_semantics_and_reads_from_are_exercised() -> None:
    result = run()
    assert result["status"] == "PASS"
    assert result["all_six_relation_types_exercised"] is True
    assert run_reads_from()["reads_from_relation_count"] == 1


def test_capture_complete_is_measured() -> None:
    builder = build()
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    audit = audit_capture(builder.capture_contract(), receipts, validated)
    assert audit["overall_status"] == CAPTURE_COMPLETE
    assert audit["scopes"][0]["measured_audit_complete"] is True
    scope = audit["scopes"][0]
    assert scope["program_order_exactness"]["status"] == "PASS"
    assert (
        scope["program_order_exactness"][
            "receipt_relation_evidence_one_to_one"
        ]
        is True
    )
    assert (
        scope["scheduler_completeness_basis"]
        == "DECLARED_CONTROLLED_EXECUTOR_PROFILE"
    )
    assert scope["global_scheduler_completeness_machine_proved"] is False
    assert scope["concurrency_scope"] == "CONTROLLED_CAPTURE_SCOPE_ONLY"


def test_capture_unknown_edge_is_partial() -> None:
    builder = build()
    receipts = builder.runtime_receipts()
    receipts["unknown_edges"].append(
        {"scope_id": "semantic-validation", "unknown_edge_id": "unknown"}
    )
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    audit = audit_capture(builder.capture_contract(), receipts, validated)
    assert audit["overall_status"] == CAPTURE_PARTIAL
    assert audit["scopes"][0]["concurrency_inference_allowed"] is False


def test_caller_cannot_override_concurrency() -> None:
    builder = build()
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    contract = deepcopy(builder.capture_contract())
    contract["allow_concurrency"] = True
    with pytest.raises(
        ExperimentError, match="CALLER_CONCURRENCY_OVERRIDE_FORBIDDEN"
    ):
        audit_capture(contract, receipts, validated)


def test_duplicate_program_order_sequence_index_fails_closed() -> None:
    builder = build()
    receipts = builder.runtime_receipts()
    receipts["occurrences"][1]["sequence_index"] = 0
    validated = validate_primitive_store(
        {"execution_run_id": builder.run_id, "primitive_relations": [], "evidence": []},
        receipts,
    )
    audit = audit_capture(builder.capture_contract(), receipts, validated)
    scope = audit["scopes"][0]
    assert audit["overall_status"] == CAPTURE_PARTIAL
    assert (
        "PROGRAM_ORDER_SEQUENCE_INDEX_DUPLICATE"
        in scope["reason_codes"]
    )
    assert scope["concurrency_inference_allowed"] is False
