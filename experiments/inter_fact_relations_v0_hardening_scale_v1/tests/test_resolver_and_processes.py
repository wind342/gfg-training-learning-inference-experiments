from __future__ import annotations

from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.mixed_dag import (
    build_mixed_dag,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.capture_auditor import (
    audit_capture,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.indexed_candidate_resolver import (
    IndexedCandidateResolver,
    resolve_candidate_payload,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.reference_process import (
    resolve_reference_payload,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.compare_process import (
    compare_outputs,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.semantic_evidence_validator import (
    validate_primitive_store,
)


def _context(scale: str = "small"):
    workload = build_mixed_dag(scale)
    builder = workload["builder"]
    receipts = builder.runtime_receipts()
    validated = validate_primitive_store(builder.primitive_store(), receipts)
    capture = audit_capture(builder.capture_contract(), receipts, validated)
    return workload, builder, receipts, validated, capture


def test_small_candidate_matches_eager_reference_exhaustively() -> None:
    workload, builder, receipts, validated, capture = _context()
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
                "queries": workload["queries"],
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
                "queries": workload["queries"],
                "reference_mode": "eager",
                "schema_version": "reference-input-v1",
            }
        ),
    }
    comparison = compare_outputs(
        {
            "candidate": candidate,
            "reference": reference,
            "query_manifest_sha256": workload["query_manifest_sha256"],
        }
    )
    assert comparison["status"] == "PASS"
    assert comparison["mismatch_count"] == 0
    assert comparison["query_count"] == 14_856


def test_candidate_does_not_materialize_closure_and_retains_multi_edges() -> None:
    _, builder, _, validated, capture = _context()
    resolver = IndexedCandidateResolver(
        execution_run_id=builder.run_id,
        primitive_store=validated,
        capture_audit=capture,
        lifting_rules={"policy": "RELATION_TYPE_SPECIFIC_LIFTING"},
    )
    metrics = resolver.metrics()
    assert metrics["full_transitive_closure_materialized"] is False
    assert metrics["global_closure_pair_count"] == 0
    assert (
        metrics["retained_primitive_relation_count"]
        > metrics["causal_pair_count"]
    )


def test_medium_meets_scale_floor() -> None:
    workload, _, receipts, validated, _ = _context("medium")
    assert len(receipts["occurrences"]) == 1_000
    assert len(receipts["facts"]) == 3_000
    assert len(validated["primitive_relations"]) >= 3_000
    assert len(workload["queries"]) == 4_200
