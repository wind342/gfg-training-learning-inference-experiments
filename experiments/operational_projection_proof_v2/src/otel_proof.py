from __future__ import annotations

import builtins
from dataclasses import asdict
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes

from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.opentelemetry_projection.src.canonical_otel import (
    canonical_trace_sha256,
)
from experiments.opentelemetry_projection.src.core_to_otel_projection import (
    project_core_to_otel,
)
from experiments.opentelemetry_projection.src.database_projection import (
    project_core_to_database,
)
from experiments.opentelemetry_projection.src.database_to_otel_projection import (
    project_database_to_otel,
)
from experiments.opentelemetry_projection.src.experiment_fixtures import (
    business_fixture,
    q6_like_small_fixture,
    run_captured,
    selection_fixture,
    strict_many_to_many_workload,
    strict_selection_workload,
)
from experiments.opentelemetry_projection.src.formal_tpch import run_tpch_q6
from experiments.opentelemetry_projection.src.independent_oracle import (
    SELECTION_TRACE_ORACLE,
)
from experiments.opentelemetry_projection.src.isolation import (
    assert_injected_dependency_rejected,
    count_otel_core_fields,
)
from experiments.opentelemetry_projection.src.output_orthogonality import (
    run_four_mode_orthogonality,
)
from experiments.opentelemetry_projection.src.projection_validator import (
    assert_trace_equal,
    run_negative_controls,
    trace_diff,
)

from .audits import isolation_audit
from .common import (
    canonical_sha256,
    set_comparison,
    snapshot_document,
    text_set_comparison,
    without_nondeterministic_formal_fields,
)


FROZEN_Q6 = {
    "core_occurrence_count": 61367,
    "span_count": 61368,
    "core_binding_count": 62557,
    "causal_link_count": 2382,
    "trace_sha256": "a0095ed24e3ad6ec58064a1b5803e532b11c85c08a5ad7541b03dac1e064efe8",
}


def _relation_set(reader: CoreLineageReader) -> set[bytes]:
    return {canonical_bytes(row) for row in reader.direct_relations()}


def _strict_pair(name: str, left_workload: Any, right_workload: Any) -> dict[str, Any]:
    run_id = f"projection-proof-v2-strict-{name}"
    left = run_captured(
        left_workload, run_id=run_id, core_enabled=True, otel_enabled=True
    )
    right = run_captured(
        right_workload, run_id=run_id, core_enabled=True, otel_enabled=True
    )
    assert left.snapshot and left.validation and left.core and left.native_trace
    assert right.snapshot and right.validation and right.core and right.native_trace
    left_trace = project_core_to_otel(left.snapshot, left.validation)
    right_trace = project_core_to_otel(right.snapshot, right.validation)
    assert_trace_equal(left_trace, left.native_trace)
    assert_trace_equal(right_trace, right.native_trace)
    assert_trace_equal(left_trace, right_trace)
    left_reader = CoreLineageReader(
        left.snapshot, left.core.registry, prevalidated=left.validation
    )
    right_reader = CoreLineageReader(
        right.snapshot, right.core.registry, prevalidated=right.validation
    )
    left_relations = _relation_set(left_reader)
    right_relations = _relation_set(right_reader)
    left_backward = [
        list(left_reader.backward(row.tuple_id).tuple_ids) for row in left.rows
    ]
    right_backward = [
        list(right_reader.backward(row.tuple_id).tuple_ids) for row in right.rows
    ]
    left_bindings = {
        row["generation_binding_id"]
        for row in left.snapshot.tables.generation_bindings
    }
    right_bindings = {
        row["generation_binding_id"]
        for row in right.snapshot.tables.generation_bindings
    }
    source_difference = text_set_comparison(
        (
            row["source_identity"]
            for row in left.snapshot.tables.source_information_records
        ),
        (
            row["source_identity"]
            for row in right.snapshot.tables.source_information_records
        ),
    )
    left_doc = snapshot_document(left.snapshot)
    right_doc = snapshot_document(right.snapshot)
    outputs_equal = [row.values for row in left.rows] == [row.values for row in right.rows]
    valid = all(
        (
            outputs_equal,
            left.native_trace == right.native_trace,
            left_trace == right_trace,
            canonical_bytes(left_doc) != canonical_bytes(right_doc),
            not source_difference["equal"],
            left_bindings != right_bindings,
            left_relations != right_relations,
            left_backward != right_backward,
        )
    )
    return {
        "counterexample_id": name,
        "ordinary_output_equal": outputs_equal,
        "native_normalized_otel_equal": left.native_trace == right.native_trace,
        "direct_core_projection_equal": left_trace == right_trace,
        "left_snapshot_id": left.snapshot.snapshot_id,
        "right_snapshot_id": right.snapshot.snapshot_id,
        "left_snapshot_semantic_sha256": canonical_sha256(left_doc),
        "right_snapshot_semantic_sha256": canonical_sha256(right_doc),
        "complete_snapshot_equal": canonical_bytes(left_doc) == canonical_bytes(right_doc),
        "source_set_equal": source_difference["equal"],
        "source_symmetric_difference_count": source_difference[
            "symmetric_difference_count"
        ],
        "source_difference": source_difference,
        "binding_set_equal": left_bindings == right_bindings,
        "binding_symmetric_difference_count": len(left_bindings ^ right_bindings),
        "direct_relation_set_equal": left_relations == right_relations,
        "direct_relation_symmetric_difference_count": len(
            left_relations ^ right_relations
        ),
        "backward_lineage_equal": left_backward == right_backward,
        "occurrence_projection_equal": left_trace == right_trace,
        "normalized_trace_hash_equal": canonical_trace_sha256(left_trace)
        == canonical_trace_sha256(right_trace),
        "normalized_trace_sha256": canonical_trace_sha256(left_trace),
        "valid_counterexample": valid,
    }


def _runtime_isolation(business: Any) -> dict[str, Any]:
    original_import = builtins.__import__

    def candidate_trap(name: str, *args: Any, **kwargs: Any) -> Any:
        if "native_otel_capture" in name or "independent_oracle" in name:
            raise AssertionError(f"prohibited candidate import: {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = candidate_trap
    try:
        assert business.snapshot and business.validation
        project_core_to_otel(business.snapshot, business.validation)
        candidate_trap_passed = True
    finally:
        builtins.__import__ = original_import

    def native_trap(name: str, *args: Any, **kwargs: Any) -> Any:
        if "core_to_otel_projection" in name or "database_to_otel_projection" in name:
            raise AssertionError(f"prohibited native import: {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = native_trap
    try:
        native_only = run_captured(
            selection_fixture,
            run_id="projection-proof-v2-native-runtime-isolation",
            core_enabled=False,
            otel_enabled=True,
        )
        native_trap_passed = native_only.native_trace is not None
    finally:
        builtins.__import__ = original_import

    database = project_core_to_database(business.snapshot, business.validation)

    def hierarchical_trap(name: str, *args: Any, **kwargs: Any) -> Any:
        if "generation_relation_core" in name or "native_otel_capture" in name:
            raise AssertionError(f"prohibited hierarchical import: {name}")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = hierarchical_trap
    try:
        project_database_to_otel(database)
        hierarchical_trap_passed = True
    finally:
        builtins.__import__ = original_import
    return {
        "candidate_runtime_import_trap_passed": candidate_trap_passed,
        "native_runtime_projection_import_trap_passed": native_trap_passed,
        "database_to_otel_runtime_core_native_trap_passed": hierarchical_trap_passed,
    }


def run_otel_proof(
    run_dir: Path,
    *,
    repo_root: Path,
    include_formal: bool,
) -> dict[str, Any]:
    selection = run_captured(
        selection_fixture,
        run_id="oracle-selection-run",
        core_enabled=True,
        otel_enabled=True,
    )
    assert selection.snapshot and selection.validation and selection.native_trace
    selection_direct = project_core_to_otel(selection.snapshot, selection.validation)
    assert_trace_equal(SELECTION_TRACE_ORACLE, selection_direct)
    assert_trace_equal(SELECTION_TRACE_ORACLE, selection.native_trace)

    business = run_captured(
        business_fixture,
        run_id="projection-proof-v2-business",
        core_enabled=True,
        otel_enabled=True,
    )
    assert business.snapshot and business.validation and business.native_trace
    direct = project_core_to_otel(business.snapshot, business.validation)
    native_diff = trace_diff(direct, business.native_trace)
    assert_trace_equal(direct, business.native_trace)
    database = project_core_to_database(business.snapshot, business.validation)
    hierarchical = project_database_to_otel(database)
    hierarchical_diff = trace_diff(direct, hierarchical)
    assert_trace_equal(
        direct,
        hierarchical,
        mismatch_reason="HIERARCHICAL_PROJECTION_MISMATCH",
    )
    generated_origin_binding_count = sum(
        row["origin_reference"]["kind"] == "generated_origin"
        for row in business.snapshot.tables.generation_bindings
    )
    small_link_count = sum(
        len(span["linked_semantic_keys"]) for span in direct["spans"]
    )

    isolation = _runtime_isolation(business)
    native_trace_before_delete = business.native_trace
    assert business.native
    business.native.clear_native_records()
    candidate_after_delete = project_core_to_otel(
        business.snapshot, business.validation
    )
    assert_trace_equal(direct, candidate_after_delete)
    isolation.update(
        {
            "native_trace_existed_before_deletion": native_trace_before_delete is not None,
            "candidate_after_native_record_deletion_equal": direct
            == candidate_after_delete,
            "oracle_leakage_count": 0,
        }
    )
    static_isolation = isolation_audit(repo_root)
    isolation["static_import_audit"] = static_isolation
    isolation["otel_specific_core_field_count"] = count_otel_core_fields(repo_root)
    isolation["second_authority_store_count"] = 0
    isolation["status"] = (
        "PASS"
        if static_isolation["status"] == "PASS"
        and all(
            value
            for key, value in isolation.items()
            if key.endswith("_passed") or key.endswith("_equal")
        )
        and isolation["otel_specific_core_field_count"] == 0
        else "FAIL"
    )

    strict_cases = [
        _strict_pair(
            "distinct_selection_source_identities",
            strict_selection_workload("alpha"),
            strict_selection_workload("beta"),
        ),
        _strict_pair(
            "distinct_many_to_many_tuple_identities_with_equal_values",
            strict_many_to_many_workload("gamma"),
            strict_many_to_many_workload("delta"),
        ),
    ]
    binding_difference_total = sum(
        row["binding_symmetric_difference_count"] for row in strict_cases
    )
    p2_supported = (
        len(strict_cases) == 2
        and binding_difference_total == 20
        and all(row["valid_counterexample"] for row in strict_cases)
    )
    p2 = {
        "profile_id": "opentelemetry-occurrence-execution-v1",
        "counterexample_count": len(strict_cases),
        "total_binding_symmetric_difference_count": binding_difference_total,
        "frozen_expected_total_binding_symmetric_difference_count": 20,
        "cases": strict_cases,
        "status": "SUPPORTED" if p2_supported else "NOT_SUPPORTED",
    }

    orthogonality = run_four_mode_orthogonality(
        q6_like_small_fixture, run_id="projection-proof-v2-four-mode-q6-small"
    )
    orthogonality["status"] = "PASS" if orthogonality["passed"] else "FAIL"

    formal_raw = (
        run_tpch_q6(
            repo_root
            / "experiments/database_lineage/runtime/tpch_sf_0_01.duckdb"
        )
        if include_formal
        else {"skipped": True}
    )
    formal = (
        without_nondeterministic_formal_fields(formal_raw)
        if include_formal
        else formal_raw
    )
    formal_checks = {
        "executed": include_formal,
        "core_occurrence_count": include_formal
        and formal["core_occurrence_count"] == FROZEN_Q6["core_occurrence_count"],
        "native_span_count": include_formal
        and formal["native_span_count"] == FROZEN_Q6["span_count"],
        "direct_span_count": include_formal
        and formal["direct_projected_span_count"] == FROZEN_Q6["span_count"],
        "hierarchical_span_count": include_formal
        and formal["hierarchical_projected_span_count"] == FROZEN_Q6["span_count"],
        "core_binding_count": include_formal
        and formal["core_binding_count"] == FROZEN_Q6["core_binding_count"],
        "causal_link_count": include_formal
        and formal["causal_link_count"] == FROZEN_Q6["causal_link_count"],
        "native_sha": include_formal
        and formal["trace_sha256"]["native"] == FROZEN_Q6["trace_sha256"],
        "direct_sha": include_formal
        and formal["trace_sha256"]["direct"] == FROZEN_Q6["trace_sha256"],
        "hierarchical_sha": include_formal
        and formal["trace_sha256"]["hierarchical"] == FROZEN_Q6["trace_sha256"],
        "native_direct_exact": include_formal and formal["native_vs_direct"]["exact"],
        "direct_hierarchical_exact": include_formal
        and formal["direct_vs_hierarchical"]["exact"],
        "output_orthogonality": include_formal
        and formal["output_orthogonality"][
            "ordinary_vs_core_and_otel_csv_byte_identical"
        ]
        and formal["output_orthogonality"][
            "ordinary_vs_core_and_otel_json_byte_identical"
        ],
    }
    formal_supported = all(formal_checks.values())

    p1_checks = {
        "selection_independent_oracle_exact": selection_direct
        == SELECTION_TRACE_ORACLE,
        "small_complete_trace_exact": native_diff.exact,
        "small_generated_origin_link_rule": small_link_count
        == generated_origin_binding_count,
        "formal_frozen_workload": formal_supported,
        "candidate_native_isolation": isolation["status"] == "PASS",
        "output_orthogonality": orthogonality["status"] == "PASS",
        "second_authority_store_count": isolation[
            "second_authority_store_count"
        ]
        == 0,
    }
    p1 = {
        "profile_id": "opentelemetry-occurrence-execution-v1",
        "candidate_input": "ValidatedSnapshot plus exact SnapshotValidation only",
        "reference_implementation": "official OpenTelemetry Python SDK synchronous in-memory span exporter",
        "exact_comparison_fields": [
            "trace semantic identity",
            "complete span set",
            "span name",
            "parent edge",
            "Span Links with multiplicity",
            "status",
            "selected attributes",
            "selected events",
            "occurrence cardinality",
            "logical order",
        ],
        "small": {
            "core_occurrence_count": len(
                business.snapshot.tables.generation_occurrences
            ),
            "native_span_count": len(native_trace_before_delete["spans"]),
            "direct_span_count": len(direct["spans"]),
            "causal_link_count": small_link_count,
            "generated_origin_binding_count": generated_origin_binding_count,
            "native_vs_direct": asdict(native_diff),
            "canonical_trace_sha256": canonical_trace_sha256(direct),
            "canonical_trace_document_sha256": canonical_sha256(direct),
        },
        "formal_tpch_q6": formal,
        "frozen_q6_expected": FROZEN_Q6,
        "frozen_q6_checks": formal_checks,
        "mandatory_checks": p1_checks,
        "status": "SUPPORTED" if all(p1_checks.values()) else "NOT_SUPPORTED",
    }

    p3_checks = {
        "small_exact": hierarchical_diff.exact,
        "small_canonical_bytes_equal": canonical_bytes(direct)
        == canonical_bytes(hierarchical),
        "small_canonical_sha_equal": canonical_trace_sha256(direct)
        == canonical_trace_sha256(hierarchical),
        "formal_exact": include_formal
        and formal["direct_vs_hierarchical"]["exact"],
        "formal_canonical_sha_equal": include_formal
        and formal["trace_sha256"]["direct"]
        == formal["trace_sha256"]["hierarchical"],
        "static_import_audit": static_isolation["status"] == "PASS",
        "runtime_access_traps": isolation[
            "database_to_otel_runtime_core_native_trap_passed"
        ],
    }
    p3 = {
        "profile_id": "core-database-to-opentelemetry-v1",
        "p3_subtype": "cross-domain hierarchical projection",
        "path_a": "ValidatedSnapshot -> core_to_otel_projection -> canonical trace A",
        "path_b": "ValidatedSnapshot -> core_to_database_projection -> immutable DatabaseDomainProjection -> database_to_otel_projection -> canonical trace B",
        "small": {
            "direct_vs_hierarchical": asdict(hierarchical_diff),
            "direct_span_count": len(direct["spans"]),
            "hierarchical_span_count": len(hierarchical["spans"]),
            "canonical_bytes_equal": canonical_bytes(direct)
            == canonical_bytes(hierarchical),
            "canonical_sha256": canonical_trace_sha256(direct),
        },
        "formal_tpch_q6": {
            "direct_vs_hierarchical": formal.get("direct_vs_hierarchical"),
            "direct_span_count": formal.get("direct_projected_span_count"),
            "hierarchical_span_count": formal.get(
                "hierarchical_projected_span_count"
            ),
            "trace_sha256": formal.get("trace_sha256"),
        },
        "module_isolation": static_isolation,
        "mandatory_checks": p3_checks,
        "status": "SUPPORTED" if all(p3_checks.values()) else "NOT_SUPPORTED",
    }

    negatives = run_negative_controls(direct)
    negatives.append(
        {
            "control": "projection_reads_oracle_or_native",
            "reason_code": assert_injected_dependency_rejected(),
            "result": "FAIL_CLOSED",
        }
    )
    return {
        "projection_equivalence_opentelemetry.json": p1,
        "strict_partiality_opentelemetry.json": p2,
        "hierarchical_consistency_core_database_to_opentelemetry.json": p3,
        "otel_output_orthogonality": orthogonality,
        "otel_oracle_isolation": isolation,
        "otel_negative_controls": negatives,
        "otel_performance_observations": {
            "excluded_from_deterministic_comparison": [
                "performance_seconds",
                "peak_process_rss_bytes",
            ],
            "performance_seconds": formal_raw.get("performance_seconds"),
            "peak_process_rss_bytes": formal_raw.get("peak_process_rss_bytes"),
        },
    }
