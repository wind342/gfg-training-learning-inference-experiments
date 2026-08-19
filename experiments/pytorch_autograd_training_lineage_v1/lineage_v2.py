from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .core_capture import CoreTrainingCollector
from .independent_reference_v2 import NativeOracleLineageReference
from .lineage import TrainingLineageIndex
from .pipeline import TrainingSpec, run_training_step


EXPERIMENT_ROOT = Path(__file__).resolve().parent
FORBIDDEN_REFERENCE_TOKENS = (
    "tracked_" + "matmul",
    "tracked_" + "mul",
    "tracked_" + "relu",
    "tracked_" + "pow",
    "tracked_" + "sin",
    "UNDECLARED_" + "LOCAL_GRADIENT_RULE",
    "REFERENCE_" + "LOCAL_GRADIENT_RULE",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _compare_paths(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[int, int, int, int, int, int]:
    actual_set = {_canonical(row) for row in actual}
    expected_set = {_canonical(row) for row in expected}
    actual_relations = [relation for path in actual for relation in path["relations"]]
    expected_relations = [relation for path in expected for relation in path["relations"]]
    actual_roles = Counter(row["role"] for row in actual_relations)
    expected_roles = Counter(row["role"] for row in expected_relations)
    actual_occurrences = Counter(row["occurrence_key"] for row in actual_relations)
    expected_occurrences = Counter(row["occurrence_key"] for row in expected_relations)
    role_delta = sum((actual_roles - expected_roles).values()) + sum(
        (expected_roles - actual_roles).values()
    )
    occurrence_delta = sum((actual_occurrences - expected_occurrences).values()) + sum(
        (expected_occurrences - actual_occurrences).values()
    )
    actual_multiplicity = {
        _canonical({key: value for key, value in row.items() if key != "multiplicity"}):
        row["multiplicity"]
        for row in actual
    }
    expected_multiplicity = {
        _canonical({key: value for key, value in row.items() if key != "multiplicity"}):
        row["multiplicity"]
        for row in expected
    }
    multiplicity_delta = sum(
        abs(actual_multiplicity.get(key, 0) - expected_multiplicity.get(key, 0))
        for key in set(actual_multiplicity) | set(expected_multiplicity)
    )
    return (
        len(actual_set - expected_set),
        len(expected_set - actual_set),
        role_delta,
        occurrence_delta,
        len(actual_set ^ expected_set),
        multiplicity_delta,
    )


def _captures() -> dict[str, Any]:
    specs = {
        name: TrainingSpec(name)
        for name in (
            "branch_and_merge",
            "duplicate_valued_distinct_sources",
            "linear_chain",
            "shared_tensor_reuse",
            "zero_gradient_and_unused_sources",
        )
    }
    specs.update({
        f"checkpoint:{mode}": TrainingSpec(
            "checkpoint_external_state",
            sample_identity="checkpoint_sample",
            checkpoint_mode=mode,
        )
        for mode in ("divergent", "no_checkpoint", "stable")
    })
    result = {}
    for workload_key, spec in sorted(specs.items()):
        collector = CoreTrainingCollector()
        run_training_step(spec, collector=collector)
        result[workload_key] = collector.finalize(
            evidence_context=spec.evidence_context
        )
    return result


def _reference_static_audit() -> dict[str, Any]:
    source = (EXPERIMENT_ROOT / "independent_reference_v2.py").read_text(
        encoding="utf-8"
    )
    matches = [token for token in FORBIDDEN_REFERENCE_TOKENS if token in source]
    return {
        "forbidden_match_count": len(matches),
        "forbidden_matches": matches,
        "operation_specific_gradient_rule_count": len(matches),
    }


def run_v2_query_comparison(oracle_result: dict[str, Any]) -> dict[str, Any]:
    captures = _captures()
    native_relations = oracle_result["native_gradient_dependency_oracle"]["relations"]
    forward_queries = []
    reverse_queries = []
    totals = [0, 0, 0, 0, 0, 0]
    for workload_key, capture in sorted(captures.items()):
        index = TrainingLineageIndex(capture.snapshot, capture.validation)
        reference = NativeOracleLineageReference(
            capture.execution_receipts,
            native_relations,
            workload_key,
        )
        for source in sorted(
            capture.snapshot.tables.source_information_records,
            key=lambda row: row["source_information_id"],
        ):
            source_ref = source["source_payload"]["source_ref"]
            actual = index.forward_lineage(source["source_information_id"])
            expected = reference.forward_paths(source_ref)
            deltas = _compare_paths(actual["paths"], expected)
            totals = [left + right for left, right in zip(totals, deltas, strict=True)]
            forward_queries.append({
                "exact": actual["paths"] == expected,
                "query": actual,
                "reference_paths": expected,
                "source_ref": source_ref,
                "workload_key": workload_key,
            })
        targets = [
            support
            for support in capture.snapshot.tables.perceptual_support_records
            if support["support_payload"]["support_kind"]
            in {"gradient", "loss", "optimizer_state_after_step", "parameter_after_step"}
        ]
        for support in sorted(targets, key=lambda row: row["support_id"]):
            support_key = support["support_payload"]["support_key"]
            actual = index.reverse_lineage(support["support_id"])
            expected = reference.reverse_paths(support_key)
            deltas = _compare_paths(actual["paths"], expected)
            totals = [left + right for left, right in zip(totals, deltas, strict=True)]
            reverse_queries.append({
                "exact": actual["paths"] == expected,
                "query": actual,
                "reference_paths": expected,
                "support_key": support_key,
                "workload_key": workload_key,
            })
    static_audit = _reference_static_audit()
    comparison = {
        "false_positive": totals[0],
        "false_negative": totals[1],
        "forward_exact_count": sum(row["exact"] for row in forward_queries),
        "forward_query_count": len(forward_queries),
        "multiplicity_mismatch": totals[5],
        "occurrence_mismatch": totals[3],
        "path_mismatch": totals[4],
        "reference_operation_specific_gradient_rule_count": static_audit[
            "operation_specific_gradient_rule_count"
        ],
        "reverse_exact_count": sum(row["exact"] for row in reverse_queries),
        "reverse_query_count": len(reverse_queries),
        "role_mismatch": totals[2],
    }
    comparison["all_exact"] = all([
        comparison["false_positive"] == 0,
        comparison["false_negative"] == 0,
        comparison["forward_exact_count"] == comparison["forward_query_count"],
        comparison["multiplicity_mismatch"] == 0,
        comparison["occurrence_mismatch"] == 0,
        comparison["path_mismatch"] == 0,
        comparison["reference_operation_specific_gradient_rule_count"] == 0,
        comparison["reverse_exact_count"] == comparison["reverse_query_count"],
        comparison["role_mismatch"] == 0,
    ])
    return {
        "bidirectional_training_lineage_v2": {
            "forward_queries": forward_queries,
            "reference_static_audit": static_audit,
            "reverse_queries": reverse_queries,
            "status": (
                "BIDIRECTIONAL_TRAINING_UPDATE_LINEAGE_NATIVE_ORACLE_VALIDATED_SUPPORTED"
                if comparison["all_exact"]
                else "BIDIRECTIONAL_TRAINING_UPDATE_LINEAGE_NATIVE_ORACLE_VALIDATED_NOT_ESTABLISHED"
            ),
        },
        "query_exact_comparison_v2": comparison,
    }
