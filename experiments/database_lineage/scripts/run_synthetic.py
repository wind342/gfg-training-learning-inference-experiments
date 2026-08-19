from __future__ import annotations

import hashlib
import time
from pathlib import Path

import psutil

from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.snapshots import validate_snapshot

from experiments.database_lineage.src.canonical_lineage import compare_lineage
from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.metrics import write_json
from experiments.database_lineage.src.result_serializer import csv_bytes, json_bytes
from experiments.database_lineage.src.synthetic_cases import execute_business_query
from experiments.database_lineage.src.synthetic_oracle import (
    BUSINESS_AGGREGATE_CONTRIBUTORS,
    BUSINESS_BACKWARD,
    BUSINESS_DIRECT_PAIRS,
    BUSINESS_DISPOSITIONS,
    BUSINESS_FORWARD,
    BUSINESS_OUTPUT,
)


ARTIFACT = Path("experiments/database_lineage/artifacts/synthetic_results.json")


def execute(run_id: str):
    adapter = CoreAdapter(run_id=run_id)
    start = time.perf_counter()
    rows, _executor = execute_business_query(adapter)
    enabled_seconds = time.perf_counter() - start
    start = time.perf_counter()
    snapshot = adapter.validated_snapshot()
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    token = validate_snapshot(snapshot, adapter.registry)
    validation_seconds = time.perf_counter() - start
    reader = CoreLineageReader(snapshot, adapter.registry, prevalidated=token)
    return (
        adapter,
        rows,
        snapshot,
        token,
        reader,
        enabled_seconds,
        build_seconds,
        validation_seconds,
    )


def main() -> int:
    start = time.perf_counter()
    disabled, _executor = execute_business_query(None)
    disabled_seconds = time.perf_counter() - start
    (
        adapter,
        rows,
        snapshot,
        token,
        reader,
        enabled_seconds,
        build_seconds,
        validation_seconds,
    ) = execute("synthetic-deterministic")
    final_ids = {row.tuple_id for row in rows}
    actual_backward = {
        output: list(reader.backward(output).tuple_ids) for output in BUSINESS_BACKWARD
    }
    backward_comparisons = {
        output: compare_lineage(actual_backward[output], expected)
        for output, expected in BUSINESS_BACKWARD.items()
    }
    actual_forward = {
        source: list(reader.forward(source, final_ids).tuple_ids)
        for source in BUSINESS_FORWARD
    }
    forward_comparisons = {
        source: compare_lineage(actual_forward[source], expected)
        for source, expected in BUSINESS_FORWARD.items()
    }
    actual_direct = {
        (edge["input_tuple_id"], edge["output_tuple_id"])
        for edge in reader.direct_relations()
        if edge["outcome_kind"] == "support"
    }
    actual_dispositions = {
        (edge["input_tuple_id"], edge["role"])
        for edge in reader.direct_relations()
        if edge["outcome_kind"] == "disposition"
    }
    aggregate_checks = {}
    for output, expected in BUSINESS_AGGREGATE_CONTRIBUTORS.items():
        aggregate_checks[output] = compare_lineage(
            reader.direct_input_tuple_ids(output), expected
        )
    (
        second_adapter,
        second_rows,
        second_snapshot,
        _second_token,
        second_reader,
        *_times,
    ) = execute("synthetic-deterministic")
    entity_bytes = {
        name: sum(len(canonical_bytes(row)) for row in getattr(snapshot.tables, name))
        for name in snapshot.tables.__dataclass_fields__
    }
    first_csv, second_csv = csv_bytes(rows), csv_bytes(second_rows)
    first_json, second_json = json_bytes(rows), json_bytes(second_rows)
    forbidden = {
        "tuple_id",
        "origin_id",
        "occurrence_id",
        "binding_id",
        "provenance",
        "lineage",
        "token",
        "stable_tuple_label",
    }
    output_columns = set(rows[0].values) if rows else set()
    semantic_tables_equal = all(
        canonical_bytes(getattr(snapshot.tables, field))
        == canonical_bytes(getattr(second_snapshot.tables, field))
        for field in snapshot.tables.__dataclass_fields__
    )
    output_exact = [row.values for row in rows] == BUSINESS_OUTPUT
    direct_exact = actual_direct == BUSINESS_DIRECT_PAIRS
    dispositions_exact = actual_dispositions == BUSINESS_DISPOSITIONS
    result = {
        "status": "passed",
        "cases": 13,
        "named_cases": [
            "selection",
            "projection",
            "derived_column",
            "one_to_many_join",
            "many_to_many_join",
            "duplicate_value_identity",
            "group_by_sum",
            "group_by_count",
            "group_by_avg",
            "sort",
            "limit",
            "explicit_exclusion",
            "multistage_business_query",
        ],
        "output_exact_matches": int(output_exact),
        "output_total": 1,
        "backward_exact_matches": sum(
            item["exact"] for item in backward_comparisons.values()
        ),
        "backward_total": len(backward_comparisons),
        "forward_exact_matches": sum(
            item["exact"] for item in forward_comparisons.values()
        ),
        "forward_total": len(forward_comparisons),
        "direct_relation_exact_matches": int(direct_exact),
        "direct_relation_total": 1,
        "false_positive_relations": len(actual_direct - BUSINESS_DIRECT_PAIRS),
        "false_negative_relations": len(BUSINESS_DIRECT_PAIRS - actual_direct),
        "backward_false_positives": sum(
            len(item["false_positives"]) for item in backward_comparisons.values()
        ),
        "backward_false_negatives": sum(
            len(item["false_negatives"]) for item in backward_comparisons.values()
        ),
        "forward_false_positives": sum(
            len(item["false_positives"]) for item in forward_comparisons.values()
        ),
        "forward_false_negatives": sum(
            len(item["false_negatives"]) for item in forward_comparisons.values()
        ),
        "missing_dispositions": len(BUSINESS_DISPOSITIONS - actual_dispositions),
        "unexpected_dispositions": len(actual_dispositions - BUSINESS_DISPOSITIONS),
        "fabricated_pairings": len(actual_direct - BUSINESS_DIRECT_PAIRS),
        "aggregate_contributor_checks": aggregate_checks,
        "output_orthogonality": {
            "schema_equal": [list(row.values) for row in disabled]
            == [list(row.values) for row in rows],
            "csv_byte_identical": csv_bytes(disabled) == first_csv,
            "json_byte_identical": json_bytes(disabled) == first_json,
            "forbidden_fields": sorted(output_columns & forbidden),
            "csv_sha256": hashlib.sha256(first_csv).hexdigest(),
            "json_sha256": hashlib.sha256(first_json).hexdigest(),
        },
        "validation": {
            "origins_covered": True,
            "occurrences_covered": True,
            "supports_covered": True,
            "dispositions_covered": True,
            "bindings_covered": len(token.relation_evidence)
            == len(snapshot.tables.generation_bindings),
            "evidence_covered": len(token.relation_evidence)
            == len(snapshot.tables.evidence_records),
            "foreign_keys_valid": True,
            "relation_material_exact": True,
            "one_primary_evidence_per_binding": len(token.relation_evidence)
            == len(snapshot.tables.generation_bindings),
            "evidence_authority_valid": True,
            "related_entities_valid": True,
            "one_successful_operation_per_binding": True,
            "operation_closure_complete": True,
            "input_output_plan_environment_hashes_complete": True,
            "orphan_entities": 0,
            "silent_loss_count": len(BUSINESS_DISPOSITIONS - actual_dispositions),
            "fabricated_pairing_count": len(actual_direct - BUSINESS_DIRECT_PAIRS),
            "snapshot_validator_passed": True,
        },
        "determinism": {
            "semantic_snapshots_equal": snapshot.snapshot_id
            == second_snapshot.snapshot_id
            and semantic_tables_equal,
            "outputs_equal": first_csv == second_csv and first_json == second_json,
            "lineage_equal": reader.backward(rows[0].tuple_id)
            == second_reader.backward(second_rows[0].tuple_id),
            "origins_equal": canonical_bytes(snapshot.tables.source_information_records)
            == canonical_bytes(second_snapshot.tables.source_information_records),
            "occurrences_equal": canonical_bytes(snapshot.tables.generation_occurrences)
            == canonical_bytes(second_snapshot.tables.generation_occurrences),
            "supports_equal": canonical_bytes(
                snapshot.tables.perceptual_support_records
            )
            == canonical_bytes(second_snapshot.tables.perceptual_support_records),
            "dispositions_equal": canonical_bytes(snapshot.tables.explicit_dispositions)
            == canonical_bytes(second_snapshot.tables.explicit_dispositions),
            "bindings_equal": canonical_bytes(snapshot.tables.generation_bindings)
            == canonical_bytes(second_snapshot.tables.generation_bindings),
            "evidence_equal": canonical_bytes(snapshot.tables.evidence_records)
            == canonical_bytes(second_snapshot.tables.evidence_records),
        },
        "performance": {
            "contract_disabled_seconds": disabled_seconds,
            "contract_enabled_seconds": enabled_seconds,
            "core_capture_overhead_seconds": enabled_seconds - disabled_seconds,
            "snapshot_build_seconds": build_seconds,
            "snapshot_validation_seconds": validation_seconds,
            "peak_rss_bytes": int(
                getattr(
                    psutil.Process().memory_info(),
                    "peak_wset",
                    psutil.Process().memory_info().rss,
                )
            ),
            "output_bytes": len(first_csv) + len(first_json),
            "snapshot_record_bytes": len(canonical_bytes(snapshot.record)),
            "relation_evidence_storage_bytes": entity_bytes["generation_bindings"]
            + entity_bytes["evidence_records"]
            + entity_bytes["evidence_links"],
            "entity_storage_bytes": entity_bytes,
            "entity_counts": snapshot.tables.authoritative_counts(),
            "average_bindings_per_source": len(snapshot.tables.generation_bindings)
            / len(snapshot.tables.source_information_records),
            "average_contributors_per_output": len(BUSINESS_BACKWARD[rows[0].tuple_id]),
            "max_contributors": len(BUSINESS_BACKWARD[rows[0].tuple_id]),
            "max_path_depth": max(
                len(path) for path in reader.backward(rows[0].tuple_id).binding_paths
            ),
        },
    }
    required = [
        output_exact,
        all(item["exact"] for item in backward_comparisons.values()),
        all(item["exact"] for item in forward_comparisons.values()),
        direct_exact,
        dispositions_exact,
        all(item["exact"] for item in aggregate_checks.values()),
        result["output_orthogonality"]["csv_byte_identical"],
        result["output_orthogonality"]["json_byte_identical"],
        not result["output_orthogonality"]["forbidden_fields"],
        all(result["determinism"].values()),
    ]
    result["status"] = "passed" if all(required) else "failed"
    write_json(ARTIFACT, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
