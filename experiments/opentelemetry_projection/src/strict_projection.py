from __future__ import annotations

from generation_relation_core.canonical import canonical_bytes

from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader

from .core_to_otel_projection import project_core_to_otel
from .experiment_fixtures import (
    run_captured,
    strict_many_to_many_workload,
    strict_selection_workload,
)
from .projection_validator import assert_trace_equal


def _relation_set(reader: CoreLineageReader) -> set[bytes]:
    return {canonical_bytes(row) for row in reader.direct_relations()}


def _evaluate_pair(name: str, left_workload, right_workload) -> dict:
    run_id = f"strict-projection-{name}"
    left = run_captured(
        left_workload,
        run_id=run_id,
        core_enabled=True,
        otel_enabled=True,
    )
    right = run_captured(
        right_workload,
        run_id=run_id,
        core_enabled=True,
        otel_enabled=True,
    )
    assert left.snapshot and left.validation and left.core and left.native_trace
    assert right.snapshot and right.validation and right.core and right.native_trace
    left_trace = project_core_to_otel(left.snapshot, left.validation)
    right_trace = project_core_to_otel(right.snapshot, right.validation)
    assert_trace_equal(left_trace, left.native_trace)
    assert_trace_equal(right_trace, right.native_trace)
    assert_trace_equal(left_trace, right_trace)

    left_bindings = {
        row["generation_binding_id"] for row in left.snapshot.tables.generation_bindings
    }
    right_bindings = {
        row["generation_binding_id"]
        for row in right.snapshot.tables.generation_bindings
    }
    left_reader = CoreLineageReader(
        left.snapshot, left.core.registry, prevalidated=left.validation
    )
    right_reader = CoreLineageReader(
        right.snapshot, right.core.registry, prevalidated=right.validation
    )
    left_backward = [
        list(left_reader.backward(row.tuple_id).tuple_ids) for row in left.rows
    ]
    right_backward = [
        list(right_reader.backward(row.tuple_id).tuple_ids) for row in right.rows
    ]
    left_relations = _relation_set(left_reader)
    right_relations = _relation_set(right_reader)
    if left_bindings == right_bindings or left_backward == right_backward:
        raise AssertionError(f"{name} is not a strict-projection counterexample")
    return {
        "name": name,
        "normalized_otel_equal": True,
        "native_otel_equal": left.native_trace == right.native_trace,
        "snapshot_ids_equal": left.snapshot.snapshot_id == right.snapshot.snapshot_id,
        "source_sets_equal": {
            row["source_identity"]
            for row in left.snapshot.tables.source_information_records
        }
        == {
            row["source_identity"]
            for row in right.snapshot.tables.source_information_records
        },
        "binding_sets_equal": left_bindings == right_bindings,
        "binding_symmetric_difference_count": len(left_bindings ^ right_bindings),
        "backward_lineage_equal": left_backward == right_backward,
        "direct_relation_sets_equal": left_relations == right_relations,
        "direct_relation_symmetric_difference_count": len(
            left_relations ^ right_relations
        ),
        "ordinary_outputs_equal": [row.values for row in left.rows]
        == [row.values for row in right.rows],
        "span_count": len(left_trace["spans"]),
    }


def run_strict_projection_counterexamples() -> list[dict]:
    return [
        _evaluate_pair(
            "distinct_selection_sources",
            strict_selection_workload("alpha"),
            strict_selection_workload("beta"),
        ),
        _evaluate_pair(
            "distinct_many_to_many_pairing_identities",
            strict_many_to_many_workload("gamma"),
            strict_many_to_many_workload("delta"),
        ),
    ]
