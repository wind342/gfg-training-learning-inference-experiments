from __future__ import annotations

import statistics
import time
import tracemalloc
from pathlib import Path

from experiments.database_lineage.src.canonical_lineage import compare_lineage
from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.core_lineage_reader import CoreLineageReader
from experiments.database_lineage.src.core_lineage_reader_reference import (
    LegacyScanCoreLineageReader,
)
from experiments.database_lineage.src.metrics import write_json
from experiments.database_lineage.src.synthetic_cases import execute_business_query
from experiments.database_lineage.src.synthetic_oracle import (
    BUSINESS_BACKWARD,
    BUSINESS_FORWARD,
)


ARTIFACT = Path("experiments/database_lineage/artifacts/reader_index_benchmark.json")


def measure(reader_type, snapshot, registry, token, rows, repeats: int = 20) -> dict:
    construction_times = []
    forward_times = []
    backward_times = []
    peaks = []
    outputs = {row.tuple_id for row in rows}
    forward_sources = sorted(BUSINESS_FORWARD)
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        reader = reader_type(snapshot, registry, prevalidated=token)
        construction_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        backward = {row.tuple_id: reader.backward(row.tuple_id) for row in rows}
        backward_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        forward = {
            source: reader.forward(source, outputs) for source in forward_sources
        }
        forward_times.append(time.perf_counter() - started)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    return {
        "repeats": repeats,
        "construction_mean_seconds": statistics.mean(construction_times),
        "backward_mean_seconds": statistics.mean(backward_times),
        "backward_minimum_seconds": min(backward_times),
        "forward_mean_seconds": statistics.mean(forward_times),
        "forward_minimum_seconds": min(forward_times),
        "mean_peak_tracemalloc_bytes": statistics.mean(peaks),
        "maximum_peak_tracemalloc_bytes": max(peaks),
        "last_backward": {
            key: {
                "tuple_ids": list(value.tuple_ids),
                "derivation_path_count": value.derivation_path_count,
                "binding_paths": [list(path) for path in value.binding_paths],
            }
            for key, value in backward.items()
        },
        "last_forward": {
            key: {
                "tuple_ids": list(value.tuple_ids),
                "derivation_path_count": value.derivation_path_count,
                "binding_paths": [list(path) for path in value.binding_paths],
            }
            for key, value in forward.items()
        },
    }


def main() -> int:
    adapter = CoreAdapter(run_id="reader-index-benchmark")
    rows, _executor = execute_business_query(adapter)
    snapshot = adapter.validated_snapshot()
    from generation_relation_core.snapshots import validate_snapshot

    validation = validate_snapshot(snapshot, adapter.registry)
    before = measure(
        LegacyScanCoreLineageReader, snapshot, adapter.registry, validation, rows
    )
    after = measure(CoreLineageReader, snapshot, adapter.registry, validation, rows)
    backward_equal = before["last_backward"] == after["last_backward"]
    forward_equal = before["last_forward"] == after["last_forward"]
    output_id = rows[0].tuple_id
    backward_oracle = compare_lineage(
        after["last_backward"][output_id]["tuple_ids"], BUSINESS_BACKWARD[output_id]
    )
    forward_oracle = {
        source: compare_lineage(after["last_forward"][source]["tuple_ids"], expected)
        for source, expected in BUSINESS_FORWARD.items()
    }
    result = {
        "fixture": "fixed adversarial multistage business query",
        "binding_count": len(snapshot.tables.generation_bindings),
        "before_scan_reader": before,
        "after_indexed_reader": after,
        "backward_results_identical": backward_equal,
        "forward_results_identical": forward_equal,
        "backward_false_positives": len(backward_oracle["false_positives"]),
        "backward_false_negatives": len(backward_oracle["false_negatives"]),
        "forward_false_positives": sum(
            len(item["false_positives"]) for item in forward_oracle.values()
        ),
        "forward_false_negatives": sum(
            len(item["false_negatives"]) for item in forward_oracle.values()
        ),
        "indexes_are_temporary_and_rebuildable": True,
        "authoritative_storage_changed": False,
        "database_specific_fields_added": False,
    }
    write_json(ARTIFACT, result)
    return (
        0
        if backward_equal
        and forward_equal
        and not any(
            (
                result["backward_false_positives"],
                result["backward_false_negatives"],
                result["forward_false_positives"],
                result["forward_false_negatives"],
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
