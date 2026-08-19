from __future__ import annotations

import copy
import statistics
import time
import tracemalloc
from pathlib import Path

from generation_relation_core.errors import CoreV3Error
from generation_relation_core.relation_evidence import RelationEvidenceResolver

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.metrics import write_json
from experiments.database_lineage.src.operators import Projection
from experiments.database_lineage.src.relational_executor import (
    RelationalExecutor,
    base_tuple,
)
from experiments.database_lineage.src.resolver_reference import (
    LegacyScanRelationEvidenceResolver,
)


ARTIFACT = Path("experiments/database_lineage/artifacts/resolver_benchmark.json")
MUTATIONS = (
    "missing_primary",
    "duplicate_primary",
    "bad_authority",
    "missing_related",
    "material_mismatch",
    "missing_operation",
    "duplicate_operation",
)


def mutate(tables, name: str) -> None:
    binding = tables.generation_bindings[0]
    binding_id = binding["generation_binding_id"]
    link = next(row for row in tables.evidence_links if row["subject_id"] == binding_id)
    evidence = next(
        row
        for row in tables.evidence_records
        if row["evidence_id"] == link["evidence_id"]
    )
    operation = next(
        row
        for row in tables.generator_operation_results
        if binding_id in row["produced_entity_ids"]
    )
    if name == "missing_primary":
        tables.evidence_links.remove(link)
    elif name == "duplicate_primary":
        tables.evidence_links.append(copy.deepcopy(link))
    elif name == "bad_authority":
        evidence["evidence_authority"] = "not_authorized"
    elif name == "missing_related":
        evidence["related_record_ids"] = []
    elif name == "material_mismatch":
        evidence["artifact_sha256"] = "0" * 64
    elif name == "missing_operation":
        operation["produced_entity_ids"].remove(binding_id)
    elif name == "duplicate_operation":
        tables.generator_operation_results.append(copy.deepcopy(operation))


def outcome(resolver, tables):
    try:
        value = resolver.resolve(tables, preverified=True)
    except CoreV3Error as exc:
        return {"status": "failure", "reason_code": exc.reason_code}
    return {"status": "success", "resolved_bindings": len(value)}


def measure(resolver_type, tables, repeats: int = 20) -> dict:
    timings = []
    peaks = []
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        value = resolver_type().resolve(tables, preverified=True)
        timings.append(time.perf_counter() - started)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        assert len(value) == len(tables.generation_bindings)
    return {
        "repeats": repeats,
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "minimum_seconds": min(timings),
        "maximum_seconds": max(timings),
        "mean_peak_tracemalloc_bytes": statistics.mean(peaks),
        "maximum_peak_tracemalloc_bytes": max(peaks),
    }


def main() -> int:
    adapter = CoreAdapter(run_id="resolver-benchmark")
    executor = RelationalExecutor(adapter)
    rows = [base_tuple("resolver:source:0", "ResolverFixture", {"value": 1}, 0)]
    for index in range(300):
        rows = executor.projection(
            rows,
            stage=f"resolver_stage_{index:04d}",
            projections=[Projection("value", "value", lambda row: row["value"])],
        )
    snapshot = adapter.validated_snapshot()
    tables = snapshot.tables
    equivalence = {
        "success": {
            "legacy": outcome(
                LegacyScanRelationEvidenceResolver(), copy.deepcopy(tables)
            ),
            "indexed": outcome(RelationEvidenceResolver(), copy.deepcopy(tables)),
        }
    }
    for mutation in MUTATIONS:
        left = copy.deepcopy(tables)
        right = copy.deepcopy(tables)
        mutate(left, mutation)
        mutate(right, mutation)
        equivalence[mutation] = {
            "legacy": outcome(LegacyScanRelationEvidenceResolver(), left),
            "indexed": outcome(RelationEvidenceResolver(), right),
        }
    all_equal = all(item["legacy"] == item["indexed"] for item in equivalence.values())
    result = {
        "fixture": "fixed 300-stage modality-independent generation chain",
        "binding_count": len(tables.generation_bindings),
        "operation_count": len(tables.generator_operation_results),
        "evidence_count": len(tables.evidence_records),
        "before": measure(LegacyScanRelationEvidenceResolver, tables),
        "after": measure(RelationEvidenceResolver, tables),
        "validation_results_identical": all_equal,
        "success_and_failure_reason_codes": equivalence,
        "indexed_structures": [
            "produced entity -> all candidate operations",
            "operation -> produced entity set",
            "operation -> evidence set",
            "binding -> all primary-evidence candidates",
            "entity id -> entity",
        ],
        "authoritative_storage_changed": False,
        "core_schema_changed": False,
        "database_specific_semantics_added": False,
    }
    write_json(ARTIFACT, result)
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
