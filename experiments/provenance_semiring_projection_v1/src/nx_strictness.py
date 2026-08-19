from __future__ import annotations

import json
from typing import Any

from .candidate_nx import project_snapshot_to_nx
from .core_capture import core_snapshot_from_events
from .native_nx import evaluate_native_nx
from .workloads import workload_by_id


PAIR_SPECS = [
    {
        "pair_id": "P2-physical-occurrence-structure",
        "dimension": "physical occurrence structure",
        "workload_id": "W11",
        "left": {"variant": "plan_a", "variant_tag": "baseline"},
        "right": {"variant": "plan_b", "variant_tag": "baseline"},
        "required_different_tables": ["generation_occurrences"],
    },
    {
        "pair_id": "P2-evidence",
        "dimension": "evidence",
        "workload_id": "W2",
        "left": {"variation": {"evidence": "method-a"}},
        "right": {"variation": {"evidence": "method-b"}},
        "required_different_tables": ["evidence_records"],
    },
    {
        "pair_id": "P2-environment",
        "dimension": "environment",
        "workload_id": "W2",
        "left": {"variation": {"environment": "runtime-a"}},
        "right": {"variation": {"environment": "runtime-b"}},
        "required_different_tables": ["environment_records"],
    },
    {
        "pair_id": "P2-disposition",
        "dimension": "disposition",
        "workload_id": "W1",
        "left": {"variation": {"disposition": "policy-a"}},
        "right": {"variation": {"disposition": "policy-b"}},
        "required_different_tables": ["explicit_dispositions"],
    },
    {
        "pair_id": "P2-operation-result",
        "dimension": "operation result",
        "workload_id": "W2",
        "left": {"variation": {"operation_result": "route-a"}},
        "right": {"variation": {"operation_result": "route-b"}},
        "required_different_tables": ["generator_operation_results"],
    },
]


def _ordinary_rows(ordinary: bytes) -> list[dict[str, Any]]:
    return json.loads(ordinary)["rows"]


def _execute(workload_id: str, case: dict[str, Any]) -> dict[str, Any]:
    workload = workload_by_id(workload_id)
    variant = case.get("variant")
    ordinary, measurements, snapshot, validation = core_snapshot_from_events(
        workload,
        variant=variant,
        run_id="p2-strictness",
        variant_tag=case.get("variant_tag", "baseline"),
        variation=case.get("variation"),
    )
    native = evaluate_native_nx(workload, variant=variant)
    candidate = project_snapshot_to_nx(snapshot, validation)
    return {
        "ordinary_rows": _ordinary_rows(ordinary),
        "measurements": measurements,
        "snapshot_id": snapshot.snapshot_id,
        "table_counts": snapshot.record["authoritative_table_counts"],
        "table_hashes": snapshot.record["authoritative_table_hashes"],
        "native_source_variables": native["source_variables"],
        "native_outputs": native["outputs"],
        "candidate_source_variables": candidate["source_variables"],
        "candidate_outputs": candidate["outputs"],
    }


def evaluate_nx_strictness() -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = []
    for spec in PAIR_SPECS:
        left = _execute(spec["workload_id"], spec["left"])
        right = _execute(spec["workload_id"], spec["right"])
        ordinary_equal = left["ordinary_rows"] == right["ordinary_rows"]
        native_candidate_exact = (
            left["native_source_variables"] == left["candidate_source_variables"]
            and right["native_source_variables"] == right["candidate_source_variables"]
            and left["native_outputs"] == left["candidate_outputs"]
            and right["native_outputs"] == right["candidate_outputs"]
        )
        nx_equal_across_pair = (
            left["native_source_variables"] == right["native_source_variables"]
            and left["native_outputs"] == right["native_outputs"]
            and left["candidate_source_variables"] == right["candidate_source_variables"]
            and left["candidate_outputs"] == right["candidate_outputs"]
        )
        different_tables = sorted(
            name
            for name in left["table_hashes"]
            if left["table_hashes"][name] != right["table_hashes"][name]
        )
        required_difference_present = set(spec["required_different_tables"]) <= set(different_tables)
        gamma_different = left["snapshot_id"] != right["snapshot_id"]
        supported = all((ordinary_equal, native_candidate_exact, nx_equal_across_pair, required_difference_present, gamma_different))
        pairs.append(
            {
                "pair_id": spec["pair_id"],
                "dimension": spec["dimension"],
                "workload_id": spec["workload_id"],
                "real_execution_count": 2,
                "ordinary_output_equal": ordinary_equal,
                "native_candidate_exact_on_both_sides": native_candidate_exact,
                "nx_equal_across_pair": nx_equal_across_pair,
                "gamma_snapshot_ids_different": gamma_different,
                "left_snapshot_id": left["snapshot_id"],
                "right_snapshot_id": right["snapshot_id"],
                "different_authoritative_tables": different_tables,
                "required_different_tables": spec["required_different_tables"],
                "required_difference_present": required_difference_present,
                "left_counts": left["table_counts"],
                "right_counts": right["table_counts"],
                "supported": supported,
            }
        )
    all_supported = len(pairs) >= 5 and all(pair["supported"] for pair in pairs)
    result = {
        "schema_version": "nx-strictness-counterexamples-v1",
        "claim": "Gamma_G(omega) is strictly richer than its N[X] projection",
        "status": "STRICTNESS_SUPPORTED" if all_supported else "NOT_ESTABLISHED",
        "required_pair_count": 5,
        "actual_pair_count": len(pairs),
        "real_execution_count": sum(pair["real_execution_count"] for pair in pairs),
        "pairs": pairs,
    }
    fibers: dict[str, list[str]] = {}
    for pair in pairs:
        fiber_key = f"{pair['workload_id']}:{pair['pair_id']}"
        fibers[fiber_key] = [pair["left_snapshot_id"], pair["right_snapshot_id"]]
    reverse = {
        "schema_version": "reverse-reconstruction-impossibility-v1",
        "claim": "No total inverse from the observed N[X] values can reconstruct a unique complete Gamma snapshot",
        "status": "NON_INJECTIVITY_WITNESSED" if all_supported else "NOT_ESTABLISHED",
        "reasoning_kind": "finite machine witness of non-injectivity, not a claim beyond the frozen execution domain",
        "projection_fiber_count": len(fibers),
        "fibers_with_multiple_gamma": sum(len(set(values)) > 1 for values in fibers.values()),
        "fibers": fibers,
    }
    return result, reverse

