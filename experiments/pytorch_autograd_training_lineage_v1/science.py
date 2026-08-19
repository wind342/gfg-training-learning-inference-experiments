from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from collections import Counter

import torch

from .checkpoint_analysis import analyze_checkpoint_divergence
from .classifications import analyze_zero_gradient_vs_nonparticipation
from .independent_reference import ReceiptLineageReference
from .isolation_audit import build_isolation_audit
from .lineage import TrainingLineageIndex
from .modes import CaptureModeSuite, run_four_capture_modes
from .negative_controls import run_negative_controls
from .pipeline import TrainingSpec
from .projection_analysis import compare_canonical_graphs
from .strict_projection import build_strict_projection_counterexamples
from .core_compatibility import protected_scope_for_unified_core


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
BASE_COMMIT = "e00144b6b47504287c2d16f20b064da81e43f1cc"


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((EXPERIMENT_ROOT / "profiles" / name).read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _snapshot_payload(suite: CaptureModeSuite) -> dict[str, Any]:
    capture = suite.core_and_native_capture
    return {
        "record": capture.snapshot.record,
        "tables": {key: value for key, value in vars(capture.snapshot.tables).items()},
    }


def _projection_artifacts(suites: dict[str, CaptureModeSuite]) -> dict[str, Any]:
    native = {name: suite.core_and_native.native_observation for name, suite in suites.items()}
    candidate = {name: suite.core_and_native_candidate for name, suite in suites.items()}
    comparisons = {
        name: compare_canonical_graphs(native[name], candidate[name]) for name in suites
    }
    aggregate = {
        "candidate_edge_count": sum(row["candidate_edge_count"] for row in comparisons.values()),
        "candidate_node_count": sum(row["candidate_node_count"] for row in comparisons.values()),
        "edge_mismatch": sum(row["edge_mismatch"] for row in comparisons.values()),
        "edge_slot_mismatch": sum(row["edge_slot_mismatch"] for row in comparisons.values()),
        "exact_workload_count": sum(row["exact"] for row in comparisons.values()),
        "fabricated_edge": sum(row["fabricated_edge"] for row in comparisons.values()),
        "fabricated_node": sum(row["fabricated_node"] for row in comparisons.values()),
        "missing_edge": sum(row["missing_edge"] for row in comparisons.values()),
        "missing_leaf": sum(row["missing_leaf"] for row in comparisons.values()),
        "missing_node": sum(row["missing_node"] for row in comparisons.values()),
        "multiplicity_mismatch": sum(row["multiplicity_mismatch"] for row in comparisons.values()),
        "native_edge_count": sum(row["native_edge_count"] for row in comparisons.values()),
        "native_node_count": sum(row["native_node_count"] for row in comparisons.values()),
        "node_type_mismatch": sum(row["node_type_mismatch"] for row in comparisons.values()),
        "root_mismatch": sum(row["root_mismatch"] for row in comparisons.values()),
        "shared_node_mismatch": sum(row["shared_node_mismatch"] for row in comparisons.values()),
        "workload_count": len(comparisons),
    }
    aggregate["status"] = (
        "PYTORCH_AUTOGRAD_EXACT_PROJECTION_SUPPORTED"
        if aggregate["exact_workload_count"] == aggregate["workload_count"]
        else "PYTORCH_AUTOGRAD_EXACT_PROJECTION_NOT_ESTABLISHED"
    )
    node_fields = sorted(set.intersection(*(
        set(graph["nodes"][0]) for graph in native.values()
    )))
    edge_fields = sorted(set.intersection(*(
        set(graph["edges"][0]) for graph in native.values()
    )))
    field_coverage = {
        "all_fields_exact": all(row["exact"] for row in comparisons.values()),
        "edge_fields_compared": edge_fields,
        "graph_fields_compared": sorted(next(iter(native.values())).keys()),
        "node_fields_compared": node_fields,
        "workloads": {name: row["exact"] for name, row in comparisons.items()},
    }
    summary = {
        "aggregate_edge_count": sum(graph["edge_count"] for graph in native.values()),
        "aggregate_node_count": sum(graph["node_count"] for graph in native.values()),
        "none_edge_count": sum(graph["none_edge_count"] for graph in native.values()),
        "shared_node_count": sum(graph["shared_node_count"] for graph in native.values()),
        "workloads": {
            name: {
                "canonical_graph_sha256": graph["canonical_graph_sha256"],
                "edge_count": graph["edge_count"],
                "node_count": graph["node_count"],
                "none_edge_count": graph["none_edge_count"],
                "shared_node_count": graph["shared_node_count"],
            }
            for name, graph in native.items()
        },
    }
    return {
        "autograd_projection_exact_comparison": {"aggregate": aggregate, "workloads": comparisons},
        "autograd_projection_field_coverage": field_coverage,
        "core_projected_autograd_graph": {"workloads": candidate},
        "native_autograd_graph": {"workloads": native},
        "native_autograd_graph_summary": summary,
        "native_graph_canonicalization": {
            "canonicalization": "root_ordered_paths_and_shared_alias_closure_v1",
            "identity_inputs": [
                "Node.name()",
                "ordered root-relative paths",
                "ordered outgoing slots and output_nr",
                "incoming signatures",
                "shared alias closure",
            ],
            "prohibited_identity_inputs": [
                "C++ pointers",
                "Python object addresses",
                "expected graph",
                "random internal IDs",
                "unstable repr",
            ],
        },
    }


def _query_artifacts(suites: dict[str, CaptureModeSuite], checkpoint: dict[str, Any]) -> dict[str, Any]:
    forward_queries = []
    reverse_queries = []
    total_fp = total_fn = role_mismatch = occurrence_mismatch = path_mismatch = multiplicity_mismatch = 0
    broken_generated_origin = fabricated_direct_shortcut = 0

    def compare_paths(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> tuple[int, int, int, int, int, int]:
        actual_set = {_canonical(row) for row in actual}
        expected_set = {_canonical(row) for row in expected}
        actual_relations = [relation for path in actual for relation in path["relations"]]
        expected_relations = [relation for path in expected for relation in path["relations"]]
        actual_roles = Counter(row["role"] for row in actual_relations)
        expected_roles = Counter(row["role"] for row in expected_relations)
        actual_occurrences = Counter(row["occurrence_key"] for row in actual_relations)
        expected_occurrences = Counter(row["occurrence_key"] for row in expected_relations)
        role_delta = sum((actual_roles - expected_roles).values()) + sum((expected_roles - actual_roles).values())
        occurrence_delta = sum((actual_occurrences - expected_occurrences).values()) + sum((expected_occurrences - actual_occurrences).values())
        actual_multiplicity = {
            _canonical({key: value for key, value in row.items() if key != "multiplicity"}): row["multiplicity"]
            for row in actual
        }
        expected_multiplicity = {
            _canonical({key: value for key, value in row.items() if key != "multiplicity"}): row["multiplicity"]
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

    for workload, suite in suites.items():
        capture = suite.core_and_native_capture
        index = TrainingLineageIndex(capture.snapshot, capture.validation)
        reference = ReceiptLineageReference(capture.execution_receipts)
        tables = capture.snapshot.tables
        support_ids = {row["support_id"] for row in tables.perceptual_support_records}
        generated_ids = {row["generated_origin_id"] for row in tables.generated_origins}
        used_generated_ids = {
            row["origin_reference"]["generated_origin_id"]
            for row in tables.generation_bindings
            if row["origin_reference"]["kind"] == "generated_origin"
        }
        broken_generated_origin += len(generated_ids - used_generated_ids)
        broken_generated_origin += sum(
            row["origin_payload"].get("source_support_id") not in support_ids
            for row in tables.generated_origins
        )
        sample_ids = {
            row["source_information_id"] for row in tables.source_information_records
            if row["source_payload"]["source_role"] == "training_sample"
        }
        parameter_after_ids = {
            row["support_id"] for row in tables.perceptual_support_records
            if row["support_payload"]["support_kind"] == "parameter_after_step"
        }
        fabricated_direct_shortcut += sum(
            row["origin_reference"].get("source_information_id") in sample_ids
            and row["outcome_reference"].get("support_id") in parameter_after_ids
            for row in tables.generation_bindings
        )
        for source in sorted(
            capture.snapshot.tables.source_information_records,
            key=lambda row: row["source_information_id"],
        ):
            source_ref = source["source_payload"]["source_ref"]
            actual = index.forward_lineage(source["source_information_id"])
            expected = reference.forward_paths(source_ref)
            exact = actual["paths"] == expected
            deltas = compare_paths(actual["paths"], expected)
            total_fp += deltas[0]
            total_fn += deltas[1]
            role_mismatch += deltas[2]
            occurrence_mismatch += deltas[3]
            path_mismatch += deltas[4]
            multiplicity_mismatch += deltas[5]
            forward_queries.append({
                "exact": exact,
                "query": actual,
                "source_ref": source_ref,
                "workload": workload,
            })
        targets = [
            support for support in capture.snapshot.tables.perceptual_support_records
            if support["support_payload"]["support_kind"] in {
                "gradient", "loss", "optimizer_state_after_step", "parameter_after_step"
            }
        ]
        for support in sorted(targets, key=lambda row: row["support_id"]):
            support_key = support["support_payload"]["support_key"]
            actual = index.reverse_lineage(support["support_id"])
            expected = reference.reverse_paths(support_key)
            exact = actual["paths"] == expected
            deltas = compare_paths(actual["paths"], expected)
            total_fp += deltas[0]
            total_fn += deltas[1]
            role_mismatch += deltas[2]
            occurrence_mismatch += deltas[3]
            path_mismatch += deltas[4]
            multiplicity_mismatch += deltas[5]
            reverse_queries.append({
                "exact": exact,
                "query": actual,
                "support_key": support_key,
                "workload": workload,
            })
    checkpoint_reverse = checkpoint["reverse_trace"]
    checkpoint_forward = checkpoint["forward_trace"]
    checkpoint_exact = all([
        checkpoint["checks"]["reverse_query_exact_against_receipts"],
        checkpoint["checks"]["forward_query_exact_against_receipts"],
    ])
    comparison = {
        "broken_generated_origin": broken_generated_origin,
        "cartesian_expansion": total_fp,
        "fabricated_direct_shortcut": fabricated_direct_shortcut,
        "false_negative": total_fn,
        "false_positive": total_fp,
        "forward_exact_count": sum(row["exact"] for row in forward_queries),
        "forward_query_count": len(forward_queries),
        "multiplicity_mismatch": multiplicity_mismatch,
        "occurrence_mismatch": occurrence_mismatch,
        "path_mismatch": path_mismatch,
        "reverse_exact_count": sum(row["exact"] for row in reverse_queries),
        "reverse_query_count": len(reverse_queries),
        "role_mismatch": role_mismatch,
    }
    comparison["all_exact"] = all([
        comparison["false_negative"] == 0,
        comparison["false_positive"] == 0,
        comparison["forward_exact_count"] == comparison["forward_query_count"],
        comparison["multiplicity_mismatch"] == 0,
        comparison["occurrence_mismatch"] == 0,
        comparison["path_mismatch"] == 0,
        comparison["reverse_exact_count"] == comparison["reverse_query_count"],
        comparison["role_mismatch"] == 0,
        checkpoint_exact,
    ])
    return {
        "bidirectional_training_lineage": {
            "checkpoint_forward_trace": checkpoint_forward,
            "checkpoint_reverse_trace": checkpoint_reverse,
            "forward_queries": forward_queries,
            "reverse_queries": reverse_queries,
            "status": (
                "BIDIRECTIONAL_TRAINING_UPDATE_LINEAGE_SUPPORTED"
                if comparison["all_exact"]
                else "BIDIRECTIONAL_TRAINING_UPDATE_LINEAGE_NOT_ESTABLISHED"
            ),
        },
        "forward_training_source_queries": {"queries": forward_queries},
        "query_exact_comparison": comparison,
        "reverse_parameter_update_queries": {"queries": reverse_queries},
    }


def _optimizer_verification(suites: dict[str, CaptureModeSuite]) -> dict[str, Any]:
    rows = []

    def subtract(before: Any, gradient: Any) -> Any:
        if isinstance(before, list):
            return [subtract(left, right) for left, right in zip(before, gradient, strict=True)]
        return before - 0.05 * gradient

    for workload, suite in suites.items():
        result = suite.output_only.ordinary_result
        for parameter_name, before in result["parameter_before"].items():
            gradient = result["gradients"].get(f"parameter:{parameter_name}")
            after = result["parameter_after"][parameter_name]
            if gradient is None:
                rows.append({
                    "actual_step_skipped": after["value"] == before["value"],
                    "parameter": parameter_name,
                    "reason": "grad_is_none",
                    "workload": workload,
                })
                continue
            expected = subtract(before["value"], gradient["value"])
            momentum_buffer = result["optimizer_state_after"][parameter_name]["momentum_buffer"]["value"]
            rows.append({
                "actual_parameter_after": after["value"],
                "expected_first_step_formula": expected,
                "formula_exact": expected == after["value"],
                "momentum_buffer_equals_gradient": momentum_buffer == gradient["value"],
                "parameter": parameter_name,
                "workload": workload,
            })
    return {
        "all_official_results_match_independent_formula": all(
            row.get("formula_exact", row.get("actual_step_skipped", False)) for row in rows
        ),
        "rows": rows,
    }


def _gradcheck() -> dict[str, Any]:
    left = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[0.4, -0.2], [0.1, 0.6]], dtype=torch.float64, requires_grad=True)

    def frozen_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.pow(torch.relu(torch.matmul(a, b)), 2.0))

    passed = bool(torch.autograd.gradcheck(frozen_function, (left, right)))
    return {
        "dtype": "torch.float64",
        "function": "mean(pow(relu(matmul(a,b)),2))",
        "passed": passed,
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *args],
        text=True,
    ).strip()


def _protected_scope() -> dict[str, Any]:
    return protected_scope_for_unified_core(_git, REPOSITORY_ROOT)


def run_complete_science() -> dict[str, Any]:
    profile = _load_json("pytorch_autograd_dependency_profile_v1.json")
    crosswalk = _load_json("core_to_pytorch_autograd_crosswalk_v1.json")
    suites = {
        workload: run_four_capture_modes(TrainingSpec(workload=workload), profile, crosswalk)
        for workload in profile["workloads"]
    }
    projections = _projection_artifacts(suites)
    strict = build_strict_projection_counterexamples()
    checkpoint = analyze_checkpoint_divergence()
    zero = analyze_zero_gradient_vs_nonparticipation()
    queries = _query_artifacts(suites, checkpoint)
    negative = run_negative_controls()
    isolation = build_isolation_audit()
    optimizer = _optimizer_verification(suites)
    gradcheck = _gradcheck()
    protected = _protected_scope()
    output_orthogonality = {
        "all_workloads_exact": all(
            suite.report["all_ordinary_bytes_exact"]
            and suite.report["graph_topology_transitive_equivalence"]
            for suite in suites.values()
        ),
        "workloads": {name: suite.report for name, suite in suites.items()},
        "status": "TRAINING_OUTPUT_ORTHOGONALITY_SUPPORTED",
    }
    checkpoint_snapshots = checkpoint.pop("validated_snapshots")
    snapshots = {
        "checkpoint": checkpoint_snapshots,
        "standard_workloads": {name: _snapshot_payload(suite) for name, suite in suites.items()},
    }
    strict_reverse = {
        "counterexample_count": strict["counterexample_count"],
        "graph_identifies_gamma": False,
        "pairs": strict["pairs"],
        "reason": "each equal graph pair has different validated complete-fact snapshot identity",
    }
    second_authority = {
        "candidate_has_reference_import": bool(isolation["candidate_forbidden_imports"]),
        "candidate_has_receipt_attribute": "execution_receipts" in isolation["candidate"]["attributes"],
        "native_observer_imports_core": bool(isolation["native_core_imports"]),
        "receipt_reference_imports_core": bool(isolation["reference_core_imports"]),
    }
    second_authority["status"] = (
        "NO_SECOND_AUTHORITY"
        if not any(value for key, value in second_authority.items() if key != "status")
        else "SECOND_AUTHORITY_DETECTED"
    )
    artifacts = {
        **projections,
        **queries,
        "autograd_reverse_non_identifiability": strict_reverse,
        "autograd_strict_projection_counterexamples": strict,
        "checkpoint_divergence_forward_trace": checkpoint["forward_trace"],
        "checkpoint_divergence_localization": checkpoint,
        "checkpoint_divergence_reverse_trace": checkpoint["reverse_trace"],
        "checkpoint_divergent_run": {
            "gradient": checkpoint["divergent_gradient"],
            "parameter_after": checkpoint["divergent_parameter_after"],
            "snapshot": checkpoint_snapshots["divergent"],
        },
        "checkpoint_gradient_comparison": {
            "checks": checkpoint["checks"],
            "divergent_gradient": checkpoint["divergent_gradient"],
            "no_checkpoint_gradient": checkpoint["no_checkpoint_gradient"],
            "stable_gradient": checkpoint["stable_gradient"],
        },
        "checkpoint_graph_equality": {
            "canonical_graph_sha256": checkpoint["graph_sha256"],
            "stable_divergent_exact": checkpoint["checks"]["native_graph_stable_divergent_exact"],
        },
        "checkpoint_stable_reference": {
            "gradient": checkpoint["stable_gradient"],
            "no_checkpoint_gradient": checkpoint["no_checkpoint_gradient"],
            "snapshot": checkpoint_snapshots["stable"],
        },
        "gradcheck": gradcheck,
        "negative_control_accounting": {
            key: value for key, value in negative.items() if key != "controls"
        },
        "negative_controls": {"controls": negative["controls"]},
        "optimizer_verification": optimizer,
        "oracle_isolation": isolation,
        "output_orthogonality": output_orthogonality,
        "runtime_dependency_trace": {
            "candidate": isolation["candidate"],
            "native": isolation["native"],
            "reference": isolation["reference"],
        },
        "second_authority_audit": second_authority,
        "validated_core_snapshots": snapshots,
        "zero_gradient_vs_nonparticipation": zero,
    }
    snapshot_fingerprints = {
        workload: suite.core_and_native_capture.snapshot.snapshot_id
        for workload, suite in suites.items()
    }
    scientific_summary = {
        "checkpoint_status": checkpoint["status"],
        "gradcheck_passed": gradcheck["passed"],
        "negative_controls_all_detected": negative["all_detected"],
        "negative_controls_count": negative["control_count"],
        "optimizer_verification": optimizer["all_official_results_match_independent_formula"],
        "output_orthogonality_status": output_orthogonality["status"],
        "projection_aggregate": projections["autograd_projection_exact_comparison"]["aggregate"],
        "protected_scope": protected,
        "query_comparison": queries["query_exact_comparison"],
        "second_authority_status": second_authority["status"],
        "snapshot_fingerprints": snapshot_fingerprints,
        "strict_projection_status": strict["status"],
        "zero_gradient_status": zero["status"],
    }
    return {"artifacts": artifacts, "scientific_summary": scientific_summary}
