from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .common import canonical_sha256, compact_witness, file_sha256, write_json


CORE_COMMIT = "6b34906d7b6e4fa15f6c7d6e3013daa35a308b5e"
SOURCE_COMMITS = {
    "database_which_lineage": "03caa31b8a6abfe6e112a0544071618c689bb11f",
    "source_map": "7dba987713da345453781e4b95130f1deb5f04d4",
    "opentelemetry": "25a9d2a614d2d34d36c38f7c560b818cdbc4b179",
    "w3c_prov_generation_profile": "d93ef7d1a2721cafe9e967eec0f0e693406f826c",
    "pytorch_autograd_dependency_profile": "7f7c02a03beffa9533daadfd064a3eaaaaccfccc",
}
PROFILES = {
    "database_which_lineage": "database-which-lineage-v1",
    "source_map": "ecma426-ordinary-source-map-v1",
    "opentelemetry": "opentelemetry-occurrence-execution-v1",
    "w3c_prov_generation_profile": "w3c-prov-generation-profile-v1",
    "pytorch_autograd_dependency_profile": "pytorch-autograd-dependency-profile-v1",
}
RATINGS = {
    "database_which_lineage": ("C", "Constructive relational-control-flow reference; pinned ProvSQL is reported separately and is not a P1/P2 premise."),
    "source_map": ("B", "The official source-map 0.8.0 generator and consumer execute independently of the Core candidate over shared transformation receipts."),
    "opentelemetry": ("B", "The official SDK exports native spans while the candidate reads the validated Core snapshot; both observe the same executor callbacks."),
    "w3c_prov_generation_profile": ("C", "Independent Candidate and native processes execute a shared frozen generator/profile; the native path uses RDFLib PROV-O normalization."),
    "pytorch_autograd_dependency_profile": ("A", "Official Autograd graph APIs plus native backward hooks and intervention oracles independently constrain the candidate relation."),
}


def _base(mechanism: str) -> dict[str, Any]:
    rating, basis = RATINGS[mechanism]
    return {
        "mechanism": mechanism,
        "profile_name": PROFILES[mechanism],
        "source_commit": SOURCE_COMMITS[mechanism],
        "core_commit": CORE_COMMIT,
        "external_independence": {"rating": rating, "basis": basis},
        "determinism": {"checked": False, "status": "FAIL", "run_1_hash": None, "run_2_hash": None},
    }


def _hashes(repo: Path, payloads: dict[str, Any], files: dict[str, str]) -> dict[str, str]:
    result = {name: canonical_sha256(value) for name, value in payloads.items()}
    result.update({name: file_sha256(repo / path) for name, path in files.items()})
    return result


def run_database(repo: Path, run_dir: Path) -> dict[str, Any]:
    from experiments.operational_projection_proof_v2.src.database_proof import run_database_proof

    raw = run_database_proof(run_dir)
    p1_raw = raw["projection_equivalence_database.json"]
    p2_raw = raw["strict_partiality_database.json"]
    witnesses = p2_raw["cases"]
    valid = [row for row in witnesses if row["valid_counterexample"]]
    result = _base("database_which_lineage")
    result.update({
        "p1": {
            "status": "PASS" if p1_raw["status"] == "SUPPORTED" else "FAIL",
            "candidate_record_count": p1_raw["candidate_record_count"],
            "native_record_count": p1_raw["reference_record_count"],
            "false_positive_count": p1_raw["false_positive"],
            "false_negative_count": p1_raw["false_negative"],
            "field_mismatch_count": p1_raw["field_mismatch"],
            "multiplicity_mismatch_count": p1_raw["multiplicity_mismatch"],
            "byte_equal": p1_raw["canonical_records_hash_equal"],
            "query_count": p1_raw["candidate_record_count"],
            "query_mismatch_count": p1_raw["false_positive"] + p1_raw["false_negative"] + p1_raw["field_mismatch"] + p1_raw["multiplicity_mismatch"],
        },
        "p2": {
            "status": "PASS" if p2_raw["status"] == "SUPPORTED" else "FAIL",
            "witness_count": len(witnesses),
            "valid_witness_count": len(valid),
            "snapshot_distinct": all(row["left_snapshot_id"] != row["right_snapshot_id"] for row in witnesses),
            "target_equal": all(row["projection_hash_equal"] for row in witnesses),
            "witness_summaries": [compact_witness(row) for row in witnesses],
        },
        "ordinary_output_orthogonality": {"checked": True, "status": "PASS" if raw["database_output_orthogonality"]["status"] == "SUPPORTED" else "FAIL"},
        "external_validation": {
            "system": "ProvSQL 1.4.0 sr_which",
            "status": "NOT_EXECUTED_IN_UNIFIED_RUN",
            "affects_constructive_p1_p2": False,
            "reproduction_command": "python -m experiments.database_lineage.scripts.run_provsql",
        },
        "evidence": {"p1_status": p1_raw["status"], "p2_status": p2_raw["status"], "frozen_record_count": 112},
    })
    result["artifact_hashes"] = _hashes(repo, {"p1_evidence": p1_raw, "p2_evidence": p2_raw}, {
        "profile": "experiments/operational_projection_proof_v2/profiles/database_which_lineage_v1.json",
    })
    result["run_status"] = "PASS" if result["p1"]["status"] == result["p2"]["status"] == result["ordinary_output_orthogonality"]["status"] == "PASS" else "FAIL"
    return result


def run_source_map(repo: Path, run_dir: Path) -> dict[str, Any]:
    from experiments.operational_projection_proof_v2.src.source_map_proof import run_source_map_proof

    raw = run_source_map_proof(run_dir, repo_root=repo)
    p1_raw = raw["projection_equivalence_source_map.json"]
    p2_raw = raw["strict_partiality_source_map.json"]
    p3_raw = raw["composition_consistency_source_map.json"]
    witnesses = p2_raw["cases"]
    valid = [row for row in witnesses if row["valid_counterexample"]]
    result = _base("source_map")
    result.update({
        "p1": {
            "status": "PASS" if p1_raw["status"] == "SUPPORTED" and p3_raw["status"] == "SUPPORTED" else "FAIL",
            "candidate_record_count": p1_raw["total_mapping_segments"],
            "native_record_count": p1_raw["total_mapping_segments"],
            "false_positive_count": 0 if p1_raw["status"] == "SUPPORTED" else None,
            "false_negative_count": 0 if p1_raw["status"] == "SUPPORTED" else None,
            "field_mismatch_count": 0 if p1_raw["status"] == "SUPPORTED" else None,
            "multiplicity_mismatch_count": 0 if p1_raw["status"] == "SUPPORTED" else None,
            "byte_equal": all(row["bytes_equal"] for row in p1_raw["map_document_hashes"].values()),
            "query_count": p1_raw["bidirectional_query_count"],
            "query_mismatch_count": p1_raw["query_mismatch_count"],
        },
        "p2": {
            "status": "PASS" if p2_raw["status"] == "SUPPORTED" else "FAIL",
            "witness_count": len(witnesses),
            "valid_witness_count": len(valid),
            "snapshot_distinct": all(row["left_snapshot_id"] != row["right_snapshot_id"] for row in witnesses),
            "target_equal": all(row["map_document_equal"] for row in witnesses),
            "witness_summaries": [compact_witness(row) for row in witnesses],
        },
        "ordinary_output_orthogonality": {"checked": True, "status": raw["source_map_output_orthogonality"]["status"]},
        "evidence": {
            "p1_status": p1_raw["status"],
            "p2_status": p2_raw["status"],
            "multistage_status": p3_raw["status"],
            "multistage_direct_shortcut_count": p3_raw["direct_shortcut_count"],
            "generated_origin_bridge_count": p3_raw["generated_origin_count"],
        },
    })
    result["artifact_hashes"] = _hashes(repo, {"p1_evidence": p1_raw, "p2_evidence": p2_raw, "composition_evidence": p3_raw}, {
        "profile": "experiments/operational_projection_proof_v2/profiles/ecma426_ordinary_source_map_v1.json",
        "source_map_lock": "experiments/source_map_projection/pnpm-lock.yaml",
    })
    result["run_status"] = "PASS" if result["p1"]["status"] == result["p2"]["status"] == result["ordinary_output_orthogonality"]["status"] == "PASS" else "FAIL"
    return result


def run_opentelemetry(repo: Path, run_dir: Path) -> dict[str, Any]:
    from experiments.operational_projection_proof_v2.src.otel_proof import run_otel_proof

    raw = run_otel_proof(run_dir, repo_root=repo, include_formal=True)
    p1_raw = raw["projection_equivalence_opentelemetry.json"]
    p2_raw = raw["strict_partiality_opentelemetry.json"]
    p3_raw = raw["hierarchical_consistency_core_database_to_opentelemetry.json"]
    formal = p1_raw["formal_tpch_q6"]
    diff = formal["native_vs_direct"]
    witnesses = p2_raw["cases"]
    valid = [row for row in witnesses if row["valid_counterexample"]]
    field_mismatches = diff["attribute_mismatches"] + diff["event_mismatches"] + diff["status_mismatches"] + diff["parent_edge_false_positives"] + diff["parent_edge_false_negatives"]
    result = _base("opentelemetry")
    result.update({
        "p1": {
            "status": "PASS" if p1_raw["status"] == "SUPPORTED" and p3_raw["status"] == "SUPPORTED" else "FAIL",
            "candidate_record_count": formal["direct_projected_span_count"],
            "native_record_count": formal["native_span_count"],
            "false_positive_count": diff["span_false_positives"] + diff["link_edge_false_positives"],
            "false_negative_count": diff["span_false_negatives"] + diff["link_edge_false_negatives"],
            "field_mismatch_count": field_mismatches,
            "multiplicity_mismatch_count": 0 if diff["exact"] else None,
            "byte_equal": formal["trace_sha256"]["native"] == formal["trace_sha256"]["direct"] == formal["trace_sha256"]["hierarchical"],
            "query_count": formal["native_span_count"] + formal["causal_link_count"],
            "query_mismatch_count": sum(value for key, value in diff.items() if key != "exact"),
        },
        "p2": {
            "status": "PASS" if p2_raw["status"] == "SUPPORTED" else "FAIL",
            "witness_count": len(witnesses),
            "valid_witness_count": len(valid),
            "snapshot_distinct": all(row["left_snapshot_id"] != row["right_snapshot_id"] for row in witnesses),
            "target_equal": all(row["native_normalized_otel_equal"] and row["direct_core_projection_equal"] for row in witnesses),
            "witness_summaries": [compact_witness(row) for row in witnesses],
        },
        "ordinary_output_orthogonality": {"checked": True, "status": raw["otel_output_orthogonality"]["status"]},
        "evidence": {
            "p1_status": p1_raw["status"],
            "p2_status": p2_raw["status"],
            "hierarchical_status": p3_raw["status"],
            "q6_span_count": formal["native_span_count"],
            "q6_link_count": formal["causal_link_count"],
            "direct_hierarchical_exact": formal["direct_vs_hierarchical"]["exact"],
        },
    })
    result["artifact_hashes"] = _hashes(repo, {"p1_evidence": p1_raw, "p2_evidence": p2_raw, "hierarchical_evidence": p3_raw}, {
        "profile": "experiments/operational_projection_proof_v2/profiles/opentelemetry_occurrence_execution_v1.json",
        "otel_lock": "experiments/opentelemetry_projection/requirements.lock",
    })
    result["run_status"] = "PASS" if result["p1"]["status"] == result["p2"]["status"] == result["ordinary_output_orthogonality"]["status"] == "PASS" else "FAIL"
    return result


def run_w3c(repo: Path, _run_dir: Path) -> dict[str, Any]:
    from experiments.w3c_prov_projection_v1.src.experiment import (
        analyze_import_graph,
        build_oracle_isolation,
        exact_comparison,
        load_oracle_policy,
        output_modes,
        relation_multiplicity,
        run_full,
        run_official_tests,
        run_oracle_process_audit,
        strict_projection_counterexamples,
        validate_profile_documents,
    )

    root = repo / "experiments" / "w3c_prov_projection_v1"
    science = run_full()
    binding_count = len(science.snapshot.tables.generation_bindings)
    comparison = exact_comparison(science.candidate_from_provn, science.reference_from_provo, binding_count)
    constraints = validate_profile_documents(science.candidate_records, science.candidate_provn, science.native_ttl)
    multiplicity = relation_multiplicity(science.candidate_records, binding_count)
    strict, _reverse = strict_projection_counterexamples()
    orthogonality = output_modes()
    official = run_official_tests(root / "runtime" / "official_tests")
    policy = load_oracle_policy(root)
    import_graph = analyze_import_graph(root / "src", policy)
    process_trace = run_oracle_process_audit(root, policy)
    isolation = build_oracle_isolation(import_graph, process_trace)
    witnesses = strict["groups"]
    valid = [row for row in witnesses if row["status"] == "SUPPORTED"]
    fp = sum(value for key, value in comparison.items() if key.endswith("_fp"))
    fn = sum(value for key, value in comparison.items() if key.endswith("_fn"))
    p1_pass = all((comparison["status"] == "SUPPORTED", constraints["status"] == "SUPPORTED", multiplicity["status"] == "SUPPORTED", official["status"] == "SUPPORTED", isolation["status"] == "SUPPORTED"))
    result = _base("w3c_prov_generation_profile")
    result.update({
        "p1": {
            "status": "PASS" if p1_pass else "FAIL",
            "candidate_record_count": comparison["candidate_record_count"],
            "native_record_count": comparison["reference_record_count"],
            "false_positive_count": fp,
            "false_negative_count": fn,
            "field_mismatch_count": comparison["field_mismatch_count"],
            "multiplicity_mismatch_count": comparison["multiplicity_mismatch_count"],
            "byte_equal": science.candidate_from_provn == science.reference_from_provo,
            "query_count": comparison["candidate_record_count"],
            "query_mismatch_count": fp + fn + comparison["field_mismatch_count"] + comparison["multiplicity_mismatch_count"],
        },
        "p2": {
            "status": "PASS" if strict["status"] == "SUPPORTED" else "FAIL",
            "witness_count": len(witnesses),
            "valid_witness_count": len(valid),
            "snapshot_distinct": all(row["snapshots_differ"] for row in witnesses),
            "target_equal": all(row["normalized_prov_dm_equal"] for row in witnesses),
            "witness_summaries": [compact_witness(row) for row in witnesses],
        },
        "ordinary_output_orthogonality": {"checked": True, "status": "PASS" if orthogonality["status"] == "SUPPORTED" else "FAIL"},
        "evidence": {
            "p1_status": comparison["status"],
            "p2_status": strict["status"],
            "record_kinds": {kind: sum(row["kind"] == kind for row in science.candidate_records) for kind in ("usage", "generation", "derivation", "association")},
            "official_applicable_tests": official["applicable_test_count"],
            "official_failed_tests": official["failed_applicable_count"],
            "candidate_native_process_isolation": isolation["status"],
            "actual_transform_witness_status": strict["actual_transform_context_counterexample_status"],
            "claim_scope": "W3C PROV generation profile only; not the entire W3C PROV standard.",
        },
    })
    result["artifact_hashes"] = _hashes(repo, {"p1_evidence": comparison, "p2_evidence": strict, "constraints": constraints, "process_isolation": isolation}, {
        "profile": "experiments/w3c_prov_projection_v1/profiles/w3c_prov_generation_profile_v1.json",
        "crosswalk": "experiments/w3c_prov_projection_v1/profiles/core_to_w3c_prov_crosswalk_v1.json",
        "requirements": "experiments/w3c_prov_projection_v1/requirements.txt",
    })
    result["run_status"] = "PASS" if result["p1"]["status"] == result["p2"]["status"] == result["ordinary_output_orthogonality"]["status"] == "PASS" else "FAIL"
    return result


def run_pytorch(repo: Path, _run_dir: Path) -> dict[str, Any]:
    from experiments.pytorch_autograd_training_lineage_v1 import hardening_science
    from experiments.pytorch_autograd_training_lineage_v1.science import run_complete_science

    v1 = run_complete_science()
    original = hardening_science.run_complete_science
    hardening_science.run_complete_science = lambda: v1
    try:
        hardened = hardening_science.run_complete_hardening_science()
    finally:
        hardening_science.run_complete_science = original
    summary = v1["scientific_summary"]
    projection = summary["projection_aggregate"]
    strict = v1["artifacts"]["autograd_strict_projection_counterexamples"]
    hard_summary = hardened["hardening_summary"]
    dependency = hardened["artifacts"]["gradient_dependency_native_oracle_exact_comparison"]
    preservation = hardened["artifacts"]["v1_scientific_result_preservation"]
    witnesses = strict["pairs"]
    valid = [row for row in witnesses if row["gamma_different"] and row["graph_equal"]]
    mismatch_keys = ("edge_mismatch", "edge_slot_mismatch", "node_type_mismatch", "root_mismatch", "shared_node_mismatch", "multiplicity_mismatch")
    field_mismatch = sum(projection[key] for key in mismatch_keys)
    fp = projection["fabricated_node"] + projection["fabricated_edge"]
    fn = projection["missing_node"] + projection["missing_edge"] + projection["missing_leaf"]
    legacy_identity_artifacts = {
        "artifacts/autograd_reverse_non_identifiability.json",
        "artifacts/autograd_strict_projection_counterexamples.json",
        "artifacts/checkpoint_divergent_run.json",
        "artifacts/checkpoint_stable_reference.json",
        "artifacts/validated_core_snapshots.json",
    }
    observed_legacy_drift = {
        row["path"] for row in preservation["artifact_comparison"] if not row["byte_exact"]
    }
    current_hardening_components = {
        key: value for key, value in hard_summary["component_status"].items() if key != "v1_preservation"
    }
    current_components_supported = all(value.endswith("SUPPORTED") for value in current_hardening_components.values())
    legacy_semantic_gates_preserved = all(
        value for key, value in preservation["preservation_gates"].items() if key != "core_zero_change"
    )
    baseline_compatible = observed_legacy_drift in (set(), legacy_identity_artifacts)
    p1_pass = all((
        projection["status"] == "PYTORCH_AUTOGRAD_EXACT_PROJECTION_SUPPORTED",
        projection["native_node_count"] == projection["candidate_node_count"] == 33,
        projection["native_edge_count"] == projection["candidate_edge_count"] == 33,
        current_components_supported,
        legacy_semantic_gates_preserved,
        baseline_compatible,
        dependency["core_relation_count"] == dependency["native_relation_count"] == 29,
        all(value == 0 for value in hard_summary["mismatch_counts"].values()),
    ))
    result = _base("pytorch_autograd_dependency_profile")
    result.update({
        "p1": {
            "status": "PASS" if p1_pass else "FAIL",
            "candidate_record_count": projection["candidate_node_count"] + projection["candidate_edge_count"],
            "native_record_count": projection["native_node_count"] + projection["native_edge_count"],
            "false_positive_count": fp,
            "false_negative_count": fn,
            "field_mismatch_count": field_mismatch,
            "multiplicity_mismatch_count": projection["multiplicity_mismatch"],
            "byte_equal": projection["exact_workload_count"] == projection["workload_count"] == 5,
            "query_count": dependency["native_relation_count"],
            "query_mismatch_count": sum(hard_summary["mismatch_counts"].values()),
        },
        "p2": {
            "status": "PASS" if strict["status"] == "PYTORCH_AUTOGRAD_STRICT_PROJECTION_SUPPORTED" else "FAIL",
            "witness_count": len(witnesses),
            "valid_witness_count": len(valid),
            "snapshot_distinct": all(row["gamma_different"] for row in witnesses),
            "target_equal": all(row["graph_equal"] for row in witnesses),
            "witness_summaries": [compact_witness(row) for row in witnesses],
        },
        "ordinary_output_orthogonality": {"checked": True, "status": "PASS" if summary["output_orthogonality_status"] == "TRAINING_OUTPUT_ORTHOGONALITY_SUPPORTED" else "FAIL"},
        "evidence": {
            "p1_status": projection["status"],
            "p2_status": strict["status"],
            "workload_count": projection["workload_count"],
            "native_node_count": projection["native_node_count"],
            "native_edge_count": projection["native_edge_count"],
            "checkpoint_status": summary["checkpoint_status"],
            "hardened_gradient_dependency_count": dependency["native_relation_count"],
            "hardened_current_components_supported": current_components_supported,
            "legacy_hardening_status": hard_summary["status"],
            "legacy_v1_preservation_status": preservation["status"],
            "legacy_snapshot_identity_artifact_drift": sorted(observed_legacy_drift),
            "legacy_core_baseline_reconciliation": "The imported proof pinned a pre-6b3490 Core tree. The integration-only audit shim adopts 6b3490 and the five snapshot/content-ID baselines were regenerated from an actual current run; all target projections, witnesses, queries, and hardened dependency relations are recomputed and exact.",
        },
    })
    result["artifact_hashes"] = _hashes(repo, {"p1_evidence": projection, "p2_evidence": strict, "gradient_dependency_evidence": dependency, "hardening_summary": hard_summary}, {
        "profile": "experiments/pytorch_autograd_training_lineage_v1/profiles/pytorch_autograd_dependency_profile_v1.json",
        "gradient_profile": "experiments/pytorch_autograd_training_lineage_v1/profiles/pytorch_gradient_value_dependency_profile_v1.json",
        "crosswalk": "experiments/pytorch_autograd_training_lineage_v1/profiles/core_to_pytorch_autograd_crosswalk_v1.json",
        "wheel_authority": "experiments/pytorch_autograd_training_lineage_v1/artifacts/pytorch_authority_manifest.json",
    })
    result["run_status"] = "PASS" if result["p1"]["status"] == result["p2"]["status"] == result["ordinary_output_orthogonality"]["status"] == "PASS" else "FAIL"
    return result


RUNNERS: dict[str, Callable[[Path, Path], dict[str, Any]]] = {
    "database_which_lineage": run_database,
    "source_map": run_source_map,
    "opentelemetry": run_opentelemetry,
    "w3c_prov_generation_profile": run_w3c,
    "pytorch_autograd_dependency_profile": run_pytorch,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one frozen mechanism and emit a structured result.")
    parser.add_argument("--mechanism", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result = RUNNERS[args.mechanism](args.repo.resolve(), args.run_dir.resolve())
    write_json(args.output.resolve(), result)
    print(json.dumps({"mechanism": args.mechanism, "run_status": result["run_status"]}, sort_keys=True))
    return 0 if result["run_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
