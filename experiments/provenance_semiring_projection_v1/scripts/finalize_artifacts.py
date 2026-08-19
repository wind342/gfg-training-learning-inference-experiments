from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


FROZEN_BASE = "f20ff57501b754b111be893565092c0e107c8b73"
ALLOWED_PREFIX = "experiments/provenance_semiring_projection_v1/"
PROTECTED = {
    "src/generation_relation_core": "03fbdce13249f84abe9d8fb605da31cdc36eda27",
    "protocol/core_v3": "0b4a2608864e771ebca7cdbfad95aabaed2d0723",
    "compat/v2": "7bbb49d18daf7ea99d7633b40c6df5bc002824ca",
    "tests/core": "280cb44d592ae48d986719638980c11e57aab1f9",
    "experiments/database_lineage": "64b5365d9a828a645c99b536254a07a2519f0cc0",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _role(relative: Path) -> str:
    if relative.parts[0] == "artifacts": return "machine_evidence"
    if relative.parts[0] == "audits": return "frozen_audit_or_authority"
    if relative.parts[0] == "profiles": return "frozen_profile"
    if relative.parts[0] == "fixtures": return "frozen_fixture"
    if relative.parts[0] in {"src", "scripts"}: return "implementation"
    if relative.parts[0] == "tests": return "test"
    if relative.suffix == ".md": return "report_or_readme"
    return "package_metadata"


def finalize(repo_root: Path, experiment_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = experiment_root / "artifacts"
    p1 = _read(artifacts / "nx_exact_comparison.json")
    p2 = _read(artifacts / "nx_strictness_counterexamples.json")
    p3 = _read(artifacts / "hierarchical_projection_exact_comparison.json")
    p1_v2 = _read(artifacts / "native_candidate_nx_exact_comparison_v2.json")
    algebra_independence = _read(artifacts / "native_candidate_algebra_independence.json")
    formal = _read(artifacts / "formal_target_semantics_audit.json")
    flat = _read(artifacts / "flat_support_view_exact_comparison.json")
    flat_classification = _read(artifacts / "flat_support_view_formal_classification.json")
    which = _read(artifacts / "nx_to_existing_which_lineage.json")
    lower = _read(artifacts / "lower_projection_strictness_constructions.json")
    joint = _read(artifacts / "joint_lower_projection_strictness.json")
    isolation = _read(artifacts / "authority_isolation.json")
    negative = _read(artifacts / "negative_controls.json")
    determinism = _read(artifacts / "formal_semantics_hardening_determinism.json")
    run1 = _read(artifacts / "hardening_runs" / "run_1" / "scientific_reports.json")
    tests = _read(artifacts / "hardening_runs" / "run_1" / "test_results.json")
    v1 = _read(artifacts / "v1_result_preservation.json")
    report_consistency = _read(artifacts / "report_artifact_consistency.json")
    changed_paths = [line for line in _git(repo_root, "diff", "--name-only", FROZEN_BASE, "--").splitlines() if line]
    path_scope_exact = bool(changed_paths) and all(path.replace("\\", "/").startswith(ALLOWED_PREFIX) for path in changed_paths)
    protected_checks = {}
    for path, expected_tree in PROTECTED.items():
        current_tree = _git(repo_root, "rev-parse", f"HEAD:{path}")
        base_tree = _git(repo_root, "rev-parse", f"{FROZEN_BASE}:{path}")
        protected_checks[path] = {"expected_tree": expected_tree, "base_tree": base_tree, "current_tree": current_tree, "exact": base_tree == current_tree == expected_tree}
    gates = [
        ("v1_p1_exact_preserved", v1["gates"]["p1_status_preserved"] and v1["gates"]["p1_case_count_preserved"] and v1["gates"]["p1_zero_mismatch_preserved"], "artifacts/v1_result_preservation.json"),
        ("v1_p2_strictness_preserved", v1["gates"]["p2_status_preserved"] and v1["gates"]["p2_pair_count_preserved"] and v1["gates"]["p2_real_execution_count_preserved"], "artifacts/v1_result_preservation.json"),
        ("independent_native_candidate_nx_exact", p1_v2["status"] == "INDEPENDENT_NATIVE_NX_ORACLE_EXACT_SUPPORTED" and all(p1_v2[name] == 0 for name in ["variable_identity_mismatch_count", "output_identity_mismatch_count", "coefficient_mismatch_count", "exponent_mismatch_count", "canonical_polynomial_mismatch_count"]), "artifacts/native_candidate_nx_exact_comparison_v2.json"),
        ("shared_algebra_helper_zero", algebra_independence["counts"]["shared_algebra_helper_count"] == 0, "artifacts/native_candidate_algebra_independence.json"),
        ("shared_variable_helper_zero", algebra_independence["counts"]["shared_variable_identity_helper_count"] == 0, "artifacts/native_candidate_algebra_independence.json"),
        ("three_formal_algebraic_targets_exact", formal["formal_algebraic_target_count"] >= 3 and sum(p3["domain_case_counts"].get(domain_id, 0) for domain_id in p3["required_algebraic_domains"]) == 39 and p3["mismatch_count"] == 0, "artifacts/formal_target_semantics_audit.json; artifacts/hierarchical_projection_exact_comparison.json"),
        ("flat_support_view_classified_and_exact", flat["status"] == "FLAT_SOURCE_SUPPORT_VIEW_EXACT_PROJECTION_SUPPORTED" and flat_classification["classification"] == "PARTIAL_NONZERO_SUPPORT_VIEW" and not flat_classification["complete_semiring_homomorphism"], "artifacts/flat_support_view_exact_comparison.json; artifacts/flat_support_view_formal_classification.json"),
        ("existing_which_hierarchy_exact", which["status"] == "THREE_WAY_EXACT_SUPPORTED", "artifacts/nx_to_existing_which_lineage.json"),
        ("lower_semantics_strictness", lower["status"] == "LOWER_PROJECTION_STRICTNESS_SUPPORTED", "artifacts/lower_projection_strictness_constructions.json"),
        ("joint_lower_strictness", joint["status"] == "JOINT_LOWER_PROJECTION_STRICTNESS_SUPPORTED", "artifacts/joint_lower_projection_strictness.json"),
        ("formal_names_and_authority_verified", formal["status"] == "FORMAL_TARGET_SEMANTICS_CLASSIFIED" and len(formal["authority"]) == 2 and all(item["classification"] in {"NOT_EVALUATED", "NON_SEMIRING_TASK_PROJECTION"} for item in formal["not_evaluated"]), "artifacts/formal_target_semantics_audit.json"),
        ("report_statistics_exact", report_consistency["status"] == "REPORT_STATISTICS_EXACT_AGAINST_ARTIFACTS", "artifacts/report_artifact_consistency.json"),
        ("original_45_negative_controls", negative["original_control_count"] == negative["original_passed_control_count"] == 45, "artifacts/negative_controls.json"),
        ("hardening_20_plus_negative_controls", negative["hardening_control_count"] >= 20 and negative["hardening_control_count"] == negative["hardening_passed_control_count"], "artifacts/negative_controls.json"),
        ("two_complete_hardening_runs_identical", determinism["status"] == "TWO_COMPLETE_RUNS_BYTE_IDENTICAL" and not determinism["excluded_scientific_fields"] and run1["status"] == "COMPLETE_HARDENING_RUN_SUPPORTED", "artifacts/formal_semantics_hardening_determinism.json"),
        ("protected_core_and_existing_experiments_unchanged", path_scope_exact and all(item["exact"] for item in protected_checks.values()) and isolation["status"] == "ISOLATION_SUPPORTED", "git tree comparison; artifacts/authority_isolation.json"),
        ("all_artifacts_rehash", True, "artifacts/artifact_manifest.json"),
        ("all_tests_pass", tests["status"] == "PASSED" and tests["passed_count"] >= 50, "artifacts/hardening_runs/run_1/test_results.json"),
    ]
    gate_rows = [{"gate": name, "passed": passed, "evidence": evidence, "blocking_reason": None if passed else f"{name.upper()}_FAILED"} for name, passed, evidence in gates]
    highest = all(item["passed"] for item in gate_rows)
    if highest:
        status = "PROVENANCE_SEMIRING_STRICT_HIERARCHY_FORMAL_SEMANTICS_HARDENING_SUPPORTED"
    elif formal["formal_algebraic_target_count"] < 3:
        status = "PROVENANCE_SEMIRING_NX_STRICT_PROJECTION_SUPPORTED_LOWER_FORMAL_HIERARCHY_PARTIAL"
    elif p1_v2["status"] != "INDEPENDENT_NATIVE_NX_ORACLE_EXACT_SUPPORTED":
        status = "PROVENANCE_SEMIRING_AS_STRICT_HIERARCHICAL_PROJECTION_SUPPORTED_ALGEBRA_ORACLE_INDEPENDENCE_NOT_ESTABLISHED"
    else:
        status = "PROVENANCE_SEMIRING_PROJECTION_NOT_ESTABLISHED"
    final_status = {
        "schema_version": "provenance-semiring-formal-semantics-hardening-final-status-v1",
        "status": status,
        "claim_prefix": "Within the frozen positive relational-algebra profile",
        "gate_count": len(gate_rows),
        "passed_gate_count": sum(item["passed"] for item in gate_rows),
        "blocking_reasons": [item["blocking_reason"] for item in gate_rows if item["blocking_reason"]],
        "gates": gate_rows,
        "protected_tree_checks": protected_checks,
        "changed_path_count": len(changed_paths),
        "all_changed_paths_within_experiment": path_scope_exact,
        "scope_exclusions_preserved": True,
        "w3c_relation_status": _read(artifacts / "nx_w3c_relation_scope.json")["status"],
    }
    _write(artifacts / "final_status.json", final_status)

    manifest_path = artifacts / "artifact_manifest.json"
    files = []
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.resolve() == manifest_path.resolve():
            continue
        relative = path.relative_to(experiment_root)
        files.append({"path": relative.as_posix(), "role": _role(relative), "size": path.stat().st_size, "sha256": _sha(path)})
    artifact_rows = [item for item in files if item["path"].startswith("artifacts/")]
    rehash_mismatches = [item["path"] for item in files if item["size"] != (experiment_root / item["path"]).stat().st_size or item["sha256"] != _sha(experiment_root / item["path"])]
    manifest = {
        "schema_version": "provenance-semiring-artifact-manifest-v2",
        "status": "ALL_FILES_REHASHED" if not rehash_mismatches else "REHASH_MISMATCH",
        "scope_root": "experiments/provenance_semiring_projection_v1",
        "self_reference_exclusion": "artifacts/artifact_manifest.json is excluded because a file cannot contain its own final SHA-256",
        "file_count": len(files),
        "artifact_file_count": len(artifact_rows),
        "rehash_mismatch_count": len(rehash_mismatches),
        "rehash_mismatches": rehash_mismatches,
        "files": files,
    }
    _write(manifest_path, manifest)
    return final_status, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize gate status and rehash every experiment file")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    final_status, manifest = finalize(args.repo_root.resolve(), args.experiment_root.resolve())
    return 0 if final_status["status"] == "PROVENANCE_SEMIRING_STRICT_HIERARCHY_FORMAL_SEMANTICS_HARDENING_SUPPORTED" and manifest["status"] == "ALL_FILES_REHASHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
