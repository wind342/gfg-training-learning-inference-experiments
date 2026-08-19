from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.profile_runtime import load_profile
from experiments.provenance_semiring_projection_v1.src.semiring_homomorphisms import compare_lower_hierarchy


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"isolated lower path failed: {command}\n{completed.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove classified N[X] algebraic and task projections")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    direct_path = args.artifact_root / "native_direct_k_relation_results.json"
    native_path = args.artifact_root / "native_nx_polynomials.json"
    derived_path = args.artifact_root / "nx_derived_domain_results.json"
    if not native_path.is_file():
        raise ValueError("P1 Native N[X] artifact must exist before isolated P3 evaluation")
    python = sys.executable
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.direct_lower_k_path", "--output", str(direct_path)])
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.nx_derived_lower_path", "--input", str(native_path), "--output", str(derived_path)])
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    comparison = compare_lower_hierarchy(direct, derived)
    profile = load_profile("formal_projection_family_v2.json")
    profile_artifact = {
        "schema_version": "evaluated-formal-projection-profiles-v2",
        "profile": profile,
        "direct_domain_invocation_counts": comparison["domain_case_counts"],
        "all_algebraic_targets_exercised": all(
            comparison["domain_case_counts"].get(item["domain_id"]) == 13
            for item in profile["algebraic_targets"]
        ),
        "all_task_projections_exercised": all(
            comparison["domain_case_counts"].get(item["domain_id"]) == 13
            for item in profile["task_projections"]
        ),
    }
    flat_cases = [item for item in comparison["cases"] if item["domain_id"] == "flat_source_support_view"]
    flat_comparison = {
        "schema_version": "flat-support-view-exact-comparison-v1",
        "status": "FLAT_SOURCE_SUPPORT_VIEW_EXACT_PROJECTION_SUPPORTED"
        if len(flat_cases) == 13 and all(item["exact"] for item in flat_cases)
        else "NOT_ESTABLISHED",
        "scope": "frozen nonzero output support",
        "direct_path_computes_nx_first": False,
        "derived_operation": "Vars(N[X])",
        "case_count": len(flat_cases),
        "mismatch_count": sum(not item["exact"] for item in flat_cases),
        "cases": flat_cases,
    }
    flat_classification = {
        "schema_version": "flat-support-view-formal-classification-v1",
        "status": "FLAT_SOURCE_SUPPORT_VIEW_CLASSIFICATION_SUPPORTED",
        "domain_id": "flat_source_support_view",
        "classification": "PARTIAL_NONZERO_SUPPORT_VIEW",
        "carrier": "finite source-variable sets",
        "zero": "empty set",
        "one": "empty set",
        "addition": "set union",
        "multiplication": "set union",
        "zero_equals_one": True,
        "multiplicative_zero_annihilates_nonempty_values": False,
        "complete_commutative_semiring_target": False,
        "complete_semiring_homomorphism": False,
        "task_projection": "union of all variables appearing in N[X]",
        "observation_scope": "frozen nonzero output support",
        "former_domain_id": "why_powerset",
    }
    _write(args.artifact_root / "semiring_homomorphism_profiles.json", profile_artifact)
    _write(args.artifact_root / "hierarchical_projection_exact_comparison.json", comparison)
    _write(args.artifact_root / "flat_support_view_exact_comparison.json", flat_comparison)
    _write(args.artifact_root / "flat_support_view_formal_classification.json", flat_classification)
    return 0 if comparison["status"] == "FORMAL_PROJECTION_HIERARCHY_EXACT_SUPPORTED" and flat_comparison["status"] == "FLAT_SOURCE_SUPPORT_VIEW_EXACT_PROJECTION_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
