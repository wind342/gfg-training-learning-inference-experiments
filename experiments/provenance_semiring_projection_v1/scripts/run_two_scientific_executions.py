from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and compare two complete scientific executions")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--reuse-complete-run-1", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    for name in ("run_1", "run_2"):
        run_dir = args.artifact_root / "hardening_runs" / name
        if (
            name == "run_1"
            and args.reuse_complete_run_1
            and (run_dir / "scientific_reports.json").is_file()
            and (run_dir / "test_results.json").is_file()
        ):
            continue
        completed = subprocess.run(
            [sys.executable, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_scientific_execution", "--repo-root", str(repo_root), "--run-dir", str(run_dir)],
            cwd=repo_root,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
    run1 = args.artifact_root / "hardening_runs" / "run_1"
    run2 = args.artifact_root / "hardening_runs" / "run_2"
    report_equal = (run1 / "scientific_reports.json").read_bytes() == (run2 / "scientific_reports.json").read_bytes()
    tests_equal = (run1 / "test_results.json").read_bytes() == (run2 / "test_results.json").read_bytes()
    determinism = {
        "schema_version": "formal-semantics-hardening-determinism-v1",
        "status": "TWO_COMPLETE_RUNS_BYTE_IDENTICAL" if report_equal and tests_equal else "NOT_ESTABLISHED",
        "run_1_scientific_sha256": _sha(run1 / "scientific_reports.json"),
        "run_2_scientific_sha256": _sha(run2 / "scientific_reports.json"),
        "scientific_reports_byte_equal": report_equal,
        "run_1_test_sha256": _sha(run1 / "test_results.json"),
        "run_2_test_sha256": _sha(run2 / "test_results.json"),
        "test_results_byte_equal": tests_equal,
        "excluded_scientific_fields": [],
        "included_non_excludable_fields": [
            "variable identity", "polynomial terms", "coefficient", "exponent", "source identity",
            "output identity", "Snapshot identity", "strictness pairs", "homomorphism results", "relation classification",
        ],
    }
    path = args.artifact_root / "formal_semantics_hardening_determinism.json"
    path.write_text(json.dumps(determinism, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if determinism["status"] == "TWO_COMPLETE_RUNS_BYTE_IDENTICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
