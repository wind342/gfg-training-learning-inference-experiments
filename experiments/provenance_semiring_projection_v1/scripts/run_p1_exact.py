from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"isolated path failed ({completed.returncode}): {completed.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P1 exact projection in isolated processes")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    evidence = root / "evidence"
    native = root / "native_nx_polynomials.json"
    snapshots = evidence / "core_validated_snapshot_corpus.json"
    candidate = root / "core_projected_nx_polynomials.json"
    comparison = root / "nx_exact_comparison.json"
    coverage = root / "nx_field_coverage.json"
    comparison_v2 = root / "native_candidate_nx_exact_comparison_v2.json"
    python = sys.executable
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.native_nx_path", "--output", str(native)])
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.core_capture_path", "--output", str(snapshots)])
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.candidate_nx_path", "--input", str(snapshots), "--output", str(candidate)])
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.compare_nx_paths", "--native", str(native), "--candidate", str(candidate), "--comparison", str(comparison), "--coverage", str(coverage), "--comparison-v2", str(comparison_v2)])
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.run_algebra_independence_audit", "--artifact-root", str(root)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
