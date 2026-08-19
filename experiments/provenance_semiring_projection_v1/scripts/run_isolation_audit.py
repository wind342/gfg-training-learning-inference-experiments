from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.isolation_audit import evaluate_isolation


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run static and dynamic authority-isolation audits")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    probe = args.artifact_root / "evidence" / "isolation_probe"
    traces = probe / "traces"
    python = sys.executable
    core = probe / "core.json"
    native = probe / "native.json"
    candidate = probe / "candidate.json"
    comparison = probe / "comparison.json"
    coverage = probe / "coverage.json"
    direct_lower = probe / "direct_lower.json"
    derived_lower = probe / "derived_lower.json"
    report_statistics = probe / "report_statistics.json"
    report_consistency = probe / "report_artifact_consistency.json"
    rendered_report = probe / "generated_report.md"
    _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.core_capture_path", "--workload", "W6", "--output", str(core)])

    def traced(name: str, module: str, forwarded: list[str]) -> None:
        _run([python, "-m", "experiments.provenance_semiring_projection_v1.scripts.trace_entrypoint", "--trace", str(traces / f"{name}.json"), "--module", module, "--", *forwarded])

    traced("native", "experiments.provenance_semiring_projection_v1.scripts.native_nx_path", ["--workload", "W6", "--output", str(native)])
    traced("candidate", "experiments.provenance_semiring_projection_v1.scripts.candidate_nx_path", ["--input", str(core), "--output", str(candidate)])
    traced("direct_lower", "experiments.provenance_semiring_projection_v1.scripts.direct_lower_k_path", ["--output", str(direct_lower)])
    traced("derived_lower", "experiments.provenance_semiring_projection_v1.scripts.nx_derived_lower_path", ["--input", str(args.artifact_root / "native_nx_polynomials.json"), "--output", str(derived_lower)])
    traced(
        "comparison",
        "experiments.provenance_semiring_projection_v1.scripts.compare_nx_paths",
        [
            "--native", str(args.artifact_root / "native_nx_polynomials.json"),
            "--candidate", str(args.artifact_root / "core_projected_nx_polynomials.json"),
            "--comparison", str(comparison),
            "--coverage", str(coverage),
        ],
    )
    traced(
        "report_statistics",
        "experiments.provenance_semiring_projection_v1.scripts.run_report_statistics",
        [
            "--artifact-root", str(args.artifact_root),
            "--report", str(repo_root / "experiments" / "provenance_semiring_projection_v1" / "EXPERIMENT_REPORT.md"),
            "--statistics-output", str(report_statistics),
            "--consistency-output", str(report_consistency),
            "--rendered-report-output", str(rendered_report),
        ],
    )
    authority, static, classification, direct_independence = evaluate_isolation(repo_root, traces)
    _write(args.artifact_root / "authority_isolation.json", authority)
    _write(args.artifact_root / "static_isolation_audit.json", static)
    _write(args.artifact_root / "persisted_artifact_classification.json", classification)
    _write(args.artifact_root / "direct_lower_k_independence_v2.json", direct_independence)
    return 0 if authority["status"] == "ISOLATION_SUPPORTED" and static["status"] == "SUPPORTED" and classification["status"] == "SUPPORTED" and direct_independence["status"] == "DIRECT_LOWER_K_INDEPENDENCE_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
