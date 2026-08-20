from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).parent


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    formal = read(root / "FORMAL_RESULT.json")
    checker = read(root / "INDEPENDENT_CHECK.json")
    diagnostic = read(root / "DIAGNOSTIC_ONLY_TEMPORAL_ANALYSIS.json")
    public_formal = deepcopy(formal)
    public_formal["environment"]["artifact_root"] = "<external-formal-artifact-root>"
    write(PACKAGE / "FORMAL_RESULT_SUMMARY.json", public_formal)
    write(PACKAGE / "INDEPENDENT_CHECK_SUMMARY.json", checker)
    diagnostic_summary = {
        key: diagnostic[key]
        for key in (
            "schema",
            "classification",
            "frozen_scientific_status",
            "reason_for_diagnostic",
            "seed_count",
            "counts",
            "means",
            "interpretation",
        )
    }
    write(PACKAGE / "DIAGNOSTIC_ONLY_TEMPORAL_ANALYSIS_SUMMARY.json", diagnostic_summary)
    failure = {
        "schema": "rl-e05-failure-case-analysis-v1",
        "formal_status": formal["scientific_status"],
        "passed_gate_count": sum(formal["decision_gates"].values()),
        "total_gate_count": len(formal["decision_gates"]),
        "failed_gates": [name for name, passed in formal["decision_gates"].items() if not passed],
        "temporal_precedence_seed_count": formal["counts"]["temporal_precedence"],
        "formal_required_temporal_precedence_seed_count": 9,
        "retained_seed_count": formal["seed_count"],
        "diagnostic_only_explanation": diagnostic_summary,
        "scientific_consequence": (
            "The strict composite hypothesis is not supported because the preregistered 0.03 support-share event "
            "usually followed the first coarse behavioural error. The other preregistered concentration, crowding, "
            "control and version-intervention relations passed. A narrower mechanism is supported, but the frozen "
            "composite verdict remains NOT_SUPPORTED."
        ),
    }
    write(PACKAGE / "FAILURE_CASE_ANALYSIS.json", failure)
    files = [
        "MODEL_CONTRACT.json",
        "PROTOCOL_FREEZE.md",
        "CONTRACT_FREEZE.json",
        "runtime.py",
        "runner.py",
        "independent_checker.py",
        "diagnostic_analysis.py",
        "FORMAL_RESULT_SUMMARY.json",
        "INDEPENDENT_CHECK_SUMMARY.json",
        "DIAGNOSTIC_ONLY_TEMPORAL_ANALYSIS_SUMMARY.json",
        "FAILURE_CASE_ANALYSIS.json",
        "RESULTS.md",
        "SCIENTIFIC_ASSESSMENT.md",
        "READY",
        "README.md",
    ]
    manifest = {
        "schema": "rl-e05-artifact-manifest-v1",
        "formal_artifact_storage": "external formal artifacts; compact verified summaries committed here",
        "files": {name: {"bytes": (PACKAGE / name).stat().st_size, "sha256": sha(PACKAGE / name)} for name in files},
        "formal_result_sha256": sha(root / "FORMAL_RESULT.json"),
        "independent_check_sha256": sha(root / "INDEPENDENT_CHECK.json"),
        "diagnostic_only_analysis_sha256": sha(root / "DIAGNOSTIC_ONLY_TEMPORAL_ANALYSIS.json"),
        "formal_seed_gfg_sha256": {
            f"seed-{row['seed']}": sha(root / f"seed-{row['seed']}" / "EXPERIMENT_GFG.json")
            for row in formal["per_seed"]
        },
    }
    write(PACKAGE / "ARTIFACT_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
