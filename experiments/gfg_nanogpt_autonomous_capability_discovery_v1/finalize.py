from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import read_json, write_json
from .negative_controls import evaluate_negative_controls
from .public_evidence import build_public_summary


def _private_instance_details(private_root: Path) -> list[dict[str, Any]]:
    aggregate = read_json(
        private_root / "formal_experiment_aggregate.json"
    )
    rows = []
    for item in aggregate["instance_attestations"]:
        root = private_root / "formal" / item["instance_id"]
        attestation = read_json(root / "instance_attestation.json")
        discovery_result = read_json(
            root / "discovery-training" / "segment_result.json"
        )
        details = {
            "branch_audit": read_json(
                root / "branch_only_difference_audit.json"
            ),
            "candidate_seal": read_json(root / "candidate_seal.json"),
            "causal_validation": read_json(
                root / "causal_validation.json"
            ),
            "discovery_counts": discovery_result["capture_manifest"][
                "counts"
            ],
            "discovery_training_sha256": discovery_result[
                "result_sha256"
            ],
            "forecast_validation": read_json(
                root / "forecast_validation.json"
            ),
            "fork_audit": read_json(root / "checkpoint_fork_audit.json"),
            "gfg_validations": [
                read_json(path)
                for path in (
                    root / "discovery-training" / "gfg_validation.json",
                    root / "unseen-prefix" / "gfg_validation.json",
                    root
                    / "baseline-continuation"
                    / "gfg_validation.json",
                    root
                    / "intervention-continuation"
                    / "gfg_validation.json",
                )
            ],
            "instance_attestation_sha256": attestation[
                "attestation_sha256"
            ],
            "instance_id": item["instance_id"],
            "participant_gfg_id": discovery_result["capture_manifest"][
                "manifest_sha256"
            ],
            "participant_gfg_validation_sha256": discovery_result[
                "gfg_validation"
            ]["validation_sha256"],
            "query_log_sha256": attestation["query_log_sha256"],
            "seal_checks": attestation["seal_verifications"],
            "sealed_forecast": read_json(root / "sealed_forecast.json"),
            "session": read_json(
                root / "ai-session" / "session_attestation.json"
            ),
            "status": item["status"],
        }
        rows.append(details)
    return rows


def finalize(
    *,
    repository: Path,
    private_root: Path,
    trainer_root: Path,
    test_summary_path: Path,
) -> dict[str, Any]:
    aggregate = read_json(
        private_root / "formal_experiment_aggregate.json"
    )
    details = _private_instance_details(private_root)
    controls = evaluate_negative_controls(details)
    write_json(private_root / "negative_control_summary.json", controls)
    if controls["status"] != "PASS":
        raise RuntimeError("NEGATIVE_CONTROL_FAILURE")
    summary = build_public_summary(
        repository=repository,
        private_root=private_root,
        trainer_root=trainer_root,
        aggregate=aggregate,
        instance_details=details,
        negative_controls=controls,
        test_summary=read_json(test_summary_path),
    )
    if summary["protected_paths_changed_files"]:
        raise RuntimeError("PROTECTED_PATH_CHANGE_DETECTED")
    output = (
        repository
        / "experiments"
        / "gfg_nanogpt_autonomous_capability_discovery_v1"
        / "reports"
        / "public_summary.json"
    )
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        repository=args.repository_root.resolve(),
        private_root=args.private_root.resolve(),
        trainer_root=args.trainer_root.resolve(),
        test_summary_path=args.test_summary.resolve(),
    )
    print(result["overall_status"], result["public_summary_sha256"])


if __name__ == "__main__":
    main()
