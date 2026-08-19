from __future__ import annotations

import argparse
from pathlib import Path
import secrets
import subprocess
from typing import Any

from .common import (
    BASELINE_COMMIT,
    NANOGPT_COMMIT,
    file_sha256,
    payload_sha256,
    write_json,
)
from .experiment_instance import build_formal_task


def _seed() -> int:
    return 100000 + secrets.randbelow(900000000)


def _new_private_plan() -> dict[str, Any]:
    instances = []
    for index in range(1, 4):
        discovery = {
            "data_order_seed": _seed(),
            "model_seed": _seed(),
            "split_seed": _seed(),
            "token_seed": _seed(),
        }
        validation = {
            "data_order_seed": _seed(),
            "model_seed": _seed(),
            "split_seed": _seed(),
            "token_seed": _seed(),
        }
        instances.append(
            {
                "discovery": discovery,
                "instance_id": (
                    f"gfg-nanogpt-dual-dynamics-discovery-{index:02d}"
                ),
                "validation": validation,
            }
        )
    return {
        "instances": instances,
        "schema": "private-formal-instance-plan-v1",
    }


def _task_commitment(instance_id: str, seeds: dict[str, int]) -> str:
    return build_formal_task(
        instance_id=instance_id,
        token_seed=seeds["token_seed"],
        split_seed=seeds["split_seed"],
    ).private_generation_commitment


def freeze_protocol(
    *,
    repository_root: Path,
    private_root: Path,
    trainer_root: Path,
    capture_equivalence_report: Path,
) -> dict[str, Any]:
    private_root.mkdir(parents=True, exist_ok=True)
    plan_path = private_root / "private_formal_instance_plan.json"
    if plan_path.exists():
        from .common import read_json

        private_plan = read_json(plan_path)
    else:
        private_plan = _new_private_plan()
        write_json(plan_path, private_plan)
    private_plan["plan_sha256"] = payload_sha256(
        {
            key: value
            for key, value in private_plan.items()
            if key != "plan_sha256"
        }
    )
    write_json(plan_path, private_plan)

    commitments = []
    for row in private_plan["instances"]:
        discovery_task_id = row["instance_id"] + "-discovery-task"
        validation_task_id = row["instance_id"] + "-validation-task"
        commitments.append(
            {
                "discovery_task_commitment": _task_commitment(
                    discovery_task_id, row["discovery"]
                ),
                "discovery_training_seed_commitment": payload_sha256(
                    {
                        "data_order_seed": row["discovery"][
                            "data_order_seed"
                        ],
                        "model_seed": row["discovery"]["model_seed"],
                    }
                ),
                "instance_id": row["instance_id"],
                "validation_task_commitment": _task_commitment(
                    validation_task_id, row["validation"]
                ),
                "validation_training_seed_commitment": payload_sha256(
                    {
                        "data_order_seed": row["validation"][
                            "data_order_seed"
                        ],
                        "model_seed": row["validation"]["model_seed"],
                    }
                ),
            }
        )
    commitment_document = {
        "formal_instance_count": 3,
        "instances": commitments,
        "private_plan_commitment": private_plan["plan_sha256"],
        "schema": "formal-instance-commitments-v1",
    }
    commitment_document["commitments_sha256"] = payload_sha256(
        commitment_document
    )
    experiment_root = Path(__file__).resolve().parent
    contracts = experiment_root / "contracts"
    reports = experiment_root / "reports"
    reports.mkdir(exist_ok=True)
    write_json(
        contracts / "formal_instance_commitments.json",
        commitment_document,
    )

    contract_hashes = {
        path.name: file_sha256(path)
        for path in sorted(contracts.glob("*.json"))
    }
    platform_hashes = {
        name: file_sha256(experiment_root / name)
        for name in (
            "ai_session.py",
            "candidate_seal.py",
            "candidate_validator.py",
            "causal_evaluator.py",
            "core_validation.py",
            "experiment_instance.py",
            "forecast_evaluator.py",
            "forecast_runner.py",
            "intervention_runtime.py",
            "model_proxy.py",
            "nanogpt_adapter.py",
            "participant_evidence.py",
            "participant_repository.py",
            "recover_submitted_instance.py",
            "resume_sealed_instance.py",
            "run_formal_experiments.py",
            "session_recovery.py",
            "training_capture.py",
            "training_gfg.py",
        )
    }
    platform_hashes["GFG_MACHINE_SEMANTICS.md"] = file_sha256(
        experiment_root / "orientation" / "GFG_MACHINE_SEMANTICS.md"
    )
    platform_hashes["EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md"] = file_sha256(
        experiment_root
        / "orientation"
        / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md"
    )
    platform_hashes["DUAL_DYNAMICS_SUBMISSION_CHECKER.py"] = file_sha256(
        experiment_root
        / "orientation"
        / "DUAL_DYNAMICS_SUBMISSION_CHECKER.py"
    )
    platform_hashes["ORIENTATION_RECEIPT_CHECKER.py"] = file_sha256(
        experiment_root
        / "orientation"
        / "ORIENTATION_RECEIPT_CHECKER.py"
    )
    platform_hashes["orientation_gate.py"] = file_sha256(
        experiment_root / "orientation" / "orientation_gate.py"
    )
    platform_hashes["unrelated_example.json"] = file_sha256(
        experiment_root / "orientation" / "unrelated_example.json"
    )
    engineering_reports = sorted(
        private_root.glob(
            "preflight-p23-f60-*/engineering-preflight/result.json"
        )
    )
    nanogpt_head = subprocess.check_output(
        ["git", "-C", str(trainer_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if nanogpt_head != NANOGPT_COMMIT:
        raise RuntimeError("NANOGPT_COMMIT_DRIFT")
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        text=True,
    ).splitlines()
    changed = [
        line[3:].replace("\\", "/")
        for line in status_lines
        if len(line) >= 4
    ]
    allowed_prefixes = (
        "experiments/gfg_nanogpt_autonomous_capability_discovery_v1/",
        "tests/experiments/gfg_nanogpt_autonomous_capability_discovery_v1/",
    )
    protected = [
        path for path in changed if not path.startswith(allowed_prefixes)
    ]
    material = {
        "ai_model": "gpt-5.6-sol",
        "baseline_commit": BASELINE_COMMIT,
        "capture_on_off_equivalence_sha256": file_sha256(
            capture_equivalence_report
        ),
        "contract_hashes": contract_hashes,
        "engineering_preflight_reports": [
            {
                "path_commitment": payload_sha256(str(path)),
                "sha256": file_sha256(path),
            }
            for path in engineering_reports
        ],
        "formal_instance_commitments_sha256": commitment_document[
            "commitments_sha256"
        ],
        "formal_scientific_timeout_seconds": 7200,
        "industrial_platform_reuse_source_commit": (
            "847e0eefef7842d87d2e5cbd9562b0239b53b123"
        ),
        "multipass_instance": "gfg-lab-ubuntu-v5-stability",
        "multipass_host_proxy": "192.168.96.1:7890",
        "nanoGPT_commit": nanogpt_head,
        "nanoGPT_model_py_sha256": file_sha256(
            trainer_root / "model.py"
        ),
        "orientation_minimum_seconds": 300,
        "participant_model_proxy_allowed_hosts": [
            "auth.openai.com",
            "chatgpt.com",
        ],
        "platform_hashes": platform_hashes,
        "protected_path_changed_files": protected,
        "reasoning_effort": "xhigh",
        "response_transport": "https-only",
        "schema": "nanogpt-autonomous-discovery-protocol-freeze-v7",
        "supersedes_protocol_freeze_commit": (
            "e4be50ebc788036896b95a75a0b498a76db0c341"
        ),
        "status": (
            "PROTOCOL_FROZEN" if not protected else "PROTOCOL_REVIEW_REQUIRED"
        ),
    }
    material["protocol_freeze_sha256"] = payload_sha256(material)
    write_json(reports / "protocol_freeze_manifest.json", material)
    write_json(
        private_root / "protocol_freeze_private_receipt.json",
        {
            "private_plan_sha256": file_sha256(plan_path),
            "protocol_freeze_sha256": material[
                "protocol_freeze_sha256"
            ],
        },
    )
    return material


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument(
        "--capture-equivalence-report", type=Path, required=True
    )
    args = parser.parse_args()
    result = freeze_protocol(
        repository_root=args.repository_root.resolve(),
        private_root=args.private_root.resolve(),
        trainer_root=args.trainer_root.resolve(),
        capture_equivalence_report=(
            args.capture_equivalence_report.resolve()
        ),
    )
    print(result["status"], result["protocol_freeze_sha256"])


if __name__ == "__main__":
    main()
