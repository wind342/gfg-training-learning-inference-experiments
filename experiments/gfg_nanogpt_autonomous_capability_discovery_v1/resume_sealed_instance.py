from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import time
from typing import Any

from .ai_session import FormalAIConfig
from .candidate_seal import verify_candidate_seal
from .causal_evaluator import evaluate_causal_intervention
from .checkpoint_fork import fork_audit, load_checkpoint
from .common import payload_sha256, read_json, require, write_json
from .experiment_instance import (
    build_formal_task,
    frozen_training_config,
    run_captured_segment,
)
from .intervention_runtime import AuditedIntervention
from .knowledge_archive import archive_instance, rebuild_archive_index
from .run_formal_experiments import (
    _branch_only_difference_audit,
    _core_report,
    _instance_attestation,
    _query_log_hash,
    _require_branch_isolation_integrity,
    _write_stage,
)
from .session_recovery import recover_transferred_ai_session
from .forecast_evaluator import evaluate_forecast


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def resume_sealed_instance(
    *,
    repository: Path,
    private_root: Path,
    trainer_root: Path,
    auth_file: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    instance_id = row["instance_id"]
    instance_root = private_root / "formal" / instance_id
    require(instance_root.is_dir(), "RESUME_INSTANCE_ROOT_MISSING")
    require(
        (instance_root / "candidate_seal.json").is_file(),
        "RESUME_CANDIDATE_SEAL_MISSING",
    )
    require(
        (instance_root / "sealed_forecast.json").is_file(),
        "RESUME_SEALED_FORECAST_MISSING",
    )
    require(
        not (instance_root / "instance_attestation.json").exists(),
        "RESUME_INSTANCE_ALREADY_COMPLETE",
    )
    for stage in (
        "01-discovery-training.json",
        "02-discovery-gfg-validation.json",
        "03-isolated-inputs.json",
        "04-formal-ai-session.json",
        "05-candidate-sealed.json",
        "06-unseen-prefix.json",
        "07-forecast-sealed-before-future.json",
    ):
        require(
            (instance_root / "stage-receipts" / stage).is_file(),
            "RESUME_PREREQUISITE_STAGE_MISSING:" + stage,
        )

    discovery_dir = instance_root / "discovery-training"
    discovery = read_json(discovery_dir / "segment_result.json")
    discovery_core = read_json(
        discovery_dir / "core_representative_validation.json"
    )
    discovery_evidence_dir = instance_root / "discovery-participant-gfg"
    discovery_evidence = read_json(discovery_evidence_dir / "manifest.json")
    participant_repo = instance_root / "participant-repository"
    participant = {
        "participant_repository_commit": _git(
            participant_repo, "rev-parse", "HEAD"
        )
    }
    session = recover_transferred_ai_session(
        config=FormalAIConfig(auth_file=auth_file),
        experiment_id=instance_id,
        participant_repository=participant_repo,
        participant_baseline_commit=participant[
            "participant_repository_commit"
        ],
        evidence_manifest=discovery_evidence,
        session_directory=instance_root / "ai-session",
    )
    submission = session["_submission"]
    candidate_validation = read_json(
        instance_root / "candidate_validation.json"
    )
    require(
        candidate_validation["status"] == "PASS",
        "RESUME_CANDIDATE_VALIDATION_NOT_PASS",
    )
    candidate_seal = read_json(instance_root / "candidate_seal.json")
    sealed_forecast = read_json(instance_root / "sealed_forecast.json")
    first_seal_check = verify_candidate_seal(
        submission=submission, seal=candidate_seal
    )
    require(first_seal_check["status"] == "PASS", "CANDIDATE_SEAL_TAMPER")

    prefix_dir = instance_root / "unseen-prefix"
    prefix = read_json(prefix_dir / "segment_result.json")
    prefix_checkpoint_path = prefix_dir / "checkpoint.pt"
    parent_checkpoint = load_checkpoint(prefix_checkpoint_path)
    baseline_initial = load_checkpoint(prefix_checkpoint_path)
    intervention_initial = load_checkpoint(prefix_checkpoint_path)
    fork = fork_audit(
        parent_checkpoint, baseline_initial, intervention_initial
    )
    write_json(instance_root / "checkpoint_fork_audit.json", fork)
    require(fork["identical"], "EXACT_CHECKPOINT_FORK_FAILURE")

    baseline_dir = instance_root / "baseline-continuation"
    intervention_dir = instance_root / "intervention-continuation"
    validation_config = frozen_training_config(
        model_seed=row["validation"]["model_seed"],
        data_order_seed=row["validation"]["data_order_seed"],
    )
    task = build_formal_task(
        instance_id=instance_id + "-validation-task",
        token_seed=row["validation"]["token_seed"],
        split_seed=row["validation"]["split_seed"],
    )
    baseline_reused = (baseline_dir / "segment_result.json").is_file()
    if baseline_reused:
        baseline = read_json(baseline_dir / "segment_result.json")
        baseline_core = _core_report(baseline_dir)
        require(
            baseline["start_step"] == prefix["stop_step"],
            "RESUME_BASELINE_START_STEP_MISMATCH",
        )
        require(
            baseline["run_id"] == instance_id + "-baseline-continuation",
            "RESUME_BASELINE_RUN_ID_MISMATCH",
        )
    else:
        require(
            not baseline_dir.exists(), "RESUME_BASELINE_DIRECTORY_INCOMPLETE"
        )
        baseline = run_captured_segment(
            trainer_root=trainer_root,
            task=task,
            config=validation_config,
            run_id=instance_id + "-baseline-continuation",
            output_directory=baseline_dir,
            initial_checkpoint=baseline_initial,
        )
        baseline_core = _core_report(baseline_dir)

    second_seal_check = verify_candidate_seal(
        submission=submission, seal=candidate_seal
    )
    require(second_seal_check["status"] == "PASS", "CANDIDATE_SEAL_TAMPER")
    intervention_reused = (
        intervention_dir / "segment_result.json"
    ).is_file()
    if intervention_reused:
        intervention = read_json(intervention_dir / "segment_result.json")
        intervention_core = _core_report(intervention_dir)
        intervention_receipt = read_json(
            instance_root / "intervention_runtime_receipt.json"
        )
        require(
            intervention["start_step"] == prefix["stop_step"],
            "RESUME_INTERVENTION_START_STEP_MISMATCH",
        )
        require(
            intervention["run_id"]
            == instance_id + "-intervention-continuation",
            "RESUME_INTERVENTION_RUN_ID_MISMATCH",
        )
    else:
        require(
            not intervention_dir.exists(),
            "RESUME_INTERVENTION_DIRECTORY_INCOMPLETE",
        )
        audited_intervention = AuditedIntervention(
            submission=submission,
            mechanism_state=sealed_forecast["mechanism_state"],
            forecast=sealed_forecast["forecast"],
        )
        intervention = run_captured_segment(
            trainer_root=trainer_root,
            task=task,
            config=validation_config,
            run_id=instance_id + "-intervention-continuation",
            output_directory=intervention_dir,
            initial_checkpoint=intervention_initial,
            intervention_hook=audited_intervention,
            intervention_state=audited_intervention.state,
        )
        intervention_core = _core_report(intervention_dir)
        intervention_receipt = audited_intervention.receipt()
        write_json(
            instance_root / "intervention_runtime_receipt.json",
            intervention_receipt,
        )
    third_seal_check = verify_candidate_seal(
        submission=submission, seal=candidate_seal
    )
    require(third_seal_check["status"] == "PASS", "CANDIDATE_SEAL_TAMPER")

    branch_audit = _branch_only_difference_audit(
        prefix_checkpoint=prefix["checkpoint_receipt"],
        baseline=baseline,
        intervention=intervention,
        fork=fork,
        intervention_receipt=intervention_receipt,
    )
    write_json(
        instance_root / "branch_only_difference_audit.json", branch_audit
    )
    _require_branch_isolation_integrity(branch_audit)

    forecast_validation = evaluate_forecast(
        sealed_forecast=sealed_forecast,
        prefix_metrics=prefix["metrics"],
        future_metrics=baseline["metrics"],
        candidate_seal_sha256=candidate_seal["candidate_seal_sha256"],
    )
    causal_validation = evaluate_causal_intervention(
        prefix_metrics=prefix["metrics"],
        baseline_metrics=baseline["metrics"],
        intervention_metrics=intervention["metrics"],
        intervention_spec=read_json(submission / "intervention_spec.json"),
        fork_audit=fork,
        intervention_receipt=intervention_receipt,
    )
    write_json(instance_root / "forecast_validation.json", forecast_validation)
    write_json(instance_root / "causal_validation.json", causal_validation)
    _write_stage(
        instance_root,
        "08-future-and-causal-validation",
        {
            "baseline_core_validation_sha256": baseline_core[
                "validation_sha256"
            ],
            "causal_status": causal_validation["status"],
            "forecast_status": forecast_validation["status"],
            "intervention_core_validation_sha256": intervention_core[
                "validation_sha256"
            ],
            "reused_completed_baseline": baseline_reused,
            "resumed_from_sealed_forecast": True,
        },
    )

    attestation = _instance_attestation(
        instance_id=instance_id,
        discovery=discovery,
        discovery_core=discovery_core,
        participant=participant,
        session=session,
        candidate_validation=candidate_validation,
        candidate_seal=candidate_seal,
        sealed_forecast=sealed_forecast,
        forecast_validation=forecast_validation,
        causal_validation=causal_validation,
        branch_audit=branch_audit,
        seal_verifications=[
            first_seal_check,
            second_seal_check,
            third_seal_check,
        ],
        query_log_sha256=_query_log_hash(submission),
    )
    attestation["elapsed_seconds"] = None
    attestation["resume_execution_seconds"] = time.monotonic() - started
    attestation["resumed_after_sealed_forecast_platform_abort"] = True
    attestation["reused_completed_baseline"] = baseline_reused
    attestation["reused_completed_intervention"] = intervention_reused
    attestation["recovery_repository_commit"] = _git(
        repository, "rev-parse", "HEAD"
    )
    attestation["model"] = asdict(validation_config)
    attestation["discovery_training_seconds"] = discovery["elapsed_seconds"]
    attestation["validation_prefix_seconds"] = prefix["elapsed_seconds"]
    attestation["baseline_continuation_seconds"] = baseline["elapsed_seconds"]
    attestation["intervention_continuation_seconds"] = intervention[
        "elapsed_seconds"
    ]
    attestation["discovery_counts"] = discovery["capture_manifest"]["counts"]
    attestation["validation_prefix_counts"] = prefix["capture_manifest"][
        "counts"
    ]
    attestation["baseline_counts"] = baseline["capture_manifest"]["counts"]
    attestation["intervention_counts"] = intervention["capture_manifest"][
        "counts"
    ]
    attestation["attestation_sha256"] = payload_sha256(
        {
            key: value
            for key, value in attestation.items()
            if key != "attestation_sha256"
        }
    )
    write_json(instance_root / "instance_attestation.json", attestation)
    archive_instance(
        repository=repository,
        instance_root=instance_root,
        run_name=private_root.name,
    )
    rebuild_archive_index(repository)
    return attestation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args()
    private_root = args.private_root.resolve()
    plan = read_json(private_root / "private_formal_instance_plan.json")
    row = next(
        (
            item
            for item in plan["instances"]
            if item["instance_id"] == args.instance_id
        ),
        None,
    )
    require(row is not None, "RESUME_INSTANCE_NOT_IN_FROZEN_PLAN")
    result = resume_sealed_instance(
        repository=args.repository_root.resolve(),
        private_root=private_root,
        trainer_root=args.trainer_root.resolve(),
        auth_file=args.auth_file.resolve(),
        row=row,
    )
    print(result["status"], result["attestation_sha256"])


if __name__ == "__main__":
    main()
