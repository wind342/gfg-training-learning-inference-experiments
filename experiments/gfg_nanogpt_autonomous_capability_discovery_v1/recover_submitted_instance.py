from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import time
from typing import Any

from .ai_session import FormalAIConfig
from .candidate_seal import seal_candidate, verify_candidate_seal
from .candidate_validator import validate_candidate
from .causal_evaluator import evaluate_causal_intervention
from .checkpoint_fork import fork_audit, load_checkpoint
from .common import file_sha256, read_json, require, write_json
from .experiment_instance import (
    build_formal_task,
    frozen_training_config,
    run_captured_segment,
)
from .forecast_evaluator import evaluate_forecast
from .forecast_runner import run_and_seal_forecast
from .intervention_runtime import AuditedIntervention
from .knowledge_archive import archive_instance, rebuild_archive_index
from .run_formal_experiments import (
    EXPERIMENT_NAME,
    _branch_only_difference_audit,
    _core_report,
    _instance_attestation,
    _query_log_hash,
    _require_branch_isolation_integrity,
    _write_stage,
)
from .session_recovery import recover_transferred_ai_session


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def recover_submitted_instance(
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
    require(instance_root.is_dir(), "RECOVERY_INSTANCE_ROOT_MISSING")
    candidate_seal_path = instance_root / "candidate_seal.json"
    resuming_from_candidate_seal = candidate_seal_path.is_file()
    require(
        not (instance_root / "instance_attestation.json").exists(),
        "RECOVERY_INSTANCE_ALREADY_COMPLETE",
    )
    require(
        not (
            resuming_from_candidate_seal
            and (instance_root / "sealed_forecast.json").exists()
        ),
        "RECOVERY_USE_SEALED_FORECAST_RESUME",
    )
    for stage in (
        "01-discovery-training.json",
        "02-discovery-gfg-validation.json",
        "03-isolated-inputs.json",
    ):
        require(
            (instance_root / "stage-receipts" / stage).is_file(),
            "RECOVERY_PREREQUISITE_STAGE_MISSING:" + stage,
        )

    experiment_root = repository / "experiments" / EXPERIMENT_NAME
    contracts = experiment_root / "contracts"
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
    stage_04 = instance_root / "stage-receipts" / "04-formal-ai-session.json"
    if not stage_04.is_file():
        _write_stage(
            instance_root,
            "04-formal-ai-session",
            {
                "attestation_sha256": session["attestation_sha256"],
                "formal_work_seconds": session["formal_work_seconds"],
                "orientation_elapsed_seconds": session[
                    "orientation_validation"
                ]["elapsed_seconds"],
                "recovered_after_submission": True,
            },
        )

    submission = session["_submission"]
    candidate_validation_path = instance_root / "candidate_validation.json"
    if resuming_from_candidate_seal:
        require(
            candidate_validation_path.is_file(),
            "RECOVERY_CANDIDATE_VALIDATION_MISSING",
        )
        candidate_validation = read_json(candidate_validation_path)
    else:
        candidate_validation = validate_candidate(
            submission=submission,
            interface_gfg_prefix=discovery_evidence_dir,
            participant_repository=participant_repo,
        )
        write_json(candidate_validation_path, candidate_validation)
    require(
        candidate_validation["status"] == "PASS",
        "CANDIDATE_EXECUTION_PLATFORM_FAILURE",
    )
    if resuming_from_candidate_seal:
        candidate_seal = read_json(candidate_seal_path)
    else:
        candidate_seal = seal_candidate(
            submission=submission,
            validation=candidate_validation,
            participant_repository=participant_repo,
            participant_gfg_id=discovery["capture_manifest"][
                "manifest_sha256"
            ],
            task_contract_hash=file_sha256(contracts / "task_family.json"),
            time_alignment_hash=file_sha256(
                contracts / "training_time_alignment.json"
            ),
            session_attestation_hash=session["attestation_sha256"],
        )
    first_seal_check = verify_candidate_seal(
        submission=submission, seal=candidate_seal
    )
    require(first_seal_check["status"] == "PASS", "CANDIDATE_SEAL_FAILURE")
    if not resuming_from_candidate_seal:
        write_json(candidate_seal_path, candidate_seal)
        _write_stage(
            instance_root,
            "05-candidate-sealed",
            {
                "candidate_seal_sha256": candidate_seal[
                    "candidate_seal_sha256"
                ],
                "candidate_validation_sha256": candidate_validation[
                    "validation_sha256"
                ],
            },
        )

    validation_task = build_formal_task(
        instance_id=instance_id + "-validation-task",
        token_seed=row["validation"]["token_seed"],
        split_seed=row["validation"]["split_seed"],
    )
    validation_config = frozen_training_config(
        model_seed=row["validation"]["model_seed"],
        data_order_seed=row["validation"]["data_order_seed"],
    )
    prefix_dir = instance_root / "unseen-prefix"
    prefix = run_captured_segment(
        trainer_root=trainer_root,
        task=validation_task,
        config=validation_config,
        run_id=instance_id + "-unseen-prefix",
        output_directory=prefix_dir,
        stop_at_prediction_cut=True,
    )
    prefix_core = _core_report(prefix_dir)
    prefix_evidence_dir = instance_root / "unseen-prefix-participant-gfg"
    from .participant_evidence import build_participant_evidence_bundle

    prefix_evidence = build_participant_evidence_bundle(
        captured_directory=prefix_dir,
        bundle_directory=prefix_evidence_dir,
    )
    _write_stage(
        instance_root,
        "06-unseen-prefix",
        {
            "core_validation_sha256": prefix_core["validation_sha256"],
            "prediction_cut_step": prefix["stop_step"],
            "prefix_result_sha256": prefix["result_sha256"],
            "prefix_participant_gfg_sha256": prefix_evidence[
                "bundle_manifest_sha256"
            ],
        },
    )

    forecast_path = instance_root / "sealed_forecast.json"
    sealed_forecast = run_and_seal_forecast(
        submission=submission,
        participant_repository=participant_repo,
        prefix_directory=prefix_evidence_dir,
        prediction_cut_step=prefix["stop_step"],
        candidate_seal=candidate_seal,
        output_path=forecast_path,
    )
    second_seal_check = verify_candidate_seal(
        submission=submission, seal=candidate_seal
    )
    require(second_seal_check["status"] == "PASS", "CANDIDATE_SEAL_TAMPER")
    _write_stage(
        instance_root,
        "07-forecast-sealed-before-future",
        {
            "forecast_seal_sha256": sealed_forecast[
                "forecast_seal_sha256"
            ],
            "prediction_cut_step": sealed_forecast[
                "prediction_cut_step"
            ],
        },
    )

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
    baseline = run_captured_segment(
        trainer_root=trainer_root,
        task=validation_task,
        config=validation_config,
        run_id=instance_id + "-baseline-continuation",
        output_directory=baseline_dir,
        initial_checkpoint=baseline_initial,
    )
    baseline_core = _core_report(baseline_dir)

    audited_intervention = AuditedIntervention(
        submission=submission,
        mechanism_state=sealed_forecast["mechanism_state"],
        forecast=sealed_forecast["forecast"],
    )
    intervention_dir = instance_root / "intervention-continuation"
    intervention = run_captured_segment(
        trainer_root=trainer_root,
        task=validation_task,
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
    write_json(instance_root / "branch_only_difference_audit.json", branch_audit)
    _require_branch_isolation_integrity(branch_audit)

    forecast_validation = evaluate_forecast(
        sealed_forecast=sealed_forecast,
        prefix_metrics=prefix["metrics"],
        future_metrics=baseline["metrics"],
        candidate_seal_sha256=candidate_seal[
            "candidate_seal_sha256"
        ],
    )
    causal_validation = evaluate_causal_intervention(
        prefix_metrics=prefix["metrics"],
        baseline_metrics=baseline["metrics"],
        intervention_metrics=intervention["metrics"],
        intervention_spec=read_json(submission / "intervention_spec.json"),
        fork_audit=fork,
        intervention_receipt=intervention_receipt,
    )
    write_json(
        instance_root / "forecast_validation.json",
        forecast_validation,
    )
    write_json(
        instance_root / "causal_validation.json",
        causal_validation,
    )
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
    attestation["recovery_execution_seconds"] = time.monotonic() - started
    attestation["recovered_after_post_submission_platform_abort"] = True
    attestation["resumed_from_existing_candidate_seal"] = (
        resuming_from_candidate_seal
    )
    attestation["model"] = asdict(validation_config)
    attestation["discovery_training_seconds"] = discovery["elapsed_seconds"]
    attestation["validation_prefix_seconds"] = prefix["elapsed_seconds"]
    attestation["baseline_continuation_seconds"] = baseline[
        "elapsed_seconds"
    ]
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
    from .common import payload_sha256

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
    require(row is not None, "RECOVERY_INSTANCE_NOT_IN_FROZEN_PLAN")
    result = recover_submitted_instance(
        repository=args.repository_root.resolve(),
        private_root=private_root,
        trainer_root=args.trainer_root.resolve(),
        auth_file=args.auth_file.resolve(),
        row=row,
    )
    print(result["status"], result["attestation_sha256"])


if __name__ == "__main__":
    main()
