from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import time
from typing import Any

from .ai_session import FormalAIConfig, run_formal_ai_session
from .candidate_seal import seal_candidate, verify_candidate_seal
from .candidate_validator import validate_candidate
from .causal_evaluator import evaluate_causal_intervention
from .checkpoint_fork import fork_audit, load_checkpoint
from .common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from .core_validation import validate_core_representatives
from .experiment_instance import (
    build_formal_task,
    frozen_training_config,
    run_captured_segment,
)
from .forecast_evaluator import evaluate_forecast
from .forecast_runner import run_and_seal_forecast
from .intervention_runtime import AuditedIntervention
from .knowledge_archive import archive_instance, rebuild_archive_index
from .participant_repository import prepare_participant_repository
from .participant_evidence import build_participant_evidence_bundle


EXPERIMENT_NAME = "gfg_nanogpt_autonomous_capability_discovery_v1"


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repository, text=True
    ).strip()


def _assert_protocol_frozen(
    repository: Path, expected_protocol_freeze_commit: str
) -> dict[str, Any]:
    require(
        not _git(repository, "status", "--porcelain"),
        "SCIENCE_REPOSITORY_NOT_CLEAN",
    )
    head = _git(repository, "rev-parse", "HEAD")
    require(
        head == expected_protocol_freeze_commit,
        "CURRENT_HEAD_IS_NOT_PROTOCOL_FREEZE_COMMIT",
    )
    branch = _git(repository, "symbolic-ref", "--short", "HEAD")
    require(
        branch.startswith("codex/") or branch.startswith("experiment/"),
        "PROTOCOL_FREEZE_BRANCH_NOT_ALLOWED",
    )
    remote = _git(
        repository,
        "ls-remote",
        "--heads",
        "origin",
        "refs/heads/" + branch,
    )
    require(bool(remote.strip()), "PROTOCOL_FREEZE_REMOTE_BRANCH_MISSING")
    remote_head = remote.split()[0]
    require(
        remote_head == expected_protocol_freeze_commit,
        "REMOTE_HEAD_IS_NOT_PROTOCOL_FREEZE_COMMIT",
    )
    manifest_path = (
        repository
        / "experiments"
        / EXPERIMENT_NAME
        / "reports"
        / "protocol_freeze_manifest.json"
    )
    manifest = read_json(manifest_path)
    require(manifest["status"] == "PROTOCOL_FROZEN", "PROTOCOL_NOT_FROZEN")
    return {
        "current_head": head,
        "protocol_freeze_branch": branch,
        "protocol_freeze_commit": expected_protocol_freeze_commit,
        "protocol_freeze_manifest_sha256": file_sha256(manifest_path),
        "protocol_freeze_sha256": manifest["protocol_freeze_sha256"],
        "remote_branch_head": remote_head,
        "status": "PASS",
    }


def _write_stage(
    instance_root: Path,
    stage: str,
    payload: dict[str, Any],
) -> None:
    stages = instance_root / "stage-receipts"
    stages.mkdir(exist_ok=True)
    material = {
        "payload": payload,
        "schema": "formal-instance-stage-receipt-v1",
        "stage": stage,
    }
    material["receipt_sha256"] = payload_sha256(material)
    write_json(stages / f"{stage}.json", material)


def _core_report(directory: Path) -> dict[str, Any]:
    report_path = directory / "core_representative_validation.json"
    report = validate_core_representatives(
        directory / "participant_gfg.sqlite3"
    )
    write_json(report_path, report)
    require(report["status"] == "PASS", "GFG_CORE_CLOSURE_FAILURE")
    return report


def _query_log_hash(submission: Path) -> str:
    path = submission / "query_log.jsonl"
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return file_sha256(path)


def _branch_only_difference_audit(
    *,
    prefix_checkpoint: dict[str, Any],
    baseline: dict[str, Any],
    intervention: dict[str, Any],
    fork: dict[str, Any],
    intervention_receipt: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "same_exact_initial_checkpoint": fork["identical"],
        "same_start_step": (
            baseline["start_step"] == intervention["start_step"]
        ),
        "same_stop_step": baseline["stop_step"] == intervention["stop_step"],
        "same_task_commitment": (
            baseline["task_participant_commitment"]
            == intervention["task_participant_commitment"]
        ),
        "same_training_config": (
            baseline["training_config"] == intervention["training_config"]
        ),
        "intervention_executed": intervention_receipt["event_count"] > 0,
        "parent_receipt_matches": (
            prefix_checkpoint["checkpoint_commitment"]
            == fork["parent_checkpoint"]
        ),
    }
    material = {
        "gates": gates,
        "schema": "baseline-intervention-only-difference-audit-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    material["audit_sha256"] = payload_sha256(material)
    return material


def _require_branch_isolation_integrity(
    branch_audit: dict[str, Any],
) -> None:
    """Reject branch contamination without reclassifying candidate failure.

    Whether the sealed intervention actually executes is an evaluated property
    of the candidate.  Every other gate in this audit establishes that the two
    continuations differ only through that candidate intervention path.
    """
    structural_gates = {
        key: value
        for key, value in branch_audit["gates"].items()
        if key != "intervention_executed"
    }
    require(
        structural_gates and all(structural_gates.values()),
        "BRANCH_ISOLATION_FAILURE",
    )


def _instance_attestation(
    *,
    instance_id: str,
    discovery: dict[str, Any],
    discovery_core: dict[str, Any],
    participant: dict[str, Any],
    session: dict[str, Any],
    candidate_validation: dict[str, Any],
    candidate_seal: dict[str, Any],
    sealed_forecast: dict[str, Any],
    forecast_validation: dict[str, Any],
    causal_validation: dict[str, Any],
    branch_audit: dict[str, Any],
    seal_verifications: list[dict[str, Any]],
    query_log_sha256: str,
) -> dict[str, Any]:
    both = (
        forecast_validation["status"] == "FORECAST_VALIDATION_PASS"
        and causal_validation["status"] == "CAUSAL_INTERVENTION_PASS"
    )
    material = {
        "branch_only_difference_audit_sha256": branch_audit[
            "audit_sha256"
        ],
        "candidate_seal_sha256": candidate_seal[
            "candidate_seal_sha256"
        ],
        "candidate_validation_sha256": candidate_validation[
            "validation_sha256"
        ],
        "causal_validation": causal_validation,
        "discovery_core_validation_sha256": discovery_core[
            "validation_sha256"
        ],
        "discovery_gfg_sha256": discovery["gfg_validation"][
            "database_sha256"
        ],
        "discovery_training_sha256": discovery["result_sha256"],
        "forecast_seal_sha256": sealed_forecast[
            "forecast_seal_sha256"
        ],
        "forecast_validation": forecast_validation,
        "instance_id": instance_id,
        "participant_gfg_id": discovery["capture_manifest"][
            "manifest_sha256"
        ],
        "participant_repository_commit": participant[
            "participant_repository_commit"
        ],
        "query_log_sha256": query_log_sha256,
        "schema": "autonomous-dual-dynamics-discovery-instance-attestation-v2",
        "seal_verifications": seal_verifications,
        "session_attestation_sha256": session["attestation_sha256"],
        "status": (
            "AUTONOMOUS_CAPABILITY_AND_STABILITY_DYNAMICS_DISCOVERY_PASS"
            if both
            else "AUTONOMOUS_CAPABILITY_AND_STABILITY_DYNAMICS_DISCOVERY_NOT_ESTABLISHED"
        ),
    }
    material["attestation_sha256"] = payload_sha256(material)
    return material


def run_instance(
    *,
    repository: Path,
    private_root: Path,
    trainer_root: Path,
    auth_file: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    instance_id = row["instance_id"]
    instance_root = private_root / "formal" / instance_id
    require(not instance_root.exists(), "FORMAL_INSTANCE_ALREADY_EXISTS")
    instance_root.mkdir(parents=True)
    started = time.monotonic()
    experiment_root = repository / "experiments" / EXPERIMENT_NAME
    contracts = experiment_root / "contracts"

    discovery_task = build_formal_task(
        instance_id=instance_id + "-discovery-task",
        token_seed=row["discovery"]["token_seed"],
        split_seed=row["discovery"]["split_seed"],
    )
    discovery_config = frozen_training_config(
        model_seed=row["discovery"]["model_seed"],
        data_order_seed=row["discovery"]["data_order_seed"],
    )
    discovery_dir = instance_root / "discovery-training"
    discovery = run_captured_segment(
        trainer_root=trainer_root,
        task=discovery_task,
        config=discovery_config,
        run_id=instance_id + "-discovery",
        output_directory=discovery_dir,
    )
    _write_stage(
        instance_root,
        "01-discovery-training",
        {
            "result_sha256": discovery["result_sha256"],
            "transition_step": discovery["transition_step"],
        },
    )
    discovery_core = _core_report(discovery_dir)
    discovery_evidence_dir = instance_root / "discovery-participant-gfg"
    discovery_evidence = build_participant_evidence_bundle(
        captured_directory=discovery_dir,
        bundle_directory=discovery_evidence_dir,
    )
    _write_stage(
        instance_root,
        "02-discovery-gfg-validation",
        {
            "gfg_validation_sha256": discovery["gfg_validation"][
                "validation_sha256"
            ],
            "core_validation_sha256": discovery_core[
                "validation_sha256"
            ],
        },
    )

    participant_repo = instance_root / "participant-repository"
    participant = prepare_participant_repository(
        repository=participant_repo,
        evidence_directory=discovery_evidence_dir,
        instance_id=instance_id,
        contracts_directory=contracts,
    )
    _write_stage(
        instance_root,
        "03-isolated-inputs",
        {
            "participant_evidence_bundle_sha256": discovery_evidence[
                "bundle_manifest_sha256"
            ],
            "participant_repository_commit": participant[
                "participant_repository_commit"
            ],
        },
    )

    session = run_formal_ai_session(
        config=FormalAIConfig(auth_file=auth_file),
        experiment_id=instance_id,
        participant_repository=participant_repo,
        participant_baseline_commit=participant[
            "participant_repository_commit"
        ],
        evidence_directory=discovery_evidence_dir,
        evidence_manifest=discovery_evidence,
        session_directory=instance_root / "ai-session",
    )
    _write_stage(
        instance_root,
        "04-formal-ai-session",
        {
            "attestation_sha256": session["attestation_sha256"],
            "formal_work_seconds": session["formal_work_seconds"],
            "orientation_elapsed_seconds": session[
                "orientation_validation"
            ]["elapsed_seconds"],
        },
    )

    submission = session["_submission"]
    candidate_validation = validate_candidate(
        submission=submission,
        interface_gfg_prefix=discovery_evidence_dir,
        participant_repository=participant_repo,
    )
    write_json(
        instance_root / "candidate_validation.json",
        candidate_validation,
    )
    require(
        candidate_validation["status"] == "PASS",
        "CANDIDATE_EXECUTION_PLATFORM_FAILURE",
    )
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
    write_json(instance_root / "candidate_seal.json", candidate_seal)
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
    attestation["elapsed_seconds"] = time.monotonic() - started
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
    attestation["attestation_sha256"] = payload_sha256(
        {
            key: value
            for key, value in attestation.items()
            if key != "attestation_sha256"
        }
    )
    write_json(instance_root / "instance_attestation.json", attestation)
    return attestation


def run_all(
    *,
    repository: Path,
    private_root: Path,
    trainer_root: Path,
    auth_file: Path,
    expected_protocol_freeze_commit: str,
) -> dict[str, Any]:
    freeze = _assert_protocol_frozen(
        repository, expected_protocol_freeze_commit
    )
    private_plan = read_json(
        private_root / "private_formal_instance_plan.json"
    )
    aggregate_path = private_root / "formal_experiment_aggregate.json"
    require(not aggregate_path.exists(), "FORMAL_AGGREGATE_ALREADY_EXISTS")
    results = []
    for row in private_plan["instances"]:
        attestation_path = (
            private_root
            / "formal"
            / row["instance_id"]
            / "instance_attestation.json"
        )
        if attestation_path.is_file():
            result = read_json(attestation_path)
            require(
                result["instance_id"] == row["instance_id"],
                "RESUMED_INSTANCE_ATTESTATION_ID_MISMATCH",
            )
        else:
            result = run_instance(
                repository=repository,
                private_root=private_root,
                trainer_root=trainer_root,
                auth_file=auth_file,
                row=row,
            )
        results.append(result)
    pass_count = sum(
        row["status"]
        == "AUTONOMOUS_CAPABILITY_AND_STABILITY_DYNAMICS_DISCOVERY_PASS"
        for row in results
    )
    material = {
        "formal_instance_count": len(results),
        "instance_attestations": [
            {
                "attestation_sha256": row["attestation_sha256"],
                "instance_id": row["instance_id"],
                "status": row["status"],
            }
            for row in results
        ],
        "pass_count": pass_count,
        "protocol_freeze": freeze,
        "schema": "formal-autonomous-dual-dynamics-discovery-aggregate-v2",
        "status": (
            "GFG_AI_AUTONOMOUS_SCIENTIFIC_DISCOVERY_FEASIBLE"
            if pass_count >= 2
            else "GFG_AI_AUTONOMOUS_SCIENTIFIC_DISCOVERY_NOT_ESTABLISHED"
        ),
        "total_instances": 3,
    }
    material["aggregate_sha256"] = payload_sha256(material)
    write_json(aggregate_path, material)
    for row in private_plan["instances"]:
        archive_instance(
            repository=repository,
            instance_root=private_root / "formal" / row["instance_id"],
            run_name=private_root.name,
        )
    rebuild_archive_index(repository)
    return material


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--protocol-freeze-commit", required=True)
    args = parser.parse_args()
    result = run_all(
        repository=args.repository_root.resolve(),
        private_root=args.private_root.resolve(),
        trainer_root=args.trainer_root.resolve(),
        auth_file=args.auth_file.resolve(),
        expected_protocol_freeze_commit=args.protocol_freeze_commit,
    )
    print(result["status"], result["aggregate_sha256"])


if __name__ == "__main__":
    main()
