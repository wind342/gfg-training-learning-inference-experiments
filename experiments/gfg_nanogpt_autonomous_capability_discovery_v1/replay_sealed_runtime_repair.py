from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from .candidate_seal import verify_candidate_seal
from .causal_evaluator import evaluate_causal_intervention
from .checkpoint_fork import fork_audit, load_checkpoint
from .common import payload_sha256, read_json, require, write_json
from .experiment_instance import (
    build_formal_task,
    frozen_training_config,
    run_captured_segment,
)
from .forecast_evaluator import evaluate_forecast
from .intervention_runtime import AuditedIntervention
from .knowledge_archive import archive_instance, rebuild_archive_index
from .run_formal_experiments import (
    _branch_only_difference_audit,
    _core_report,
    _require_branch_isolation_integrity,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def replay_sealed_runtime_repair(
    *,
    repository: Path,
    private_root: Path,
    trainer_root: Path,
    row: dict[str, Any],
    replay_label: str,
) -> dict[str, Any]:
    instance_id = row["instance_id"]
    instance_root = private_root / "formal" / instance_id
    require(instance_root.is_dir(), "RUNTIME_REPAIR_INSTANCE_MISSING")
    repair_commit = _git(repository, "rev-parse", "HEAD")
    require(
        not _git(repository, "status", "--porcelain"),
        "RUNTIME_REPAIR_REPOSITORY_DIRTY",
    )
    require(
        replay_label.replace("-", "").isalnum(),
        "RUNTIME_REPAIR_LABEL_INVALID",
    )
    repair_root = instance_root / (
        "runtime-repair-" + repair_commit[:12] + "-" + replay_label
    )
    require(not repair_root.exists(), "RUNTIME_REPAIR_ALREADY_EXISTS")
    repair_root.mkdir()

    submission = instance_root / "ai-session" / "result" / "submission"
    candidate_seal = read_json(instance_root / "candidate_seal.json")
    sealed_forecast = read_json(instance_root / "sealed_forecast.json")
    seal_before = verify_candidate_seal(
        submission=submission, seal=candidate_seal
    )
    require(seal_before["status"] == "PASS", "CANDIDATE_SEAL_TAMPER")

    prefix_dir = instance_root / "unseen-prefix"
    prefix = read_json(prefix_dir / "segment_result.json")
    checkpoint_path = prefix_dir / "checkpoint.pt"
    parent = load_checkpoint(checkpoint_path)
    baseline_initial = load_checkpoint(checkpoint_path)
    intervention_initial = load_checkpoint(checkpoint_path)
    fork = fork_audit(parent, baseline_initial, intervention_initial)
    require(fork["identical"], "EXACT_CHECKPOINT_FORK_FAILURE")
    write_json(repair_root / "checkpoint_fork_audit.json", fork)

    validation_config = frozen_training_config(
        model_seed=row["validation"]["model_seed"],
        data_order_seed=row["validation"]["data_order_seed"],
    )
    task = build_formal_task(
        instance_id=instance_id + "-validation-task",
        token_seed=row["validation"]["token_seed"],
        split_seed=row["validation"]["split_seed"],
    )

    baseline_dir = repair_root / "baseline-continuation"
    baseline = run_captured_segment(
        trainer_root=trainer_root,
        task=task,
        config=validation_config,
        run_id=instance_id + "-runtime-repair-baseline-continuation",
        output_directory=baseline_dir,
        initial_checkpoint=baseline_initial,
    )
    baseline_core = _core_report(baseline_dir)
    original_baseline = read_json(
        instance_root / "baseline-continuation" / "segment_result.json"
    )
    baseline_equivalence = {
        "checkpoint_commitment_equal": (
            baseline["checkpoint_receipt"]["checkpoint_commitment"]
            == original_baseline["checkpoint_receipt"]["checkpoint_commitment"]
        ),
        "metrics_equal": baseline["metrics"] == original_baseline["metrics"],
        "start_step_equal": (
            baseline["start_step"] == original_baseline["start_step"]
        ),
        "stop_step_equal": (
            baseline["stop_step"] == original_baseline["stop_step"]
        ),
    }
    require(
        all(baseline_equivalence.values()),
        "RUNTIME_REPAIR_CHANGED_BASELINE_EXECUTION",
    )

    audited_intervention = AuditedIntervention(
        submission=submission,
        mechanism_state=sealed_forecast["mechanism_state"],
        forecast=sealed_forecast["forecast"],
    )
    intervention_dir = repair_root / "intervention-continuation"
    intervention = run_captured_segment(
        trainer_root=trainer_root,
        task=task,
        config=validation_config,
        run_id=instance_id + "-runtime-repair-intervention-continuation",
        output_directory=intervention_dir,
        initial_checkpoint=intervention_initial,
        intervention_hook=audited_intervention,
        intervention_state=audited_intervention.state,
    )
    intervention_core = _core_report(intervention_dir)
    intervention_receipt = audited_intervention.receipt()
    write_json(
        repair_root / "intervention_runtime_receipt.json",
        intervention_receipt,
    )
    require(
        intervention_receipt["event_count"] > 0,
        "RUNTIME_REPAIR_INTERVENTION_STILL_NOT_EXECUTED",
    )

    branch_audit = _branch_only_difference_audit(
        prefix_checkpoint=prefix["checkpoint_receipt"],
        baseline=baseline,
        intervention=intervention,
        fork=fork,
        intervention_receipt=intervention_receipt,
    )
    _require_branch_isolation_integrity(branch_audit)
    require(branch_audit["status"] == "PASS", "RUNTIME_REPAIR_BRANCH_FAILURE")
    write_json(repair_root / "branch_only_difference_audit.json", branch_audit)

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
    write_json(repair_root / "forecast_validation.json", forecast_validation)
    write_json(repair_root / "causal_validation.json", causal_validation)

    seal_after = verify_candidate_seal(
        submission=submission, seal=candidate_seal
    )
    require(seal_after["status"] == "PASS", "CANDIDATE_SEAL_TAMPER")
    material = {
        "baseline_core_validation_sha256": baseline_core["validation_sha256"],
        "baseline_equivalence": baseline_equivalence,
        "branch_audit_sha256": branch_audit["audit_sha256"],
        "candidate_seal_sha256": candidate_seal["candidate_seal_sha256"],
        "candidate_unchanged": seal_before == seal_after,
        "causal_validation": causal_validation,
        "forecast_validation": forecast_validation,
        "instance_id": instance_id,
        "intervention_core_validation_sha256": intervention_core[
            "validation_sha256"
        ],
        "intervention_event_count": intervention_receipt["event_count"],
        "original_intervention_event_count": read_json(
            instance_root / "intervention_runtime_receipt.json"
        )["event_count"],
        "repair_commit": repair_commit,
        "repair_reason": "GRADIENT_CLIPPING_CONTROL_WAS_DECLARED_BUT_NOT_WIRED",
        "schema": "sealed-runtime-repair-replay-attestation-v1",
        "status": (
            "SEALED_RUNTIME_REPAIR_REPLAY_PASS"
            if (
                forecast_validation["status"] == "FORECAST_VALIDATION_PASS"
                and causal_validation["status"] == "CAUSAL_INTERVENTION_PASS"
            )
            else "SEALED_RUNTIME_REPAIR_REPLAY_NOT_ESTABLISHED"
        ),
    }
    material["attestation_sha256"] = payload_sha256(material)
    write_json(repair_root / "runtime_repair_attestation.json", material)
    archive_instance(
        repository=repository,
        instance_root=instance_root,
        run_name=private_root.name,
    )
    rebuild_archive_index(repository)
    return material


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--replay-label", default="replay-01")
    args = parser.parse_args()
    private_root = args.private_root.resolve()
    plan = read_json(private_root / "private_formal_instance_plan.json")
    rows = [
        row
        for row in plan["instances"]
        if row["instance_id"] == args.instance_id
    ]
    require(len(rows) == 1, "RUNTIME_REPAIR_INSTANCE_PLAN_MISMATCH")
    result = replay_sealed_runtime_repair(
        repository=args.repository_root.resolve(),
        private_root=private_root,
        trainer_root=args.trainer_root.resolve(),
        row=rows[0],
        replay_label=args.replay_label,
    )
    print(result["status"], result["attestation_sha256"])


if __name__ == "__main__":
    main()
