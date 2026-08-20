from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
from statistics import mean
from typing import Any

import torch

from .runtime import (
    MultiSkillGRUPolicy,
    combined_state_sha256,
    compact_evaluation,
    deterministic_allocated_feedback_batch,
    evaluate_policy,
    make_optimizer,
    model_sha256,
    positive_feedback_update,
    restore_state,
    seed_everything,
    state_payload,
    supervised_pretrain_step,
    support_profile,
)


PACKAGE = Path(__file__).resolve().parent
DOSE_CONDITIONS = ("balanced", "mild", "high", "exclusive", "frozen")
RECOVERY_CONDITIONS = ("rebalance_recovery", "repair_recovery")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _skill_mean(evaluation: dict[str, Any], key: str, skills: list[int]) -> float:
    return mean(float(evaluation["per_skill"][skill][key]) for skill in skills)


def _measurement(
    model: MultiSkillGRUPolicy,
    *,
    condition: str,
    phase_update: int,
    global_update: int,
    device: torch.device,
) -> dict[str, Any]:
    evaluation = evaluate_policy(model, device)
    return {
        "schema": "rl-e06-support-checkpoint-v1",
        "condition": condition,
        "phase_update": int(phase_update),
        "global_update": int(global_update),
        "model_sha256": model_sha256(model),
        "evaluation": evaluation,
        "support": support_profile(model, device),
    }


def _trajectory_row(
    model: MultiSkillGRUPolicy,
    *,
    condition: str,
    phase_update: int,
    global_update: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema": "rl-e06-continuous-boundary-trajectory-v1",
        "condition": condition,
        "phase_update": int(phase_update),
        "global_update": int(global_update),
        "evaluation": compact_evaluation(evaluate_policy(model, device)),
    }


def _save_checkpoint(
    path: Path,
    model: MultiSkillGRUPolicy,
    optimizer: torch.optim.AdamW,
    *,
    seed: int,
    condition: str,
    phase_update: int,
    global_update: int,
) -> dict[str, str]:
    payload = {
        "schema": "rl-e06-training-state-v1",
        "seed": int(seed),
        "condition": condition,
        "phase_update": int(phase_update),
        "global_update": int(global_update),
        "state": state_payload(model, optimizer),
        "sealed_at_utc": utc_now(),
    }
    torch.save(payload, path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _run_phase(
    *,
    condition: str,
    seed: int,
    starting_state: dict[str, Any],
    allocation: list[int],
    updates: int,
    global_offset: int,
    support_checkpoints: list[int],
    apply_update: bool,
    contract: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    ledger_seed_offset: int,
    seal_phase_update: int | None = None,
) -> tuple[dict[str, Any], MultiSkillGRUPolicy, torch.optim.AdamW, dict[str, Any] | None]:
    model, optimizer = restore_state(
        starting_state,
        hidden_size=int(contract["model"]["hidden_size"]),
        learning_rate=float(contract["model"]["learning_rate"]),
        weight_decay=float(contract["model"]["weight_decay"]),
        device=device,
    )
    initial_state_sha256 = combined_state_sha256(model, optimizer)
    checkpoint_set = set(int(value) for value in support_checkpoints)
    measurements = []
    if 0 in checkpoint_set:
        measurements.append(
            _measurement(
                model,
                condition=condition,
                phase_update=0,
                global_update=global_offset,
                device=device,
            )
        )
    receipt_path = run_dir / f"{condition}-feedback-receipts.jsonl"
    trajectory_path = run_dir / f"{condition}-boundary-trajectory.jsonl"
    total_positive = 0
    total_episodes = 0
    fork_state: dict[str, Any] | None = None
    with receipt_path.open("w", encoding="utf-8", newline="\n") as receipt_handle, trajectory_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as trajectory_handle:
        _append_jsonl(
            trajectory_handle,
            _trajectory_row(
                model,
                condition=condition,
                phase_update=0,
                global_update=global_offset,
                device=device,
            ),
        )
        for index in range(updates):
            phase_update = index + 1
            global_update = global_offset + phase_update
            batch = deterministic_allocated_feedback_batch(
                seed=seed + ledger_seed_offset,
                update=global_update - 1,
                allocation=allocation,
                device=device,
            )
            receipt = positive_feedback_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                apply_update=apply_update,
            )
            receipt.update({
                "schema": "rl-e06-positive-feedback-occurrence-v1",
                "seed": int(seed),
                "condition": condition,
                "phase_update": phase_update,
                "global_update": global_update,
                "skill_allocation": allocation,
                "reward_authority": "exact_complete_chain_rule",
            })
            _append_jsonl(receipt_handle, receipt)
            total_positive += int(receipt["positive_consequence_count"])
            total_episodes += int(receipt["episode_count"])
            _append_jsonl(
                trajectory_handle,
                _trajectory_row(
                    model,
                    condition=condition,
                    phase_update=phase_update,
                    global_update=global_update,
                    device=device,
                ),
            )
            if phase_update in checkpoint_set:
                measurements.append(
                    _measurement(
                        model,
                        condition=condition,
                        phase_update=phase_update,
                        global_update=global_update,
                        device=device,
                    )
                )
            if seal_phase_update is not None and phase_update == int(seal_phase_update):
                fork_state = state_payload(model, optimizer)
                _save_checkpoint(
                    run_dir / f"{condition}-fork-{phase_update}.pt",
                    model,
                    optimizer,
                    seed=seed,
                    condition=condition,
                    phase_update=phase_update,
                    global_update=global_update,
                )
    final_checkpoint = _save_checkpoint(
        run_dir / f"{condition}-final.pt",
        model,
        optimizer,
        seed=seed,
        condition=condition,
        phase_update=updates,
        global_update=global_offset + updates,
    )
    final_measurement = measurements[-1]
    require(int(final_measurement["phase_update"]) == updates, f"RL_E06_FINAL_SUPPORT_CHECKPOINT_MISSING:{condition}")
    result = {
        "schema": "rl-e06-condition-result-v1",
        "condition": condition,
        "allocation": allocation,
        "phase_updates": updates,
        "global_offset": global_offset,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": combined_state_sha256(model, optimizer),
        "feedback_receipts": str(receipt_path),
        "feedback_receipts_sha256": file_sha256(receipt_path),
        "boundary_trajectory": str(trajectory_path),
        "boundary_trajectory_sha256": file_sha256(trajectory_path),
        "support_measurements": measurements,
        "final_evaluation": final_measurement["evaluation"],
        "final_support": final_measurement["support"],
        "final_checkpoint": final_checkpoint,
        "total_positive_consequences": total_positive,
        "total_episodes": total_episodes,
    }
    write_json(run_dir / f"{condition}-result.json", result)
    return result, model, optimizer, fork_state


def _ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for position in range(start, stop):
            result[ordered[position]] = rank
        start = stop
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return 0.0 if denominator <= 1e-15 else numerator / denominator


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _l2(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _seed_diagnostics(summary: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    ordered = ["balanced", "mild", "high", "exclusive"]
    doses = [float(contract["feedback"]["dose_levels"][name]) for name in ordered]
    shares = [float(summary["dose_conditions"][name]["final_support"]["task_support_shares"][0]) for name in ordered]
    margins = [
        _skill_mean(summary["dose_conditions"][name]["final_evaluation"], "mean_margin", [1, 2, 3])
        for name in ordered
    ]
    accuracies = [
        _skill_mean(summary["dose_conditions"][name]["final_evaluation"], "chain_accuracy", [1, 2, 3])
        for name in ordered
    ]
    exclusive = summary["dose_conditions"]["exclusive"]
    balanced = summary["dose_conditions"]["balanced"]
    rebalance = summary["recovery_conditions"]["rebalance_recovery"]
    repair = summary["recovery_conditions"]["repair_recovery"]
    exclusive_unreinforced = accuracies[-1]
    balanced_unreinforced = accuracies[0]
    rebalance_unreinforced = _skill_mean(rebalance["final_evaluation"], "chain_accuracy", [1, 2, 3])
    repair_unreinforced = _skill_mean(repair["final_evaluation"], "chain_accuracy", [1, 2, 3])
    balanced_shares = [float(value) for value in balanced["final_support"]["task_support_shares"]]
    exclusive_shares = [float(value) for value in exclusive["final_support"]["task_support_shares"]]
    rebalance_shares = [float(value) for value in rebalance["final_support"]["task_support_shares"]]
    repair_shares = [float(value) for value in repair["final_support"]["task_support_shares"]]
    return {
        "dose_levels": doses,
        "task0_support_shares": shares,
        "unreinforced_mean_margins": margins,
        "unreinforced_mean_accuracies": accuracies,
        "support_dose_spearman": _spearman(doses, shares),
        "unreinforced_margin_dose_spearman": _spearman(doses, margins),
        "exclusive_unreinforced_accuracy_deficit_vs_balanced": balanced_unreinforced - exclusive_unreinforced,
        "exclusive_task0_support_share_excess_vs_balanced": shares[-1] - shares[0],
        "rebalance_unreinforced_accuracy_gain_vs_exclusive": rebalance_unreinforced - exclusive_unreinforced,
        "repair_unreinforced_accuracy_gain_vs_exclusive": repair_unreinforced - exclusive_unreinforced,
        "rebalance_target0_final_accuracy": float(rebalance["final_evaluation"]["per_skill"][0]["chain_accuracy"]),
        "repair_target0_final_accuracy": float(repair["final_evaluation"]["per_skill"][0]["chain_accuracy"]),
        "exclusive_support_distance_from_balanced": _l2(exclusive_shares, balanced_shares),
        "rebalance_support_distance_from_balanced": _l2(rebalance_shares, balanced_shares),
        "repair_support_distance_from_balanced": _l2(repair_shares, balanced_shares),
        "rebalance_support_reversal": _l2(rebalance_shares, balanced_shares) < _l2(exclusive_shares, balanced_shares),
        "repair_support_reversal": _l2(repair_shares, balanced_shares) < _l2(exclusive_shares, balanced_shares),
    }


def compile_gfg(seed_summary: dict[str, Any]) -> dict[str, Any]:
    seed = int(seed_summary["seed"])
    facts: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []

    def add(occurrence_id: str, origin: str, transformation: str, outcome: str, role: str) -> None:
        fact_id = f"fact:{occurrence_id}"
        occurrences.append({"occurrence_id": occurrence_id, "realizes_fact": [fact_id]})
        facts.append({"fact_id": fact_id, "u": origin, "tau": transformation, "omega": occurrence_id, "z": outcome, "rho": role})

    add(
        f"seed-{seed}:baseline-seal",
        "balanced-supervised-formation",
        "seal_parameter_adamw_receiving_state",
        seed_summary["baseline"]["state_sha256"],
        "receiving_state",
    )
    for family in ("dose_conditions", "recovery_conditions"):
        for condition, result in seed_summary[family].items():
            with Path(result["feedback_receipts"]).open("r", encoding="utf-8") as handle:
                for line in handle:
                    receipt = json.loads(line)
                    add(
                        f"seed-{seed}:{condition}:update-{receipt['global_update']}",
                        json.dumps({
                            "receiving_state": receipt["pre_state_sha256"],
                            "allocation": receipt["skill_allocation"],
                            "cue_batch": receipt["cue_batch_sha256"],
                            "sampling_uniforms": receipt["uniform_batch_sha256"],
                            "positive_consequence_count": receipt["positive_consequence_count"],
                        }, sort_keys=True),
                        f"exact_positive_feedback:{condition}",
                        receipt["post_state_sha256"],
                        "actual_training_action" if condition != "frozen" else "explicit_no_persistent_update",
                    )
            for row in result["support_measurements"]:
                add(
                    f"seed-{seed}:{condition}:support-{row['global_update']}",
                    row["model_sha256"],
                    "all_16_component_coalitions_and_exact_shapley",
                    row["support"]["coalition_value_sha256"],
                    "functional_support_state",
                )
    return {
        "schema": "rl-e06-generation-fact-graph-v1",
        "seed": seed,
        "atomic_generation_facts": facts,
        "concrete_occurrences": occurrences,
        "validation": {
            "fact_count": len(facts),
            "occurrence_count": len(occurrences),
            "all_facts_realized_once": len(facts) == len(occurrences),
            "all_five_coordinates_present": all(
                all(fact.get(name) not in (None, "") for name in ("u", "tau", "omega", "z", "rho"))
                for fact in facts
            ),
        },
    }


def run_seed(seed: int, contract: dict[str, Any], artifact_root: Path, device: torch.device) -> dict[str, Any]:
    run_dir = artifact_root / f"seed-{seed}"
    require(not run_dir.exists(), f"RL_E06_RUN_DIR_EXISTS:{run_dir}")
    run_dir.mkdir(parents=True)
    seed_everything(seed)
    model = MultiSkillGRUPolicy(int(contract["model"]["hidden_size"])).to(device)
    optimizer = make_optimizer(model, float(contract["model"]["learning_rate"]), float(contract["model"]["weight_decay"]))
    pretrain_path = run_dir / "pretrain-receipts.jsonl"
    pretrain_updates = 0
    with pretrain_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(int(contract["formation"]["pretrain_max_steps"])):
            receipt = supervised_pretrain_step(model, optimizer, device)
            receipt.update({"schema": "rl-e06-pretrain-occurrence-v1", "update": index + 1})
            _append_jsonl(handle, receipt)
            pretrain_updates = index + 1
            if pretrain_updates >= int(contract["formation"]["pretrain_min_steps"]):
                evaluation = evaluate_policy(model, device)
                if (
                    float(evaluation["macro_chain_accuracy"]) == float(contract["formation"]["required_chain_accuracy"])
                    and min(float(row["minimum_margin"]) for row in evaluation["per_skill"])
                    >= float(contract["formation"]["pretrain_minimum_margin"])
                ):
                    break
    baseline_evaluation = evaluate_policy(model, device)
    require(float(baseline_evaluation["macro_chain_accuracy"]) == 1.0, f"RL_E06_BASELINE_MASTERY_FAILED:{seed}")
    baseline_support = support_profile(model, device)
    baseline_state = state_payload(model, optimizer)
    baseline_checkpoint = _save_checkpoint(
        run_dir / "baseline-seal.pt",
        model,
        optimizer,
        seed=seed,
        condition="baseline",
        phase_update=pretrain_updates,
        global_update=0,
    )
    dose_results: dict[str, Any] = {}
    exclusive_fork: dict[str, Any] | None = None
    initial_hashes = []
    for condition in DOSE_CONDITIONS:
        allocation = [int(value) for value in contract["feedback"]["dose_conditions"][condition]]
        result, _, _, fork_state = _run_phase(
            condition=condition,
            seed=seed,
            starting_state=baseline_state,
            allocation=allocation,
            updates=int(contract["feedback"]["updates"]),
            global_offset=0,
            support_checkpoints=[int(value) for value in contract["feedback"]["support_checkpoints"]],
            apply_update=condition != "frozen",
            contract=contract,
            device=device,
            run_dir=run_dir,
            ledger_seed_offset=700_001,
            seal_phase_update=int(contract["recovery"]["fork_update"]) if condition == "exclusive" else None,
        )
        dose_results[condition] = result
        initial_hashes.append(result["initial_state_sha256"])
        if condition == "exclusive":
            exclusive_fork = fork_state
    require(len(set(initial_hashes)) == 1, f"RL_E06_DOSE_BRANCH_CLONE_MISMATCH:{seed}")
    require(exclusive_fork is not None, f"RL_E06_EXCLUSIVE_FORK_MISSING:{seed}")
    recovery_results: dict[str, Any] = {}
    recovery_initial_hashes = []
    for condition in RECOVERY_CONDITIONS:
        allocation = [int(value) for value in contract["recovery"]["conditions"][condition]]
        result, _, _, _ = _run_phase(
            condition=condition,
            seed=seed,
            starting_state=exclusive_fork,
            allocation=allocation,
            updates=int(contract["recovery"]["updates"]),
            global_offset=int(contract["recovery"]["fork_update"]),
            support_checkpoints=[int(value) for value in contract["recovery"]["support_checkpoints"]],
            apply_update=True,
            contract=contract,
            device=device,
            run_dir=run_dir,
            ledger_seed_offset=900_001,
        )
        recovery_results[condition] = result
        recovery_initial_hashes.append(result["initial_state_sha256"])
    require(len(set(recovery_initial_hashes)) == 1, f"RL_E06_RECOVERY_BRANCH_CLONE_MISMATCH:{seed}")
    require(
        dose_results["frozen"]["final_state_sha256"] == baseline_state["combined_state_sha256"],
        f"RL_E06_FROZEN_STATE_CHANGED:{seed}",
    )
    summary = {
        "schema": "rl-e06-seed-result-v1",
        "seed": int(seed),
        "pretrain_updates": pretrain_updates,
        "pretrain_receipts": str(pretrain_path),
        "pretrain_receipts_sha256": file_sha256(pretrain_path),
        "baseline": {
            "state_sha256": baseline_state["combined_state_sha256"],
            "checkpoint": baseline_checkpoint,
            "evaluation": baseline_evaluation,
            "support": baseline_support,
        },
        "dose_conditions": dose_results,
        "recovery_conditions": recovery_results,
    }
    summary["diagnostics"] = _seed_diagnostics(summary, contract)
    write_json(run_dir / "SEED_RESULT.json", summary)
    gfg = compile_gfg(summary)
    write_json(run_dir / "EXPERIMENT_GFG.json", gfg)
    return summary


def aggregate(rows: list[dict[str, Any]], contract: dict[str, Any], *, formal: bool) -> dict[str, Any]:
    diagnostics = [row["diagnostics"] for row in rows]
    means = {
        "support_dose_spearman": mean(float(row["support_dose_spearman"]) for row in diagnostics),
        "unreinforced_margin_dose_spearman": mean(float(row["unreinforced_margin_dose_spearman"]) for row in diagnostics),
        "exclusive_unreinforced_accuracy_deficit_vs_balanced": mean(float(row["exclusive_unreinforced_accuracy_deficit_vs_balanced"]) for row in diagnostics),
        "exclusive_task0_support_share_excess_vs_balanced": mean(float(row["exclusive_task0_support_share_excess_vs_balanced"]) for row in diagnostics),
        "rebalance_unreinforced_accuracy_gain_vs_exclusive": mean(float(row["rebalance_unreinforced_accuracy_gain_vs_exclusive"]) for row in diagnostics),
        "repair_unreinforced_accuracy_gain_vs_exclusive": mean(float(row["repair_unreinforced_accuracy_gain_vs_exclusive"]) for row in diagnostics),
        "rebalance_target0_final_accuracy": mean(float(row["rebalance_target0_final_accuracy"]) for row in diagnostics),
        "repair_target0_final_accuracy": mean(float(row["repair_target0_final_accuracy"]) for row in diagnostics),
    }
    counts = {
        "positive_support_dose_association": sum(float(row["support_dose_spearman"]) > 0.0 for row in diagnostics),
        "negative_unreinforced_margin_dose_association": sum(float(row["unreinforced_margin_dose_spearman"]) < 0.0 for row in diagnostics),
        "rebalance_support_reversal": sum(bool(row["rebalance_support_reversal"]) for row in diagnostics),
        "repair_support_reversal": sum(bool(row["repair_support_reversal"]) for row in diagnostics),
    }
    result = {
        "schema": "rl-e06-aggregate-v1",
        "experiment_id": contract["experiment_id"],
        "formal": formal,
        "seed_count": len(rows),
        "per_seed": [{"seed": row["seed"], **row["diagnostics"]} for row in rows],
        "means": means,
        "counts": counts,
    }
    if not formal:
        result.update({"scientific_status": "DEVELOPMENT_ONLY_NO_FORMAL_DECISION", "decision_gates": {}})
        return result
    gates = contract["decision_gates"]
    gate_results = {
        "all_formal_seeds_retained": len(rows) == len(contract["formal_seeds"]),
        "initial_mastery_every_seed": all(float(row["baseline"]["evaluation"]["macro_chain_accuracy"]) == 1.0 for row in rows),
        "frozen_state_exact_every_seed": all(
            row["dose_conditions"]["frozen"]["final_state_sha256"] == row["baseline"]["state_sha256"] for row in rows
        ),
        "balanced_capability_preserved": all(
            min(float(skill["chain_accuracy"]) for skill in row["dose_conditions"]["balanced"]["final_evaluation"]["per_skill"])
            >= float(gates["minimum_balanced_final_accuracy_each_skill"])
            for row in rows
        ),
        "positive_support_dose_association": counts["positive_support_dose_association"]
        >= int(gates["minimum_seeds_with_positive_support_dose_association"]),
        "negative_unreinforced_margin_dose_association": counts["negative_unreinforced_margin_dose_association"]
        >= int(gates["minimum_seeds_with_negative_unreinforced_margin_dose_association"]),
        "exclusive_accuracy_deficit": means["exclusive_unreinforced_accuracy_deficit_vs_balanced"]
        >= float(gates["minimum_mean_exclusive_unreinforced_accuracy_deficit_vs_balanced"]),
        "exclusive_support_excess": means["exclusive_task0_support_share_excess_vs_balanced"]
        >= float(gates["minimum_mean_exclusive_task0_support_share_excess_vs_balanced"]),
        "rebalance_accuracy_recovery": sum(
            float(row["rebalance_unreinforced_accuracy_gain_vs_exclusive"])
            >= float(gates["minimum_per_seed_rebalance_unreinforced_accuracy_gain_vs_exclusive"])
            for row in diagnostics
        ) >= int(gates["minimum_seeds_with_rebalance_accuracy_recovery"]),
        "mean_rebalance_accuracy_recovery": means["rebalance_unreinforced_accuracy_gain_vs_exclusive"]
        >= float(gates["minimum_mean_rebalance_unreinforced_accuracy_gain_vs_exclusive"]),
        "rebalance_target0_retained": all(
            float(row["rebalance_target0_final_accuracy"]) >= float(gates["minimum_rebalance_target0_final_accuracy"])
            for row in diagnostics
        ),
        "recovery_support_reversal": counts["rebalance_support_reversal"]
        >= int(gates["minimum_seeds_with_recovery_support_reversal"]),
    }
    result.update({
        "decision_gates": gate_results,
        "scientific_status": "SUPPORTED" if all(gate_results.values()) else "NOT_SUPPORTED",
        "bounded_claim": (
            "in the executed shared GRU policy, increasing concentration of exact positive feedback caused a dose-dependent functional-support and capability trade-off, and redistributing feedback reversed part of that trade-off"
            if all(gate_results.values())
            else "the frozen experiment did not satisfy every preregistered gate for the dose-duration-recovery mechanism"
        ),
    })
    return result


def verify_freeze() -> dict[str, Any]:
    freeze = read_json(PACKAGE / "CONTRACT_FREEZE.json")
    for name, expected in freeze["files"].items():
        require(file_sha256(PACKAGE / name) == expected, f"RL_E06_FROZEN_FILE_CHANGED:{name}")
    return freeze


def run(artifact_root: Path, device_name: str, mode: str) -> dict[str, Any]:
    require(not artifact_root.exists(), f"RL_E06_ARTIFACT_ROOT_EXISTS:{artifact_root}")
    contract = read_json(PACKAGE / "MODEL_CONTRACT.json")
    formal = mode == "formal"
    if formal:
        require(contract["status"] == "FROZEN_BEFORE_FORMAL_EXECUTION", "RL_E06_CONTRACT_NOT_FROZEN")
        freeze = verify_freeze()
        seeds = [int(value) for value in contract["formal_seeds"]]
    else:
        require(contract["status"] == "DRAFT_DEVELOPMENT_ONLY", "RL_E06_DEVELOPMENT_CONTRACT_STATUS_INVALID")
        freeze = None
        seeds = [int(value) for value in contract["development_seeds"]]
    artifact_root.mkdir(parents=True)
    device = torch.device(device_name)
    started = utc_now()
    rows = []
    for seed in seeds:
        rows.append(run_seed(seed, contract, artifact_root, device))
        print(json.dumps({"mode": mode, "seed_complete": seed, "completed": len(rows), "total": len(seeds)}), flush=True)
    result = aggregate(rows, contract, formal=formal)
    result.update({
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "artifact_root": str(artifact_root),
            "free_gib_at_start": shutil.disk_usage(artifact_root).free / (1024 ** 3),
        },
    })
    if freeze is not None:
        result["freeze_sha256"] = file_sha256(PACKAGE / "CONTRACT_FREEZE.json")
        result["frozen_files"] = freeze["files"]
    write_json(artifact_root / ("FORMAL_RESULT.json" if formal else "DEVELOPMENT_RESULT.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.artifact_root.resolve(), args.device, args.mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

