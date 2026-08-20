from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
from statistics import mean
from typing import Any

import torch

from .runtime import (
    CONDITIONS,
    MultiSkillGRUPolicy,
    combined_state_sha256,
    deterministic_feedback_batch,
    evaluate_policy,
    make_optimizer,
    positive_feedback_update,
    restore_state,
    rollback_components_state,
    seed_everything,
    state_payload,
    supervised_pretrain_step,
    support_profile,
)


PACKAGE = Path(__file__).parent


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _checkpoint_payload(
    model: MultiSkillGRUPolicy,
    optimizer: torch.optim.AdamW,
    *,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    return {
        "schema": "rl-e05-training-state-v1",
        "seed": seed,
        "condition": condition,
        "update": update,
        "state": state_payload(model, optimizer),
        "sealed_at_utc": utc_now(),
    }


def _save_checkpoint(
    path: Path,
    model: MultiSkillGRUPolicy,
    optimizer: torch.optim.AdamW,
    *,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, str]:
    torch.save(
        _checkpoint_payload(model, optimizer, seed=seed, condition=condition, update=update),
        path,
    )
    return {"path": str(path), "sha256": file_sha256(path)}


def _skill_means(evaluation: dict[str, Any], key: str, skills: list[int]) -> float:
    return mean(float(evaluation["per_skill"][skill][key]) for skill in skills)


def _checkpoint_measurement(
    model: MultiSkillGRUPolicy,
    *,
    condition: str,
    update: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema": "rl-e05-checkpoint-measurement-v1",
        "condition": condition,
        "update": update,
        "model_sha256": hashlib.sha256(
            b"".join(value.detach().cpu().contiguous().numpy().tobytes() for value in model.state_dict().values())
        ).hexdigest(),
        "evaluation": evaluate_policy(model, device),
        "support": support_profile(model, device),
    }


def _run_branch(
    *,
    condition: str,
    seed: int,
    baseline: dict[str, Any],
    contract: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> tuple[dict[str, Any], MultiSkillGRUPolicy, torch.optim.AdamW]:
    model, optimizer = restore_state(
        baseline,
        hidden_size=int(contract["model"]["hidden_size"]),
        learning_rate=float(contract["model"]["learning_rate"]),
        weight_decay=float(contract["model"]["weight_decay"]),
        device=device,
    )
    initial_hash = combined_state_sha256(model, optimizer)
    checkpoints = set(int(value) for value in contract["feedback"]["checkpoints"])
    measurements = [_checkpoint_measurement(model, condition=condition, update=0, device=device)]
    receipts: list[dict[str, Any]] = []
    for index in range(int(contract["feedback"]["updates"])):
        batch = deterministic_feedback_batch(
            seed=seed + 500_009,
            update=index,
            batch_size=int(contract["feedback"]["batch_size"]),
            condition=condition,
            device=device,
        )
        receipt = positive_feedback_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            apply_update=condition != "frozen",
        )
        receipt.update({
            "schema": "rl-e05-feedback-occurrence-v1",
            "seed": seed,
            "condition": condition,
            "update": index + 1,
            "reward_authority": "exact_complete_chain_rule",
        })
        receipts.append(receipt)
        completed = index + 1
        if completed in checkpoints:
            measurements.append(
                _checkpoint_measurement(model, condition=condition, update=completed, device=device)
            )
    receipt_path = run_dir / f"{condition}-feedback-receipts.jsonl"
    write_jsonl(receipt_path, receipts)
    checkpoint = _save_checkpoint(
        run_dir / f"{condition}-final.pt",
        model,
        optimizer,
        seed=seed,
        condition=condition,
        update=int(contract["feedback"]["updates"]),
    )
    result = {
        "schema": "rl-e05-condition-result-v1",
        "condition": condition,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": combined_state_sha256(model, optimizer),
        "feedback_receipts": str(receipt_path),
        "feedback_receipts_sha256": file_sha256(receipt_path),
        "checkpoint": checkpoint,
        "measurements": measurements,
        "final_evaluation": measurements[-1]["evaluation"],
        "final_support": measurements[-1]["support"],
        "total_positive_consequences": sum(row["positive_consequence_count"] for row in receipts),
        "total_episodes": sum(row["episode_count"] for row in receipts),
    }
    write_json(run_dir / f"{condition}-result.json", result)
    return result, model, optimizer


def _run_rollbacks(
    *,
    trained: MultiSkillGRUPolicy,
    baseline_state: dict[str, torch.Tensor],
    final_evaluation: dict[str, Any],
    contract: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    tolerance = float(contract["causal_rollback"]["directional_tolerance"])
    final_reinforced_margin = _skill_means(final_evaluation, "mean_margin", [0])
    final_unreinforced_margin = _skill_means(final_evaluation, "mean_margin", [1, 2, 3])
    final_unreinforced_accuracy = _skill_means(final_evaluation, "chain_accuracy", [1, 2, 3])
    rows = []
    for mask in contract["causal_rollback"]["subset_masks"]:
        components = [component for component in range(4) if int(mask) & (1 << component)]
        rollback = rollback_components_state(trained, baseline_state, components)
        evaluation = evaluate_policy(rollback, device)
        profile = support_profile(rollback, device)
        reinforced_margin = _skill_means(evaluation, "mean_margin", [0])
        unreinforced_margin = _skill_means(evaluation, "mean_margin", [1, 2, 3])
        unreinforced_accuracy = _skill_means(evaluation, "chain_accuracy", [1, 2, 3])
        directional = (
            int(mask) != 15
            and reinforced_margin < final_reinforced_margin - tolerance
            and (
                unreinforced_margin > final_unreinforced_margin + tolerance
                or unreinforced_accuracy > final_unreinforced_accuracy + tolerance
            )
        )
        rows.append({
            "mask": int(mask),
            "components": components,
            "evaluation": evaluation,
            "support": profile,
            "reinforced_mean_margin_change": reinforced_margin - final_reinforced_margin,
            "unreinforced_mean_margin_change": unreinforced_margin - final_unreinforced_margin,
            "unreinforced_mean_accuracy_change": unreinforced_accuracy - final_unreinforced_accuracy,
            "directional_proper_subset": directional,
        })
    return {
        "schema": "rl-e05-exhaustive-component-version-rollback-v1",
        "subset_count": len(rows),
        "directional_proper_subset_exists": any(row["directional_proper_subset"] for row in rows),
        "rows": rows,
    }


def _temporal_and_capture(summary: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    baseline_support = summary["baseline"]["support"]
    selective = summary["conditions"]["selective"]["measurements"]
    support_threshold = float(contract["decision_gates"]["mean_selective_reinforced_support_share_increase"])
    baseline_share = float(baseline_support["task_support_shares"][0])
    first_support = next(
        (
            int(row["update"])
            for row in selective
            if float(row["support"]["task_support_shares"][0]) >= baseline_share + support_threshold
        ),
        None,
    )
    first_drop = next(
        (
            int(row["update"])
            for row in selective
            if _skill_means(row["evaluation"], "chain_accuracy", [1, 2, 3]) < 1.0 - 1e-12
        ),
        None,
    )
    baseline_primary = baseline_support["primary_task_by_component"]
    final_primary = summary["conditions"]["selective"]["final_support"]["primary_task_by_component"]
    captured = [
        component for component in range(4)
        if int(baseline_primary[component]) != 0 and int(final_primary[component]) == 0
    ]
    return {
        "support_share_event_delta": support_threshold,
        "first_support_concentration_update": first_support,
        "first_unreinforced_accuracy_drop_update": first_drop,
        "support_change_before_or_at_first_drop": (
            first_support is not None and first_drop is not None and first_support <= first_drop
        ),
        "newly_captured_components": captured,
        "new_component_capture": bool(captured),
    }


def compile_gfg(seed_summary: dict[str, Any]) -> dict[str, Any]:
    seed = int(seed_summary["seed"])
    facts: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []

    def add(occurrence_id: str, origin: str, transformation: str, outcome: str, role: str) -> None:
        fact_id = f"fact:{occurrence_id}"
        occurrences.append({"occurrence_id": occurrence_id, "realizes_fact": [fact_id]})
        facts.append({
            "fact_id": fact_id,
            "u": origin,
            "tau": transformation,
            "omega": occurrence_id,
            "z": outcome,
            "rho": role,
        })

    add(f"seed-{seed}:baseline-seal", "balanced-supervised-formation", "seal_receiving_state", seed_summary["baseline"]["state_sha256"], "receiving_state")
    for condition, result in seed_summary["conditions"].items():
        with Path(result["feedback_receipts"]).open("r", encoding="utf-8") as handle:
            for line in handle:
                receipt = json.loads(line)
                add(
                    f"seed-{seed}:{condition}:feedback-update-{receipt['update']}",
                    json.dumps({
                        "receiving_state": receipt["pre_state_sha256"],
                        "cue_batch": receipt["cue_batch_sha256"],
                        "action_uniforms": receipt["uniform_batch_sha256"],
                        "positive_consequence_count": receipt["positive_consequence_count"],
                    }, sort_keys=True),
                    f"exact_consequence_binding_and_positive_feedback:{condition}",
                    receipt["post_state_sha256"],
                    "actual_training_action" if condition != "frozen" else "explicit_no_persistent_update",
                )
        for row in result["measurements"]:
            update = int(row["update"])
            add(
                f"seed-{seed}:{condition}:checkpoint-{update}",
                result["initial_state_sha256"],
                f"{condition}_positive_feedback_through_{update}",
                row["model_sha256"],
                "checkpoint_state",
            )
        add(
            f"seed-{seed}:{condition}:support-adjudication",
            result["checkpoint"]["sha256"],
            "all_16_component_coalition_interventions_and_exact_shapley",
            result["final_support"]["coalition_value_sha256"],
            "functional_support",
        )
    for row in seed_summary["rollbacks"]["rows"]:
        add(
            f"seed-{seed}:rollback-mask-{row['mask']}",
            seed_summary["conditions"]["selective"]["checkpoint"]["sha256"],
            f"restore_baseline_component_versions:{row['components']}",
            hashlib.sha256(json.dumps(row["evaluation"], sort_keys=True).encode()).hexdigest(),
            "causal_version_intervention",
        )
    return {
        "schema": "rl-e05-generation-fact-graph-v1",
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
    require(not run_dir.exists(), f"RL_E05_RUN_DIR_EXISTS:{run_dir}")
    run_dir.mkdir(parents=True)
    seed_everything(seed)
    model = MultiSkillGRUPolicy(int(contract["model"]["hidden_size"])).to(device)
    optimizer = make_optimizer(
        model,
        float(contract["model"]["learning_rate"]),
        float(contract["model"]["weight_decay"]),
    )
    pretrain_receipts = []
    for index in range(int(contract["formation"]["pretrain_max_steps"])):
        receipt = supervised_pretrain_step(model, optimizer, device)
        receipt["update"] = index + 1
        pretrain_receipts.append(receipt)
        if index + 1 >= int(contract["formation"]["pretrain_min_steps"]):
            evaluation = evaluate_policy(model, device)
            if (
                evaluation["macro_chain_accuracy"] == float(contract["formation"]["required_chain_accuracy"])
                and min(row["minimum_margin"] for row in evaluation["per_skill"])
                >= float(contract["formation"]["pretrain_minimum_margin"])
            ):
                break
    baseline_evaluation = evaluate_policy(model, device)
    require(baseline_evaluation["macro_chain_accuracy"] == 1.0, f"RL_E05_BASELINE_MASTERY_FAILED:{seed}")
    baseline_support = support_profile(model, device)
    baseline = state_payload(model, optimizer)
    baseline_model_state = deepcopy(model.state_dict())
    write_jsonl(run_dir / "pretrain-receipts.jsonl", pretrain_receipts)
    baseline_checkpoint = _save_checkpoint(
        run_dir / "baseline-seal.pt", model, optimizer, seed=seed, condition="baseline", update=len(pretrain_receipts)
    )
    branch_results: dict[str, Any] = {}
    selective_model: MultiSkillGRUPolicy | None = None
    initial_hashes = []
    for condition in CONDITIONS:
        result, branch_model, _ = _run_branch(
            condition=condition,
            seed=seed,
            baseline=baseline,
            contract=contract,
            device=device,
            run_dir=run_dir,
        )
        branch_results[condition] = result
        initial_hashes.append(result["initial_state_sha256"])
        if condition == "selective":
            selective_model = branch_model
    require(len(set(initial_hashes)) == 1, f"RL_E05_BRANCH_CLONE_MISMATCH:{seed}")
    require(selective_model is not None, "RL_E05_SELECTIVE_MODEL_MISSING")
    rollbacks = _run_rollbacks(
        trained=selective_model,
        baseline_state=baseline_model_state,
        final_evaluation=branch_results["selective"]["final_evaluation"],
        contract=contract,
        device=device,
    )
    summary = {
        "schema": "rl-e05-seed-result-v1",
        "seed": seed,
        "pretrain_updates": len(pretrain_receipts),
        "baseline": {
            "state_sha256": baseline["combined_state_sha256"],
            "checkpoint": baseline_checkpoint,
            "evaluation": baseline_evaluation,
            "support": baseline_support,
        },
        "branch_initial_state_sha256": initial_hashes[0],
        "conditions": branch_results,
        "rollbacks": rollbacks,
    }
    summary["temporal_and_capture"] = _temporal_and_capture(summary, contract)
    write_json(run_dir / "SEED_RESULT.json", summary)
    gfg = compile_gfg(summary)
    write_json(run_dir / "EXPERIMENT_GFG.json", gfg)
    return summary


def aggregate(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    selective_unreinforced = [
        _skill_means(row["conditions"]["selective"]["final_evaluation"], "chain_accuracy", [1, 2, 3])
        for row in rows
    ]
    balanced_unreinforced = [
        _skill_means(row["conditions"]["balanced"]["final_evaluation"], "chain_accuracy", [1, 2, 3])
        for row in rows
    ]
    baseline_share = [float(row["baseline"]["support"]["task_support_shares"][0]) for row in rows]
    selective_share = [float(row["conditions"]["selective"]["final_support"]["task_support_shares"][0]) for row in rows]
    balanced_share = [float(row["conditions"]["balanced"]["final_support"]["task_support_shares"][0]) for row in rows]
    frozen_exact = [
        row["conditions"]["frozen"]["final_state_sha256"] == row["baseline"]["state_sha256"]
        for row in rows
    ]
    gates = contract["decision_gates"]
    gate_results = {
        "all_formal_seeds_retained": len(rows) == len(contract["formal_seeds"]),
        "initial_mastery_every_seed": all(row["baseline"]["evaluation"]["macro_chain_accuracy"] == 1.0 for row in rows),
        "frozen_state_exact_every_seed": all(frozen_exact),
        "balanced_final_each_skill": all(
            min(skill["chain_accuracy"] for skill in row["conditions"]["balanced"]["final_evaluation"]["per_skill"])
            >= float(gates["balanced_final_minimum_chain_accuracy_each_skill"])
            for row in rows
        ),
        "selective_reinforced_final": all(
            row["conditions"]["selective"]["final_evaluation"]["per_skill"][0]["chain_accuracy"]
            >= float(gates["selective_final_minimum_reinforced_chain_accuracy"])
            for row in rows
        ),
        "unreinforced_accuracy_deficit_vs_balanced": mean(
            balanced - selective for balanced, selective in zip(balanced_unreinforced, selective_unreinforced)
        ) >= float(gates["mean_selective_unreinforced_accuracy_deficit_vs_balanced"]),
        "reinforced_support_share_increase": mean(
            final - initial for final, initial in zip(selective_share, baseline_share)
        ) >= float(gates["mean_selective_reinforced_support_share_increase"]),
        "support_share_excess_vs_balanced": mean(
            selective - balanced for selective, balanced in zip(selective_share, balanced_share)
        ) >= float(gates["mean_selective_support_share_excess_vs_balanced"]),
        "new_component_capture": sum(row["temporal_and_capture"]["new_component_capture"] for row in rows)
        >= int(gates["minimum_seeds_with_new_component_capture"]),
        "temporal_precedence": sum(row["temporal_and_capture"]["support_change_before_or_at_first_drop"] for row in rows)
        >= int(gates["minimum_seeds_with_support_change_before_or_at_first_unreinforced_drop"]),
        "directional_proper_subset_rollback": sum(row["rollbacks"]["directional_proper_subset_exists"] for row in rows)
        >= int(gates["minimum_seeds_with_directional_rollback_evidence"]),
    }
    return {
        "schema": "rl-e05-formal-aggregate-v1",
        "experiment_id": contract["experiment_id"],
        "seed_count": len(rows),
        "per_seed": [{
            "seed": row["seed"],
            "baseline_task0_support_share": float(row["baseline"]["support"]["task_support_shares"][0]),
            "selective_task0_support_share": float(row["conditions"]["selective"]["final_support"]["task_support_shares"][0]),
            "balanced_task0_support_share": float(row["conditions"]["balanced"]["final_support"]["task_support_shares"][0]),
            "selective_reinforced_accuracy": float(row["conditions"]["selective"]["final_evaluation"]["per_skill"][0]["chain_accuracy"]),
            "selective_unreinforced_mean_accuracy": selective_unreinforced[index],
            "balanced_unreinforced_mean_accuracy": balanced_unreinforced[index],
            "selective_cross_task_hhi": float(row["conditions"]["selective"]["final_support"]["cross_task_hhi"]),
            "balanced_cross_task_hhi": float(row["conditions"]["balanced"]["final_support"]["cross_task_hhi"]),
            "newly_captured_components": row["temporal_and_capture"]["newly_captured_components"],
            "first_support_concentration_update": row["temporal_and_capture"]["first_support_concentration_update"],
            "first_unreinforced_accuracy_drop_update": row["temporal_and_capture"]["first_unreinforced_accuracy_drop_update"],
            "directional_proper_subset_rollback": row["rollbacks"]["directional_proper_subset_exists"],
        } for index, row in enumerate(rows)],
        "means": {
            "baseline_task0_support_share": mean(baseline_share),
            "selective_task0_support_share": mean(selective_share),
            "balanced_task0_support_share": mean(balanced_share),
            "selective_support_share_increase": mean(final - initial for final, initial in zip(selective_share, baseline_share)),
            "selective_support_share_excess_vs_balanced": mean(final - balanced for final, balanced in zip(selective_share, balanced_share)),
            "selective_unreinforced_mean_accuracy": mean(selective_unreinforced),
            "balanced_unreinforced_mean_accuracy": mean(balanced_unreinforced),
            "unreinforced_accuracy_deficit_vs_balanced": mean(balanced - selective for balanced, selective in zip(balanced_unreinforced, selective_unreinforced)),
        },
        "counts": {
            "new_component_capture": sum(row["temporal_and_capture"]["new_component_capture"] for row in rows),
            "temporal_precedence": sum(row["temporal_and_capture"]["support_change_before_or_at_first_drop"] for row in rows),
            "directional_proper_subset_rollback": sum(row["rollbacks"]["directional_proper_subset_exists"] for row in rows),
        },
        "decision_gates": gate_results,
        "scientific_status": "SUPPORTED" if all(gate_results.values()) else "NOT_SUPPORTED",
        "bounded_claim": "correct selective positive feedback caused functional-support concentration and crowding in the executed shared finite GRU policy" if all(gate_results.values()) else "the frozen experiment did not establish the proposed mechanism",
    }


def verify_freeze() -> dict[str, Any]:
    freeze = read_json(PACKAGE / "CONTRACT_FREEZE.json")
    for name, expected in freeze["files"].items():
        require(file_sha256(PACKAGE / name) == expected, f"RL_E05_FROZEN_FILE_CHANGED:{name}")
    return freeze


def run(artifact_root: Path, device_name: str) -> dict[str, Any]:
    require(not artifact_root.exists(), f"RL_E05_ARTIFACT_ROOT_EXISTS:{artifact_root}")
    freeze = verify_freeze()
    contract = read_json(PACKAGE / "MODEL_CONTRACT.json")
    require(contract["status"] == "FROZEN_BEFORE_FORMAL_EXECUTION", "RL_E05_CONTRACT_NOT_FROZEN")
    artifact_root.mkdir(parents=True)
    free_gib = shutil.disk_usage(artifact_root).free / (1024 ** 3)
    device = torch.device(device_name)
    started = utc_now()
    rows = []
    for seed in contract["formal_seeds"]:
        rows.append(run_seed(int(seed), contract, artifact_root, device))
        print(json.dumps({"seed_complete": int(seed), "completed": len(rows), "total": len(contract["formal_seeds"])}), flush=True)
    result = aggregate(rows, contract)
    result.update({
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "artifact_root": str(artifact_root),
            "free_gib_at_start": free_gib,
        },
        "freeze_sha256": file_sha256(PACKAGE / "CONTRACT_FREEZE.json"),
        "frozen_files": freeze["files"],
    })
    write_json(artifact_root / "FORMAL_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.artifact_root.resolve(), args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
