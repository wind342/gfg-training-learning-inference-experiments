from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .runtime import (
    CONDITIONS,
    MultiSkillGRUPolicy,
    deterministic_feedback_batch,
    evaluate_policy,
    make_optimizer,
    positive_feedback_update,
    restore_state,
    seed_everything,
    state_payload,
    supervised_pretrain_step,
    support_profile,
)


def run_configuration(config: dict, device: torch.device) -> dict:
    seed = int(config["seed"])
    seed_everything(seed)
    model = MultiSkillGRUPolicy(int(config["hidden_size"])).to(device)
    optimizer = make_optimizer(model, float(config["learning_rate"]), float(config["weight_decay"]))
    pretrain_steps = 0
    for step in range(int(config["pretrain_max_steps"])):
        supervised_pretrain_step(model, optimizer, device)
        pretrain_steps = step + 1
        if pretrain_steps >= int(config["pretrain_min_steps"]):
            evaluation = evaluate_policy(model, device)
            margins = [row["minimum_margin"] for row in evaluation["per_skill"]]
            if evaluation["macro_chain_accuracy"] == 1.0 and min(margins) >= float(config["pretrain_min_margin"]):
                break
    baseline = state_payload(model, optimizer)
    branches = {}
    for condition in CONDITIONS:
        branch_model, branch_optimizer = restore_state(
            baseline,
            hidden_size=int(config["hidden_size"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            device=device,
        )
        checkpoints = []
        for update in range(int(config["feedback_updates"])):
            batch = deterministic_feedback_batch(
                seed=seed + 500_009,
                update=update,
                batch_size=int(config["batch_size"]),
                condition=condition,
                device=device,
            )
            positive_feedback_update(
                model=branch_model,
                optimizer=branch_optimizer,
                batch=batch,
                apply_update=condition != "frozen",
            )
            completed = update + 1
            if completed in set(config["checkpoints"]):
                checkpoints.append({
                    "update": completed,
                    "evaluation": evaluate_policy(branch_model, device),
                    "support": support_profile(branch_model, device),
                })
        branches[condition] = {
            "final_evaluation": evaluate_policy(branch_model, device),
            "final_support": support_profile(branch_model, device),
            "checkpoints": checkpoints,
        }
    return {
        "configuration": config,
        "pretrain_steps": pretrain_steps,
        "baseline_evaluation": evaluate_policy(model, device),
        "baseline_support": support_profile(model, device),
        "branches": branches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    configurations = []
    for hidden_size in (8, 12, 16):
        for learning_rate in (0.001, 0.003):
            configurations.append({
                "seed": 20260851,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "weight_decay": 0.0001,
                "pretrain_max_steps": 3000,
                "pretrain_min_steps": 100,
                "pretrain_min_margin": 0.5,
                "feedback_updates": 800,
                "batch_size": 64,
                "checkpoints": [100, 200, 400, 800],
            })
    rows = [run_configuration(config, device) for config in configurations]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
