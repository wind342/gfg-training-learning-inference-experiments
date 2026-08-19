from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .common import write_json
from .nanogpt_adapter import TrainingConfig, detect_transition, train_plain
from .task_generator import TaskInstanceSpec, build_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--modulus", type=int, default=23)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--token-seed", type=int, default=11003)
    parser.add_argument("--split-seed", type=int, default=11027)
    parser.add_argument("--model-seed", type=int, default=11047)
    parser.add_argument("--data-order-seed", type=int, default=11057)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    args = parser.parse_args()

    task_spec = TaskInstanceSpec(
        instance_id="engineering-preflight-only",
        modulus=args.modulus,
        train_fraction=args.train_fraction,
        token_permutation_seed=args.token_seed,
        split_seed=args.split_seed,
    )
    task = build_task(task_spec)
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        evaluation_interval=args.evaluation_interval,
        seed=args.model_seed,
        data_order_seed=args.data_order_seed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    result = train_plain(args.trainer_root.resolve(), task, config)
    transition = detect_transition(
        result["metrics"],
        train_threshold=0.99,
        pre_transition_validation_max=0.30,
        validation_threshold=0.90,
        sustained_points=3,
    )
    report = {
        "checkpoint_sha256": result["checkpoint_sha256"],
        "elapsed_seconds": result["elapsed_seconds"],
        "engineering_only": True,
        "evaluation_interval": args.evaluation_interval,
        "final_metric": result["metrics"][-1],
        "metrics": result["metrics"],
        "model_parameter_count": result["model_parameter_count"],
        "schema": "nanogpt-capability-engineering-preflight-v1",
        "status": "PASS",
        "task_commitment": task.private_generation_commitment,
        "task_spec": asdict(task_spec),
        "training_config": asdict(config),
        "transition_step": transition,
    }
    path = (
        args.external_root.resolve()
        / "engineering-preflight"
        / "result.json"
    )
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
