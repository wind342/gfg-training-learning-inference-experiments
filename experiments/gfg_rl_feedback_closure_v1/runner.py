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
    DelayedGRUPolicy,
    clone_training_state,
    combined_state_sha256,
    deterministic_batch,
    evaluate_policy,
    make_optimizer,
    object_sha256,
    seed_everything,
    train_update,
)


PACKAGE = Path(__file__).parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


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


def preflight(artifact_root: Path, require_cuda: bool) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(artifact_root).free / (1024 ** 3)
    cuda = torch.cuda.is_available()
    if require_cuda:
        require(cuda, "RL_CUDA_REQUIRED")
    return {
        "timestamp_utc": utc_now(),
        "artifact_root": str(artifact_root),
        "artifact_free_gib": free_gib,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
    }


def _training_state_payload(model: DelayedGRUPolicy, optimizer: torch.optim.AdamW) -> dict[str, Any]:
    return {
        "model": deepcopy(model.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
        "combined_state_sha256": combined_state_sha256(model, optimizer),
    }


def _load_training_state(
    payload: dict[str, Any], contract: dict[str, Any], device: torch.device,
) -> tuple[DelayedGRUPolicy, torch.optim.AdamW]:
    model = DelayedGRUPolicy(int(contract["model"]["hidden_size"])).to(device)
    model.load_state_dict(deepcopy(payload["model"]))
    optimizer = make_optimizer(model, contract)
    # torch optimizer loading may normalize the supplied state dictionary in
    # place.  A sealed payload is reused for all three conditions, so it must
    # never be handed to a loader by reference.
    optimizer.load_state_dict(deepcopy(payload["optimizer"]))
    require(combined_state_sha256(model, optimizer) == payload["combined_state_sha256"], "RL_STATE_RESTORE_MISMATCH")
    return model, optimizer


def _auc(curve: list[dict[str, Any]]) -> float:
    return float(mean(float(row["chain_accuracy"]) for row in curve))


def _episodes_to_threshold(curve: list[dict[str, Any]], batch_size: int, threshold: float) -> int | None:
    for row in curve:
        if float(row["chain_accuracy"]) >= threshold:
            return int(row["update"]) * batch_size
    return None


def _run_condition(
    *,
    condition: str,
    seed: int,
    state_payload: dict[str, Any],
    contract: dict[str, Any],
    run_config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    fork_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model, optimizer = _load_training_state(state_payload, contract, device)
    initial_hash = combined_state_sha256(model, optimizer)
    updates = int(run_config["reversal_updates"])
    batch_size = int(run_config["batch_size"])
    interval = int(run_config.get("evaluation_interval", 5))
    fork_updates = set(int(value) for value in run_config.get("fork_updates", []))
    entropy = float(contract["model"]["entropy_coefficient"])
    exploration = float(contract["model"]["behavior_exploration_epsilon"])
    ledger: list[dict[str, Any]] = []
    curve = [{"update": 0, **evaluate_policy(model, "B", device)}]
    for update in range(updates):
        batch = deterministic_batch(seed + 400_009, update, batch_size, device)
        if condition == "A" and update in fork_updates:
            common_hash = combined_state_sha256(model, optimizer)
            fork_outputs: dict[str, dict[str, Any]] = {}
            for fork_condition in CONDITIONS:
                fork_model, fork_optimizer = clone_training_state(model, optimizer, contract, device)
                fork_result = train_update(
                    model=fork_model,
                    optimizer=fork_optimizer,
                    batch=batch,
                    phase="B",
                    condition=fork_condition,
                    entropy_coefficient=entropy,
                    exploration_epsilon=exploration,
                    include_episode_ledger=False,
                )
                fork_outputs[fork_condition] = fork_result
            fork_rows.append({
                "schema": "rl-one-step-causal-fork-v1",
                "seed": seed,
                "update": update,
                "common_receiving_state_sha256": common_hash,
                "cue_batch_sha256": fork_outputs["A"]["cue_batch_sha256"],
                "uniform_batch_sha256": fork_outputs["A"]["uniform_batch_sha256"],
                "conditions": fork_outputs,
            })
        result = train_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            phase="B",
            condition=condition,
            entropy_coefficient=entropy,
            exploration_epsilon=exploration,
            include_episode_ledger=True,
        )
        result["schema"] = "rl-update-receipt-v1"
        result["seed"] = seed
        result["update"] = update
        ledger.append(result)
        completed = update + 1
        if completed % interval == 0 or completed == updates:
            curve.append({"update": completed, **evaluate_policy(model, "B", device)})
    sealed_at = utc_now()
    checkpoint = run_dir / f"condition-{condition}.pt"
    torch.save({
        "schema": "rl-condition-checkpoint-v1",
        "condition": condition,
        "seed": seed,
        "state": _training_state_payload(model, optimizer),
        "sealed_at_utc": sealed_at,
    }, checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)
    final_evaluation = evaluate_policy(model, "B", device)
    final_evaluation["sealed_checkpoint_sha256"] = checkpoint_sha256
    ledger_path = run_dir / f"condition-{condition}-events.jsonl"
    write_jsonl(ledger_path, ledger)
    result_path = run_dir / f"condition-{condition}-result.json"
    result = {
        "schema": "rl-condition-result-v1",
        "condition": condition,
        "seed": seed,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": combined_state_sha256(model, optimizer),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "ledger": str(ledger_path),
        "ledger_sha256": file_sha256(ledger_path),
        "updates": updates,
        "batch_size": batch_size,
        "episodes": updates * batch_size,
        "curve": curve,
        "auc": _auc(curve),
        "episodes_to_threshold": _episodes_to_threshold(
            curve, batch_size, float(contract["success_gates"]["condition_a_final_min_chain_accuracy"]),
        ),
        "final_evaluation": final_evaluation,
        "sealed_at_utc": sealed_at,
        "evaluated_at_utc": utc_now(),
    }
    write_json(result_path, result)
    result["result_path"] = str(result_path)
    result["result_sha256"] = file_sha256(result_path)
    return result


def run_seed(
    *,
    seed: int,
    mode: str,
    contract: dict[str, Any],
    artifact_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_config = contract[mode]
    run_dir = artifact_root / mode / f"seed-{seed}"
    require(not run_dir.exists(), f"RL_RUN_DIR_ALREADY_EXISTS:{run_dir}")
    run_dir.mkdir(parents=True)
    seed_everything(seed)
    model = DelayedGRUPolicy(int(contract["model"]["hidden_size"])).to(device)
    optimizer = make_optimizer(model, contract)
    base_curve = [{"update": 0, **evaluate_policy(model, "A", device)}]
    base_updates = int(run_config["base_updates"])
    batch_size = int(run_config["batch_size"])
    for update in range(base_updates):
        batch = deterministic_batch(seed + 100_003, update, batch_size, device)
        train_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            phase="A",
            condition="A",
            entropy_coefficient=float(contract["model"]["entropy_coefficient"]),
            exploration_epsilon=float(contract["model"]["behavior_exploration_epsilon"]),
            include_episode_ledger=False,
        )
        if (update + 1) % 10 == 0 or update + 1 == base_updates:
            base_curve.append({"update": update + 1, **evaluate_policy(model, "A", device)})
    base_evaluation = evaluate_policy(model, "A", device)
    base_state = _training_state_payload(model, optimizer)
    base_checkpoint = run_dir / "phase-a-seal.pt"
    torch.save({
        "schema": "rl-phase-a-seal-v1",
        "seed": seed,
        "state": base_state,
        "evaluation": base_evaluation,
        "sealed_at_utc": utc_now(),
    }, base_checkpoint)
    clone_hashes = []
    for _ in CONDITIONS:
        clone_model, clone_optimizer = _load_training_state(base_state, contract, device)
        clone_hashes.append(combined_state_sha256(clone_model, clone_optimizer))
    require(len(set(clone_hashes)) == 1, "RL_INITIAL_CLONES_DIFFER")
    fork_rows: list[dict[str, Any]] = []
    condition_results = {
        condition: _run_condition(
            condition=condition,
            seed=seed,
            state_payload=base_state,
            contract=contract,
            run_config=run_config,
            device=device,
            run_dir=run_dir,
            fork_rows=fork_rows,
        )
        for condition in CONDITIONS
    }
    fork_path = run_dir / "one-step-causal-forks.json"
    write_json(fork_path, fork_rows)
    summary = {
        "schema": "rl-seed-result-v1",
        "mode": mode,
        "seed": seed,
        "phase_a": {
            "curve": base_curve,
            "final_evaluation": base_evaluation,
            "seal_checkpoint": str(base_checkpoint),
            "seal_checkpoint_sha256": file_sha256(base_checkpoint),
            "combined_state_sha256": base_state["combined_state_sha256"],
        },
        "clone_state_sha256": clone_hashes[0],
        "conditions": condition_results,
        "forks": str(fork_path),
        "forks_sha256": file_sha256(fork_path),
    }
    write_json(run_dir / "SEED_RESULT.json", summary)
    return summary


def aggregate(mode: str, rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        finals = [row["conditions"][condition]["final_evaluation"]["chain_accuracy"] for row in rows]
        aucs = [row["conditions"][condition]["auc"] for row in rows]
        by_condition[condition] = {
            "mean_final_chain_accuracy": mean(finals),
            "per_seed_final_chain_accuracy": finals,
            "mean_auc": mean(aucs),
            "per_seed_auc": aucs,
        }
    a_wins_b = sum(
        row["conditions"]["A"]["auc"] > row["conditions"]["B"]["auc"] for row in rows
    )
    a_wins_c = sum(
        row["conditions"]["A"]["auc"] > row["conditions"]["C"]["auc"] for row in rows
    )
    fork_total = 0
    fork_nonidentical_ab = 0
    fork_nonidentical_ac = 0
    for row in rows:
        forks = read_json(Path(row["forks"]))
        for fork in forks:
            fork_total += 1
            values = fork["conditions"]
            fork_nonidentical_ab += values["A"]["actual_update_sha256"] != values["B"]["actual_update_sha256"]
            fork_nonidentical_ac += values["A"]["actual_update_sha256"] != values["C"]["actual_update_sha256"]
    gates = contract["success_gates"]
    scientific_gates = {
        "phase_a_seal": all(
            row["phase_a"]["final_evaluation"]["chain_accuracy"] >= gates["phase_a_seal_min_chain_accuracy"]
            for row in rows
        ),
        "condition_a_final": all(
            row["conditions"]["A"]["final_evaluation"]["chain_accuracy"]
            >= gates["condition_a_final_min_chain_accuracy"] for row in rows
        ),
        "a_auc_advantage_over_b": by_condition["A"]["mean_auc"] - by_condition["B"]["mean_auc"]
        >= gates["condition_a_min_auc_advantage_over_each_control"],
        "a_auc_advantage_over_c": by_condition["A"]["mean_auc"] - by_condition["C"]["mean_auc"]
        >= gates["condition_a_min_auc_advantage_over_each_control"],
        "a_seed_wins_over_b": a_wins_b >= min(len(rows), int(gates["condition_a_min_seed_wins_over_each_control"])),
        "a_seed_wins_over_c": a_wins_c >= min(len(rows), int(gates["condition_a_min_seed_wins_over_each_control"])),
        "fork_update_identity_ab": fork_total > 0 and fork_nonidentical_ab / fork_total
        >= gates["one_step_fork_min_nonidentical_update_fraction"],
        "fork_update_identity_ac": fork_total > 0 and fork_nonidentical_ac / fork_total
        >= gates["one_step_fork_min_nonidentical_update_fraction"],
    }
    return {
        "schema": "rl-feedback-closure-aggregate-v1",
        "mode": mode,
        "seed_count": len(rows),
        "conditions": by_condition,
        "a_seed_wins_over_b": a_wins_b,
        "a_seed_wins_over_c": a_wins_c,
        "fork_count": fork_total,
        "fork_nonidentical_update_fraction_ab": fork_nonidentical_ab / fork_total if fork_total else None,
        "fork_nonidentical_update_fraction_ac": fork_nonidentical_ac / fork_total if fork_total else None,
        "scientific_gates": scientific_gates,
        "scientific_status": "PASS" if all(scientific_gates.values()) else "FAIL",
    }


def run(mode: str, artifact_root: Path, require_cuda: bool) -> dict[str, Any]:
    contract = read_json(PACKAGE / "EXPERIMENT_CONTRACT.json")
    require(mode in {"development", "formal"}, "RL_MODE_INVALID")
    if mode == "formal":
        freeze_path = PACKAGE / "CONTRACT_FREEZE.json"
        require(freeze_path.exists(), "RL_FORMAL_FREEZE_MISSING")
        freeze = read_json(freeze_path)
        require(file_sha256(PACKAGE / "EXPERIMENT_CONTRACT.json") == freeze["experiment_contract_sha256"], "RL_FORMAL_CONTRACT_CHANGED")
        require(file_sha256(PACKAGE / "PROTOCOL_FREEZE.md") == freeze["protocol_sha256"], "RL_FORMAL_PROTOCOL_CHANGED")
        for name, expected in freeze["source_hashes"].items():
            require(file_sha256(PACKAGE / name) == expected, f"RL_FORMAL_SOURCE_CHANGED:{name}")
    require(not artifact_root.exists(), f"RL_ARTIFACT_ROOT_ALREADY_EXISTS:{artifact_root}")
    environment = preflight(artifact_root, require_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = utc_now()
    rows = [
        run_seed(seed=int(seed), mode=mode, contract=contract, artifact_root=artifact_root, device=device)
        for seed in contract[mode]["seeds"]
    ]
    result = aggregate(mode, rows, contract)
    result.update({
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "environment": environment,
        "contract_sha256": file_sha256(PACKAGE / "EXPERIMENT_CONTRACT.json"),
        "protocol_sha256": file_sha256(PACKAGE / "PROTOCOL_FREEZE.md"),
        "seed_result_paths": [
            str(artifact_root / mode / f"seed-{row['seed']}" / "SEED_RESULT.json") for row in rows
        ],
    })
    write_json(artifact_root / mode / "AGGREGATE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = run(args.mode, args.artifact_root.resolve(), args.require_cuda)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
