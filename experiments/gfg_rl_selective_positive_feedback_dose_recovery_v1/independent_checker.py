from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .runner import (
    DOSE_CONDITIONS,
    PACKAGE,
    RECOVERY_CONDITIONS,
    file_sha256,
    read_json,
    run_seed,
    verify_freeze,
    write_json,
)
from .runtime import combined_state_sha256, evaluate_policy, restore_state, support_profile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _skill_mean(evaluation: dict[str, Any], key: str, skills: list[int]) -> float:
    return mean(float(evaluation["per_skill"][skill][key]) for skill in skills)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _checkpoint_state(path: Path, contract: dict[str, Any], device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model, optimizer = restore_state(
        payload["state"],
        hidden_size=int(contract["model"]["hidden_size"]),
        learning_rate=float(contract["model"]["learning_rate"]),
        weight_decay=float(contract["model"]["weight_decay"]),
        device=device,
    )
    return model, optimizer, payload


def _verify_condition_files(
    result: dict[str, Any],
    *,
    expected_updates: int,
    expected_episodes_per_update: int,
    contract: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    receipt_path = Path(result["feedback_receipts"])
    trajectory_path = Path(result["boundary_trajectory"])
    require(file_sha256(receipt_path) == result["feedback_receipts_sha256"], f"RECEIPT_HASH_MISMATCH:{result['condition']}")
    require(file_sha256(trajectory_path) == result["boundary_trajectory_sha256"], f"TRAJECTORY_HASH_MISMATCH:{result['condition']}")
    receipts = _read_jsonl(receipt_path)
    trajectory = _read_jsonl(trajectory_path)
    require(len(receipts) == expected_updates, f"RECEIPT_COUNT_MISMATCH:{result['condition']}")
    require(len(trajectory) == expected_updates + 1, f"TRAJECTORY_COUNT_MISMATCH:{result['condition']}")
    require(sum(int(row["episode_count"]) for row in receipts) == expected_updates * expected_episodes_per_update, f"EPISODE_COUNT_MISMATCH:{result['condition']}")
    require(all(row["reward_authority"] == "exact_complete_chain_rule" for row in receipts), f"REWARD_AUTHORITY_MISMATCH:{result['condition']}")
    require(all(min(int(value) for value in row["skill_allocation"]) >= 0 for row in receipts), f"NEGATIVE_ALLOCATION:{result['condition']}")
    final_path = Path(result["final_checkpoint"]["path"])
    require(file_sha256(final_path) == result["final_checkpoint"]["sha256"], f"CHECKPOINT_HASH_MISMATCH:{result['condition']}")
    model, optimizer, _ = _checkpoint_state(final_path, contract, device)
    require(combined_state_sha256(model, optimizer) == result["final_state_sha256"], f"FINAL_STATE_MISMATCH:{result['condition']}")
    require(evaluate_policy(model, device) == result["final_evaluation"], f"FINAL_EVALUATION_MISMATCH:{result['condition']}")
    recomputed_support = support_profile(model, device)
    require(recomputed_support == result["final_support"], f"FINAL_SUPPORT_MISMATCH:{result['condition']}")
    tolerance = float(contract["support"]["shapley_efficiency_tolerance"])
    require(all(float(row["support"]["shapley_efficiency_max_error"]) <= tolerance for row in result["support_measurements"]), f"SHAPLEY_EFFICIENCY_FAILURE:{result['condition']}")
    return {
        "receipt_count": len(receipts),
        "trajectory_count": len(trajectory),
        "support_checkpoint_count": len(result["support_measurements"]),
        "receipts": receipts,
    }


def _verify_common_ledgers(seed_result: dict[str, Any]) -> None:
    dose_receipts = {
        name: _read_jsonl(Path(seed_result["dose_conditions"][name]["feedback_receipts"]))
        for name in DOSE_CONDITIONS
    }
    for update in range(len(dose_receipts["balanced"])):
        cue_hashes = {dose_receipts[name][update]["cue_batch_sha256"] for name in DOSE_CONDITIONS}
        uniform_hashes = {dose_receipts[name][update]["uniform_batch_sha256"] for name in DOSE_CONDITIONS}
        require(len(cue_hashes) == 1, f"DOSE_CUE_LEDGER_MISMATCH:{seed_result['seed']}:{update + 1}")
        require(len(uniform_hashes) == 1, f"DOSE_UNIFORM_LEDGER_MISMATCH:{seed_result['seed']}:{update + 1}")
        require(
            dose_receipts["exclusive"][update]["skill_batch_sha256"] == dose_receipts["frozen"][update]["skill_batch_sha256"],
            f"EXCLUSIVE_FROZEN_SKILL_LEDGER_MISMATCH:{seed_result['seed']}:{update + 1}",
        )
    recovery_receipts = {
        name: _read_jsonl(Path(seed_result["recovery_conditions"][name]["feedback_receipts"]))
        for name in RECOVERY_CONDITIONS
    }
    for update in range(len(recovery_receipts["rebalance_recovery"])):
        require(
            recovery_receipts["rebalance_recovery"][update]["cue_batch_sha256"]
            == recovery_receipts["repair_recovery"][update]["cue_batch_sha256"],
            f"RECOVERY_CUE_LEDGER_MISMATCH:{seed_result['seed']}:{update + 1}",
        )
        require(
            recovery_receipts["rebalance_recovery"][update]["uniform_batch_sha256"]
            == recovery_receipts["repair_recovery"][update]["uniform_batch_sha256"],
            f"RECOVERY_UNIFORM_LEDGER_MISMATCH:{seed_result['seed']}:{update + 1}",
        )


def _recompute_aggregate(seeds: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    ordered = ["balanced", "mild", "high", "exclusive"]
    rows = []
    for seed in seeds:
        shares = [float(seed["dose_conditions"][name]["final_support"]["task_support_shares"][0]) for name in ordered]
        margins = [_skill_mean(seed["dose_conditions"][name]["final_evaluation"], "mean_margin", [1, 2, 3]) for name in ordered]
        accuracies = [_skill_mean(seed["dose_conditions"][name]["final_evaluation"], "chain_accuracy", [1, 2, 3]) for name in ordered]
        rebalance = seed["recovery_conditions"]["rebalance_recovery"]
        rebalance_accuracy = _skill_mean(rebalance["final_evaluation"], "chain_accuracy", [1, 2, 3])
        balanced_shares = [float(value) for value in seed["dose_conditions"]["balanced"]["final_support"]["task_support_shares"]]
        exclusive_shares = [float(value) for value in seed["dose_conditions"]["exclusive"]["final_support"]["task_support_shares"]]
        rebalance_shares = [float(value) for value in rebalance["final_support"]["task_support_shares"]]
        distance = lambda left, right: math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
        rows.append({
            "support_direction": all(shares[index] < shares[index + 1] for index in range(3)),
            "margin_direction": all(margins[index] > margins[index + 1] for index in range(3)),
            "accuracy_deficit": accuracies[0] - accuracies[-1],
            "support_excess": shares[-1] - shares[0],
            "rebalance_gain": rebalance_accuracy - accuracies[-1],
            "rebalance_target0": float(rebalance["final_evaluation"]["per_skill"][0]["chain_accuracy"]),
            "support_reversal": distance(rebalance_shares, balanced_shares) < distance(exclusive_shares, balanced_shares),
        })
    gates = contract["decision_gates"]
    return {
        "seed_count": len(rows),
        "strict_support_dose_order_count": sum(row["support_direction"] for row in rows),
        "strict_unreinforced_margin_dose_order_count": sum(row["margin_direction"] for row in rows),
        "mean_exclusive_accuracy_deficit": mean(row["accuracy_deficit"] for row in rows),
        "mean_exclusive_support_excess": mean(row["support_excess"] for row in rows),
        "mean_rebalance_accuracy_gain": mean(row["rebalance_gain"] for row in rows),
        "minimum_rebalance_target0_accuracy": min(row["rebalance_target0"] for row in rows),
        "support_reversal_count": sum(row["support_reversal"] for row in rows),
        "gates_independently_pass": {
            "support_dose": sum(row["support_direction"] for row in rows) >= int(gates["minimum_seeds_with_positive_support_dose_association"]),
            "margin_dose": sum(row["margin_direction"] for row in rows) >= int(gates["minimum_seeds_with_negative_unreinforced_margin_dose_association"]),
            "accuracy_deficit": mean(row["accuracy_deficit"] for row in rows) >= float(gates["minimum_mean_exclusive_unreinforced_accuracy_deficit_vs_balanced"]),
            "support_excess": mean(row["support_excess"] for row in rows) >= float(gates["minimum_mean_exclusive_task0_support_share_excess_vs_balanced"]),
            "recovery_gain": mean(row["rebalance_gain"] for row in rows) >= float(gates["minimum_mean_rebalance_unreinforced_accuracy_gain_vs_exclusive"]),
            "target0_retained": min(row["rebalance_target0"] for row in rows) >= float(gates["minimum_rebalance_target0_final_accuracy"]),
            "support_reversal": sum(row["support_reversal"] for row in rows) >= int(gates["minimum_seeds_with_recovery_support_reversal"]),
        },
    }


def run(formal_root: Path, audit_root: Path, device_name: str) -> dict[str, Any]:
    require(not audit_root.exists(), f"RL_E06_AUDIT_ROOT_EXISTS:{audit_root}")
    verify_freeze()
    contract = read_json(PACKAGE / "MODEL_CONTRACT.json")
    formal_result = read_json(formal_root / "FORMAL_RESULT.json")
    require(formal_result["scientific_status"] == "SUPPORTED", "FORMAL_STATUS_NOT_SUPPORTED")
    device = torch.device(device_name)
    audit_root.mkdir(parents=True)
    stored_seeds = []
    replayed_seeds = []
    file_counts = {}
    for seed in [int(value) for value in contract["formal_seeds"]]:
        stored = read_json(formal_root / f"seed-{seed}" / "SEED_RESULT.json")
        stored_seeds.append(stored)
        condition_counts = {}
        for name in DOSE_CONDITIONS:
            condition_counts[name] = _verify_condition_files(
                stored["dose_conditions"][name],
                expected_updates=int(contract["feedback"]["updates"]),
                expected_episodes_per_update=int(contract["feedback"]["batch_size"]),
                contract=contract,
                device=device,
            )
        for name in RECOVERY_CONDITIONS:
            condition_counts[name] = _verify_condition_files(
                stored["recovery_conditions"][name],
                expected_updates=int(contract["recovery"]["updates"]),
                expected_episodes_per_update=int(contract["feedback"]["batch_size"]),
                contract=contract,
                device=device,
            )
        _verify_common_ledgers(stored)
        gfg = read_json(formal_root / f"seed-{seed}" / "EXPERIMENT_GFG.json")
        require(gfg["validation"]["all_facts_realized_once"], f"GFG_REALIZATION_FAILURE:{seed}")
        require(gfg["validation"]["all_five_coordinates_present"], f"GFG_COORDINATE_FAILURE:{seed}")
        file_counts[str(seed)] = {
            "conditions": {name: {key: value for key, value in row.items() if key != "receipts"} for name, row in condition_counts.items()},
            "gfg_fact_count": int(gfg["validation"]["fact_count"]),
        }
        replayed = run_seed(seed, contract, audit_root, device)
        replayed_seeds.append(replayed)
        require(replayed["diagnostics"] == stored["diagnostics"], f"NATIVE_REPLAY_DIAGNOSTIC_MISMATCH:{seed}")
        for family in ("dose_conditions", "recovery_conditions"):
            for name in stored[family]:
                left = stored[family][name]
                right = replayed[family][name]
                for key in (
                    "initial_state_sha256",
                    "final_state_sha256",
                    "feedback_receipts_sha256",
                    "boundary_trajectory_sha256",
                    "support_measurements",
                    "final_evaluation",
                    "final_support",
                    "total_positive_consequences",
                    "total_episodes",
                ):
                    require(left[key] == right[key], f"NATIVE_REPLAY_MISMATCH:{seed}:{name}:{key}")
        print(json.dumps({"checked_seed": seed, "completed": len(replayed_seeds), "total": len(contract["formal_seeds"])}), flush=True)
    independent = _recompute_aggregate(stored_seeds, contract)
    replay_independent = _recompute_aggregate(replayed_seeds, contract)
    require(independent == replay_independent, "REPLAYED_AGGREGATE_MISMATCH")
    require(all(independent["gates_independently_pass"].values()), "INDEPENDENT_GATE_FAILURE")
    require(_close(independent["mean_exclusive_accuracy_deficit"], formal_result["means"]["exclusive_unreinforced_accuracy_deficit_vs_balanced"]), "FORMAL_ACCURACY_MEAN_MISMATCH")
    require(_close(independent["mean_exclusive_support_excess"], formal_result["means"]["exclusive_task0_support_share_excess_vs_balanced"]), "FORMAL_SUPPORT_MEAN_MISMATCH")
    require(_close(independent["mean_rebalance_accuracy_gain"], formal_result["means"]["rebalance_unreinforced_accuracy_gain_vs_exclusive"]), "FORMAL_RECOVERY_MEAN_MISMATCH")
    result = {
        "schema": "rl-e06-independent-check-v1",
        "status": "PASS",
        "formal_root": str(formal_root),
        "audit_root": str(audit_root),
        "formal_result_sha256": file_sha256(formal_root / "FORMAL_RESULT.json"),
        "seed_count": len(stored_seeds),
        "full_native_replays": len(replayed_seeds),
        "independent_recalculation": independent,
        "file_and_gfg_checks": file_counts,
    }
    write_json(audit_root / "INDEPENDENT_CHECK_SUMMARY.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.formal_root.resolve(), args.audit_root.resolve(), args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

