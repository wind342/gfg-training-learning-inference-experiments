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
import time
from typing import Any

import numpy as np
import torch

from .runtime import (
    HORIZON,
    METHODS,
    build_base_gfg,
    candidate_metrics,
    credit_training_examples,
    deterministic_behavior_actions,
    evaluate_credit_policy,
    exact_shapley_credits,
    execute_episode,
    fit_trace_decomposition,
    make_episode_spec,
    pair_interactions,
    retrieve_formation_candidates,
    rewire_graph,
    top_k_trace_candidates,
    train_credit_policy,
    validate_base_gfg,
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


def _episode_rows(seed: int, start: int, count: int, with_graph: bool) -> list[tuple]:
    rows = []
    for episode_id in range(start, start + count):
        spec = make_episode_spec(seed, episode_id)
        actions = deterministic_behavior_actions(seed, episode_id)
        result = execute_episode(spec, actions)
        if with_graph:
            graph = build_base_gfg(spec, actions, result)
            validate_base_gfg(graph)
            rows.append((spec, actions, result, graph))
        else:
            rows.append((spec, actions, result))
    return rows


def _selected_candidates(
    method: str,
    spec: Any,
    actions: tuple[int, ...],
    graph: dict[str, Any],
    trace_model: Any,
    budget: int,
    seed: int,
) -> tuple[int, ...]:
    if method == "gfg_forks" or method == "gfg_ancestry_only":
        return retrieve_formation_candidates(graph)
    if method == "trace_decomposition_forks":
        return top_k_trace_candidates(trace_model, spec, actions, budget)
    if method == "temporal_recency_forks":
        return tuple(range(HORIZON - budget, HORIZON))
    if method == "rewired_gfg_forks":
        return retrieve_formation_candidates(rewire_graph(graph, seed))[:budget]
    if method == "oracle_forks":
        return spec.functional_positions
    if method == "terminal_all_actions":
        return tuple(range(HORIZON))
    raise ValueError(f"unknown method {method}")


def _sign(value: float, tolerance: float = 1e-10) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _candidate_and_credit_evaluation(
    *,
    seed: int,
    rows: list[tuple],
    trace_model: Any,
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    accumulators = {
        method: {
            "candidate_tp": 0,
            "candidate_fp": 0,
            "candidate_fn": 0,
            "credit_tp": 0,
            "credit_fp": 0,
            "credit_fn": 0,
            "credit_tn": 0,
            "credit_sign_correct": 0,
            "credit_sign_total": 0,
            "interaction_tp": 0,
            "interaction_fp": 0,
            "interaction_fn": 0,
            "interaction_tn": 0,
            "replays": 0,
        }
        for method in METHODS
    }
    for row_index, (spec, actions, result, graph) in enumerate(rows):
        oracle_credit, oracle_replays = exact_shapley_credits(spec, actions, spec.functional_positions)
        oracle_interactions, oracle_pair_replays = pair_interactions(spec, actions, spec.functional_positions)
        truth_credit_steps = {step for step, value in oracle_credit.items() if _sign(value)}
        truth_interaction_pairs = {key for key, value in oracle_interactions.items() if _sign(value)}
        detail = {
            "episode_id": spec.episode_id,
            "terminal_consequence": result.consequence,
            "terminal_success": result.success,
            "functional_positions": spec.functional_positions,
            "ancestry_positions": spec.ancestry_positions,
            "graph_sha256": graph["graph_sha256"],
            "methods": {},
        }
        for method_index, method in enumerate(METHODS):
            selected = _selected_candidates(
                method,
                spec,
                actions,
                graph,
                trace_model,
                budget,
                seed * 10_000 + row_index * 101 + method_index,
            )
            c_metrics = candidate_metrics(selected, spec.functional_positions)
            acc = accumulators[method]
            acc["candidate_tp"] += c_metrics["tp"]
            acc["candidate_fp"] += c_metrics["fp"]
            acc["candidate_fn"] += c_metrics["fn"]
            method_detail: dict[str, Any] = {"selected": selected, "candidate": c_metrics}
            if method.endswith("_forks"):
                predicted_credit, replay_count = exact_shapley_credits(spec, actions, selected)
                predicted_interactions, pair_replays = pair_interactions(spec, actions, selected)
                acc["replays"] += replay_count + pair_replays
                predicted_credit_steps = {step for step, value in predicted_credit.items() if _sign(value)}
                for step in range(HORIZON):
                    actual_nonzero = step in truth_credit_steps
                    predicted_nonzero = step in predicted_credit_steps
                    if actual_nonzero and predicted_nonzero:
                        acc["credit_tp"] += 1
                    elif predicted_nonzero:
                        acc["credit_fp"] += 1
                    elif actual_nonzero:
                        acc["credit_fn"] += 1
                    else:
                        acc["credit_tn"] += 1
                    if actual_nonzero:
                        acc["credit_sign_total"] += 1
                        acc["credit_sign_correct"] += _sign(predicted_credit.get(step, 0.0)) == _sign(oracle_credit[step])
                predicted_interaction_pairs = {key for key, value in predicted_interactions.items() if _sign(value)}
                all_pairs = {f"{left}:{right}" for left in range(HORIZON) for right in range(left + 1, HORIZON)}
                acc["interaction_tp"] += len(truth_interaction_pairs & predicted_interaction_pairs)
                acc["interaction_fp"] += len(predicted_interaction_pairs - truth_interaction_pairs)
                acc["interaction_fn"] += len(truth_interaction_pairs - predicted_interaction_pairs)
                acc["interaction_tn"] += len(all_pairs - truth_interaction_pairs - predicted_interaction_pairs)
                method_detail.update({
                    "credit": {str(key): value for key, value in predicted_credit.items()},
                    "oracle_credit": {str(key): value for key, value in oracle_credit.items()},
                    "interactions": predicted_interactions,
                    "oracle_interactions": oracle_interactions,
                    "counterfactual_replays": replay_count + pair_replays,
                })
            detail["methods"][method] = method_detail
        # Oracle cost is reported but not double-counted into non-oracle methods.
        detail["hidden_oracle_replays"] = oracle_replays + oracle_pair_replays
        details.append(detail)
    aggregate: dict[str, Any] = {}
    for method, acc in accumulators.items():
        candidate = _binary_metrics(
            acc["candidate_tp"], acc["candidate_fp"], acc["candidate_fn"],
            len(rows) * HORIZON - acc["candidate_tp"] - acc["candidate_fp"] - acc["candidate_fn"],
        )
        if method.endswith("_forks"):
            credit = _binary_metrics(acc["credit_tp"], acc["credit_fp"], acc["credit_fn"], acc["credit_tn"])
            interaction = _binary_metrics(
                acc["interaction_tp"], acc["interaction_fp"], acc["interaction_fn"], acc["interaction_tn"],
            )
            sign_accuracy = acc["credit_sign_correct"] / acc["credit_sign_total"] if acc["credit_sign_total"] else 0.0
        else:
            credit = None
            interaction = None
            sign_accuracy = None
        aggregate[method] = {
            "candidate": candidate,
            "candidate_history_fraction": (acc["candidate_tp"] + acc["candidate_fp"]) / (len(rows) * HORIZON),
            "causal_credit": credit,
            "credit_sign_accuracy_on_oracle_nonzero": sign_accuracy,
            "pair_interaction": interaction,
            "counterfactual_replay_count": acc["replays"],
            "counterfactual_transition_count": acc["replays"] * HORIZON,
        }
    return aggregate, details


def run_seed(seed: int, mode: str, contract: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    config = contract[mode]
    started = time.perf_counter()
    run_dir = artifact_root / mode / f"seed-{seed}"
    require(not run_dir.exists(), f"TCD_RUN_DIR_EXISTS:{run_dir}")
    run_dir.mkdir(parents=True)
    trace_count = int(config["trace_fit_episodes"])
    train_count = int(config["credit_training_episodes"])
    candidate_count = int(config["candidate_test_episodes"])
    policy_count = int(config["policy_test_episodes"])
    trace_rows = _episode_rows(seed, 0, trace_count, with_graph=False)
    train_rows = _episode_rows(seed, 100_000, train_count, with_graph=True)
    candidate_rows = _episode_rows(seed, 200_000, candidate_count, with_graph=True)
    policy_specs = [make_episode_spec(seed, 300_000 + index) for index in range(policy_count)]
    trace_model = fit_trace_decomposition(trace_rows, float(contract["trace_decomposition_ridge"]))
    trace_model_payload = {
        "schema": "trace-return-decomposition-v1",
        "coefficients": trace_model.coefficients,
        "intercept": trace_model.intercept,
        "fit_episode_count": trace_count,
    }
    trace_model_path = run_dir / "TRACE_DECOMPOSITION_MODEL.json"
    write_json(trace_model_path, trace_model_payload)
    graph_path = run_dir / "BASE_GFG_SAMPLES.jsonl"
    write_jsonl(graph_path, [row[3] for row in candidate_rows])
    candidate_aggregate, candidate_details = _candidate_and_credit_evaluation(
        seed=seed,
        rows=candidate_rows,
        trace_model=trace_model,
        budget=int(contract["candidate_budget"]),
    )
    detail_path = run_dir / "CANDIDATE_AND_CREDIT_DETAILS.jsonl"
    write_jsonl(detail_path, candidate_details)
    method_results: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        examples, accounting = credit_training_examples(
            method,
            train_rows,
            trace_model,
            int(contract["candidate_budget"]),
            rewiring_seed=seed * 1000 + method_index * 10_000,
        )
        policy, training = train_credit_policy(
            examples,
            seed=seed + 500_000,
            epochs=int(config["policy_epochs"]),
            learning_rate=float(contract["policy"]["learning_rate"]),
            hidden_size=int(contract["policy"]["hidden_size"]),
        )
        checkpoint_path = run_dir / f"POLICY_{method}.pt"
        torch.save({
            "schema": "temporal-credit-policy-checkpoint-v1",
            "seed": seed,
            "method": method,
            "state_dict": policy.state_dict(),
        }, checkpoint_path)
        evaluation = evaluate_credit_policy(policy, policy_specs)
        method_results[method] = {
            "training": training,
            "accounting": {
                **accounting,
                "counterfactual_transition_count": accounting["counterfactual_replay_count"] * HORIZON,
            },
            "evaluation": evaluation,
            "candidate_and_credit": candidate_aggregate[method],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
        }
    result = {
        "schema": "gfg-temporal-credit-seed-result-v1",
        "mode": mode,
        "seed": seed,
        "dataset_ranges": {
            "trace_fit": [0, trace_count - 1],
            "credit_training": [100_000, 100_000 + train_count - 1],
            "candidate_test": [200_000, 200_000 + candidate_count - 1],
            "policy_test": [300_000, 300_000 + policy_count - 1],
        },
        "trace_decomposition_model": str(trace_model_path),
        "trace_decomposition_model_sha256": file_sha256(trace_model_path),
        "base_gfg_samples": str(graph_path),
        "base_gfg_samples_sha256": file_sha256(graph_path),
        "candidate_and_credit_details": str(detail_path),
        "candidate_and_credit_details_sha256": file_sha256(detail_path),
        "methods": method_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = run_dir / "SEED_RESULT.json"
    write_json(path, result)
    result["result_path"] = str(path)
    result["result_sha256"] = file_sha256(path)
    return result


def _mean_ci95(values: list[float]) -> dict[str, float]:
    value = float(mean(values))
    if len(values) < 2:
        return {"mean": value, "ci95_low": value, "ci95_high": value}
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return {"mean": value, "ci95_low": value - 1.96 * standard_error, "ci95_high": value + 1.96 * standard_error}


def aggregate(mode: str, rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        terminal = [row["methods"][method]["evaluation"]["mean_terminal_consequence"] for row in rows]
        success = [row["methods"][method]["evaluation"]["terminal_success_rate"] for row in rows]
        functional = [row["methods"][method]["evaluation"]["functional_action_accuracy"] for row in rows]
        candidate_recall = [row["methods"][method]["candidate_and_credit"]["candidate"]["recall"] for row in rows]
        replay_transitions = [row["methods"][method]["accounting"]["counterfactual_transition_count"] for row in rows]
        methods[method] = {
            "mean_terminal_consequence": _mean_ci95(terminal),
            "terminal_success_rate": _mean_ci95(success),
            "functional_action_accuracy": _mean_ci95(functional),
            "candidate_recall": _mean_ci95(candidate_recall),
            "counterfactual_transition_count": _mean_ci95(replay_transitions),
            "per_seed": {
                "mean_terminal_consequence": terminal,
                "terminal_success_rate": success,
                "functional_action_accuracy": functional,
                "candidate_recall": candidate_recall,
                "counterfactual_transition_count": replay_transitions,
            },
        }
    result: dict[str, Any] = {
        "schema": "gfg-temporal-credit-aggregate-v1",
        "mode": mode,
        "seed_count": len(rows),
        "methods": methods,
        "paired_differences": {},
    }
    for comparator in (
        "trace_decomposition_forks",
        "temporal_recency_forks",
        "rewired_gfg_forks",
        "gfg_ancestry_only",
        "terminal_all_actions",
    ):
        differences = [
            row["methods"]["gfg_forks"]["evaluation"]["terminal_success_rate"]
            - row["methods"][comparator]["evaluation"]["terminal_success_rate"]
            for row in rows
        ]
        result["paired_differences"][f"gfg_minus_{comparator}_success_rate"] = _mean_ci95(differences)
    if mode == "formal":
        gates = contract["success_gates"]
        scientific_gates = {
            "candidate_recall": all(
                row["methods"]["gfg_forks"]["candidate_and_credit"]["candidate"]["recall"]
                >= gates["gfg_candidate_recall_min"] for row in rows
            ),
            "history_reduction": all(
                row["methods"]["gfg_forks"]["candidate_and_credit"]["candidate_history_fraction"]
                <= gates["gfg_candidate_history_fraction_max"] for row in rows
            ),
            "causal_credit_f1": all(
                row["methods"]["gfg_forks"]["candidate_and_credit"]["causal_credit"]["f1"]
                >= gates["gfg_causal_credit_f1_min"] for row in rows
            ),
            "pair_interaction_f1": all(
                row["methods"]["gfg_forks"]["candidate_and_credit"]["pair_interaction"]["f1"]
                >= gates["gfg_pair_interaction_f1_min"] for row in rows
            ),
            "absolute_learning_result": result["methods"]["gfg_forks"]["terminal_success_rate"]["mean"]
            >= gates["gfg_mean_terminal_success_min"],
            "oracle_equivalence": abs(
                result["methods"]["gfg_forks"]["terminal_success_rate"]["mean"]
                - result["methods"]["oracle_forks"]["terminal_success_rate"]["mean"]
            ) <= gates["gfg_oracle_success_gap_max"],
            "learning_over_trace": result["paired_differences"]["gfg_minus_trace_decomposition_forks_success_rate"]["ci95_low"]
            >= gates["gfg_policy_advantage_over_trace_min"],
            "learning_over_terminal": result["paired_differences"]["gfg_minus_terminal_all_actions_success_rate"]["ci95_low"]
            >= gates["gfg_policy_advantage_over_terminal_min"],
            "rewiring_reduces_result": result["paired_differences"]["gfg_minus_rewired_gfg_forks_success_rate"]["ci95_low"]
            >= gates["gfg_policy_advantage_over_rewired_min"],
        }
        result["scientific_gates"] = scientific_gates
        result["scientific_status"] = "PASS" if all(scientific_gates.values()) else "FAIL"
    return result


def run(mode: str, artifact_root: Path) -> dict[str, Any]:
    contract_path = PACKAGE / "EXPERIMENT_CONTRACT.json"
    protocol_path = PACKAGE / "PROTOCOL_FREEZE.md"
    contract = read_json(contract_path)
    require(mode in {"development", "formal"}, "TCD_MODE_INVALID")
    if mode == "formal":
        freeze = read_json(PACKAGE / "CONTRACT_FREEZE.json")
        require(file_sha256(contract_path) == freeze["experiment_contract_sha256"], "TCD_CONTRACT_CHANGED")
        require(file_sha256(protocol_path) == freeze["protocol_sha256"], "TCD_PROTOCOL_CHANGED")
        for name, expected in freeze["source_hashes"].items():
            require(file_sha256(PACKAGE / name) == expected, f"TCD_SOURCE_CHANGED:{name}")
    require(not artifact_root.exists(), f"TCD_ARTIFACT_ROOT_EXISTS:{artifact_root}")
    artifact_root.mkdir(parents=True)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "free_gib": shutil.disk_usage(artifact_root).free / (1024 ** 3),
        "started_at_utc": utc_now(),
    }
    rows = [run_seed(int(seed), mode, contract, artifact_root) for seed in contract[mode]["seeds"]]
    result = aggregate(mode, rows, contract)
    result.update({
        "environment": environment,
        "completed_at_utc": utc_now(),
        "contract_sha256": file_sha256(contract_path),
        "protocol_sha256": file_sha256(protocol_path),
        "seed_result_paths": [row["result_path"] for row in rows],
    })
    write_json(artifact_root / mode / "AGGREGATE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.artifact_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
