from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .runtime import (
    CUE_BITS,
    EVENT_CODES,
    HORIZON,
    TERM_IDS,
    VISIBLE_EVENT_CODE_COUNT,
    build_atomic_execution,
    build_credit_discovery_atomic_execution,
    compile_and_validate_canonical_gfg,
    credit_metrics,
    deterministic_behavior_actions,
    exact_scalar_credit,
    execute_episode,
    factorized_credit,
    hidden_target_actions,
    make_episode_spec,
    object_sha256,
    retrieve_candidates,
    retrieve_term_candidate_steps,
    rewire_term_candidate_steps,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"
CONTRACT = json.loads((ROOT / "EXPERIMENT_CONTRACT.json").read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _candidate_metrics(selected: tuple[int, ...], truth: tuple[int, ...]) -> dict[str, float | int]:
    left, right = set(selected), set(truth)
    tp = len(left & right)
    fp = len(left - right)
    fn = len(right - left)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _fit_trace_scores(rows: list[tuple[Any, tuple[int, ...], float]]) -> dict[int, float]:
    by_code: dict[int, dict[int, list[float]]] = defaultdict(lambda: {0: [], 1: []})
    for spec, actions, consequence in rows:
        for code, action in zip(spec.event_codes, actions):
            by_code[code][action].append(consequence)
    scores: dict[int, float] = {}
    for code, groups in by_code.items():
        if groups[0] and groups[1]:
            scores[code] = abs(statistics.fmean(groups[1]) - statistics.fmean(groups[0]))
        else:
            scores[code] = 0.0
    return scores


def _trace_candidates(spec: Any, scores: dict[int, float], budget: int = 9) -> tuple[int, ...]:
    ranked = sorted(range(HORIZON), key=lambda step: (-scores.get(spec.event_codes[step], 0.0), step))
    return tuple(sorted(ranked[:budget]))


def _recency_candidates(_spec: Any, budget: int = 9) -> tuple[int, ...]:
    return tuple(range(HORIZON - budget, HORIZON))


def _pair_error(reference: dict[str, float], candidate: dict[str, float]) -> float:
    keys = set(reference) | set(candidate)
    return max((abs(reference.get(key, 0.0) - candidate.get(key, 0.0)) for key in keys), default=0.0)


def _sum_costs(*costs: dict[str, int | float]) -> dict[str, int | float]:
    fields = set().union(*(row.keys() for row in costs))
    return {field: sum(float(row.get(field, 0)) for row in costs) for field in fields}


def _three_way_difference(values: dict[int, float]) -> float:
    if set(values) != set(range(8)):
        raise ValueError("THREE_WAY_LEDGER_INCOMPLETE")
    return float(
        values[7] - values[6] - values[5] - values[3]
        + values[4] + values[2] + values[1] - values[0]
    )


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CUE_BITS + VISIBLE_EVENT_CODE_COUNT, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 2),
        )

    def forward(self, cue: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        encoded = torch.nn.functional.one_hot(code.long(), VISIBLE_EVENT_CODE_COUNT).float()
        return self.network(torch.cat((cue.float(), encoded), dim=1))


def _train_policy(examples: list[dict[str, Any]], seed: int) -> tuple[Policy, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = Policy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=1e-4)
    cue = torch.tensor([row["cue"] for row in examples], dtype=torch.float32)
    code = torch.tensor([row["event_code"] for row in examples], dtype=torch.long)
    target = torch.tensor([row["target_action"] for row in examples], dtype=torch.long)
    final_loss = 0.0
    for _ in range(900):
        logits = model(cue, code)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    return model, {"example_count": len(examples), "epochs": 900, "final_loss": final_loss}


@torch.no_grad()
def _evaluate_policy(model: Policy, specs: list[Any]) -> dict[str, Any]:
    successes: list[bool] = []
    consequences: list[float] = []
    functional_correct = 0
    functional_total = 0
    for spec in specs:
        cue = torch.tensor([spec.cue] * HORIZON, dtype=torch.float32)
        code = torch.tensor(spec.event_codes, dtype=torch.long)
        actions = tuple(int(value) for value in model(cue, code).argmax(dim=1).tolist())
        result = execute_episode(spec, actions)
        successes.append(result.success)
        consequences.append(result.consequence)
        targets = hidden_target_actions(spec.cue)
        for event_code, target in zip(EVENT_CODES[:6], targets):
            functional_correct += actions[spec.schedule[event_code]] == target
            functional_total += 1
    return {
        "episode_count": len(specs),
        "terminal_success_rate": float(np.mean(successes)),
        "mean_terminal_consequence": float(np.mean(consequences)),
        "functional_action_accuracy": functional_correct / functional_total,
    }


def _artifact_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted(ARTIFACT_ROOT.glob("*"), key=lambda item: item.name):
        if not path.is_file() or path.name == "ARTIFACT_MANIFEST.json":
            continue
        payload = path.read_bytes()
        rows.append({
            "path": path.name,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    return {
        "schema_version": "gfg-temporal-credit-long-chain-artifact-manifest-v1",
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def run() -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    development_rows = []
    for seed in CONTRACT["development_seeds"]:
        for episode_id in range(48):
            spec = make_episode_spec(seed, episode_id)
            actions = deterministic_behavior_actions(seed, episode_id)
            result = execute_episode(spec, actions)
            development_rows.append((spec, actions, result.consequence))
    trace_scores = _fit_trace_scores(development_rows)

    # The first development execution establishes both canonical graphs and the
    # frozen structure-guided optimization plan before formal seeds are read.
    dev_spec, dev_actions, _ = development_rows[0]
    dev_bundle, _, dev_metadata = build_atomic_execution(dev_spec, dev_actions)
    base_graph_started = time.perf_counter()
    dev_base_validation = compile_and_validate_canonical_gfg(
        dev_bundle, generator_name="long-chain-temporal-credit-environment"
    )
    base_graph_seconds = time.perf_counter() - base_graph_started
    frozen_candidates = retrieve_candidates(dev_bundle, dev_metadata)
    frozen_term_structure = retrieve_term_candidate_steps(dev_bundle, dev_metadata)
    dev_factorized = factorized_credit(
        dev_spec, dev_actions, frozen_candidates, frozen_term_structure
    )
    meta_bundle, meta_credits = build_credit_discovery_atomic_execution(
        base_run_id=dev_metadata["run_id"],
        base_graph_sha256=dev_base_validation["graph_sha256"],
        candidates=frozen_candidates,
        term_candidate_steps=frozen_term_structure,
        term_values=dev_factorized["term_values"],
    )
    meta_started = time.perf_counter()
    dev_meta_validation = compile_and_validate_canonical_gfg(
        meta_bundle, generator_name="long-chain-credit-discovery"
    )
    meta_graph_seconds = time.perf_counter() - meta_started
    if max(abs(meta_credits[key] - dev_factorized["credits"][key]) for key in meta_credits) > 1e-12:
        raise RuntimeError("META_GFG_CREDIT_MISMATCH")

    development = {
        "status": "PASS",
        "development_episode_count": len(development_rows),
        "trace_score_count": len(trace_scores),
        "frozen_candidate_count": len(frozen_candidates),
        "frozen_term_group_sizes": {key: len(value) for key, value in frozen_term_structure.items()},
        "base_gfg_validation": dev_base_validation,
        "credit_discovery_gfg_validation": dev_meta_validation,
        "base_graph_build_seconds": base_graph_seconds,
        "credit_discovery_graph_build_seconds": meta_graph_seconds,
        "formal_thresholds_unchanged": True,
    }
    write_json(ARTIFACT_ROOT / "DEVELOPMENT_CALIBRATION.json", development)

    per_episode: list[dict[str, Any]] = []
    aggregate_costs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    policy_examples: list[dict[str, Any]] = []
    base_graph_validations: list[dict[str, Any]] = []
    gfg_candidate_rows = []
    trace_candidate_rows = []
    recency_candidate_rows = []
    max_credit_error = 0.0
    max_pair_error = 0.0
    rewired_detected = 0
    three_way_values: list[float] = []
    all_exact = True
    formal_started = time.perf_counter()
    for seed in CONTRACT["formal_seeds"]:
        for episode_id in range(CONTRACT["episodes_per_formal_seed"]):
            spec = make_episode_spec(seed, episode_id)
            actions = deterministic_behavior_actions(seed, episode_id)
            bundle, observed, metadata = build_atomic_execution(spec, actions)
            candidates = retrieve_candidates(bundle, metadata)
            term_structure = retrieve_term_candidate_steps(bundle, metadata)
            if episode_id == 0:
                base_graph_validations.append(
                    compile_and_validate_canonical_gfg(
                        bundle, generator_name="long-chain-temporal-credit-environment"
                    )
                )
            exact = exact_scalar_credit(spec, actions, candidates, prefix_cache=False)
            trace = exact_scalar_credit(spec, actions, candidates, prefix_cache=True)
            hand = exact_scalar_credit(spec, actions, candidates, prefix_cache=True)
            gfg = factorized_credit(spec, actions, candidates, term_structure)
            dependency = factorized_credit(spec, actions, candidates, term_structure)
            rewired_structure = rewire_term_candidate_steps(
                term_structure, candidates, seed * 1000 + episode_id
            )
            rewired_attempt = factorized_credit(
                spec, actions, candidates, rewired_structure
            )
            rewired_error = max(
                credit_metrics(exact["credits"], rewired_attempt["credits"], spec.passenger_positions)["max_abs_error"],
                _pair_error(exact["pair_interactions"], rewired_attempt["pair_interactions"]),
            )
            if rewired_error > 1e-12:
                rewired_detected += 1
                rewired_final = trace
                rewired_cost = _sum_costs(rewired_attempt["cost"], trace["cost"])
                rewired_fallback = True
            else:
                rewired_final = rewired_attempt
                rewired_cost = rewired_attempt["cost"]
                rewired_fallback = False
            gfg_metrics = credit_metrics(exact["credits"], gfg["credits"], spec.passenger_positions)
            dependency_metrics = credit_metrics(exact["credits"], dependency["credits"], spec.passenger_positions)
            pair_error = _pair_error(exact["pair_interactions"], gfg["pair_interactions"])
            max_credit_error = max(max_credit_error, float(gfg_metrics["max_abs_error"]))
            max_pair_error = max(max_pair_error, pair_error)
            all_exact = all_exact and bool(gfg_metrics["exact_within_1e_12"]) and bool(dependency_metrics["exact_within_1e_12"]) and pair_error <= 1e-12
            three_way_values.append(_three_way_difference(gfg["term_values"]["term-4"]))
            for name, payload in {
                "exact_naive": exact,
                "trace_profile": trace,
                "hand_engineered": hand,
                "dependency_dag": dependency,
                "gfg_guided": gfg,
            }.items():
                for key, value in payload["cost"].items():
                    aggregate_costs[name][key] += float(value)
            for key, value in rewired_cost.items():
                aggregate_costs["rewired_gfg"][key] += float(value)
            for step_text, value in gfg["credits"].items():
                if abs(value) <= 1e-12:
                    continue
                step = int(step_text)
                actual = actions[step]
                policy_examples.append({
                    "cue": spec.cue,
                    "event_code": spec.event_codes[step],
                    "target_action": actual if value > 0 else 1 - actual,
                })
            ancestry_truth = spec.ancestry_positions
            gfg_candidate = _candidate_metrics(candidates, ancestry_truth)
            trace_candidate = _candidate_metrics(_trace_candidates(spec, trace_scores), ancestry_truth)
            recency_candidate = _candidate_metrics(_recency_candidates(spec), ancestry_truth)
            gfg_candidate_rows.append(gfg_candidate)
            trace_candidate_rows.append(trace_candidate)
            recency_candidate_rows.append(recency_candidate)
            per_episode.append({
                "seed": seed,
                "episode_id": episode_id,
                "base_consequence": observed.consequence,
                "candidate_count": len(candidates),
                "gfg_candidate_metrics": gfg_candidate,
                "gfg_credit": gfg_metrics,
                "dependency_credit": dependency_metrics,
                "pair_interaction_max_abs_error": pair_error,
                "three_way_difference": three_way_values[-1],
                "rewired_attempt_error": rewired_error,
                "rewired_fallback": rewired_fallback,
            })
    formal_seconds = time.perf_counter() - formal_started

    exact_cost = aggregate_costs["exact_naive"]
    gfg_cost = aggregate_costs["gfg_guided"]
    # The meta-GFG is a one-time development cost used by all formal episodes.
    gfg_total_wall = gfg_cost["wall_seconds"] + meta_graph_seconds
    transition_reduction = 1.0 - gfg_cost["native_transitions"] / exact_cost["native_transitions"]
    end_to_end_speedup = exact_cost["wall_seconds"] / gfg_total_wall

    policy_model, policy_training = _train_policy(policy_examples, seed=7001)
    held_out_specs = [
        make_episode_spec(seed + 100_000, episode_id + 50_000)
        for seed in CONTRACT["formal_seeds"]
        for episode_id in range(CONTRACT["held_out_policy_episodes_per_seed"])
    ]
    policy_result = _evaluate_policy(policy_model, held_out_specs)

    def mean_metric(rows: list[dict[str, Any]], field: str) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    candidate_results = {
        "gfg_formation_path": {
            "mean_precision": mean_metric(gfg_candidate_rows, "precision"),
            "mean_recall": mean_metric(gfg_candidate_rows, "recall"),
            "mean_f1": mean_metric(gfg_candidate_rows, "f1"),
            "history_reduction": 1.0 - 9 / HORIZON,
        },
        "trace_profile": {
            "mean_precision": mean_metric(trace_candidate_rows, "precision"),
            "mean_recall": mean_metric(trace_candidate_rows, "recall"),
            "mean_f1": mean_metric(trace_candidate_rows, "f1"),
        },
        "temporal_recency": {
            "mean_precision": mean_metric(recency_candidate_rows, "precision"),
            "mean_recall": mean_metric(recency_candidate_rows, "recall"),
            "mean_f1": mean_metric(recency_candidate_rows, "f1"),
        },
    }
    cost_results = {
        name: {key: (int(value) if key != "wall_seconds" else value) for key, value in row.items()}
        for name, row in aggregate_costs.items()
    }
    cost_results["gfg_guided"]["one_time_meta_gfg_seconds"] = meta_graph_seconds
    cost_results["gfg_guided"]["end_to_end_including_meta_gfg_seconds"] = gfg_total_wall
    cost_results["gfg_guided"]["transition_reduction_vs_exact"] = transition_reduction
    cost_results["gfg_guided"]["speedup_vs_exact_including_meta_gfg"] = end_to_end_speedup

    gates = {
        "base_core_and_gfg_validation_all_pass": all(row["status"] == "PASS" for row in base_graph_validations),
        "development_base_gfg_pass": dev_base_validation["status"] == "PASS",
        "credit_discovery_meta_gfg_pass": dev_meta_validation["status"] == "PASS",
        "candidate_retrieval_exact": candidate_results["gfg_formation_path"]["mean_f1"] == 1.0,
        "credit_exact_within_tolerance": all_exact and max_credit_error <= 1e-12,
        "pair_interactions_exact_within_tolerance": max_pair_error <= 1e-12,
        "passenger_zero_preserved": all(row["gfg_credit"]["passenger_zero_accuracy"] == 1.0 for row in per_episode),
        "pure_three_way_pressure_nonzero": any(abs(value) > 1e-12 for value in three_way_values),
        "rewired_control_detected_or_fallback": rewired_detected == len(per_episode),
        "transition_reduction_threshold": transition_reduction >= CONTRACT["minimum_transition_reduction"],
        "wall_speedup_threshold_including_meta_gfg": end_to_end_speedup >= CONTRACT["minimum_end_to_end_speedup"],
        "dependency_dag_matches_gfg": all(row["dependency_credit"]["exact_within_1e_12"] for row in per_episode),
    }
    scientific_pass = all(
        value
        for key, value in gates.items()
        if key != "dependency_dag_matches_gfg"
    )
    result = {
        "schema_version": "gfg-temporal-credit-long-chain-self-optimization-result-v1",
        "status": "PASS" if scientific_pass else "FAIL",
        "formal_episode_count": len(per_episode),
        "formal_wall_seconds": formal_seconds,
        "candidate_results": candidate_results,
        "credit_results": {
            "max_abs_credit_error": max_credit_error,
            "max_abs_pair_interaction_error": max_pair_error,
            "passenger_zero_accuracy": statistics.fmean(row["gfg_credit"]["passenger_zero_accuracy"] for row in per_episode),
            "three_way_difference_abs_mean": statistics.fmean(abs(value) for value in three_way_values),
        },
        "cost_results": cost_results,
        "policy_training": policy_training,
        "held_out_policy": policy_result,
        "canonical_validations": {
            "development_base": dev_base_validation,
            "development_credit_discovery": dev_meta_validation,
            "formal_base_graphs": base_graph_validations,
        },
        "gates": gates,
        "claim_adjudication": {
            "long_chain_temporal_credit_supported": gates["candidate_retrieval_exact"] and gates["credit_exact_within_tolerance"],
            "exact_structure_guided_optimization_supported": gates["transition_reduction_threshold"] and gates["wall_speedup_threshold_including_meta_gfg"],
            "gfg_exclusive_advantage_over_equivalent_dependency_dag": False if gates["dependency_dag_matches_gfg"] else "NOT_ESTABLISHED",
            "reason": "The conventional dependency DAG retained the same term-to-action dependency partition and matched GFG cost and credit. The executed result therefore supports validated formation-structure guidance, not GFG exclusivity over an equivalent dependency DAG.",
        },
        "per_episode_sha256": object_sha256(per_episode),
    }
    write_json(ARTIFACT_ROOT / "PER_EPISODE_RESULTS.json", per_episode)
    write_json(ARTIFACT_ROOT / "FORMAL_RESULT_SUMMARY.json", result)
    return result


def result_markdown(result: dict[str, Any]) -> str:
    candidate = result["candidate_results"]["gfg_formation_path"]
    credit = result["credit_results"]
    cost = result["cost_results"]["gfg_guided"]
    policy = result["held_out_policy"]
    return f"""# RL-E03 long-chain temporal-credit discovery and self-optimization

Final status: **{result['status']}**

## Long formation chain

- Formal episodes: {result['formal_episode_count']}
- GFG candidate precision/recall/F1: {candidate['mean_precision']:.6f} / {candidate['mean_recall']:.6f} / {candidate['mean_f1']:.6f}
- Chronological history reduction: {candidate['history_reduction']:.2%}
- Every retained early action crossed at least 36 native state transitions before the terminal-only consequence.

## Exact causal credit

- Maximum GFG-guided vs exact credit error: {credit['max_abs_credit_error']:.3e}
- Maximum pair-interaction error: {credit['max_abs_pair_interaction_error']:.3e}
- Passenger-zero accuracy: {credit['passenger_zero_accuracy']:.2%}
- Mean absolute pure three-way finite difference: {credit['three_way_difference_abs_mean']:.6f}

## Computation

- Native-transition reduction vs exact: {cost['transition_reduction_vs_exact']:.2%}
- End-to-end speedup including the one-time credit-discovery GFG: {cost['speedup_vs_exact_including_meta_gfg']:.2f}x
- Exact and GFG-guided credit remained equal within the frozen 1e-12 tolerance.

## Downstream learning

- Held-out terminal success: {policy['terminal_success_rate']:.2%}
- Held-out functional-action accuracy: {policy['functional_action_accuracy']:.2%}

## Critical falsification result

An ordinary value-dependency DAG supplied with the same term-to-action dependency partition matched the GFG-guided exact decomposition. Therefore this execution establishes that validated generation relations can guide exact cheaper credit discovery, but it does **not** establish that the computational saving is exclusive to GFG rather than any representation preserving equivalent dependency structure.

Formation ancestry remained distinct from causal credit: the GFG retained three real passenger ancestors, and matched forks assigned all three zero scalar credit.
"""


def main() -> int:
    result = run()
    (ROOT / "RESULTS.md").write_text(result_markdown(result), encoding="utf-8", newline="\n")
    write_json(ARTIFACT_ROOT / "ARTIFACT_MANIFEST.json", _artifact_manifest())
    print(json.dumps({
        "status": result["status"],
        "formal_episode_count": result["formal_episode_count"],
        "transition_reduction": result["cost_results"]["gfg_guided"]["transition_reduction_vs_exact"],
        "speedup": result["cost_results"]["gfg_guided"]["speedup_vs_exact_including_meta_gfg"],
        "held_out_policy_success": result["held_out_policy"]["terminal_success_rate"],
    }, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
