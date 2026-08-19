from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

from experiments.gfg_temporal_credit_long_chain_self_optimization_v1.runner import (
    Policy,
    _train_policy,
)

from .runtime import (
    CUE_BITS,
    EVENT_CODES,
    FUNCTIONAL_CODES,
    HORIZON,
    TERM_IDS,
    VISIBLE_EVENT_CODE_COUNT,
    build_atomic_execution,
    build_credit_discovery_atomic_execution,
    compile_and_validate_canonical_gfg,
    credit_metrics,
    deterministic_behavior_actions,
    exact_conditional_credit,
    execute_episode,
    factorized_conditional_credit,
    hidden_target_actions,
    make_episode_spec,
    make_stochastic_tape,
    object_sha256,
    permute_stochastic_bindings,
    retrieve_candidates,
    retrieve_term_candidate_steps,
    rewire_term_candidate_steps,
    sign,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"
CONTRACT_PATH = ROOT / "EXPERIMENT_CONTRACT.json"
SOURCE_FILES = ("runtime.py", "runner.py", "independent_checker.py")


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def executable_source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def _sum_costs(*costs: dict[str, int | float]) -> dict[str, float]:
    fields = set().union(*(row.keys() for row in costs))
    return {field: sum(float(row.get(field, 0)) for row in costs) for field in fields}


def _add_cost(target: dict[str, float], source: dict[str, int | float]) -> None:
    for key, value in source.items():
        target[key] += float(value)


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
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: {0: [], 1: []})
    for spec, actions, consequence in rows:
        for code, action in zip(spec.event_codes, actions):
            grouped[code][action].append(consequence)
    return {
        code: (
            abs(statistics.fmean(values[1]) - statistics.fmean(values[0]))
            if values[0] and values[1]
            else 0.0
        )
        for code, values in grouped.items()
    }


def _trace_candidates(spec: Any, scores: dict[int, float], budget: int = 9) -> tuple[int, ...]:
    ranked = sorted(
        range(HORIZON),
        key=lambda step: (-scores.get(spec.event_codes[step], 0.0), step),
    )
    return tuple(sorted(ranked[:budget]))


def _recency_candidates(budget: int = 9) -> tuple[int, ...]:
    return tuple(range(HORIZON - budget, HORIZON))


def _mean_map(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set().union(*(row.keys() for row in rows)), key=lambda value: tuple(map(int, value.split(":"))))
    return {key: float(statistics.fmean(row.get(key, 0.0) for row in rows)) for key in keys}


def _map_error(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    keys = set(reference) | set(candidate)
    errors = [abs(reference.get(key, 0.0) - candidate.get(key, 0.0)) for key in keys]
    return {
        "mae": statistics.fmean(errors) if errors else 0.0,
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)) if errors else 0.0,
        "max_abs_error": max(errors, default=0.0),
    }


def _pair_error(reference: dict[str, float], candidate: dict[str, float]) -> float:
    return _map_error(reference, candidate)["max_abs_error"]


def _three_way_difference(values: dict[int, float]) -> float:
    if set(values) != set(range(8)):
        raise ValueError("STOCHASTIC_THREE_WAY_LEDGER_INCOMPLETE")
    return float(
        values[7] - values[6] - values[5] - values[3]
        + values[4] + values[2] + values[1] - values[0]
    )


def _t_critical_95(n: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 8: 2.365, 16: 2.131, 32: 2.040, 64: 2.000}
    if n in table:
        return table[n]
    return 1.96 if n >= 120 else 2.0


def _sample_stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    half = _t_critical_95(len(values)) * math.sqrt(variance / len(values)) if len(values) > 1 else 0.0
    return {
        "mean": float(mean),
        "variance": float(variance),
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
    }


def _build_trace_scores(contract: dict[str, Any]) -> dict[int, float]:
    rows: list[tuple[Any, tuple[int, ...], float]] = []
    realization_count = int(contract["development_stochastic_realizations"])
    for seed in contract["development_seeds"]:
        for episode_id in range(contract["development_episodes_per_seed"]):
            spec = make_episode_spec(seed, episode_id)
            actions = deterministic_behavior_actions(seed, episode_id)
            consequences = [
                execute_episode(
                    spec,
                    actions,
                    make_stochastic_tape(seed=seed, episode_id=episode_id, realization_id=index),
                ).consequence
                for index in range(realization_count)
            ]
            rows.append((spec, actions, statistics.fmean(consequences)))
    return _fit_trace_scores(rows)


def _effect_deltas(
    exact_rows: list[dict[str, Any]], candidates: tuple[int, ...]
) -> tuple[list[float], list[float], dict[str, float]]:
    full = (1 << len(candidates)) - 1
    matched: list[float] = []
    independent: list[float] = []
    independent_credit: dict[str, list[float]] = defaultdict(list)
    for tape_index, row in enumerate(exact_rows):
        other = exact_rows[(tape_index + 1) % len(exact_rows)]
        factual = row["scalar_values"][full]
        for candidate_index, step in enumerate(candidates):
            alternative_mask = full ^ (1 << candidate_index)
            matched_delta = factual - row["scalar_values"][alternative_mask]
            independent_delta = factual - other["scalar_values"][alternative_mask]
            matched.append(float(matched_delta))
            independent.append(float(independent_delta))
            independent_credit[str(step)].append(float(independent_delta))
    return matched, independent, {
        key: statistics.fmean(values) for key, values in independent_credit.items()
    }


def _policy_examples(
    spec: Any,
    actions: tuple[int, ...],
    credits: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step_text, value in credits.items():
        if sign(value) == 0:
            continue
        step = int(step_text)
        action = actions[step]
        rows.append(
            {
                "cue": spec.cue,
                "event_code": spec.event_codes[step],
                "target_action": action if value > 0 else 1 - action,
            }
        )
    return rows


@torch.no_grad()
def _evaluate_policy(model: Policy, specs: list[Any], *, tape_offset: int) -> dict[str, Any]:
    successes: list[bool] = []
    consequences: list[float] = []
    oracle_consequences: list[float] = []
    functional_correct = 0
    functional_total = 0
    for index, spec in enumerate(specs):
        cue = torch.tensor([spec.cue] * HORIZON, dtype=torch.float32)
        code = torch.tensor(spec.event_codes, dtype=torch.long)
        actions = tuple(int(value) for value in model(cue, code).argmax(dim=1).tolist())
        tape = make_stochastic_tape(
            seed=spec.seed,
            episode_id=spec.episode_id,
            realization_id=tape_offset + index,
        )
        result = execute_episode(spec, actions, tape)
        successes.append(result.success)
        consequences.append(result.consequence)
        targets = hidden_target_actions(spec.cue)
        oracle_actions = list(actions)
        for event_code, target in zip(FUNCTIONAL_CODES, targets):
            step = spec.schedule[event_code]
            functional_correct += actions[step] == target
            functional_total += 1
            oracle_actions[step] = target
        oracle_consequences.append(execute_episode(spec, tuple(oracle_actions), tape).consequence)
    return {
        "episode_count": len(specs),
        "terminal_success_rate": float(np.mean(successes)),
        "mean_terminal_consequence": float(np.mean(consequences)),
        "oracle_mean_terminal_consequence": float(np.mean(oracle_consequences)),
        "mean_consequence_regret": float(np.mean(oracle_consequences) - np.mean(consequences)),
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
        "schema_version": "gfg-temporal-credit-stochastic-artifact-manifest-v1",
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def run_experiment(*, phase: str) -> dict[str, Any]:
    contract = load_contract()
    if phase == "formal":
        if contract["status"] != "FROZEN_BEFORE_FORMAL_EXECUTION":
            raise RuntimeError("FORMAL_CONTRACT_NOT_FROZEN")
        if contract["execution_authorization"] is not True:
            raise RuntimeError("FORMAL_EXECUTION_NOT_AUTHORIZED")
        if contract["executable_source_hashes"] != executable_source_hashes():
            raise RuntimeError("FORMAL_EXECUTABLE_SOURCE_HASH_MISMATCH")
        seeds = contract["formal_seeds"]
        episodes_per_seed = int(contract["formal_episodes_per_seed"])
        realization_count = int(contract["formal_stochastic_realizations"])
        reference_count = int(contract["formal_reference_realizations"])
    elif phase == "development":
        seeds = contract["development_seeds"]
        episodes_per_seed = int(contract["development_episodes_per_seed"])
        realization_count = int(contract["development_stochastic_realizations"])
        reference_count = int(contract["development_reference_realizations"])
    else:
        raise ValueError("EXPERIMENT_PHASE_INVALID")

    trace_scores = _build_trace_scores(contract)
    started = time.perf_counter()
    aggregate_costs: dict[str, dict[str, float]] = {
        name: defaultdict(float)
        for name in (
            "exact_matched_reference",
            "gfg_guided_matched",
            "dependency_dag_supplied_equivalent",
            "expected_reference_factorized",
            "stochastic_binding_permuted",
            "rewired_gfg",
        )
    }
    per_episode: list[dict[str, Any]] = []
    gfg_candidate_rows: list[dict[str, Any]] = []
    trace_candidate_rows: list[dict[str, Any]] = []
    recency_candidate_rows: list[dict[str, Any]] = []
    ancestry_credit_rows: list[dict[str, Any]] = []
    canonical_validations: list[dict[str, Any]] = []
    conditional_credit_max_error = 0.0
    conditional_pair_max_error = 0.0
    expected_same_sample_max_error = 0.0
    expected_reference_errors: list[float] = []
    expected_reference_squared_errors: list[float] = []
    expected_reference_signs: list[bool] = []
    expected_ci_coverage: list[bool] = []
    passenger_zero: list[bool] = []
    sign_stabilities: list[float] = []
    three_way_values: list[float] = []
    matched_deltas: list[float] = []
    independent_deltas: list[float] = []
    binding_changed_results: list[bool] = []
    binding_changed_credit: list[bool] = []
    binding_reproduction_exact: list[bool] = []
    structure_stable: list[bool] = []
    rewired_detected = 0
    native_capture_seconds = 0.0
    canonical_graph_seconds = 0.0
    policy_examples_gfg: list[dict[str, Any]] = []
    policy_examples_exact: list[dict[str, Any]] = []
    policy_examples_independent: list[dict[str, Any]] = []
    meta_material: tuple[dict[str, Any], dict[str, Any], list[dict[str, dict[int, float]]]] | None = None

    for seed in seeds:
        for episode_id in range(episodes_per_seed):
            spec = make_episode_spec(seed, episode_id)
            actions = deterministic_behavior_actions(seed, episode_id)
            tapes = [
                make_stochastic_tape(seed=seed, episode_id=episode_id, realization_id=index)
                for index in range(realization_count)
            ]
            capture_started = time.perf_counter()
            bundle, observed, metadata = build_atomic_execution(spec, actions, tapes[0])
            candidates = retrieve_candidates(bundle, metadata)
            term_structure = retrieve_term_candidate_steps(bundle, metadata)
            tape_structures = []
            stochastic_fact_counts = []
            for tape in tapes:
                tape_bundle, _, tape_metadata = build_atomic_execution(spec, actions, tape)
                tape_structures.append(
                    (
                        retrieve_candidates(tape_bundle, tape_metadata),
                        retrieve_term_candidate_steps(tape_bundle, tape_metadata),
                    )
                )
                stochastic_fact_counts.append(
                    sum(
                        row["coordinates"]["rho"]["role"] == "exogenous_stochastic_input"
                        for row in tape_bundle["facts"]
                    )
                )
            native_capture_seconds += time.perf_counter() - capture_started
            structure_stable.append(
                all(row == (candidates, term_structure) for row in tape_structures)
            )
            if episode_id == 0:
                graph_started = time.perf_counter()
                validation = compile_and_validate_canonical_gfg(
                    bundle,
                    generator_name=f"rl-e04-{phase}-seed-{seed}",
                )
                canonical_graph_seconds += time.perf_counter() - graph_started
                canonical_validations.append(validation)

            exact_rows: list[dict[str, Any]] = []
            gfg_rows: list[dict[str, Any]] = []
            dependency_rows: list[dict[str, Any]] = []
            for tape in tapes:
                exact = exact_conditional_credit(
                    spec, actions, tape, candidates, prefix_cache=False
                )
                gfg = factorized_conditional_credit(
                    spec, actions, tape, candidates, term_structure
                )
                dependency = factorized_conditional_credit(
                    spec, actions, tape, candidates, term_structure
                )
                exact_rows.append(exact)
                gfg_rows.append(gfg)
                dependency_rows.append(dependency)
                _add_cost(aggregate_costs["exact_matched_reference"], exact["cost"])
                _add_cost(aggregate_costs["gfg_guided_matched"], gfg["cost"])
                _add_cost(aggregate_costs["dependency_dag_supplied_equivalent"], dependency["cost"])
                metrics = credit_metrics(exact["credits"], gfg["credits"], spec.passenger_positions)
                conditional_credit_max_error = max(conditional_credit_max_error, metrics["max_abs_error"])
                conditional_pair_max_error = max(
                    conditional_pair_max_error,
                    _pair_error(exact["pair_interactions"], gfg["pair_interactions"]),
                )
                passenger_zero.extend(
                    sign(gfg["credits"][str(step)]) == 0 for step in spec.passenger_positions
                )
                three_way_values.append(_three_way_difference(gfg["term_values"]["term-4"]))

            expected_exact = _mean_map([row["credits"] for row in exact_rows])
            expected_gfg = _mean_map([row["credits"] for row in gfg_rows])
            same_sample = _map_error(expected_exact, expected_gfg)
            expected_same_sample_max_error = max(
                expected_same_sample_max_error, same_sample["max_abs_error"]
            )

            reference_rows: list[dict[str, Any]] = []
            for index in range(reference_count):
                tape = make_stochastic_tape(
                    seed=seed,
                    episode_id=episode_id,
                    realization_id=100_000 + index,
                )
                reference = factorized_conditional_credit(
                    spec, actions, tape, candidates, term_structure
                )
                reference_rows.append(reference)
                _add_cost(aggregate_costs["expected_reference_factorized"], reference["cost"])
            expected_reference = _mean_map([row["credits"] for row in reference_rows])
            for step in candidates:
                key = str(step)
                error = abs(expected_gfg[key] - expected_reference[key])
                expected_reference_errors.append(error)
                expected_reference_squared_errors.append(error * error)
                expected_reference_signs.append(sign(expected_gfg[key]) == sign(expected_reference[key]))
                stats = _sample_stats([row["credits"][key] for row in gfg_rows])
                expected_ci_coverage.append(
                    stats["ci95_low"] <= expected_reference[key] <= stats["ci95_high"]
                )
                reference_sign = sign(expected_reference[key])
                if reference_sign == 0:
                    sign_stabilities.append(
                        statistics.fmean(sign(row["credits"][key]) == 0 for row in gfg_rows)
                    )
                else:
                    sign_stabilities.append(
                        statistics.fmean(sign(row["credits"][key]) == reference_sign for row in gfg_rows)
                    )

            matched, independent, independent_credit = _effect_deltas(exact_rows, candidates)
            matched_deltas.extend(matched)
            independent_deltas.extend(independent)

            permuted_tape = permute_stochastic_bindings(
                tapes[0], salt=seed * 10_000 + episode_id
            )
            factual_replay = execute_episode(spec, actions, tapes[0])
            permuted_result = execute_episode(spec, actions, permuted_tape)
            permuted_credit = factorized_conditional_credit(
                spec, actions, permuted_tape, candidates, term_structure
            )
            _add_cost(aggregate_costs["stochastic_binding_permuted"], permuted_credit["cost"])
            binding_reproduction_exact.append(
                factual_replay.consequence == observed.consequence
                and factual_replay.final_state == observed.final_state
            )
            binding_changed_results.append(
                permuted_result.final_state != observed.final_state
                or abs(permuted_result.consequence - observed.consequence) > 1e-15
            )
            binding_changed_credit.append(
                _map_error(gfg_rows[0]["credits"], permuted_credit["credits"])["max_abs_error"] > 1e-12
            )

            rewired_structure = rewire_term_candidate_steps(
                term_structure, candidates, seed * 1000 + episode_id
            )
            rewired_attempt = factorized_conditional_credit(
                spec, actions, tapes[0], candidates, rewired_structure
            )
            rewired_error = max(
                _map_error(exact_rows[0]["credits"], rewired_attempt["credits"])["max_abs_error"],
                _pair_error(exact_rows[0]["pair_interactions"], rewired_attempt["pair_interactions"]),
            )
            if rewired_error > 1e-12:
                rewired_detected += 1
            _add_cost(aggregate_costs["rewired_gfg"], rewired_attempt["cost"])

            gfg_candidate = _candidate_metrics(candidates, spec.ancestry_positions)
            trace_candidate = _candidate_metrics(
                _trace_candidates(spec, trace_scores), spec.ancestry_positions
            )
            recency_candidate = _candidate_metrics(
                _recency_candidates(), spec.ancestry_positions
            )
            ancestry_credit = _candidate_metrics(candidates, spec.functional_positions)
            gfg_candidate_rows.append(gfg_candidate)
            trace_candidate_rows.append(trace_candidate)
            recency_candidate_rows.append(recency_candidate)
            ancestry_credit_rows.append(ancestry_credit)

            policy_examples_gfg.extend(_policy_examples(spec, actions, expected_gfg))
            policy_examples_exact.extend(_policy_examples(spec, actions, expected_exact))
            policy_examples_independent.extend(_policy_examples(spec, actions, independent_credit))

            if meta_material is None:
                meta_material = (
                    bundle,
                    metadata,
                    [row["term_values"] for row in gfg_rows[: min(4, len(gfg_rows))]],
                )

            per_episode.append(
                {
                    "seed": seed,
                    "episode_id": episode_id,
                    "candidate_count": len(candidates),
                    "candidate_steps": list(candidates),
                    "stochastic_realization_count": realization_count,
                    "reference_realization_count": reference_count,
                    "native_fact_count": metadata["native_fact_count"],
                    "exogenous_transition_fact_count_per_tape": stochastic_fact_counts[0],
                    "candidate_structure_stable_across_tapes": structure_stable[-1],
                    "expected_credit": expected_gfg,
                    "reference_expected_credit": expected_reference,
                    "expected_credit_error": _map_error(expected_reference, expected_gfg),
                    "rewired_attempt_error": rewired_error,
                    "binding_permutation_changed_realized_result": binding_changed_results[-1],
                    "binding_permutation_changed_conditional_credit": binding_changed_credit[-1],
                }
            )

    meta_graph_seconds = 0.0
    meta_validation: dict[str, Any] | None = None
    if meta_material is not None:
        bundle, metadata, tape_term_values = meta_material
        meta_bundle, _ = build_credit_discovery_atomic_execution(
            base_run_id=metadata["run_id"],
            base_graph_sha256=object_sha256(bundle),
            candidates=retrieve_candidates(bundle, metadata),
            tape_term_values=tape_term_values,
            term_candidate_steps=retrieve_term_candidate_steps(bundle, metadata),
        )
        meta_started = time.perf_counter()
        meta_validation = compile_and_validate_canonical_gfg(
            meta_bundle,
            generator_name=f"rl-e04-{phase}-stochastic-credit-discovery",
            enforce_participant_labels=False,
        )
        meta_graph_seconds = time.perf_counter() - meta_started

    exact_cost = aggregate_costs["exact_matched_reference"]
    gfg_cost = aggregate_costs["gfg_guided_matched"]
    replay_transition_reduction = 1.0 - gfg_cost["native_transitions"] / exact_cost["native_transitions"]
    replay_speedup = exact_cost["wall_seconds"] / gfg_cost["wall_seconds"]
    gfg_end_to_end_seconds = (
        gfg_cost["wall_seconds"]
        + native_capture_seconds
        + canonical_graph_seconds
        + meta_graph_seconds
    )
    end_to_end_speedup = exact_cost["wall_seconds"] / gfg_end_to_end_seconds

    policy_gfg, training_gfg = _train_policy(policy_examples_gfg, seed=7101)
    policy_exact, training_exact = _train_policy(policy_examples_exact, seed=7101)
    policy_independent, training_independent = _train_policy(policy_examples_independent, seed=7101)
    held_out_specs = [
        make_episode_spec(seed + 100_000, episode_id + 50_000)
        for seed in seeds
        for episode_id in range(int(load_contract()["held_out_policy_episodes_per_seed"]))
    ]
    policy_results = {
        "gfg_guided_matched": {
            "training": training_gfg,
            "evaluation": _evaluate_policy(policy_gfg, held_out_specs, tape_offset=700_000),
        },
        "exact_matched_reference": {
            "training": training_exact,
            "evaluation": _evaluate_policy(policy_exact, held_out_specs, tape_offset=700_000),
        },
        "independent_random_replay": {
            "training": training_independent,
            "evaluation": _evaluate_policy(policy_independent, held_out_specs, tape_offset=700_000),
        },
    }

    def mean_metric(rows: list[dict[str, Any]], field: str) -> float:
        return statistics.fmean(float(row[field]) for row in rows)

    candidate_results = {
        "gfg_formation_path": {
            "mean_precision": mean_metric(gfg_candidate_rows, "precision"),
            "mean_recall": mean_metric(gfg_candidate_rows, "recall"),
            "mean_f1": mean_metric(gfg_candidate_rows, "f1"),
            "history_reduction": 1.0 - 9 / HORIZON,
        },
        "trace_history": {
            "mean_precision": mean_metric(trace_candidate_rows, "precision"),
            "mean_recall": mean_metric(trace_candidate_rows, "recall"),
            "mean_f1": mean_metric(trace_candidate_rows, "f1"),
        },
        "temporal_recency": {
            "mean_precision": mean_metric(recency_candidate_rows, "precision"),
            "mean_recall": mean_metric(recency_candidate_rows, "recall"),
            "mean_f1": mean_metric(recency_candidate_rows, "f1"),
        },
        "formation_ancestry_without_causal_adjudication": {
            "mean_credit_precision": mean_metric(ancestry_credit_rows, "precision"),
            "mean_credit_recall": mean_metric(ancestry_credit_rows, "recall"),
            "mean_credit_f1": mean_metric(ancestry_credit_rows, "f1"),
        },
    }
    matched_variance = statistics.variance(matched_deltas)
    independent_variance = statistics.variance(independent_deltas)
    stochastic_results = {
        "conditional_credit_max_abs_error": conditional_credit_max_error,
        "conditional_pair_interaction_max_abs_error": conditional_pair_max_error,
        "expected_credit_same_sample_max_abs_error": expected_same_sample_max_error,
        "expected_credit_vs_disjoint_reference_mae": statistics.fmean(expected_reference_errors),
        "expected_credit_vs_disjoint_reference_rmse": math.sqrt(statistics.fmean(expected_reference_squared_errors)),
        "expected_credit_vs_disjoint_reference_max_abs_error": max(expected_reference_errors),
        "expected_credit_sign_accuracy": statistics.fmean(expected_reference_signs),
        "expected_credit_ci95_coverage": statistics.fmean(expected_ci_coverage),
        "mean_conditional_sign_stability": statistics.fmean(sign_stabilities),
        "passenger_zero_accuracy": statistics.fmean(passenger_zero),
        "mean_absolute_three_way_interaction": statistics.fmean(abs(value) for value in three_way_values),
        "matched_effect_variance": matched_variance,
        "independent_effect_variance": independent_variance,
        "matched_to_independent_variance_ratio": matched_variance / independent_variance,
        "correct_binding_reproduction_rate": statistics.fmean(binding_reproduction_exact),
        "permuted_binding_realized_result_change_rate": statistics.fmean(binding_changed_results),
        "permuted_binding_conditional_credit_change_rate": statistics.fmean(binding_changed_credit),
    }
    cost_results = {
        name: {
            key: (int(value) if key != "wall_seconds" else value)
            for key, value in row.items()
        }
        for name, row in aggregate_costs.items()
    }
    cost_results["gfg_guided_matched"].update(
        {
            "stochastic_realization_count": len(per_episode) * realization_count,
            "native_capture_seconds": native_capture_seconds,
            "canonical_base_gfg_seconds": canonical_graph_seconds,
            "credit_discovery_meta_gfg_seconds": meta_graph_seconds,
            "end_to_end_including_all_gfg_seconds": gfg_end_to_end_seconds,
            "native_transition_reduction_vs_exact": replay_transition_reduction,
            "replay_speedup_vs_exact": replay_speedup,
            "end_to_end_speedup_vs_exact": end_to_end_speedup,
        }
    )

    result: dict[str, Any] = {
        "schema_version": "gfg-temporal-credit-stochastic-long-chain-result-v1",
        "phase": phase,
        "status": "DEVELOPMENT_COMPLETE" if phase == "development" else "PENDING_GATES",
        "episode_count": len(per_episode),
        "stochastic_realizations_per_episode": realization_count,
        "disjoint_reference_realizations_per_episode": reference_count,
        "wall_seconds": time.perf_counter() - started,
        "candidate_results": candidate_results,
        "stochastic_credit_results": stochastic_results,
        "cost_results": cost_results,
        "policy_results": policy_results,
        "canonical_validations": {
            "base_graphs": canonical_validations,
            "credit_discovery_meta_graph": meta_validation,
        },
        "capture_results": {
            "all_candidate_structures_stable_across_tapes": all(structure_stable),
            "all_base_core_and_gfg_validations_pass": all(
                row["status"] == "PASS" and not row["forbidden_label_hits"]
                for row in canonical_validations
            ),
            "meta_core_and_gfg_validation_pass": bool(meta_validation and meta_validation["status"] == "PASS"),
        },
        "rewired_gfg": {
            "detected_count": rewired_detected,
            "attempt_count": len(per_episode),
        },
        "dependency_dag_control": {
            "structure_source": "supplied_equivalent_term_to_action_partition",
            "discovery_claimed": False,
            "matches_gfg_by_construction_and_execution": True,
        },
        "per_episode_sha256": object_sha256(per_episode),
    }

    if phase == "formal":
        thresholds = contract["formal_thresholds"]
        gates = {
            "base_and_meta_gfg_validation_pass": (
                result["capture_results"]["all_base_core_and_gfg_validations_pass"]
                and result["capture_results"]["meta_core_and_gfg_validation_pass"]
            ),
            "candidate_retrieval_exact": candidate_results["gfg_formation_path"]["mean_f1"] == 1.0,
            "candidate_structure_stable_across_tapes": all(structure_stable),
            "conditional_credit_exact": conditional_credit_max_error <= contract["credit_tolerance"],
            "conditional_interactions_exact": conditional_pair_max_error <= contract["credit_tolerance"],
            "expected_same_sample_exact": expected_same_sample_max_error <= contract["credit_tolerance"],
            "expected_reference_mae": stochastic_results["expected_credit_vs_disjoint_reference_mae"] <= thresholds["maximum_expected_credit_mae"],
            "expected_sign_accuracy": stochastic_results["expected_credit_sign_accuracy"] >= thresholds["minimum_expected_credit_sign_accuracy"],
            "expected_ci_coverage": stochastic_results["expected_credit_ci95_coverage"] >= thresholds["minimum_expected_credit_ci95_coverage"],
            "passenger_zero_exact": stochastic_results["passenger_zero_accuracy"] == 1.0,
            "three_way_interaction_present": stochastic_results["mean_absolute_three_way_interaction"] >= thresholds["minimum_mean_absolute_three_way_interaction"],
            "matched_variance_reduction": stochastic_results["matched_to_independent_variance_ratio"] <= thresholds["maximum_matched_variance_ratio"],
            "binding_reproduction_exact": stochastic_results["correct_binding_reproduction_rate"] == 1.0,
            "binding_permutation_changes_realized_execution": stochastic_results["permuted_binding_realized_result_change_rate"] >= thresholds["minimum_binding_result_change_rate"],
            "binding_permutation_changes_conditional_credit": stochastic_results["permuted_binding_conditional_credit_change_rate"] >= thresholds["minimum_binding_credit_change_rate"],
            "rewired_gfg_detected": rewired_detected == len(per_episode),
            "held_out_policy_success": policy_results["gfg_guided_matched"]["evaluation"]["terminal_success_rate"] >= thresholds["minimum_held_out_policy_success"],
            "held_out_functional_accuracy": policy_results["gfg_guided_matched"]["evaluation"]["functional_action_accuracy"] >= thresholds["minimum_held_out_functional_action_accuracy"],
        }
        computational_gates = {
            "native_transition_reduction": replay_transition_reduction >= thresholds["minimum_transition_reduction"],
            "replay_only_speedup": replay_speedup >= thresholds["minimum_replay_speedup"],
            "end_to_end_speedup_including_all_gfg_work": end_to_end_speedup >= thresholds["minimum_end_to_end_speedup"],
        }
        result["gates"] = gates
        result["computational_gates"] = computational_gates
        result["claim_adjudication"] = {
            "stochastic_temporal_credit_supported": all(gates.values()),
            "replay_computation_reduced": (
                computational_gates["native_transition_reduction"]
                and computational_gates["replay_only_speedup"]
            ),
            "end_to_end_computational_advantage_supported": computational_gates[
                "end_to_end_speedup_including_all_gfg_work"
            ],
            "dependency_dag_exclusive_advantage_supported": False,
        }
        result["status"] = "PASS" if all(gates.values()) else "FAIL"
        write_json(ARTIFACT_ROOT / "PER_EPISODE_RESULTS.json", per_episode)
        write_json(ARTIFACT_ROOT / "FORMAL_RESULT_SUMMARY.json", result)
    else:
        write_json(ARTIFACT_ROOT / "DEVELOPMENT_RESULTS.json", result)
    return result


def result_markdown(result: dict[str, Any]) -> str:
    candidate = result["candidate_results"]["gfg_formation_path"]
    stochastic = result["stochastic_credit_results"]
    cost = result["cost_results"]["gfg_guided_matched"]
    policy = result["policy_results"]["gfg_guided_matched"]["evaluation"]
    return f"""# RL-E04 stochastic long-chain temporal credit

Final status: **{result['status']}**

## Formation retrieval and conditional credit

- Formal episodes: {result['episode_count']}
- Stochastic realizations per episode: {result['stochastic_realizations_per_episode']}
- GFG candidate precision/recall/F1: {candidate['mean_precision']:.6f} / {candidate['mean_recall']:.6f} / {candidate['mean_f1']:.6f}
- Maximum conditional credit error: {stochastic['conditional_credit_max_abs_error']:.3e}
- Maximum conditional pair-interaction error: {stochastic['conditional_pair_interaction_max_abs_error']:.3e}
- Passenger-zero accuracy: {stochastic['passenger_zero_accuracy']:.2%}

## Expected stochastic credit

- Expected-credit MAE against disjoint reference tapes: {stochastic['expected_credit_vs_disjoint_reference_mae']:.6f}
- Expected-credit sign accuracy: {stochastic['expected_credit_sign_accuracy']:.2%}
- 95% interval coverage of disjoint reference means: {stochastic['expected_credit_ci95_coverage']:.2%}
- Conditional sign stability: {stochastic['mean_conditional_sign_stability']:.2%}

## Matched stochastic replay and occurrence binding

- Matched/independent effect-variance ratio: {stochastic['matched_to_independent_variance_ratio']:.4f}
- Correct-binding factual reproduction: {stochastic['correct_binding_reproduction_rate']:.2%}
- Permuted binding changed the realized result: {stochastic['permuted_binding_realized_result_change_rate']:.2%}
- Permuted binding changed conditional credit: {stochastic['permuted_binding_conditional_credit_change_rate']:.2%}

## Computation

- Native-transition reduction vs complete matched reference: {cost['native_transition_reduction_vs_exact']:.2%}
- Replay-only speedup: {cost['replay_speedup_vs_exact']:.2f}x
- End-to-end speedup including native capture, canonical base GFG validation and the representative credit-discovery GFG: {cost['end_to_end_speedup_vs_exact']:.2f}x

## Downstream learning

- Held-out terminal success: {policy['terminal_success_rate']:.2%}
- Held-out functional-action accuracy: {policy['functional_action_accuracy']:.2%}
- Mean consequence regret to hidden evaluator oracle: {policy['mean_consequence_regret']:.6f}

## Interpretation

The stochastic-binding control is a realized-execution test.  Because the
exogenous inputs are exchangeable, the protocol does not require permutation
to change the population expectation.  The supplied-equivalent dependency DAG
control receives the already established term/action partition and therefore
does not claim independent structure discovery.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--development", action="store_true")
    group.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    phase = "development" if args.development else "formal"
    result = run_experiment(phase=phase)
    if phase == "formal":
        (ROOT / "RESULTS.md").write_text(
            result_markdown(result), encoding="utf-8", newline="\n"
        )
        write_json(ARTIFACT_ROOT / "ARTIFACT_MANIFEST.json", _artifact_manifest())
    print(
        json.dumps(
            {
                "phase": phase,
                "status": result["status"],
                "episode_count": result["episode_count"],
                "conditional_credit_max_abs_error": result["stochastic_credit_results"]["conditional_credit_max_abs_error"],
                "expected_credit_reference_mae": result["stochastic_credit_results"]["expected_credit_vs_disjoint_reference_mae"],
                "matched_variance_ratio": result["stochastic_credit_results"]["matched_to_independent_variance_ratio"],
            },
            sort_keys=True,
        )
    )
    return 0 if phase == "development" or result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
