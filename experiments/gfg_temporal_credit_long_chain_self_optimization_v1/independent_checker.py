from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

from .runtime import (
    EVENT_CODES,
    HORIZON,
    TERM_IDS,
    build_atomic_execution,
    deterministic_behavior_actions,
    hidden_target_actions,
    make_episode_spec,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"


def _load(name: str) -> Any:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def _mask(step: int, code: int) -> int:
    return ((step + 1) * (code + 3) + code * code + 7) & 1


def independent_terms(spec: Any, actions: tuple[int, ...]) -> tuple[float, ...]:
    final: list[int] = []
    for code in EVENT_CODES:
        value = actions[spec.schedule[code]]
        for step in range(spec.schedule[code], HORIZON):
            value ^= _mask(step, code)
        for step in range(spec.schedule[code], HORIZON):
            value ^= _mask(step, code)
        final.append(value)
    target = hidden_target_actions(spec.cue)
    matched = [final[index] == target[index] for index in range(6)]
    criteria = (
        matched[0],
        matched[1] or matched[2],
        matched[3] and matched[4],
        matched[5],
    )
    p7, p8, p9 = final[6:9]
    return (
        0.2 * float(criteria[0]),
        0.2 * float(criteria[1]),
        0.2 * float(criteria[2]),
        0.2 * float(criteria[3]),
        0.2 * float(matched[0] and matched[3] and matched[5]),
        1.0 * float(all(criteria)),
        0.0 * float((p7 - p7) + (p8 ^ p8) + (p9 & (1 - p9))),
    )


def independent_shapley(values: dict[int, float], n: int) -> tuple[float, ...]:
    answer = []
    for index in range(n):
        total = 0.0
        for mask in range(1 << n):
            if mask & (1 << index):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            total += weight * (values[mask | (1 << index)] - values[mask])
        answer.append(total)
    return tuple(answer)


def _coalition_actions(actions: tuple[int, ...], group: tuple[int, ...], mask: int) -> tuple[int, ...]:
    changed = list(actions)
    for index, step in enumerate(group):
        if not mask & (1 << index):
            changed[step] = 1 - changed[step]
    return tuple(changed)


def _independent_ancestor_steps(bundle: dict[str, Any], start: list[str]) -> tuple[int, ...]:
    producers: dict[str, list[str]] = {}
    for fact in bundle["facts"]:
        producers.setdefault(fact["result_id"], []).append(fact["fact_id"])
    reverse: dict[str, list[str]] = {}
    by_id = {fact["fact_id"]: fact for fact in bundle["facts"]}
    for fact in bundle["facts"]:
        u = fact["coordinates"]["u"]
        if u.get("kind") == "generated_origin":
            reverse[fact["fact_id"]] = list(producers[u["prior_support_id"]])
    pending = list(start)
    visited: set[str] = set()
    steps: set[int] = set()
    while pending:
        fact_id = pending.pop()
        if fact_id in visited:
            continue
        visited.add(fact_id)
        fact = by_id[fact_id]
        if fact["coordinates"]["rho"].get("role") == "current_action":
            steps.add(int(fact["coordinates"]["omega_bar"]["step"]))
        pending.extend(reverse.get(fact_id, []))
    return tuple(sorted(steps))


def run_checks() -> dict[str, Any]:
    result = _load("FORMAL_RESULT_SUMMARY.json")
    manifest = _load("ARTIFACT_MANIFEST.json")
    manifest_exact = True
    for row in manifest["artifacts"]:
        path = ARTIFACT_ROOT / row["path"]
        manifest_exact &= path.is_file()
        if path.is_file():
            manifest_exact &= len(path.read_bytes()) == row["byte_count"]
            manifest_exact &= hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]

    samples = [(9101, 0), (9102, 7), (9104, 13), (9108, 23)]
    max_credit_error = 0.0
    ancestry_exact = True
    terminal_has_no_direct_action_source = True
    pure_three_way_seen = False
    for seed, episode_id in samples:
        spec = make_episode_spec(seed, episode_id)
        actions = deterministic_behavior_actions(seed, episode_id)
        bundle, _observed, metadata = build_atomic_execution(spec, actions)
        candidates = spec.ancestry_positions
        terminal_ids = metadata["scalar_fact_ids"]
        independent_candidates = _independent_ancestor_steps(bundle, terminal_ids)
        ancestry_exact &= independent_candidates == candidates
        by_id = {fact["fact_id"]: fact for fact in bundle["facts"]}
        terminal_has_no_direct_action_source &= all(
            by_id[fact_id]["coordinates"]["u"].get("kind") == "generated_origin"
            for fact_id in terminal_ids
        )

        scalar_values: dict[int, float] = {}
        for mask in range(1 << len(candidates)):
            ledger = _coalition_actions(actions, candidates, mask)
            scalar_values[mask] = math.fsum(independent_terms(spec, ledger))
        exact = independent_shapley(scalar_values, len(candidates))

        accumulated = {step: 0.0 for step in candidates}
        for term_index, term_id in enumerate(TERM_IDS):
            term_fact_ids = metadata["term_fact_ids"][term_id]
            group = _independent_ancestor_steps(bundle, term_fact_ids)
            values = {
                mask: independent_terms(spec, _coalition_actions(actions, group, mask))[term_index]
                for mask in range(1 << len(group))
            }
            partial = independent_shapley(values, len(group))
            for index, step in enumerate(group):
                accumulated[step] += partial[index]
            if term_id == "term-4":
                pure_three_way_seen |= abs(
                    values[7] - values[6] - values[5] - values[3]
                    + values[4] + values[2] + values[1] - values[0]
                ) > 1e-12
        for index, step in enumerate(candidates):
            max_credit_error = max(max_credit_error, abs(exact[index] - accumulated[step]))

    expected_exact_transitions = (
        result["formal_episode_count"] * (1 << 9) * HORIZON
    )
    gates = {
        "artifact_manifest_exact": manifest_exact,
        "independent_formation_ancestry_exact": ancestry_exact,
        "terminal_scalar_has_no_direct_early_action_source": terminal_has_no_direct_action_source,
        "independent_factorized_credit_exact": max_credit_error <= 1e-12,
        "pure_three_way_control_present": pure_three_way_seen,
        "exact_transition_accounting_exact": result["cost_results"]["exact_naive"]["native_transitions"] == expected_exact_transitions,
        "reported_status_pass": result["status"] == "PASS",
        "dependency_dag_non_exclusivity_disclosed": result["claim_adjudication"]["gfg_exclusive_advantage_over_equivalent_dependency_dag"] is False,
    }
    return {
        "schema_version": "gfg-temporal-credit-long-chain-independent-check-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "sample_count": len(samples),
        "independent_max_abs_credit_error": max_credit_error,
        "expected_exact_transition_count": expected_exact_transitions,
    }


def main() -> int:
    report = run_checks()
    path = ARTIFACT_ROOT / "INDEPENDENT_CHECK.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
