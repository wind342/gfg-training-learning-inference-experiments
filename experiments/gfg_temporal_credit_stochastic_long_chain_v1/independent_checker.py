from __future__ import annotations

from collections import defaultdict, deque
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
    STOCHASTIC_CHANNEL_COUNT,
    STOCHASTIC_MODULUS,
    TERM_CODE_GROUPS,
    TERM_IDS,
    build_atomic_execution,
    deterministic_behavior_actions,
    hidden_target_actions,
    make_episode_spec,
    make_stochastic_tape,
    object_sha256,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"


def _load(name: str) -> Any:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def _mask(step: int, code: int) -> int:
    return ((step + 1) * (code + 3) + code * code + 7) & 1


def independent_terms(spec: Any, actions: tuple[int, ...], tape: Any) -> tuple[float, ...]:
    action_values = [0] * len(EVENT_CODES)
    active = [False] * len(EVENT_CODES)
    stochastic = [0] * STOCHASTIC_CHANNEL_COUNT
    for step, action in enumerate(actions):
        for code in EVENT_CODES:
            index = code - 1
            if active[index]:
                action_values[index] ^= _mask(step, code)
        event_code = spec.event_codes[step]
        if event_code in EVENT_CODES:
            index = event_code - 1
            active[index] = True
            action_values[index] = action ^ _mask(step, event_code)
        for channel, prior in enumerate(stochastic):
            stochastic[channel] = (
                prior * (17 + 2 * channel)
                + (tape[step].value + 1) * (channel + 3)
                + (step + 1) * (channel * channel + 5)
            ) % STOCHASTIC_MODULUS
    schedule = dict(spec.schedule)
    decoded = []
    for code in EVENT_CODES:
        value = action_values[code - 1]
        for step in range(schedule[code], HORIZON):
            value ^= _mask(step, code)
        decoded.append(value)
    expected = hidden_target_actions(spec.cue)
    matched = [decoded[index] == expected[index] for index in range(6)]
    criteria = (
        matched[0],
        matched[1] or matched[2],
        matched[3] and matched[4],
        matched[5],
    )
    scales = [
        0.70 + 0.60 * value / (STOCHASTIC_MODULUS - 1)
        for value in stochastic
    ]
    p7, p8, p9 = decoded[6:9]
    return (
        0.20 * scales[0] * float(criteria[0]),
        0.20 * scales[1] * float(criteria[1]),
        0.20 * scales[2] * float(criteria[2]),
        0.20 * scales[3] * float(criteria[3]),
        0.20 * scales[4] * float(matched[0] and matched[3] and matched[5]),
        1.00 * scales[5] * float(all(criteria)),
        0.00 * scales[6] * float((p7 - p7) + (p8 ^ p8) + (p9 & (1 - p9))),
        0.40 * (scales[7] - 1.0),
    )


def independent_shapley(values: dict[int, float], n: int) -> tuple[float, ...]:
    result = []
    for index in range(n):
        total = 0.0
        for mask in range(1 << n):
            if mask & (1 << index):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            total += weight * (values[mask | (1 << index)] - values[mask])
        result.append(total)
    return tuple(result)


def _coalition_actions(actions: tuple[int, ...], group: tuple[int, ...], mask: int) -> tuple[int, ...]:
    changed = list(actions)
    for index, step in enumerate(group):
        if not mask & (1 << index):
            changed[step] = 1 - changed[step]
    return tuple(changed)


def _independent_ancestor_steps(bundle: dict[str, Any], start: list[str]) -> tuple[int, ...]:
    producers: dict[str, list[str]] = defaultdict(list)
    for fact in bundle["facts"]:
        producers[fact["result_id"]].append(fact["fact_id"])
    reverse: dict[str, list[str]] = defaultdict(list)
    by_id = {fact["fact_id"]: fact for fact in bundle["facts"]}
    for fact in bundle["facts"]:
        source = fact["coordinates"]["u"]
        if source.get("kind") == "generated_origin":
            reverse[fact["fact_id"]].extend(producers[source["prior_support_id"]])
    queue = deque(start)
    visited: set[str] = set()
    steps: set[int] = set()
    while queue:
        fact_id = queue.popleft()
        if fact_id in visited:
            continue
        visited.add(fact_id)
        fact = by_id[fact_id]
        if fact["coordinates"]["rho"]["role"] == "current_action":
            steps.add(int(fact["coordinates"]["omega_bar"]["step"]))
        queue.extend(reverse.get(fact_id, []))
    return tuple(sorted(steps))


def run_checks() -> dict[str, Any]:
    result = _load("FORMAL_RESULT_SUMMARY.json")
    per_episode = _load("PER_EPISODE_RESULTS.json")
    manifest = _load("ARTIFACT_MANIFEST.json")
    contract = json.loads((ROOT / "EXPERIMENT_CONTRACT.json").read_text(encoding="utf-8"))
    manifest_exact = True
    for row in manifest["artifacts"]:
        path = ARTIFACT_ROOT / row["path"]
        manifest_exact &= path.is_file()
        if path.is_file():
            payload = path.read_bytes()
            manifest_exact &= len(payload) == row["byte_count"]
            manifest_exact &= hashlib.sha256(payload).hexdigest() == row["sha256"]

    source_hashes_exact = all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        for name, expected in contract["executable_source_hashes"].items()
    )
    samples = [(9201, 0, 0), (9204, 5, 7), (9208, 11, 15)]
    max_factorized_error = 0.0
    ancestry_exact = True
    terminal_has_no_direct_action_or_random_source = True
    stochastic_binding_complete = True
    for seed, episode_id, realization_id in samples:
        spec = make_episode_spec(seed, episode_id)
        actions = deterministic_behavior_actions(seed, episode_id)
        tape = make_stochastic_tape(
            seed=seed, episode_id=episode_id, realization_id=realization_id
        )
        bundle, _, metadata = build_atomic_execution(spec, actions, tape)
        candidates = _independent_ancestor_steps(bundle, metadata["scalar_fact_ids"])
        ancestry_exact &= candidates == spec.ancestry_positions
        by_id = {fact["fact_id"]: fact for fact in bundle["facts"]}
        terminal_has_no_direct_action_or_random_source &= all(
            by_id[fact_id]["coordinates"]["u"].get("kind") == "generated_origin"
            for fact_id in metadata["scalar_fact_ids"]
        )
        stochastic_binding_complete &= sum(
            fact["coordinates"]["rho"]["role"] == "exogenous_stochastic_input"
            for fact in bundle["facts"]
        ) == HORIZON * STOCHASTIC_CHANNEL_COUNT

        scalar_values: dict[int, float] = {}
        for mask in range(1 << len(candidates)):
            ledger = _coalition_actions(actions, candidates, mask)
            scalar_values[mask] = math.fsum(independent_terms(spec, ledger, tape))
        scalar_credit = independent_shapley(scalar_values, len(candidates))
        accumulated = {step: 0.0 for step in candidates}
        for term_index, term_id in enumerate(TERM_IDS):
            group_codes = TERM_CODE_GROUPS[term_id]
            group = tuple(sorted(spec.schedule[code] for code in group_codes))
            values = {
                mask: independent_terms(
                    spec, _coalition_actions(actions, group, mask), tape
                )[term_index]
                for mask in range(1 << len(group))
            }
            partial = independent_shapley(values, len(group))
            for index, step in enumerate(group):
                accumulated[step] += partial[index]
        for index, step in enumerate(candidates):
            max_factorized_error = max(
                max_factorized_error, abs(scalar_credit[index] - accumulated[step])
            )

    expected_exact_transitions = (
        result["episode_count"]
        * result["stochastic_realizations_per_episode"]
        * (1 << 9)
        * HORIZON
    )
    gates = {
        "artifact_manifest_exact": manifest_exact,
        "executable_source_hashes_exact": source_hashes_exact,
        "per_episode_hash_exact": object_sha256(per_episode) == result["per_episode_sha256"],
        "independent_formation_ancestry_exact": ancestry_exact,
        "terminal_has_no_direct_early_action_or_random_source": terminal_has_no_direct_action_or_random_source,
        "every_transition_has_stochastic_source_facts": stochastic_binding_complete,
        "independent_conditional_factorization_exact": max_factorized_error <= 1e-12,
        "exact_transition_accounting_exact": result["cost_results"]["exact_matched_reference"]["native_transitions"] == expected_exact_transitions,
        "reported_status_pass": result["status"] == "PASS",
        "dependency_dag_supplied_status_disclosed": (
            result["dependency_dag_control"]["structure_source"]
            == "supplied_equivalent_term_to_action_partition"
            and result["dependency_dag_control"]["discovery_claimed"] is False
        ),
    }
    return {
        "schema_version": "gfg-temporal-credit-stochastic-independent-check-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "sample_count": len(samples),
        "independent_max_abs_factorized_credit_error": max_factorized_error,
        "expected_exact_transition_count": expected_exact_transitions,
    }


def main() -> int:
    report = run_checks()
    path = ARTIFACT_ROOT / "INDEPENDENT_CHECK.json"
    path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
