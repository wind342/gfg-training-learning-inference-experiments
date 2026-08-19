from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import random
import time
from typing import Any, Iterable


HORIZON = 64
CUE_BITS = 6
ACTION_COUNT = 2
EVENT_CODES = tuple(range(1, 10))
FUNCTIONAL_CODES = tuple(range(1, 7))
PASSENGER_CODES = tuple(range(7, 10))
DISTRACTOR_CODE = 0
VISIBLE_EVENT_CODE_COUNT = 64
TERM_IDS = tuple(f"term-{index}" for index in range(7))
DOMAIN_SCOPE_ID = "gfg-temporal-credit-long-chain-v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _bit(cue: tuple[int, ...], *indices: int) -> int:
    value = 0
    for index in indices:
        value ^= int(cue[index])
    return value


def hidden_target_actions(cue: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluator authority. These labels are never written to a GFG payload."""
    return (
        _bit(cue, 0, 3),
        _bit(cue, 1, 4),
        _bit(cue, 2, 5),
        _bit(cue, 0, 2, 4),
        _bit(cue, 1, 3, 5),
        _bit(cue, 0, 1, 5),
    )


def deterministic_schedule(seed: int, episode_id: int) -> dict[int, int]:
    rng = random.Random(seed * 1_000_003 + episode_id * 97 + 31)
    windows = {
        1: (1, 3),
        7: (4, 6),
        2: (7, 9),
        3: (10, 12),
        8: (13, 15),
        4: (16, 18),
        5: (19, 21),
        9: (22, 24),
        6: (25, 27),
    }
    return {
        code: rng.randrange(low, high + 1)
        for code, (low, high) in windows.items()
    }


def event_codes_from_schedule(
    schedule: dict[int, int], *, seed: int, episode_id: int
) -> tuple[int, ...]:
    # Every chronological action receives an opaque nonzero event identity.
    # Thus a flat trace cannot retrieve candidates merely by testing code != 0.
    result = [10 + ((step * 17 + seed * 3 + episode_id * 5) % 54) for step in range(HORIZON)]
    for code, step in schedule.items():
        result[step] = code
    return tuple(result)


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    episode_id: int
    cue: tuple[int, ...]
    schedule: dict[int, int]
    event_codes: tuple[int, ...]

    @property
    def functional_positions(self) -> tuple[int, ...]:
        return tuple(self.schedule[code] for code in FUNCTIONAL_CODES)

    @property
    def passenger_positions(self) -> tuple[int, ...]:
        return tuple(self.schedule[code] for code in PASSENGER_CODES)

    @property
    def ancestry_positions(self) -> tuple[int, ...]:
        return tuple(sorted((*self.functional_positions, *self.passenger_positions)))


@dataclass(frozen=True)
class ChainState:
    step: int
    cue: tuple[int, ...]
    schedule: tuple[tuple[int, int], ...]
    values: tuple[int, ...]
    active: tuple[bool, ...]


@dataclass(frozen=True)
class EpisodeResult:
    consequence: float
    success: bool
    terms: tuple[float, ...]
    criteria: tuple[bool, ...]
    final_state: ChainState
    decoded_actions: tuple[int, ...]


@dataclass
class ReplayCost:
    native_replays: int = 0
    native_transitions: int = 0
    checkpoint_restores: int = 0
    unique_cached_states: int = 0
    cache_hits: int = 0
    scalar_coalitions: int = 0
    term_coalitions: int = 0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "native_replays": self.native_replays,
            "native_transitions": self.native_transitions,
            "checkpoint_restores": self.checkpoint_restores,
            "unique_cached_states": self.unique_cached_states,
            "cache_hits": self.cache_hits,
            "scalar_coalitions": self.scalar_coalitions,
            "term_coalitions": self.term_coalitions,
            "wall_seconds": self.wall_seconds,
        }


def make_episode_spec(seed: int, episode_id: int) -> EpisodeSpec:
    rng = random.Random(seed * 2_000_033 + episode_id * 193 + 11)
    cue = tuple(rng.randrange(2) for _ in range(CUE_BITS))
    schedule = deterministic_schedule(seed, episode_id)
    return EpisodeSpec(
        seed=seed,
        episode_id=episode_id,
        cue=cue,
        schedule=schedule,
        event_codes=event_codes_from_schedule(
            schedule, seed=seed, episode_id=episode_id
        ),
    )


def deterministic_behavior_actions(seed: int, episode_id: int) -> tuple[int, ...]:
    rng = random.Random(seed * 3_000_017 + episode_id * 389 + 101)
    return tuple(rng.randrange(ACTION_COUNT) for _ in range(HORIZON))


def _mask(step: int, code: int) -> int:
    return ((step + 1) * (code + 3) + code * code + 7) & 1


def initial_state(spec: EpisodeSpec) -> ChainState:
    return ChainState(
        step=0,
        cue=spec.cue,
        schedule=tuple(sorted(spec.schedule.items())),
        values=(0,) * len(EVENT_CODES),
        active=(False,) * len(EVENT_CODES),
    )


def transition_state(
    state: ChainState, *, event_code: int, action: int
) -> ChainState:
    if action not in (0, 1) or event_code not in (DISTRACTOR_CODE, *EVENT_CODES):
        raise ValueError("LONG_CHAIN_TRANSITION_INPUT_INVALID")
    step = state.step
    values = list(state.values)
    active = list(state.active)
    for code in EVENT_CODES:
        index = code - 1
        if active[index]:
            values[index] ^= _mask(step, code)
    if event_code:
        index = event_code - 1
        active[index] = True
        values[index] = action ^ _mask(step, event_code)
    return ChainState(
        step=step + 1,
        cue=state.cue,
        schedule=state.schedule,
        values=tuple(values),
        active=tuple(active),
    )


def decode_final_actions(state: ChainState) -> tuple[int, ...]:
    if state.step != HORIZON or not all(state.active):
        raise ValueError("LONG_CHAIN_TERMINAL_STATE_INCOMPLETE")
    schedule = dict(state.schedule)
    decoded: list[int] = []
    for code in EVENT_CODES:
        value = state.values[code - 1]
        for step in range(schedule[code], HORIZON):
            value ^= _mask(step, code)
        decoded.append(value)
    return tuple(decoded)


def terminal_terms(state: ChainState) -> tuple[tuple[float, ...], tuple[bool, ...], bool, tuple[int, ...]]:
    decoded = decode_final_actions(state)
    expected = hidden_target_actions(state.cue)
    matched = tuple(decoded[index] == expected[index] for index in range(6))
    route = matched[0]
    source = matched[1] or matched[2]
    catalyst = matched[3] and matched[4]
    finish = matched[5]
    criteria = (route, source, catalyst, finish)
    pure_three_way = matched[0] and matched[3] and matched[5]
    closure = all(criteria)
    # Passenger channels are deliberately read by the native evaluator. Their
    # algebraic contribution is exactly zero, so ancestry is not causal credit.
    p7, p8, p9 = decoded[6:9]
    passenger_balance = (p7 - p7) + (p8 ^ p8) + (p9 & (1 - p9))
    terms = (
        0.20 * float(route),
        0.20 * float(source),
        0.20 * float(catalyst),
        0.20 * float(finish),
        0.20 * float(pure_three_way),
        1.00 * float(closure),
        0.00 * float(passenger_balance),
    )
    return terms, criteria, closure, decoded


def result_from_state(state: ChainState) -> EpisodeResult:
    terms, criteria, success, decoded = terminal_terms(state)
    return EpisodeResult(
        consequence=float(math.fsum(terms)),
        success=success,
        terms=terms,
        criteria=criteria,
        final_state=state,
        decoded_actions=decoded,
    )


def execute_episode(
    spec: EpisodeSpec,
    actions: Iterable[int],
    *,
    cost: ReplayCost | None = None,
) -> EpisodeResult:
    ledger = tuple(int(value) for value in actions)
    if len(ledger) != HORIZON or any(value not in (0, 1) for value in ledger):
        raise ValueError("LONG_CHAIN_ACTION_LEDGER_INVALID")
    state = initial_state(spec)
    if cost is not None:
        cost.native_replays += 1
        cost.checkpoint_restores += 1
    for step, action in enumerate(ledger):
        state = transition_state(
            state,
            event_code=(spec.event_codes[step] if spec.event_codes[step] in EVENT_CODES else DISTRACTOR_CODE),
            action=action,
        )
        if cost is not None:
            cost.native_transitions += 1
    return result_from_state(state)


class PrefixReplayEngine:
    """Generic checkpoint cache; no terminal formation relations are used."""

    def __init__(self, spec: EpisodeSpec) -> None:
        self.spec = spec
        self.transition_cache: dict[tuple[ChainState, int], ChainState] = {}
        self.result_cache: dict[ChainState, EpisodeResult] = {}
        self.cost = ReplayCost()

    def execute(self, actions: tuple[int, ...]) -> EpisodeResult:
        self.cost.native_replays += 1
        self.cost.checkpoint_restores += 1
        state = initial_state(self.spec)
        for step, action in enumerate(actions):
            key = (state, action)
            next_state = self.transition_cache.get(key)
            if next_state is None:
                next_state = transition_state(
                    state,
                    event_code=(
                        self.spec.event_codes[step]
                        if self.spec.event_codes[step] in EVENT_CODES
                        else DISTRACTOR_CODE
                    ),
                    action=action,
                )
                self.transition_cache[key] = next_state
                self.cost.native_transitions += 1
            else:
                self.cost.cache_hits += 1
            state = next_state
        result = self.result_cache.get(state)
        if result is None:
            result = result_from_state(state)
            self.result_cache[state] = result
        self.cost.unique_cached_states = len(self.transition_cache)
        return result


def coalition_actions(
    actions: tuple[int, ...], candidates: tuple[int, ...], mask: int
) -> tuple[int, ...]:
    changed = list(actions)
    for index, step in enumerate(candidates):
        if not (mask & (1 << index)):
            changed[step] = 1 - changed[step]
    return tuple(changed)


def _shapley(values: dict[int, float], n: int) -> tuple[float, ...]:
    factorial = [math.factorial(index) for index in range(n + 1)]
    denominator = math.factorial(n)
    result: list[float] = []
    for index in range(n):
        bit = 1 << index
        total = 0.0
        for mask in range(1 << n):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = factorial[size] * factorial[n - size - 1] / denominator
            total += weight * (values[mask | bit] - values[mask])
        result.append(float(total))
    return tuple(result)


def _pair_interactions(
    values: dict[int, float], candidates: tuple[int, ...]
) -> dict[str, float]:
    full = (1 << len(candidates)) - 1
    base = values[full]
    singles = {
        index: values[full ^ (1 << index)] - base
        for index in range(len(candidates))
    }
    result: dict[str, float] = {}
    for left, right in itertools.combinations(range(len(candidates)), 2):
        joint = values[full ^ (1 << left) ^ (1 << right)] - base
        result[f"{candidates[left]}:{candidates[right]}"] = float(
            joint - singles[left] - singles[right]
        )
    return result


def exact_scalar_credit(
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    candidates: tuple[int, ...],
    *,
    prefix_cache: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    engine = PrefixReplayEngine(spec) if prefix_cache else None
    cost = engine.cost if engine else ReplayCost()
    values: dict[int, float] = {}
    term_values: dict[int, tuple[float, ...]] = {}
    for mask in range(1 << len(candidates)):
        ledger = coalition_actions(actions, candidates, mask)
        result = engine.execute(ledger) if engine else execute_episode(spec, ledger, cost=cost)
        values[mask] = result.consequence
        term_values[mask] = result.terms
    cost.scalar_coalitions = len(values)
    cost.wall_seconds = time.perf_counter() - started
    raw = _shapley(values, len(candidates))
    return {
        "credits": {str(step): raw[index] for index, step in enumerate(candidates)},
        "pair_interactions": _pair_interactions(values, candidates),
        "scalar_values": values,
        "term_values": term_values,
        "cost": cost.to_dict(),
    }


TERM_CODE_GROUPS: dict[str, tuple[int, ...]] = {
    "term-0": (1,),
    "term-1": (2, 3),
    "term-2": (4, 5),
    "term-3": (6,),
    "term-4": (1, 4, 6),
    "term-5": (1, 2, 3, 4, 5, 6),
    "term-6": (7, 8, 9),
}


def factorized_credit(
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    candidates: tuple[int, ...],
    term_candidate_steps: dict[str, tuple[int, ...]],
    *,
    prefix_cache: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    engine = PrefixReplayEngine(spec) if prefix_cache else None
    cost = engine.cost if engine else ReplayCost()
    credit = {step: 0.0 for step in candidates}
    pair_totals = {
        f"{left}:{right}": 0.0
        for left, right in itertools.combinations(candidates, 2)
    }
    term_ledgers: dict[str, dict[int, float]] = {}
    for term_index, term_id in enumerate(TERM_IDS):
        group = tuple(sorted(term_candidate_steps[term_id]))
        if not set(group) <= set(candidates):
            raise ValueError("TERM_CANDIDATE_OUTSIDE_RETRIEVED_SET")
        values: dict[int, float] = {}
        for mask in range(1 << len(group)):
            ledger = coalition_actions(actions, group, mask)
            result = engine.execute(ledger) if engine else execute_episode(spec, ledger, cost=cost)
            values[mask] = result.terms[term_index]
        term_ledgers[term_id] = values
        contribution = _shapley(values, len(group))
        for index, step in enumerate(group):
            credit[step] += contribution[index]
        for key, value in _pair_interactions(values, group).items():
            pair_totals[key] = pair_totals.get(key, 0.0) + value
        cost.term_coalitions += len(values)
    cost.wall_seconds = time.perf_counter() - started
    return {
        "credits": {str(step): float(credit[step]) for step in candidates},
        "pair_interactions": {key: float(value) for key, value in pair_totals.items()},
        "term_values": term_ledgers,
        "cost": cost.to_dict(),
    }


def _state_payload(state: ChainState, code: int) -> dict[str, Any]:
    return {
        "step": state.step,
        "component_identity": f"component-{code}",
        "value": state.values[code - 1],
        "active": state.active[code - 1],
        "state_sha256": object_sha256(
            {
                "step": state.step,
                "component_identity": f"component-{code}",
                "value": state.values[code - 1],
                "active": state.active[code - 1],
            }
        ),
    }


def _fact(
    fact_id: str,
    result_id: str,
    *,
    u: dict[str, Any],
    tau: dict[str, Any],
    omega: dict[str, Any],
    z: dict[str, Any],
    rho: str,
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "result_id": result_id,
        "support_id": result_id,
        "coordinates": {
            "u": u,
            "tau": tau,
            "omega_bar": omega,
            "z": z,
            "rho": {"role": rho},
        },
    }


def build_atomic_execution(
    spec: EpisodeSpec, actions: tuple[int, ...]
) -> tuple[dict[str, Any], EpisodeResult, dict[str, Any]]:
    run_id = f"long-chain-seed-{spec.seed}-episode-{spec.episode_id}"
    facts: list[dict[str, Any]] = []
    state = initial_state(spec)
    previous_support = {
        code: f"{run_id}:component-{code}:version-0" for code in EVENT_CODES
    }
    # Initial component values are registered runtime sources.
    for code in EVENT_CODES:
        payload = _state_payload(state, code)
        facts.append(
            _fact(
                f"{run_id}:initial-component-{code}",
                previous_support[code],
                u={"kind": "registered_source", "source_identity": f"initial-component-{code}"},
                tau={"operation": "initialize-versioned-component"},
                omega={"concrete_occurrence_id": f"{run_id}:initialize-{code}", "step": -1},
                z={"kind": "OutcomeSupport", "payload": payload},
                rho="initial_component",
            )
        )
    action_fact_by_step: dict[int, str] = {}
    for step, action in enumerate(actions):
        next_state = transition_state(
            state,
            event_code=(spec.event_codes[step] if spec.event_codes[step] in EVENT_CODES else DISTRACTOR_CODE),
            action=action,
        )
        for code in EVENT_CODES:
            result_id = f"{run_id}:component-{code}:version-{step + 1}"
            payload = _state_payload(next_state, code)
            facts.append(
                _fact(
                    f"{run_id}:step-{step}:component-{code}:prior",
                    result_id,
                    u={"kind": "generated_origin", "prior_support_id": previous_support[code]},
                    tau={"operation": "execute-long-chain-transition"},
                    omega={"concrete_occurrence_id": f"{run_id}:transition-{step}", "step": step},
                    z={"kind": "OutcomeSupport", "payload": payload},
                    rho="prior_component_state",
                )
            )
            previous_support[code] = result_id
        event_code = spec.event_codes[step]
        if event_code in EVENT_CODES:
            result_id = previous_support[event_code]
            fact_id = f"{run_id}:step-{step}:event-action"
            action_fact_by_step[step] = fact_id
            facts.append(
                _fact(
                    fact_id,
                    result_id,
                    u={"kind": "registered_source", "source_identity": f"action-{step}", "action": action},
                    tau={"operation": "execute-long-chain-transition"},
                    omega={"concrete_occurrence_id": f"{run_id}:transition-{step}", "step": step},
                    z={"kind": "OutcomeSupport", "payload": _state_payload(next_state, event_code)},
                    rho="current_action",
                )
            )
        else:
            fact_id = f"{run_id}:step-{step}:audit-action"
            action_fact_by_step[step] = fact_id
            facts.append(
                _fact(
                    fact_id,
                    f"{run_id}:audit-{step}",
                    u={"kind": "registered_source", "source_identity": f"action-{step}", "action": action},
                    tau={"operation": "record-isolated-action-audit"},
                    omega={"concrete_occurrence_id": f"{run_id}:audit-{step}", "step": step},
                    z={"kind": "OutcomeSupport", "payload": {"step": step, "event_identity": "event-0"}},
                    rho="current_action",
                )
            )
        state = next_state
    result = result_from_state(state)
    term_sources: dict[str, tuple[int, ...]] = {
        term_id: TERM_CODE_GROUPS[term_id] for term_id in TERM_IDS
    }
    term_supports: dict[str, str] = {}
    term_fact_ids: dict[str, list[str]] = defaultdict(list)
    for term_index, term_id in enumerate(TERM_IDS):
        support_id = f"{run_id}:{term_id}:result"
        term_supports[term_id] = support_id
        for code in term_sources[term_id]:
            fact_id = f"{run_id}:{term_id}:input-{code}"
            term_fact_ids[term_id].append(fact_id)
            facts.append(
                _fact(
                    fact_id,
                    support_id,
                    u={"kind": "generated_origin", "prior_support_id": previous_support[code]},
                    tau={"operation": "evaluate-opaque-terminal-term", "term_identity": term_id},
                    omega={"concrete_occurrence_id": f"{run_id}:terminal-{term_id}", "step": HORIZON},
                    z={"kind": "OutcomeSupport", "payload": {"term_identity": term_id, "value": result.terms[term_index]}},
                    rho="terminal_state_component",
                )
            )
    scalar_support = f"{run_id}:terminal-consequence"
    scalar_fact_ids: list[str] = []
    for term_id in TERM_IDS:
        fact_id = f"{run_id}:terminal-scalar:input-{term_id}"
        scalar_fact_ids.append(fact_id)
        facts.append(
            _fact(
                fact_id,
                scalar_support,
                u={"kind": "generated_origin", "prior_support_id": term_supports[term_id]},
                tau={"operation": "aggregate-terminal-scalar"},
                omega={"concrete_occurrence_id": f"{run_id}:terminal-scalar", "step": HORIZON},
                z={"kind": "OutcomeSupport", "payload": {"terminal_consequence": result.consequence}},
                rho="terminal_term",
            )
        )
    bundle = {
        "schema_version": "long-chain-native-atomic-facts-v1",
        "execution_run_id": run_id,
        "facts": facts,
    }
    graph_index = build_native_formation_index(bundle)
    metadata = {
        "run_id": run_id,
        "action_fact_by_step": action_fact_by_step,
        "term_fact_ids": dict(term_fact_ids),
        "scalar_fact_ids": scalar_fact_ids,
        "term_supports": term_supports,
        "scalar_support": scalar_support,
        "native_fact_count": len(facts),
        "bundle_sha256": object_sha256(bundle),
        "formation_index": graph_index,
    }
    return bundle, result, metadata


def build_native_formation_index(bundle: dict[str, Any]) -> dict[str, Any]:
    producers: dict[str, list[str]] = defaultdict(list)
    for fact in bundle["facts"]:
        producers[fact["result_id"]].append(fact["fact_id"])
    edges: list[dict[str, str]] = []
    reverse: dict[str, list[str]] = defaultdict(list)
    for fact in bundle["facts"]:
        u = fact["coordinates"]["u"]
        prior = u.get("prior_support_id") if u.get("kind") == "generated_origin" else None
        if prior is None:
            continue
        for producer in producers.get(prior, []):
            edge = {
                "source": producer,
                "target": fact["fact_id"],
                "relation_type": "generated_origin_dependency",
            }
            edges.append(edge)
            reverse[fact["fact_id"]].append(producer)
    return {
        "edges": sorted(edges, key=lambda row: (row["source"], row["target"])),
        "reverse": {key: sorted(value) for key, value in reverse.items()},
    }


def _ancestor_action_steps(
    bundle: dict[str, Any], start_fact_ids: Iterable[str]
) -> tuple[int, ...]:
    facts = {row["fact_id"]: row for row in bundle["facts"]}
    reverse = build_native_formation_index(bundle)["reverse"]
    queue = deque(start_fact_ids)
    visited: set[str] = set()
    steps: set[int] = set()
    while queue:
        fact_id = queue.popleft()
        if fact_id in visited:
            continue
        visited.add(fact_id)
        fact = facts[fact_id]
        rho = fact["coordinates"]["rho"].get("role")
        if rho == "current_action":
            steps.add(int(fact["coordinates"]["omega_bar"]["step"]))
        queue.extend(reverse.get(fact_id, []))
    return tuple(sorted(steps))


def retrieve_candidates(
    bundle: dict[str, Any], metadata: dict[str, Any]
) -> tuple[int, ...]:
    return _ancestor_action_steps(bundle, metadata["scalar_fact_ids"])


def retrieve_term_candidate_steps(
    bundle: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    return {
        term_id: _ancestor_action_steps(bundle, metadata["term_fact_ids"][term_id])
        for term_id in TERM_IDS
    }


def rewire_term_candidate_steps(
    term_steps: dict[str, tuple[int, ...]], candidates: tuple[int, ...], seed: int
) -> dict[str, tuple[int, ...]]:
    rng = random.Random(seed)
    permuted = list(candidates)
    rng.shuffle(permuted)
    mapping = dict(zip(candidates, permuted))
    return {
        term_id: tuple(sorted(mapping[step] for step in steps))
        for term_id, steps in term_steps.items()
    }


def native_relation_sidecar(bundle: dict[str, Any]) -> dict[str, Any]:
    relations = []
    for edge in build_native_formation_index(bundle)["edges"]:
        material = {
            "execution_run_id": bundle["execution_run_id"],
            "relation_type": edge["relation_type"],
            "endpoint_level": "fact",
            "source_id": edge["source"],
            "target_id": edge["target"],
            "relation_payload": {"established_from": "exact prior outcome identity"},
            "establishment_source": "generator_established",
            "authority_id": "long-chain-synchronous-capture-v1",
            "evidence_refs": [],
        }
        relations.append({**material, "relation_id": "lcrel_" + object_sha256(material)})
    return {
        "schema_version": "long-chain-native-relation-sidecar-v1",
        "execution_run_id": bundle["execution_run_id"],
        "relations": relations,
        "evidence": [],
    }


def compile_and_validate_canonical_gfg(
    bundle: dict[str, Any], *, generator_name: str
) -> dict[str, Any]:
    from experiments.executable_generation_fact_graph_v1.adapters.core_snapshot_adapter import (
        build_core_snapshot_from_atomic_facts,
        normalize_relation_store,
    )
    from experiments.executable_generation_fact_graph_v2.adapters.common import (
        complete_capture_audit,
    )
    from experiments.executable_generation_fact_graph_v2.endpoint_registry import (
        build_core_occurrence_catalog,
    )
    from experiments.executable_generation_fact_graph_v2.graph_compiler import (
        compile_executable_generation_fact_graph_v2,
    )
    from experiments.executable_generation_fact_graph_v2.graph_validator import (
        load_contracts,
        validate_executable_generation_fact_graph_v2,
    )

    run_id = bundle["execution_run_id"]
    snapshot_input, mapping = build_core_snapshot_from_atomic_facts(
        atomic_fact_bundle=bundle,
        execution_run_id=run_id,
        domain_scope_id=DOMAIN_SCOPE_ID,
        generator_name=generator_name,
    )
    relation_store, normalization = normalize_relation_store(
        native_sidecar=native_relation_sidecar(bundle),
        mapping=mapping,
        require_complete=True,
    )
    relation_store["relations"] = [
        {
            **row,
            "source_endpoint_kind": "fact",
            "target_endpoint_kind": "fact",
        }
        for row in relation_store["relations"]
    ]
    inputs = [snapshot_input]
    catalog = build_core_occurrence_catalog(inputs)
    audit = complete_capture_audit(run_id, domain=DOMAIN_SCOPE_ID)
    contracts = load_contracts()
    graph = compile_executable_generation_fact_graph_v2(
        inputs,
        relation_store,
        catalog,
        audit,
        contracts["graph_profile"],
        contracts["relation_type_registry"],
    )
    validated = validate_executable_generation_fact_graph_v2(
        graph,
        inputs,
        relation_store,
        catalog,
        audit,
        contracts,
    )
    document = graph.to_dict()
    forbidden = (
        "causal_credit",
        "necessary",
        "backup",
        "substitution",
        "synergy",
        "functional_action",
        "hidden_target",
    )
    serialized = canonical_bytes(document).decode("utf-8")
    leaks = [label for label in forbidden if label in serialized]
    return {
        "status": validated.validation.status,
        "snapshot_id": snapshot_input["snapshot"].snapshot_id,
        "graph_id": validated.graph_id,
        "validation_sha256": validated.validation.validation_sha256,
        "fact_node_count": len(graph.fact_nodes),
        "occurrence_node_count": len(graph.occurrence_nodes),
        "incidence_edge_count": len(graph.incidence_edges),
        "relation_edge_count": len(graph.relation_edges),
        "coordinate_mapping_exact": mapping["coordinate_mapping_exact"],
        "relation_mapping_exact": normalization["coverage_exact"],
        "forbidden_label_hits": leaks,
        "graph_sha256": object_sha256(document),
    }


def build_credit_discovery_atomic_execution(
    *,
    base_run_id: str,
    base_graph_sha256: str,
    candidates: tuple[int, ...],
    term_candidate_steps: dict[str, tuple[int, ...]],
    term_values: dict[str, dict[int, float]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Capture the exact factorized adjudication as a second generation run."""
    run_id = base_run_id + ":credit-discovery"
    facts: list[dict[str, Any]] = []
    replay_supports: dict[tuple[str, int], str] = {}
    term_credit_supports: dict[tuple[str, int], str] = {}
    term_credits_by_step: dict[int, list[float]] = defaultdict(list)
    for term_id in TERM_IDS:
        group = tuple(sorted(term_candidate_steps[term_id]))
        values = {int(mask): float(value) for mask, value in term_values[term_id].items()}
        for mask, value in sorted(values.items()):
            support_id = f"{run_id}:{term_id}:coalition-{mask}"
            replay_supports[(term_id, mask)] = support_id
            facts.append(
                _fact(
                    f"{run_id}:{term_id}:coalition-{mask}:replay",
                    support_id,
                    u={
                        "kind": "registered_source",
                        "source_identity": f"coalition-{term_id}-{mask}",
                        "base_graph_sha256": base_graph_sha256,
                        "base_run_id": base_run_id,
                        "candidate_steps": list(group),
                        "coalition_mask": mask,
                    },
                    tau={"operation": "restore-and-replay-native-long-chain"},
                    omega={"concrete_occurrence_id": f"{run_id}:replay-{term_id}-{mask}", "coalition_mask": mask},
                    z={"kind": "OutcomeSupport", "payload": {"term_identity": term_id, "value": value}},
                    rho="coalition_assignment",
                )
            )
        n = len(group)
        factorial = [math.factorial(index) for index in range(n + 1)]
        denominator = math.factorial(n)
        term_credit = _shapley(values, n)
        for index, step in enumerate(group):
            marginal_supports: list[str] = []
            bit = 1 << index
            for mask in range(1 << n):
                if mask & bit:
                    continue
                size = mask.bit_count()
                weight = factorial[size] * factorial[n - size - 1] / denominator
                marginal = weight * (values[mask | bit] - values[mask])
                support_id = f"{run_id}:{term_id}:step-{step}:marginal-{mask}"
                marginal_supports.append(support_id)
                for side, source_mask in (("without", mask), ("with", mask | bit)):
                    facts.append(
                        _fact(
                            f"{run_id}:{term_id}:step-{step}:marginal-{mask}:{side}",
                            support_id,
                            u={"kind": "generated_origin", "prior_support_id": replay_supports[(term_id, source_mask)]},
                            tau={"operation": "form-weighted-shapley-marginal", "weight": weight},
                            omega={"concrete_occurrence_id": f"{run_id}:marginal-{term_id}-{step}-{mask}", "coalition_mask": mask},
                            z={"kind": "OutcomeSupport", "payload": {"weighted_marginal": marginal}},
                            rho=f"coalition_{side}_candidate",
                        )
                    )
            credit_support = f"{run_id}:{term_id}:step-{step}:credit"
            term_credit_supports[(term_id, step)] = credit_support
            term_credits_by_step[step].append(term_credit[index])
            for marginal_index, marginal_support in enumerate(marginal_supports):
                facts.append(
                    _fact(
                        f"{run_id}:{term_id}:step-{step}:credit-input-{marginal_index}",
                        credit_support,
                        u={"kind": "generated_origin", "prior_support_id": marginal_support},
                        tau={"operation": "sum-exact-shapley-marginals"},
                        omega={"concrete_occurrence_id": f"{run_id}:term-credit-{term_id}-{step}", "candidate_step": step},
                        z={"kind": "OutcomeSupport", "payload": {"term_identity": term_id, "candidate_step": step, "credit": term_credit[index]}},
                        rho="weighted_marginal",
                    )
                )
    final_credits = {str(step): float(math.fsum(term_credits_by_step.get(step, [0.0]))) for step in candidates}
    for step in candidates:
        sources = [
            support
            for (term_id, candidate_step), support in term_credit_supports.items()
            if candidate_step == step
        ]
        if not sources:
            facts.append(
                _fact(
                    f"{run_id}:step-{step}:zero-credit",
                    f"{run_id}:step-{step}:final-credit",
                    u={"kind": "registered_source", "source_identity": f"empty-credit-ledger-{step}"},
                    tau={"operation": "establish-zero-credit"},
                    omega={"concrete_occurrence_id": f"{run_id}:final-credit-{step}", "candidate_step": step},
                    z={"kind": "OutcomeSupport", "payload": {"candidate_step": step, "credit": 0.0}},
                    rho="empty_term_credit_ledger",
                )
            )
            continue
        for index, support in enumerate(sources):
            facts.append(
                _fact(
                    f"{run_id}:step-{step}:final-credit-input-{index}",
                    f"{run_id}:step-{step}:final-credit",
                    u={"kind": "generated_origin", "prior_support_id": support},
                    tau={"operation": "sum-term-shapley-credit"},
                    omega={"concrete_occurrence_id": f"{run_id}:final-credit-{step}", "candidate_step": step},
                    z={"kind": "OutcomeSupport", "payload": {"candidate_step": step, "credit": final_credits[str(step)]}},
                    rho="term_credit",
                )
            )
    bundle = {
        "schema_version": "credit-discovery-native-atomic-facts-v1",
        "execution_run_id": run_id,
        "facts": facts,
    }
    return bundle, final_credits


def credit_error(
    reference: dict[str, float], candidate: dict[str, float]
) -> float:
    keys = set(reference) | set(candidate)
    return max((abs(reference.get(key, 0.0) - candidate.get(key, 0.0)) for key in keys), default=0.0)


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def credit_metrics(
    reference: dict[str, float], candidate: dict[str, float], passenger_steps: tuple[int, ...]
) -> dict[str, Any]:
    keys = sorted(set(reference) | set(candidate), key=int)
    signs = [sign(reference.get(key, 0.0)) == sign(candidate.get(key, 0.0)) for key in keys]
    passenger_zero = [sign(candidate.get(str(step), 0.0)) == 0 for step in passenger_steps]
    return {
        "max_abs_error": credit_error(reference, candidate),
        "sign_accuracy": sum(signs) / len(signs) if signs else 1.0,
        "passenger_zero_accuracy": sum(passenger_zero) / len(passenger_zero) if passenger_zero else 1.0,
        "exact_within_1e_12": credit_error(reference, candidate) <= 1e-12,
    }


__all__ = [
    "ACTION_COUNT",
    "CUE_BITS",
    "DOMAIN_SCOPE_ID",
    "EVENT_CODES",
    "FUNCTIONAL_CODES",
    "HORIZON",
    "PASSENGER_CODES",
    "TERM_IDS",
    "VISIBLE_EVENT_CODE_COUNT",
    "EpisodeResult",
    "EpisodeSpec",
    "ReplayCost",
    "build_atomic_execution",
    "coalition_actions",
    "compile_and_validate_canonical_gfg",
    "build_credit_discovery_atomic_execution",
    "credit_metrics",
    "deterministic_behavior_actions",
    "exact_scalar_credit",
    "execute_episode",
    "factorized_credit",
    "hidden_target_actions",
    "make_episode_spec",
    "object_sha256",
    "retrieve_candidates",
    "retrieve_term_candidate_steps",
    "rewire_term_candidate_steps",
]
