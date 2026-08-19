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

from experiments.gfg_temporal_credit_long_chain_self_optimization_v1.runtime import (
    ACTION_COUNT,
    CUE_BITS,
    EVENT_CODES,
    FUNCTIONAL_CODES,
    HORIZON,
    PASSENGER_CODES,
    VISIBLE_EVENT_CODE_COUNT,
    EpisodeSpec,
    TERM_CODE_GROUPS as DETERMINISTIC_TERM_CODE_GROUPS,
    _pair_interactions,
    _shapley,
    canonical_bytes,
    deterministic_behavior_actions,
    hidden_target_actions,
    make_episode_spec,
    object_sha256,
    rewire_term_candidate_steps,
    sign,
)


DOMAIN_SCOPE_ID = "gfg-temporal-credit-stochastic-long-chain-v1"
STOCHASTIC_CHANNEL_COUNT = 8
STOCHASTIC_MODULUS = 65_521
TERM_IDS = tuple(f"term-{index}" for index in range(8))
TERM_CODE_GROUPS: dict[str, tuple[int, ...]] = {
    **DETERMINISTIC_TERM_CODE_GROUPS,
    "term-7": (),
}
TERM_STOCHASTIC_CHANNEL: dict[str, int] = {
    term_id: index for index, term_id in enumerate(TERM_IDS)
}


@dataclass(frozen=True)
class ExogenousInput:
    source_identity: str
    original_step: int
    value: int


StochasticTape = tuple[ExogenousInput, ...]


@dataclass(frozen=True)
class StochasticChainState:
    step: int
    cue: tuple[int, ...]
    schedule: tuple[tuple[int, int], ...]
    action_values: tuple[int, ...]
    active: tuple[bool, ...]
    stochastic_values: tuple[int, ...]


@dataclass(frozen=True)
class StochasticEpisodeResult:
    consequence: float
    success: bool
    terms: tuple[float, ...]
    criteria: tuple[bool, ...]
    final_state: StochasticChainState
    decoded_actions: tuple[int, ...]
    stochastic_scales: tuple[float, ...]


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


def _counter_value(
    *, seed: int, episode_id: int, realization_id: int, step: int
) -> int:
    material = (
        f"stochastic-environment-v1|{seed}|{episode_id}|"
        f"{realization_id}|{step}|environment-transition"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 256


def make_stochastic_tape(
    *, seed: int, episode_id: int, realization_id: int
) -> StochasticTape:
    return tuple(
        ExogenousInput(
            source_identity=(
                f"xi:seed-{seed}:episode-{episode_id}:"
                f"realization-{realization_id}:step-{step}"
            ),
            original_step=step,
            value=_counter_value(
                seed=seed,
                episode_id=episode_id,
                realization_id=realization_id,
                step=step,
            ),
        )
        for step in range(HORIZON)
    )


def permute_stochastic_bindings(tape: StochasticTape, *, salt: int) -> StochasticTape:
    if len(tape) != HORIZON:
        raise ValueError("STOCHASTIC_TAPE_LENGTH_INVALID")
    offset = 1 + (
        int.from_bytes(hashlib.sha256(f"binding-permutation|{salt}".encode()).digest()[:4], "big")
        % (HORIZON - 1)
    )
    return tuple(tape[(step + offset) % HORIZON] for step in range(HORIZON))


def _mask(step: int, code: int) -> int:
    return ((step + 1) * (code + 3) + code * code + 7) & 1


def initial_state(spec: EpisodeSpec) -> StochasticChainState:
    return StochasticChainState(
        step=0,
        cue=spec.cue,
        schedule=tuple(sorted(spec.schedule.items())),
        action_values=(0,) * len(EVENT_CODES),
        active=(False,) * len(EVENT_CODES),
        stochastic_values=(0,) * STOCHASTIC_CHANNEL_COUNT,
    )


def transition_state(
    state: StochasticChainState,
    *,
    event_code: int,
    action: int,
    exogenous_value: int,
) -> StochasticChainState:
    if action not in (0, 1):
        raise ValueError("STOCHASTIC_ACTION_INVALID")
    if event_code not in (0, *EVENT_CODES):
        raise ValueError("STOCHASTIC_EVENT_CODE_INVALID")
    if not 0 <= exogenous_value < 256:
        raise ValueError("STOCHASTIC_VALUE_INVALID")
    step = state.step
    action_values = list(state.action_values)
    active = list(state.active)
    for code in EVENT_CODES:
        index = code - 1
        if active[index]:
            action_values[index] ^= _mask(step, code)
    if event_code:
        index = event_code - 1
        active[index] = True
        action_values[index] = action ^ _mask(step, event_code)

    stochastic_values: list[int] = []
    for channel, prior in enumerate(state.stochastic_values):
        multiplier = 17 + 2 * channel
        injected = (exogenous_value + 1) * (channel + 3)
        occurrence = (step + 1) * (channel * channel + 5)
        stochastic_values.append(
            (prior * multiplier + injected + occurrence) % STOCHASTIC_MODULUS
        )
    return StochasticChainState(
        step=step + 1,
        cue=state.cue,
        schedule=state.schedule,
        action_values=tuple(action_values),
        active=tuple(active),
        stochastic_values=tuple(stochastic_values),
    )


def decode_final_actions(state: StochasticChainState) -> tuple[int, ...]:
    if state.step != HORIZON or not all(state.active):
        raise ValueError("STOCHASTIC_TERMINAL_STATE_INCOMPLETE")
    schedule = dict(state.schedule)
    decoded: list[int] = []
    for code in EVENT_CODES:
        value = state.action_values[code - 1]
        for step in range(schedule[code], HORIZON):
            value ^= _mask(step, code)
        decoded.append(value)
    return tuple(decoded)


def _stochastic_scales(state: StochasticChainState) -> tuple[float, ...]:
    # Positive multipliers preserve action-credit semantics while allowing
    # realized magnitudes and interactions to vary across stochastic worlds.
    return tuple(
        0.70 + 0.60 * value / (STOCHASTIC_MODULUS - 1)
        for value in state.stochastic_values
    )


def terminal_terms(
    state: StochasticChainState,
) -> tuple[tuple[float, ...], tuple[bool, ...], bool, tuple[int, ...], tuple[float, ...]]:
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
    scales = _stochastic_scales(state)
    p7, p8, p9 = decoded[6:9]
    passenger_balance = (p7 - p7) + (p8 ^ p8) + (p9 & (1 - p9))
    environmental_background = 0.40 * (scales[7] - 1.0)
    terms = (
        0.20 * scales[0] * float(route),
        0.20 * scales[1] * float(source),
        0.20 * scales[2] * float(catalyst),
        0.20 * scales[3] * float(finish),
        0.20 * scales[4] * float(pure_three_way),
        1.00 * scales[5] * float(closure),
        0.00 * scales[6] * float(passenger_balance),
        environmental_background,
    )
    return terms, criteria, closure, decoded, scales


def result_from_state(state: StochasticChainState) -> StochasticEpisodeResult:
    terms, criteria, success, decoded, scales = terminal_terms(state)
    return StochasticEpisodeResult(
        consequence=float(math.fsum(terms)),
        success=success,
        terms=terms,
        criteria=criteria,
        final_state=state,
        decoded_actions=decoded,
        stochastic_scales=scales,
    )


def execute_episode(
    spec: EpisodeSpec,
    actions: Iterable[int],
    tape: StochasticTape,
    *,
    cost: ReplayCost | None = None,
) -> StochasticEpisodeResult:
    ledger = tuple(int(value) for value in actions)
    if len(ledger) != HORIZON or any(value not in (0, 1) for value in ledger):
        raise ValueError("STOCHASTIC_ACTION_LEDGER_INVALID")
    if len(tape) != HORIZON:
        raise ValueError("STOCHASTIC_TAPE_LENGTH_INVALID")
    state = initial_state(spec)
    if cost is not None:
        cost.native_replays += 1
        cost.checkpoint_restores += 1
    for step, action in enumerate(ledger):
        code = spec.event_codes[step]
        state = transition_state(
            state,
            event_code=(code if code in EVENT_CODES else 0),
            action=action,
            exogenous_value=tape[step].value,
        )
        if cost is not None:
            cost.native_transitions += 1
    return result_from_state(state)


class PrefixReplayEngine:
    def __init__(self, spec: EpisodeSpec, tape: StochasticTape) -> None:
        self.spec = spec
        self.tape = tape
        self.transition_cache: dict[tuple[StochasticChainState, int, int], StochasticChainState] = {}
        self.result_cache: dict[StochasticChainState, StochasticEpisodeResult] = {}
        self.cost = ReplayCost()

    def execute(self, actions: tuple[int, ...]) -> StochasticEpisodeResult:
        self.cost.native_replays += 1
        self.cost.checkpoint_restores += 1
        state = initial_state(self.spec)
        for step, action in enumerate(actions):
            xi = self.tape[step].value
            key = (state, action, xi)
            next_state = self.transition_cache.get(key)
            if next_state is None:
                code = self.spec.event_codes[step]
                next_state = transition_state(
                    state,
                    event_code=(code if code in EVENT_CODES else 0),
                    action=action,
                    exogenous_value=xi,
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
        if not mask & (1 << index):
            changed[step] = 1 - changed[step]
    return tuple(changed)


def exact_conditional_credit(
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    tape: StochasticTape,
    candidates: tuple[int, ...],
    *,
    prefix_cache: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    engine = PrefixReplayEngine(spec, tape) if prefix_cache else None
    cost = engine.cost if engine else ReplayCost()
    scalar_values: dict[int, float] = {}
    term_values: dict[int, tuple[float, ...]] = {}
    for mask in range(1 << len(candidates)):
        ledger = coalition_actions(actions, candidates, mask)
        result = engine.execute(ledger) if engine else execute_episode(spec, ledger, tape, cost=cost)
        scalar_values[mask] = result.consequence
        term_values[mask] = result.terms
    cost.scalar_coalitions += len(scalar_values)
    cost.wall_seconds = time.perf_counter() - started
    return {
        "credits": {
            str(step): float(value)
            for step, value in zip(candidates, _shapley(scalar_values, len(candidates)))
        },
        "pair_interactions": _pair_interactions(scalar_values, candidates),
        "scalar_values": scalar_values,
        "term_values": term_values,
        "cost": cost.to_dict(),
    }


def factorized_conditional_credit(
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    tape: StochasticTape,
    candidates: tuple[int, ...],
    term_candidate_steps: dict[str, tuple[int, ...]],
    *,
    prefix_cache: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    engine = PrefixReplayEngine(spec, tape) if prefix_cache else None
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
            raise ValueError("STOCHASTIC_TERM_CANDIDATE_OUTSIDE_RETRIEVED_SET")
        values: dict[int, float] = {}
        for mask in range(1 << len(group)):
            ledger = coalition_actions(actions, group, mask)
            result = engine.execute(ledger) if engine else execute_episode(spec, ledger, tape, cost=cost)
            values[mask] = result.terms[term_index]
        term_ledgers[term_id] = values
        for index, step in enumerate(group):
            credit[step] += _shapley(values, len(group))[index]
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


def _action_state_payload(state: StochasticChainState, code: int) -> dict[str, Any]:
    payload = {
        "step": state.step,
        "component_identity": f"action-component-{code}",
        "value": state.action_values[code - 1],
        "active": state.active[code - 1],
    }
    return {**payload, "state_sha256": object_sha256(payload)}


def _stochastic_state_payload(state: StochasticChainState, channel: int) -> dict[str, Any]:
    payload = {
        "step": state.step,
        "component_identity": f"stochastic-channel-{channel}",
        "value": state.stochastic_values[channel],
        "modulus": STOCHASTIC_MODULUS,
    }
    return {**payload, "state_sha256": object_sha256(payload)}


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
    spec: EpisodeSpec,
    actions: tuple[int, ...],
    tape: StochasticTape,
) -> tuple[dict[str, Any], StochasticEpisodeResult, dict[str, Any]]:
    realization_hash = object_sha256(
        [(row.source_identity, row.original_step, row.value) for row in tape]
    )[:16]
    run_id = (
        f"stochastic-long-chain-seed-{spec.seed}-episode-{spec.episode_id}-"
        f"tape-{realization_hash}"
    )
    facts: list[dict[str, Any]] = []
    state = initial_state(spec)
    previous_action_support = {
        code: f"{run_id}:action-component-{code}:version-0" for code in EVENT_CODES
    }
    previous_stochastic_support = {
        channel: f"{run_id}:stochastic-channel-{channel}:version-0"
        for channel in range(STOCHASTIC_CHANNEL_COUNT)
    }
    for code in EVENT_CODES:
        facts.append(
            _fact(
                f"{run_id}:initial-action-component-{code}",
                previous_action_support[code],
                u={"kind": "registered_source", "source_identity": f"initial-action-component-{code}"},
                tau={"operation": "initialize-versioned-action-component"},
                omega={"concrete_occurrence_id": f"{run_id}:initialize-action-{code}", "step": -1},
                z={"kind": "OutcomeSupport", "payload": _action_state_payload(state, code)},
                rho="initial_component",
            )
        )
    for channel in range(STOCHASTIC_CHANNEL_COUNT):
        facts.append(
            _fact(
                f"{run_id}:initial-stochastic-channel-{channel}",
                previous_stochastic_support[channel],
                u={"kind": "registered_source", "source_identity": f"initial-stochastic-channel-{channel}"},
                tau={"operation": "initialize-versioned-stochastic-channel"},
                omega={"concrete_occurrence_id": f"{run_id}:initialize-stochastic-{channel}", "step": -1},
                z={"kind": "OutcomeSupport", "payload": _stochastic_state_payload(state, channel)},
                rho="initial_stochastic_state",
            )
        )

    action_fact_by_step: dict[int, str] = {}
    stochastic_draw_fact_by_step: dict[int, str] = {}
    for step, action in enumerate(actions):
        xi = tape[step]
        code = spec.event_codes[step]
        event_code = code if code in EVENT_CODES else 0
        next_state = transition_state(
            state,
            event_code=event_code,
            action=action,
            exogenous_value=xi.value,
        )
        occurrence = {"concrete_occurrence_id": f"{run_id}:transition-{step}", "step": step}
        for component_code in EVENT_CODES:
            result_id = f"{run_id}:action-component-{component_code}:version-{step + 1}"
            facts.append(
                _fact(
                    f"{run_id}:step-{step}:action-component-{component_code}:prior",
                    result_id,
                    u={"kind": "generated_origin", "prior_support_id": previous_action_support[component_code]},
                    tau={"operation": "execute-stochastic-long-chain-transition"},
                    omega=occurrence,
                    z={"kind": "OutcomeSupport", "payload": _action_state_payload(next_state, component_code)},
                    rho="prior_component_state",
                )
            )
            previous_action_support[component_code] = result_id

        draw_support = f"{run_id}:xi-{step}:result"
        draw_fact_id = f"{run_id}:xi-{step}:draw"
        stochastic_draw_fact_by_step[step] = draw_fact_id
        facts.append(
            _fact(
                draw_fact_id,
                draw_support,
                u={
                    "kind": "registered_source",
                    "source_identity": xi.source_identity,
                    "original_step": xi.original_step,
                },
                tau={"operation": "counter-addressed-exogenous-draw", "channel_name": "environment-transition"},
                omega={"concrete_occurrence_id": f"{run_id}:draw-{step}", "step": step},
                z={"kind": "OutcomeSupport", "payload": {"value": xi.value, "value_cardinality": 256}},
                rho="exogenous_random_source",
            )
        )
        for channel in range(STOCHASTIC_CHANNEL_COUNT):
            result_id = f"{run_id}:stochastic-channel-{channel}:version-{step + 1}"
            payload = _stochastic_state_payload(next_state, channel)
            facts.append(
                _fact(
                    f"{run_id}:step-{step}:stochastic-channel-{channel}:prior",
                    result_id,
                    u={"kind": "generated_origin", "prior_support_id": previous_stochastic_support[channel]},
                    tau={"operation": "execute-stochastic-long-chain-transition"},
                    omega=occurrence,
                    z={"kind": "OutcomeSupport", "payload": payload},
                    rho="prior_stochastic_state",
                )
            )
            facts.append(
                _fact(
                    f"{run_id}:step-{step}:stochastic-channel-{channel}:input",
                    result_id,
                    u={"kind": "generated_origin", "prior_support_id": draw_support},
                    tau={"operation": "execute-stochastic-long-chain-transition"},
                    omega=occurrence,
                    z={"kind": "OutcomeSupport", "payload": payload},
                    rho="exogenous_stochastic_input",
                )
            )
            previous_stochastic_support[channel] = result_id

        if event_code:
            result_id = previous_action_support[event_code]
            fact_id = f"{run_id}:step-{step}:event-action"
            action_fact_by_step[step] = fact_id
            facts.append(
                _fact(
                    fact_id,
                    result_id,
                    u={"kind": "registered_source", "source_identity": f"action-{step}", "action": action},
                    tau={"operation": "execute-stochastic-long-chain-transition"},
                    omega=occurrence,
                    z={"kind": "OutcomeSupport", "payload": _action_state_payload(next_state, event_code)},
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
    term_supports: dict[str, str] = {}
    term_fact_ids: dict[str, list[str]] = defaultdict(list)
    for term_index, term_id in enumerate(TERM_IDS):
        support_id = f"{run_id}:{term_id}:result"
        term_supports[term_id] = support_id
        for component_code in TERM_CODE_GROUPS[term_id]:
            fact_id = f"{run_id}:{term_id}:action-input-{component_code}"
            term_fact_ids[term_id].append(fact_id)
            facts.append(
                _fact(
                    fact_id,
                    support_id,
                    u={"kind": "generated_origin", "prior_support_id": previous_action_support[component_code]},
                    tau={"operation": "evaluate-stochastic-terminal-term", "term_identity": term_id},
                    omega={"concrete_occurrence_id": f"{run_id}:terminal-{term_id}", "step": HORIZON},
                    z={"kind": "OutcomeSupport", "payload": {"term_identity": term_id, "value": result.terms[term_index]}},
                    rho="terminal_action_state_component",
                )
            )
        channel = TERM_STOCHASTIC_CHANNEL[term_id]
        stochastic_fact_id = f"{run_id}:{term_id}:stochastic-input-{channel}"
        term_fact_ids[term_id].append(stochastic_fact_id)
        facts.append(
            _fact(
                stochastic_fact_id,
                support_id,
                u={"kind": "generated_origin", "prior_support_id": previous_stochastic_support[channel]},
                tau={"operation": "evaluate-stochastic-terminal-term", "term_identity": term_id},
                omega={"concrete_occurrence_id": f"{run_id}:terminal-{term_id}", "step": HORIZON},
                z={"kind": "OutcomeSupport", "payload": {"term_identity": term_id, "value": result.terms[term_index]}},
                rho="terminal_stochastic_state",
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
                tau={"operation": "aggregate-stochastic-terminal-scalar"},
                omega={"concrete_occurrence_id": f"{run_id}:terminal-scalar", "step": HORIZON},
                z={"kind": "OutcomeSupport", "payload": {"terminal_consequence": result.consequence}},
                rho="terminal_term",
            )
        )
    bundle = {
        "schema_version": "stochastic-long-chain-native-atomic-facts-v1",
        "execution_run_id": run_id,
        "facts": facts,
    }
    metadata = {
        "run_id": run_id,
        "action_fact_by_step": action_fact_by_step,
        "stochastic_draw_fact_by_step": stochastic_draw_fact_by_step,
        "term_fact_ids": dict(term_fact_ids),
        "scalar_fact_ids": scalar_fact_ids,
        "term_supports": term_supports,
        "scalar_support": scalar_support,
        "native_fact_count": len(facts),
        "bundle_sha256": object_sha256(bundle),
        "formation_index": build_native_formation_index(bundle),
        "tape_binding_sha256": object_sha256(
            [(step, row.source_identity, row.original_step, row.value) for step, row in enumerate(tape)]
        ),
    }
    return bundle, result, metadata


def build_native_formation_index(bundle: dict[str, Any]) -> dict[str, Any]:
    producers: dict[str, list[str]] = defaultdict(list)
    for fact in bundle["facts"]:
        producers[fact["result_id"]].append(fact["fact_id"])
    edges: list[dict[str, str]] = []
    reverse: dict[str, list[str]] = defaultdict(list)
    for fact in bundle["facts"]:
        source = fact["coordinates"]["u"]
        prior = source.get("prior_support_id") if source.get("kind") == "generated_origin" else None
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
        if fact["coordinates"]["rho"].get("role") == "current_action":
            steps.add(int(fact["coordinates"]["omega_bar"]["step"]))
        queue.extend(reverse.get(fact_id, []))
    return tuple(sorted(steps))


def retrieve_candidates(bundle: dict[str, Any], metadata: dict[str, Any]) -> tuple[int, ...]:
    return _ancestor_action_steps(bundle, metadata["scalar_fact_ids"])


def retrieve_term_candidate_steps(
    bundle: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    return {
        term_id: _ancestor_action_steps(bundle, metadata["term_fact_ids"][term_id])
        for term_id in TERM_IDS
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
            "authority_id": "stochastic-long-chain-synchronous-capture-v1",
            "evidence_refs": [],
        }
        relations.append({**material, "relation_id": "slcrel_" + object_sha256(material)})
    return {
        "schema_version": "stochastic-long-chain-native-relation-sidecar-v1",
        "execution_run_id": bundle["execution_run_id"],
        "relations": relations,
        "evidence": [],
    }


def compile_and_validate_canonical_gfg(
    bundle: dict[str, Any],
    *,
    generator_name: str,
    enforce_participant_labels: bool = True,
) -> dict[str, Any]:
    from experiments.executable_generation_fact_graph_v1.adapters.core_snapshot_adapter import (
        build_core_snapshot_from_atomic_facts,
        normalize_relation_store,
    )
    from experiments.executable_generation_fact_graph_v2.adapters.common import complete_capture_audit
    from experiments.executable_generation_fact_graph_v2.endpoint_registry import build_core_occurrence_catalog
    from experiments.executable_generation_fact_graph_v2.graph_compiler import compile_executable_generation_fact_graph_v2
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
        {**row, "source_endpoint_kind": "fact", "target_endpoint_kind": "fact"}
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
        graph, inputs, relation_store, catalog, audit, contracts
    )
    document = graph.to_dict()
    forbidden = (
        "causal_credit",
        "credited_to_action",
        "functional_action",
        "necessary",
        "backup",
        "substitution",
        "synergy",
        "hidden_target",
        "expected_credit",
        "oracle_action",
    )
    serialized = canonical_bytes(document).decode("utf-8")
    leaks = [label for label in forbidden if label in serialized] if enforce_participant_labels else []
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
    tape_term_values: list[dict[str, dict[int, float]]],
    term_candidate_steps: dict[str, tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, float]]:
    run_id = base_run_id + ":stochastic-credit-discovery"
    facts: list[dict[str, Any]] = []
    conditional_supports: dict[tuple[int, int], str] = {}
    conditional_credits: dict[int, list[float]] = defaultdict(list)
    for realization_index, term_values in enumerate(tape_term_values):
        term_credit_by_step: dict[int, list[float]] = defaultdict(list)
        for term_id in TERM_IDS:
            group = tuple(sorted(term_candidate_steps[term_id]))
            values = {int(mask): float(value) for mask, value in term_values[term_id].items()}
            replay_supports: dict[int, str] = {}
            for mask, value in sorted(values.items()):
                support_id = f"{run_id}:tape-{realization_index}:{term_id}:coalition-{mask}"
                replay_supports[mask] = support_id
                facts.append(
                    _fact(
                        f"{support_id}:replay",
                        support_id,
                        u={
                            "kind": "registered_source",
                            "source_identity": f"coalition-{realization_index}-{term_id}-{mask}",
                            "base_graph_sha256": base_graph_sha256,
                            "candidate_steps": list(group),
                            "coalition_mask": mask,
                            "realization_index": realization_index,
                        },
                        tau={"operation": "restore-matched-tape-and-replay-native-long-chain"},
                        omega={"concrete_occurrence_id": f"{support_id}:occurrence", "coalition_mask": mask},
                        z={"kind": "OutcomeSupport", "payload": {"term_identity": term_id, "value": value}},
                        rho="matched_coalition_assignment",
                    )
                )
            term_credit = _shapley(values, len(group))
            for index, step in enumerate(group):
                credit_support = f"{run_id}:tape-{realization_index}:{term_id}:step-{step}:credit"
                term_credit_by_step[step].append(term_credit[index])
                for mask in range(1 << len(group)):
                    if mask & (1 << index):
                        continue
                    for side, source_mask in (("without", mask), ("with", mask | (1 << index))):
                        facts.append(
                            _fact(
                                f"{credit_support}:marginal-{mask}:{side}",
                                credit_support,
                                u={"kind": "generated_origin", "prior_support_id": replay_supports[source_mask]},
                                tau={"operation": "form-conditional-shapley-credit"},
                                omega={"concrete_occurrence_id": f"{credit_support}:occurrence", "candidate_step": step},
                                z={"kind": "OutcomeSupport", "payload": {"conditional_term_credit": term_credit[index]}},
                                rho=f"coalition_{side}_candidate",
                            )
                        )
        for step in candidates:
            value = float(math.fsum(term_credit_by_step.get(step, [0.0])))
            support_id = f"{run_id}:tape-{realization_index}:step-{step}:conditional-credit"
            conditional_supports[(realization_index, step)] = support_id
            conditional_credits[step].append(value)
            facts.append(
                _fact(
                    f"{support_id}:result",
                    support_id,
                    u={"kind": "registered_source", "source_identity": f"conditional-credit-ledger-{realization_index}-{step}"},
                    tau={"operation": "sum-term-conditional-credit"},
                    omega={"concrete_occurrence_id": f"{support_id}:occurrence", "candidate_step": step},
                    z={"kind": "OutcomeSupport", "payload": {"candidate_step": step, "conditional_credit": value}},
                    rho="conditional_term_credit_ledger",
                )
            )
    expected = {
        str(step): float(math.fsum(values) / len(values))
        for step, values in conditional_credits.items()
    }
    for step in candidates:
        result_id = f"{run_id}:step-{step}:expected-credit"
        for realization_index in range(len(tape_term_values)):
            facts.append(
                _fact(
                    f"{result_id}:input-{realization_index}",
                    result_id,
                    u={"kind": "generated_origin", "prior_support_id": conditional_supports[(realization_index, step)]},
                    tau={"operation": "average-conditional-credit-across-tapes"},
                    omega={"concrete_occurrence_id": f"{result_id}:occurrence", "candidate_step": step},
                    z={"kind": "OutcomeSupport", "payload": {"candidate_step": step, "mean_credit": expected[str(step)]}},
                    rho="conditional_credit_sample",
                )
            )
    bundle = {
        "schema_version": "stochastic-credit-discovery-native-atomic-facts-v1",
        "execution_run_id": run_id,
        "facts": facts,
    }
    return bundle, expected


def credit_metrics(
    reference: dict[str, float],
    candidate: dict[str, float],
    passenger_steps: tuple[int, ...],
) -> dict[str, Any]:
    keys = sorted(set(reference) | set(candidate), key=int)
    errors = [abs(reference.get(key, 0.0) - candidate.get(key, 0.0)) for key in keys]
    signs = [sign(reference.get(key, 0.0)) == sign(candidate.get(key, 0.0)) for key in keys]
    passenger_zero = [sign(candidate.get(str(step), 0.0)) == 0 for step in passenger_steps]
    return {
        "max_abs_error": max(errors, default=0.0),
        "mae": sum(errors) / len(errors) if errors else 0.0,
        "sign_accuracy": sum(signs) / len(signs) if signs else 1.0,
        "passenger_zero_accuracy": sum(passenger_zero) / len(passenger_zero) if passenger_zero else 1.0,
        "exact_within_1e_12": max(errors, default=0.0) <= 1e-12,
    }


__all__ = [
    "ACTION_COUNT",
    "CUE_BITS",
    "DOMAIN_SCOPE_ID",
    "EVENT_CODES",
    "FUNCTIONAL_CODES",
    "HORIZON",
    "PASSENGER_CODES",
    "STOCHASTIC_CHANNEL_COUNT",
    "TERM_IDS",
    "VISIBLE_EVENT_CODE_COUNT",
    "ExogenousInput",
    "StochasticChainState",
    "StochasticEpisodeResult",
    "StochasticTape",
    "ReplayCost",
    "build_atomic_execution",
    "build_credit_discovery_atomic_execution",
    "compile_and_validate_canonical_gfg",
    "credit_metrics",
    "deterministic_behavior_actions",
    "exact_conditional_credit",
    "execute_episode",
    "factorized_conditional_credit",
    "hidden_target_actions",
    "initial_state",
    "make_episode_spec",
    "make_stochastic_tape",
    "object_sha256",
    "permute_stochastic_bindings",
    "retrieve_candidates",
    "retrieve_term_candidate_steps",
    "rewire_term_candidate_steps",
    "sign",
    "transition_state",
]
