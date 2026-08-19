from __future__ import annotations

from .runtime import (
    HORIZON,
    build_atomic_execution,
    credit_metrics,
    deterministic_behavior_actions,
    exact_conditional_credit,
    execute_episode,
    factorized_conditional_credit,
    make_episode_spec,
    make_stochastic_tape,
    permute_stochastic_bindings,
    retrieve_candidates,
    retrieve_term_candidate_steps,
    transition_state,
    initial_state,
)


def test_counter_addressed_stochastic_input_changes_native_transition() -> None:
    spec = make_episode_spec(41, 2)
    state = initial_state(spec)
    left = transition_state(state, event_code=0, action=0, exogenous_value=11)
    right = transition_state(state, event_code=0, action=0, exogenous_value=12)
    assert left.stochastic_values != right.stochastic_values
    assert left.action_values == right.action_values


def test_atomic_capture_retains_exact_action_ancestry_and_random_bindings() -> None:
    spec = make_episode_spec(42, 3)
    actions = deterministic_behavior_actions(42, 3)
    tape = make_stochastic_tape(seed=42, episode_id=3, realization_id=7)
    bundle, _, metadata = build_atomic_execution(spec, actions, tape)
    assert retrieve_candidates(bundle, metadata) == spec.ancestry_positions
    assert sum(
        fact["coordinates"]["rho"]["role"] == "exogenous_stochastic_input"
        for fact in bundle["facts"]
    ) == HORIZON * 8
    assert len(metadata["stochastic_draw_fact_by_step"]) == HORIZON
    by_id = {fact["fact_id"]: fact for fact in bundle["facts"]}
    assert all(
        by_id[fact_id]["coordinates"]["u"]["kind"] == "generated_origin"
        for fact_id in metadata["scalar_fact_ids"]
    )


def test_factorized_conditional_credit_matches_complete_reference() -> None:
    spec = make_episode_spec(43, 4)
    actions = deterministic_behavior_actions(43, 4)
    tape = make_stochastic_tape(seed=43, episode_id=4, realization_id=2)
    bundle, _, metadata = build_atomic_execution(spec, actions, tape)
    candidates = retrieve_candidates(bundle, metadata)
    term_structure = retrieve_term_candidate_steps(bundle, metadata)
    exact = exact_conditional_credit(spec, actions, tape, candidates, prefix_cache=False)
    optimized = factorized_conditional_credit(
        spec, actions, tape, candidates, term_structure
    )
    assert credit_metrics(exact["credits"], optimized["credits"], spec.passenger_positions)[
        "exact_within_1e_12"
    ]
    assert optimized["cost"]["native_transitions"] < exact["cost"]["native_transitions"]


def test_binding_permutation_preserves_multiset_but_changes_realized_state() -> None:
    spec = make_episode_spec(44, 5)
    actions = deterministic_behavior_actions(44, 5)
    tape = make_stochastic_tape(seed=44, episode_id=5, realization_id=1)
    permuted = permute_stochastic_bindings(tape, salt=44005)
    assert sorted((row.source_identity, row.value) for row in tape) == sorted(
        (row.source_identity, row.value) for row in permuted
    )
    assert tuple(row.source_identity for row in tape) != tuple(
        row.source_identity for row in permuted
    )
    assert execute_episode(spec, actions, tape).final_state != execute_episode(
        spec, actions, permuted
    ).final_state
