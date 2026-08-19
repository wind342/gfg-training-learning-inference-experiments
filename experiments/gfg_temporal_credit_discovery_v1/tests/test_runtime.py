from __future__ import annotations

import random

from experiments.gfg_temporal_credit_discovery_v1.runtime import (
    build_base_gfg,
    candidate_metrics,
    deterministic_behavior_actions,
    exact_shapley_credits,
    execute_episode,
    make_episode_spec,
    retrieve_formation_candidates,
    rewire_graph,
    terminal_consequence_only,
    validate_base_gfg,
)


def test_compact_replay_matches_full_native_execution() -> None:
    for episode_id in range(50):
        spec = make_episode_spec(1729, episode_id)
        actions = deterministic_behavior_actions(1729, episode_id)
        assert terminal_consequence_only(spec, actions) == execute_episode(spec, actions).consequence
        for step in random.Random(episode_id).sample(range(64), 5):
            changed = list(actions)
            changed[step] = 1 - changed[step]
            changed = tuple(changed)
            assert terminal_consequence_only(spec, changed) == execute_episode(spec, changed).consequence


def test_gfg_retrieval_preserves_truth_and_passengers_without_credit_labels() -> None:
    spec = make_episode_spec(2718, 10)
    actions = deterministic_behavior_actions(2718, 10)
    result = execute_episode(spec, actions)
    graph = build_base_gfg(spec, actions, result)
    assert validate_base_gfg(graph)["status"] == "PASS"
    candidates = retrieve_formation_candidates(graph)
    assert candidates == spec.ancestry_positions
    assert candidate_metrics(candidates, spec.functional_positions) == {
        "tp": 6,
        "fp": 3,
        "fn": 0,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
    }
    assert not {"credited_to_action", "causes_consequence"} & {edge["edge_kind"] for edge in graph["edges"]}


def test_matched_forks_remove_passengers_and_equal_hidden_oracle_credit() -> None:
    spec = make_episode_spec(31415, 7)
    actions = deterministic_behavior_actions(31415, 7)
    graph = build_base_gfg(spec, actions, execute_episode(spec, actions))
    candidates = retrieve_formation_candidates(graph)
    gfg_credit, _ = exact_shapley_credits(spec, actions, candidates)
    oracle_credit, _ = exact_shapley_credits(spec, actions, spec.functional_positions)
    for step in spec.passenger_positions:
        assert abs(gfg_credit[step]) <= 1e-12
    for step in spec.functional_positions:
        assert abs(gfg_credit[step] - oracle_credit[step]) <= 1e-12


def test_rewired_graph_does_not_preserve_candidates() -> None:
    spec = make_episode_spec(57721, 3)
    actions = deterministic_behavior_actions(57721, 3)
    graph = build_base_gfg(spec, actions, execute_episode(spec, actions))
    rewired = rewire_graph(graph, 99)
    assert validate_base_gfg(rewired)["status"] == "PASS"
    assert retrieve_formation_candidates(rewired) != retrieve_formation_candidates(graph)
