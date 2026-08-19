from .runtime import (
    build_atomic_execution,
    compile_and_validate_canonical_gfg,
    credit_metrics,
    deterministic_behavior_actions,
    exact_scalar_credit,
    factorized_credit,
    make_episode_spec,
    retrieve_candidates,
    retrieve_term_candidate_steps,
    rewire_term_candidate_steps,
)


def fixture():
    spec = make_episode_spec(71, 3)
    actions = deterministic_behavior_actions(71, 3)
    bundle, result, metadata = build_atomic_execution(spec, actions)
    return spec, actions, bundle, result, metadata


def test_long_chain_candidate_retrieval_is_exact():
    spec, _actions, bundle, _result, metadata = fixture()
    assert retrieve_candidates(bundle, metadata) == spec.ancestry_positions
    assert max(spec.ancestry_positions) <= 27


def test_factorized_credit_matches_full_exact():
    spec, actions, bundle, _result, metadata = fixture()
    candidates = retrieve_candidates(bundle, metadata)
    groups = retrieve_term_candidate_steps(bundle, metadata)
    exact = exact_scalar_credit(spec, actions, candidates, prefix_cache=False)
    optimized = factorized_credit(spec, actions, candidates, groups)
    metrics = credit_metrics(exact["credits"], optimized["credits"], spec.passenger_positions)
    assert metrics["exact_within_1e_12"]
    assert metrics["passenger_zero_accuracy"] == 1.0


def test_rewired_structure_is_detected_on_frozen_fixture():
    spec, actions, bundle, _result, metadata = fixture()
    candidates = retrieve_candidates(bundle, metadata)
    groups = retrieve_term_candidate_steps(bundle, metadata)
    exact = exact_scalar_credit(spec, actions, candidates, prefix_cache=False)
    rewired = factorized_credit(
        spec,
        actions,
        candidates,
        rewire_term_candidate_steps(groups, candidates, seed=9003),
    )
    assert not credit_metrics(exact["credits"], rewired["credits"], spec.passenger_positions)["exact_within_1e_12"]


def test_canonical_core_and_gfg_validation_pass():
    _spec, _actions, bundle, _result, _metadata = fixture()
    validation = compile_and_validate_canonical_gfg(bundle, generator_name="long-chain-unit-test")
    assert validation["status"] == "PASS"
    assert validation["coordinate_mapping_exact"]
    assert validation["relation_mapping_exact"]
    assert not validation["forbidden_label_hits"]
