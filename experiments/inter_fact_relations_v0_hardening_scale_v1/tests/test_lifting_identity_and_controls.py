from __future__ import annotations

from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.multi_fact_occurrence import (
    run as run_multi_fact,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.scenarios.same_output_relation_identity import (
    run as run_identity,
)
from experiments.inter_fact_relations_v0_hardening_scale_v1.src.negative_controls import (
    run_negative_controls,
)


def test_multi_fact_dependency_is_selective() -> None:
    result = run_multi_fact()
    assert result["status"] == "PASS"
    assert result["occurrence_order_lift_pair_count"] == 3
    assert len(result["fact_specific_dependency_pairs"]) == 1
    assert result["false_dependency_pairs"] == []


def test_run_identity_is_not_semantic_identity() -> None:
    result = run_identity()
    assert result["ordinary_output_equal"] is True
    assert result["semantic_fact_projection_equal"] is True
    assert result["core_fact_id_equal"] is True
    assert result["concrete_run_scoped_gamma_equal"] is False
    assert (
        result["exact_gamma_equality_status"]
        == "EXACT_GAMMA_EQUALITY_NOT_ESTABLISHED"
    )
    assert result["relation_graph_different"] is True


def test_thirty_one_controls_fail_closed_once() -> None:
    result = run_negative_controls()
    assert result["status"] == "PASS"
    assert result["control_count"] == 31
    assert result["passed_count"] == 31
    assert all(row["execution_count"] == 1 for row in result["controls"])
    assert all(row["partial_success"] is False for row in result["controls"])
    adjacency = next(
        row
        for row in result["controls"]
        if row["mutation_id"]
        == "mutation-31-program-order-same-count-wrong-adjacency"
    )
    assert (
        adjacency["observed_reason_code"]
        == "PROGRAM_ORDER_ADJACENCY_SET_MISMATCH"
    )
    assert adjacency["auto_repaired"] is False
