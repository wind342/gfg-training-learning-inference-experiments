from __future__ import annotations

from experiments.provenance_semiring_projection_v1.src.lower_strictness import evaluate_lower_strictness


def test_lower_and_joint_projection_strictness_have_real_witnesses() -> None:
    lower, joint = evaluate_lower_strictness()
    assert lower["status"] == "LOWER_PROJECTION_STRICTNESS_SUPPORTED"
    assert lower["real_execution_count"] == 6
    assert all(lower["requirements"].values())
    assert all(pair["native_candidate_exact_both_sides"] for pair in lower["pairs"])
    assert joint["status"] == "JOINT_LOWER_PROJECTION_STRICTNESS_SUPPORTED"
    assert joint["joint_witness_count"] >= 1
