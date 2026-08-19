from __future__ import annotations

from experiments.provenance_semiring_projection_v1.src.nx_strictness import evaluate_nx_strictness


def test_five_real_pair_dimensions_witness_noninjectivity() -> None:
    result, reverse = evaluate_nx_strictness()
    assert result["status"] == "STRICTNESS_SUPPORTED"
    assert result["actual_pair_count"] == 5
    assert result["real_execution_count"] == 10
    assert {pair["dimension"] for pair in result["pairs"]} == {
        "physical occurrence structure",
        "evidence",
        "environment",
        "disposition",
        "operation result",
    }
    assert all(pair["supported"] for pair in result["pairs"])
    assert reverse["status"] == "NON_INJECTIVITY_WITNESSED"
    assert reverse["fibers_with_multiple_gamma"] == 5
