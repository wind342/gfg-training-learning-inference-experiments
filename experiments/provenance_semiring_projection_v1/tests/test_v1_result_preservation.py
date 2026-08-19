from __future__ import annotations

from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.v1_preservation import (
    build_v1_result_preservation,
)


def test_pr19_p1_p2_and_protected_trees_are_preserved_from_frozen_head() -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    result = build_v1_result_preservation(experiment_root.parents[1], experiment_root / "artifacts")
    assert result["status"] == "PR19_V1_RESULTS_PRESERVED"
    assert result["blocking_reasons"] == []
    assert all(result["gates"].values())
    assert set(item["dimension"] for item in result["p2_witnesses"].values()) == {
        "physical occurrence structure",
        "evidence",
        "environment",
        "disposition",
        "operation result",
    }

