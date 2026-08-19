from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.provenance_semiring_projection_v1.src.v1_preservation import (
    FROZEN_PR19_HEAD,
    build_v1_result_preservation,
)


def test_pr19_p1_p2_and_protected_trees_are_preserved_from_frozen_head() -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = experiment_root.parents[1]
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{FROZEN_PR19_HEAD}^{{commit}}"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not available:
        pytest.skip("GF-P02 PR19 source-history object is not part of the companion clone")
    result = build_v1_result_preservation(repo_root, experiment_root / "artifacts")
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
