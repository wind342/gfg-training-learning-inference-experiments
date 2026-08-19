from __future__ import annotations

import json
from pathlib import Path


def _artifact(name: str) -> dict:
    root = Path(__file__).resolve().parents[1] / "artifacts"
    return json.loads((root / name).read_text(encoding="utf-8"))


def test_hierarchy_v2_separates_algebraic_and_task_levels() -> None:
    hierarchy = _artifact("two_level_unification_hierarchy_v2.json")
    assert hierarchy["status"] == "TWO_LEVEL_FORMAL_HIERARCHY_SUPPORTED"
    levels = {str(item["level"]): item for item in hierarchy["levels"]}
    assert levels["2A"]["domains"] == ["bag N", "Boolean B", "PosBool(X)"]
    assert levels["2B"]["domains"] == [
        "flat source-support view",
        "Vars(N[X])",
        "existing Database which-lineage",
    ]
    assert "homomorphic" in hierarchy["forbidden_conflation"]


def test_unification_result_lists_only_established_arrows() -> None:
    result = _artifact("unification_of_unification_result_v2.json")
    assert result["status"] == "UNIFICATION_OF_UNIFICATION_FORMAL_BOUNDARY_SUPPORTED"
    assert len(result["established_arrows"]) == 6
    assert all(
        "Why" not in arrow and "Trio" not in arrow and "Which(X)" not in arrow
        for arrow in result["established_arrows"]
    )
