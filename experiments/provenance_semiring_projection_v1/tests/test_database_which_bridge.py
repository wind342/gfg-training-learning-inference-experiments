from __future__ import annotations

from pathlib import Path

from experiments.provenance_semiring_projection_v1.src.database_which_bridge import evaluate_existing_database_which_bridge


def test_existing_database_native_core_and_vars_nx_are_exact() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    bridge, hierarchy = evaluate_existing_database_which_bridge(repo_root)
    assert bridge["status"] == "THREE_WAY_EXACT_SUPPORTED"
    assert bridge["existing_native_candidate_all_records_exact"] is True
    assert bridge["existing_candidate_record_count"] == 112
    assert bridge["existing_native_record_count"] == 112
    assert bridge["which_output_comparison_count"] >= 1
    assert all(item["three_way_exact"] for item in bridge["comparisons"])
    assert hierarchy["status"] == "DATABASE_WHICH_AS_NX_VARIABLE_PROJECTION_SUPPORTED"
