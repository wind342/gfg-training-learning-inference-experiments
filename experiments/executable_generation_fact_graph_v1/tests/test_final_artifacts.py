from __future__ import annotations

import json
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _load(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def test_v1_final_status_and_claim_boundary() -> None:
    result = _load("v1_final_result.json")
    assert (
        result["final_status"]
        == "EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED"
    )
    assert (
        result["failure_reason"]
        == "PURE_FACT_VERTEX_MODEL_CANNOT_PRESERVE_ALL_NATIVE_"
        "PRIMITIVE_RELATION_ENDPOINTS"
    )
    assert not result["claim_boundary"][
        "atomic_generation_facts_invalidated"
    ]
    assert result["unmappable_primitive_relation_count"] == 55


def test_signal_and_signed_subexperiments_remain_exact() -> None:
    signal = _load("signal_graph_result.json")
    signed = _load("signed_projection.json")
    assert signal["status"] == "PASS"
    assert signal["path_count"] == 2880
    assert signal["raw_source_count"] == 197
    assert signal["path_multiset_exact"]
    assert signed["status"] == "PASS"
    assert signed["execution_count"] == 8


def test_negative_controls_are_unique_and_fail_closed() -> None:
    controls = _load("negative_controls.json")
    assert controls["status"] == "PASS"
    assert controls["control_count"] == 48
    assert controls["detected_count"] == 48
    reason_codes = [row["reason_code"] for row in controls["controls"]]
    assert len(reason_codes) == len(set(reason_codes))
    assert all(row["execution_count"] == 1 for row in controls["controls"])
    assert not any(
        row["automatic_repair_performed"] for row in controls["controls"]
    )


def test_protected_trees_and_prohibited_actions_are_unchanged() -> None:
    protected = _load("protected_path_audit.json")
    census = _load("endpoint_type_census.json")
    assert protected["status"] == "PASS"
    assert all(protected["gates"].values())
    assert not any(census["prohibited_action_counts"].values())
