from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "gfg_rl_selective_positive_feedback_dose_recovery_v1"


def read_json(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def sha256(name: str) -> str:
    return hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()


def test_frozen_contract_files_are_unchanged() -> None:
    freeze = read_json("CONTRACT_FREEZE.json")
    for name, expected in freeze["files"].items():
        assert sha256(name) == expected


def test_formal_and_independent_verdicts_pass() -> None:
    formal = read_json("FORMAL_RESULT_SUMMARY.json")
    independent = read_json("INDEPENDENT_CHECK_SUMMARY.json")
    assert formal["scientific_status"] == "SUPPORTED"
    assert formal["formal_seed_count"] == 12
    assert all(formal["decision_gates"].values())
    assert independent["status"] == "PASS"
    assert independent["full_native_replays"] == 12
    assert independent["independent_recalculation"]["all_independently_recomputed_gates_pass"]


def test_dose_endpoint_has_strict_support_and_margin_order() -> None:
    result = read_json("TIME_COURSE_ANALYSIS.json")
    names = ("balanced", "mild", "high", "exclusive")
    endpoints = result["dose_endpoint_means"]
    support = [endpoints[name]["task0_support_share"] for name in names]
    margin = [endpoints[name]["unreinforced_mean_margin"] for name in names]
    assert support == sorted(support)
    assert margin == sorted(margin, reverse=True)


def test_recovery_exposes_the_bidirectional_tradeoff() -> None:
    result = read_json("TIME_COURSE_ANALYSIS.json")
    rebalance = result["rebalance_recovery_means"]
    repair = result["repair_recovery_means"]
    assert rebalance[-1]["unreinforced_accuracy"] > rebalance[0]["unreinforced_accuracy"]
    assert rebalance[-1]["task0_accuracy"] == 1.0
    assert repair[-1]["unreinforced_accuracy"] > repair[0]["unreinforced_accuracy"]
    assert repair[-1]["task0_accuracy"] < repair[0]["task0_accuracy"]
