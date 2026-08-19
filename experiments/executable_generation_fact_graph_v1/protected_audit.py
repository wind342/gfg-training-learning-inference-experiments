from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TREES = {
    "src/generation_relation_core": "03fbdce13249f84abe9d8fb605da31cdc36eda27",
    "protocol/core_v3": "0b4a2608864e771ebca7cdbfad95aabaed2d0723",
    "tests/core": "280cb44d592ae48d986719638980c11e57aab1f9",
    "experiments/signal_multistage_generated_origin_v1": (
        "9871b14722548d503324762b6dc3a222828168d0"
    ),
    "experiments/inter_fact_relations_v0": (
        "fccb595dfc0a8c7272f3e6e2af6937a57f8168b7"
    ),
    "experiments/inter_fact_relations_v0_hardening_scale_v1": (
        "587ae72e94102fe4249eb1c38fa5b54ba9e78633"
    ),
    "experiments/order_refund_freeze_inter_fact_relations_v1": (
        "68f8db905678f47b5ddc02637b175b7270556e33"
    ),
    "experiments/signed_generation_algebra_v1": (
        "8cfc18a206bde2ceecca5ccee29a18f74d7b2ea1"
    ),
    "manuscript": "3652bd5aef4cfc14f1bf5f1ea08e0f42cd523d47",
    "claims": "f6b95d5ae0d8fdb77da13f9dbd48ecaed8af27ff",
    "claim_atlas": "85203984d514bf6647f41acd8848fced47bb8bff",
}


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def protected_path_audit() -> dict[str, Any]:
    actual = {
        path: _git("rev-parse", f"HEAD:{path}")
        for path in EXPECTED_TREES
    }
    status = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *EXPECTED_TREES,
    )
    comparisons = {
        path: {
            "expected_tree_sha1": EXPECTED_TREES[path],
            "actual_tree_sha1": actual[path],
            "unchanged": EXPECTED_TREES[path] == actual[path],
        }
        for path in EXPECTED_TREES
    }
    gates = {
        "all_tree_hashes_exact": all(
            row["unchanged"] for row in comparisons.values()
        ),
        "no_protected_worktree_changes": not status,
    }
    return {
        "schema_version": "protected-path-audit-v1",
        "head": _git("rev-parse", "HEAD"),
        "comparisons": comparisons,
        "protected_status_entries": status.splitlines() if status else [],
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
