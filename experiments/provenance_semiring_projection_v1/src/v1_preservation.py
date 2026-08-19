from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


FROZEN_PR19_HEAD = "f20ff57501b754b111be893565092c0e107c8b73"
ARTIFACT_PREFIX = "experiments/provenance_semiring_projection_v1/artifacts"
PROTECTED_PATHS = [
    "src/generation_relation_core",
    "protocol/core_v3",
    "compat/v2",
    "tests/core",
    "experiments/database_lineage",
]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_json(repo_root: Path, name: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{FROZEN_PR19_HEAD}:{ARTIFACT_PREFIX}/{name}"],
        cwd=repo_root,
    )
    return json.loads(raw.decode("utf-8"))


def _tree(repo_root: Path, revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}:{path}"],
        cwd=repo_root,
        text=True,
    ).strip()


def build_v1_result_preservation(repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    current = {
        name: _read(artifact_root / name)
        for name in [
            "nx_exact_comparison.json",
            "nx_strictness_counterexamples.json",
            "nx_to_existing_which_lineage.json",
            "lower_projection_strictness_constructions.json",
            "joint_lower_projection_strictness.json",
        ]
    }
    frozen = {name: _frozen_json(repo_root, name) for name in current}
    current_p1 = current["nx_exact_comparison.json"]
    frozen_p1 = frozen["nx_exact_comparison.json"]
    current_p2 = current["nx_strictness_counterexamples.json"]
    frozen_p2 = frozen["nx_strictness_counterexamples.json"]
    current_w12 = next(item for item in current_p1["cases"] if item["workload_id"] == "W12")
    frozen_w12 = next(item for item in frozen_p1["cases"] if item["workload_id"] == "W12")

    current_pairs = {
        item["pair_id"]: {
            "dimension": item["dimension"],
            "supported": item["supported"],
            "nx_equal_across_pair": item["nx_equal_across_pair"],
            "ordinary_output_equal": item["ordinary_output_equal"],
            "required_difference_present": item["required_difference_present"],
            "real_execution_count": item["real_execution_count"],
        }
        for item in current_p2["pairs"]
    }
    frozen_pairs = {
        item["pair_id"]: {
            "dimension": item["dimension"],
            "supported": item["supported"],
            "nx_equal_across_pair": item["nx_equal_across_pair"],
            "ordinary_output_equal": item["ordinary_output_equal"],
            "required_difference_present": item["required_difference_present"],
            "real_execution_count": item["real_execution_count"],
        }
        for item in frozen_p2["pairs"]
    }
    protected = {
        path: {
            "frozen_tree": _tree(repo_root, FROZEN_PR19_HEAD, path),
            "current_tree": _tree(repo_root, "HEAD", path),
        }
        for path in PROTECTED_PATHS
    }
    gates = {
        "p1_status_preserved": current_p1["status"] == frozen_p1["status"] == "EXACT_SUPPORTED",
        "p1_case_count_preserved": current_p1["actual_case_count"] == frozen_p1["actual_case_count"] == 13,
        "p1_zero_mismatch_preserved": current_p1["mismatch_count"] == frozen_p1["mismatch_count"] == 0,
        "p1_zero_repair_preserved": current_p1["repair_count"] == frozen_p1["repair_count"] == 0,
        "w12_large_case_preserved": current_w12 == frozen_w12 and current_w12["exact"],
        "p2_status_preserved": current_p2["status"] == frozen_p2["status"] == "STRICTNESS_SUPPORTED",
        "p2_pair_count_preserved": current_p2["actual_pair_count"] == frozen_p2["actual_pair_count"] == 5,
        "p2_real_execution_count_preserved": current_p2["real_execution_count"] == frozen_p2["real_execution_count"] == 10,
        "p2_semantic_witnesses_preserved": current_pairs == frozen_pairs,
        "database_which_bridge_preserved": current["nx_to_existing_which_lineage.json"]["status"] == frozen["nx_to_existing_which_lineage.json"]["status"] == "THREE_WAY_EXACT_SUPPORTED",
        "lower_strictness_preserved": current["lower_projection_strictness_constructions.json"]["status"] == frozen["lower_projection_strictness_constructions.json"]["status"] == "LOWER_PROJECTION_STRICTNESS_SUPPORTED",
        "joint_lower_strictness_preserved": current["joint_lower_projection_strictness.json"]["status"] == frozen["joint_lower_projection_strictness.json"]["status"] == "JOINT_LOWER_PROJECTION_STRICTNESS_SUPPORTED",
        "protected_trees_preserved": all(item["frozen_tree"] == item["current_tree"] for item in protected.values()),
    }
    return {
        "schema_version": "v1-result-preservation-v1",
        "status": "PR19_V1_RESULTS_PRESERVED" if all(gates.values()) else "BLOCK",
        "frozen_pr19_head": FROZEN_PR19_HEAD,
        "comparison_policy": "compare frozen scientific conclusions and witness dimensions; regenerated Snapshot identities are retained as run evidence rather than required to equal an earlier commit",
        "gates": gates,
        "p2_witnesses": current_pairs,
        "protected_tree_checks": protected,
        "blocking_reasons": [name for name, passed in gates.items() if not passed],
        "automatic_repair": False,
    }

