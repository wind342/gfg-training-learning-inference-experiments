from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .analysis import canonical, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def rho(left: list[float], right: list[float]) -> float:
    return float(spearmanr(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)).statistic)


def check(report_root: Path) -> dict[str, Any]:
    scope = read_json(report_root / "ANALYSIS_MANIFEST.json")
    summary = read_json(report_root / "SUPPORT_REALLOCATION_RESULTS.json")
    sources = read_rows(report_root / "SOURCE_SECTION_LEDGER.jsonl.gz")
    sections = read_rows(report_root / "SECTION_UPDATE_LEDGER.jsonl.gz")
    rows = read_rows(report_root / "TARGET_SUPPORT_TRANSITION_LEDGER.jsonl.gz")
    checks: list[str] = []

    require(scope["claim_scope"] == "EXPLORATORY_POST_HOC_CAUSAL_DESCRIPTION_NOT_ADVANCE_PREDICTION", "CLAIM_SCOPE")
    for path, digest in scope["source_hashes"].items():
        require(sha256_file(Path(path)) == digest, f"SOURCE_HASH:{path}")
    require(len(sources) == len(sections) == 72, "SECTION_COUNT")
    require(len(rows) == 1656, "TARGET_GROUP_COUNT")
    checks.append("source_hashes_and_counts")

    section_paths = {row["section_id"]: Path(row["section_npz"]) for row in sources}
    cache: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        saved_hash = row["row_content_sha256"]
        material = dict(row)
        del material["row_content_sha256"]
        require(hashlib.sha256(canonical(material).encode("utf-8")).hexdigest() == saved_hash, "ROW_HASH")
        section_id = row["section_id"]
        if section_id not in cache:
            with np.load(section_paths[section_id], allow_pickle=False) as data:
                cache[section_id] = {
                    "groups": np.asarray(data["groups"], dtype=np.int64),
                    "predictions": np.asarray(data["all_predictions"], dtype=np.int64),
                    "allocation0": np.asarray(data["support_allocation"][1].T, dtype=np.float64),
                    "allocation1": np.asarray(data["support_allocation"][-1].T, dtype=np.float64),
                    "necessity0": np.asarray(data["necessity"][1].T, dtype=np.float64),
                    "necessity1": np.asarray(data["necessity"][-1].T, dtype=np.float64),
                }
        value = cache[section_id]
        group = int(row["target_group"])
        a0 = np.asarray([row["allocation_before"][name] for name in ("h0_attn", "h0_mlp", "h1_attn", "h1_mlp")])
        a1 = np.asarray([row["allocation_after"][name] for name in ("h0_attn", "h0_mlp", "h1_attn", "h1_mlp")])
        require(np.array_equal(a0, value["allocation0"][group]), "ALLOCATION0")
        require(np.array_equal(a1, value["allocation1"][group]), "ALLOCATION1")
        n0 = np.asarray([row["necessity_before"][name] for name in ("h0_attn", "h0_mlp", "h1_attn", "h1_mlp")])
        n1 = np.asarray([row["necessity_after"][name] for name in ("h0_attn", "h0_mlp", "h1_attn", "h1_mlp")])
        require(np.array_equal(n0, value["necessity0"][group]), "NECESSITY0")
        require(np.array_equal(n1, value["necessity1"][group]), "NECESSITY1")
        selector = value["groups"] == group
        c0 = float(np.mean(value["predictions"][1, 0, selector] == group))
        c1 = float(np.mean(value["predictions"][-1, 0, selector] == group))
        require(close(c0, row["capability_before"]) and close(c1, row["capability_after"]), "CAPABILITY")
        if row["allocation_valid"]:
            require(close(0.5 * float(np.sum(np.abs(a1 - a0))), row["reallocation_magnitude"]), "REALLOCATION")
    checks.append("all_rows_reconstructed_from_npz")

    valid = [row for row in rows if row["allocation_valid"]]
    require(len(valid) == summary["valid_allocation_transition_count"] == 1636, "VALID_COUNT")
    require(sum(row["primary_support_switched"] for row in valid) == summary["primary_support_switch_count"], "SWITCH_COUNT")
    relations = summary["relations"]
    require(close(rho([row["update_l2_norm"] for row in sections], [row["mean_reallocation_magnitude"] for row in sections]), relations["update_magnitude_to_section_mean_reallocation_rho"]), "UPDATE_REALLOCATION_RHO")
    require(close(rho([row["reallocation_magnitude"] for row in valid], [abs(row["capability_delta"]) for row in valid]), relations["reallocation_to_absolute_capability_change_rho"]), "REALLOCATION_CAPABILITY_RHO")
    finite_effective = [row for row in valid if row["effective_support_delta"] is not None]
    require(close(rho([row["effective_support_delta"] for row in finite_effective], [row["capability_delta"] for row in finite_effective]), relations["effective_support_change_to_capability_change_rho"]), "EFFECTIVE_CAPABILITY_RHO")
    checks.append("summary_relations_recomputed")

    result = {
        "schema": "nanogpt-support-reallocation-independent-check-v1",
        "status": "PASS",
        "checks": checks,
        "check_count": len(checks),
        "section_count": len(sections),
        "target_group_transition_count": len(rows),
        "future_leakage_claimed_as_prediction": False,
    }
    write_json(report_root / "INDEPENDENT_CHECK.json", result)
    return result


__all__ = ["check"]
