from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import file_sha256

from .runtime import COMPONENTS, COMPONENT_PAIRS


DEFAULT_REPORT_ROOT = (
    Path(__file__).parents[1]
    / "gfg_nanogpt_cumulative_scientist_v1"
    / "reports"
    / "training_learning_inference_projection_v1"
)
DEFAULT_GRAPH_ROOT = Path(r"E:\gfg-evidence\nanogpt-training-learning-inference-projection-v1\gfg")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _blocks(run_root: Path) -> list[tuple[str, dict[str, Any]]]:
    result = []
    with sqlite3.connect(run_root / "training_learning_inference_gfg.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT stage,payload_zlib FROM graph_blocks ORDER BY block_ordinal"):
            result.append((str(row["stage"]), json.loads(zlib.decompress(row["payload_zlib"]))))
    return result


def _load(run_root: Path, block: dict[str, Any], role: str, gate: list[str] | None = None) -> np.ndarray:
    candidates = [value for value in block["objects"] if value["role"] == role]
    if gate is not None:
        candidates = [value for value in candidates if value["payload"].get("gate_components") == gate]
    require(len(candidates) == 1, f"TLI_STRICT_OBJECT_NOT_UNIQUE:{role}:{gate}")
    return np.load(run_root / candidates[0]["payload"]["locator"], allow_pickle=False)


def audit(report_root: Path, graph_root: Path) -> dict[str, Any]:
    require((report_root / "READY").exists(), "TLI_STRICT_PRIMARY_NOT_READY")
    archive = json.loads((graph_root / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    require(archive["status"] == "PASS" and archive["entry_count"] == 13, "TLI_STRICT_ARCHIVE_INVALID")
    rows: list[dict[str, Any]] = []
    for entry in archive["entries"]:
        run_root = graph_root / entry["entry_id"]
        blocks = _blocks(run_root)
        target_block = next(block for stage, block in blocks if stage == "target_mapping_and_query_scope")
        targets = _load(run_root, target_block, "derived_validation_targets")[:, -1]
        for stage, block in blocks:
            if not stage.startswith("native_inference:"):
                continue
            phase = stage.split(":", 1)[1]
            baseline = _load(run_root, block, "native_inference_logits")
            singles = {
                component: _load(run_root, block, "single_component_gate_logits", [component])
                for component in COMPONENTS
            }
            pairs = {
                pair: _load(run_root, block, "pair_component_gate_logits", list(pair))
                for pair in COMPONENT_PAIRS
            }
            support_profile = np.zeros((23, len(COMPONENTS)), dtype=np.float64)
            for group in range(23):
                mask = targets == group
                for component_index, component in enumerate(COMPONENTS):
                    delta = singles[component][mask] - baseline[mask]
                    support_profile[group, component_index] = float(
                        np.sqrt(np.mean(delta.astype(np.float64) ** 2))
                    )
            profile_distances = [
                float(np.linalg.norm(support_profile[left] - support_profile[right]))
                for left in range(23)
                for right in range(left + 1, 23)
            ]
            cell_rms = []
            for pair in COMPONENT_PAIRS:
                interaction = pairs[pair] - singles[pair[0]] - singles[pair[1]] + baseline
                for group in range(23):
                    value = interaction[targets == group]
                    cell_rms.append(float(np.sqrt(np.mean(value.astype(np.float64) ** 2))))
            rows.append(
                {
                    "entry_id": entry["entry_id"],
                    "phase": phase,
                    "logit_level_nonadditive_pair_group_count_at_1e_6": int(
                        np.count_nonzero(np.asarray(cell_rms) > 1e-6)
                    ),
                    "logit_interaction_rms_min": float(min(cell_rms)),
                    "logit_interaction_rms_median": float(np.median(cell_rms)),
                    "logit_interaction_rms_max": float(max(cell_rms)),
                    "query_group_profile_distance_min": float(min(profile_distances)),
                    "query_group_profile_distance_median": float(np.median(profile_distances)),
                    "query_group_profile_distance_max": float(max(profile_distances)),
                }
            )
    formed = [row for row in rows if row["phase"] == "formed"]
    result = {
        "schema": "nanogpt-training-learning-inference-strict-logit-level-audit-v1",
        "status": "PASS"
        if len(formed) == 13
        and all(row["logit_level_nonadditive_pair_group_count_at_1e_6"] == 138 for row in rows)
        and all(row["query_group_profile_distance_min"] > 1e-6 for row in formed)
        else "FAIL",
        "purpose": "Exclude quantile-aggregation nonlinearity as the explanation for the primary pair-combination result.",
        "post_primary_hardening": True,
        "primary_results_changed": False,
        "run_count": 13,
        "phase_count": len(rows),
        "formed_logit_level_nonadditive_pair_group_minimum": min(
            row["logit_level_nonadditive_pair_group_count_at_1e_6"] for row in formed
        ),
        "all_phase_logit_level_nonadditive_pair_group_minimum": min(
            row["logit_level_nonadditive_pair_group_count_at_1e_6"] for row in rows
        ),
        "formed_logit_interaction_rms_global_minimum": min(row["logit_interaction_rms_min"] for row in formed),
        "formed_logit_interaction_rms_median_of_run_medians": float(
            np.median([row["logit_interaction_rms_median"] for row in formed])
        ),
        "formed_logit_interaction_rms_global_maximum": max(row["logit_interaction_rms_max"] for row in formed),
        "formed_query_group_profile_distance_global_minimum": min(
            row["query_group_profile_distance_min"] for row in formed
        ),
        "formed_query_group_profile_distance_median_of_run_medians": float(
            np.median([row["query_group_profile_distance_median"] for row in formed])
        ),
        "method": "For each component pair, compute L_ab-L_a-L_b+L_0 on complete 212x24 logits before any margin, quantile, or accuracy aggregation. Query-group profiles are RMS complete-logit gate effects.",
        "rows": rows,
        "audit_implementation_sha256": file_sha256(Path(__file__)),
    }
    write_json(report_root / "STRICT_LOGIT_LEVEL_AUDIT.json", result)
    final_path = report_root / "FINAL_MANIFEST.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["strict_logit_level_audit_sha256"] = file_sha256(report_root / "STRICT_LOGIT_LEVEL_AUDIT.json")
    final["strict_logit_level_audit_status"] = result["status"]
    final["status"] = "PASS" if final["status"] == result["status"] == "PASS" else "FAIL"
    write_json(final_path, final)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    args = parser.parse_args()
    result = audit(args.report_root.resolve(), args.graph_root.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

