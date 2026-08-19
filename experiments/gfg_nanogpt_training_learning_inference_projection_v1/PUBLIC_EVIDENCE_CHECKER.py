from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

import numpy as np


COMPONENTS = ("h0.attn", "h0.mlp", "h1.attn", "h1.mlp")
COMPONENT_PAIRS = tuple(
    (left, right)
    for index, left in enumerate(COMPONENTS)
    for right in COMPONENTS[index + 1 :]
)
FLOAT_TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _blocks(run_root: Path) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    with closing(sqlite3.connect(run_root / "training_learning_inference_gfg.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT stage,payload_zlib FROM graph_blocks ORDER BY block_ordinal"):
            result.append((str(row["stage"]), json.loads(zlib.decompress(row["payload_zlib"]))))
    return result


def _load(run_root: Path, block: dict[str, Any], role: str, gate: list[str] | None = None) -> np.ndarray:
    candidates = [value for value in block["objects"] if value["role"] == role]
    if gate is not None:
        candidates = [value for value in candidates if value["payload"].get("gate_components") == gate]
    if len(candidates) != 1:
        raise RuntimeError(f"PUBLIC_EVIDENCE_OBJECT_NOT_UNIQUE:{role}:{gate}")
    return np.load(run_root / candidates[0]["payload"]["locator"], allow_pickle=False)


def _validate_archive(graph_root: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    archive = _read(graph_root / "ARCHIVE_MANIFEST.json")
    if archive.get("status") != "PASS" or archive.get("entry_count") != 13:
        failures.append("archive_status_or_entry_count")
    if len(archive.get("entries", [])) != 13:
        failures.append("archive_entry_list_length")
    for entry in archive.get("entries", []):
        run_root = graph_root / str(entry["entry_id"])
        manifest_path = run_root / "GFG_MANIFEST.json"
        validation_path = run_root / "GFG_VALIDATION.json"
        if not manifest_path.is_file() or _sha256(manifest_path) != entry.get("manifest_sha256"):
            failures.append(f"manifest_hash:{entry['entry_id']}")
            continue
        if not validation_path.is_file() or _sha256(validation_path) != entry.get("validation_sha256"):
            failures.append(f"validation_hash:{entry['entry_id']}")
        validation = _read(validation_path)
        if entry.get("validation_status") != "PASS" or validation.get("status") != "PASS":
            failures.append(f"validation_status:{entry['entry_id']}")
        manifest = _read(manifest_path)
        database = run_root / str(manifest["database"])
        if not database.is_file() or _sha256(database) != manifest.get("database_sha256"):
            failures.append(f"database_hash:{entry['entry_id']}")
            continue
        for _, block in _blocks(run_root):
            for value in block.get("objects", []):
                payload = value.get("payload", {})
                locator = payload.get("locator")
                expected = payload.get("file_sha256")
                if locator is None:
                    continue
                tensor_path = run_root / str(locator)
                if not tensor_path.is_file() or (expected is not None and _sha256(tensor_path) != expected):
                    failures.append(f"tensor_hash:{entry['entry_id']}:{locator}")
    return archive, failures


def _compute(graph_root: Path, archive: dict[str, Any]) -> dict[str, Any]:
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
            cell_rms: list[float] = []
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
    return {
        "status": "PASS"
        if len(formed) == 13
        and len(rows) == 52
        and all(row["logit_level_nonadditive_pair_group_count_at_1e_6"] == 138 for row in rows)
        and all(row["query_group_profile_distance_min"] > 1e-6 for row in formed)
        else "FAIL",
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
        "rows": rows,
    }


def _compare(actual: Any, expected: Any, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(expected, dict):
        for key, expected_value in expected.items():
            if key in {"schema", "purpose", "method", "post_primary_hardening", "primary_results_changed", "audit_implementation_sha256"}:
                continue
            if key not in actual:
                failures.append(f"missing:{path}.{key}")
            else:
                failures.extend(_compare(actual[key], expected_value, f"{path}.{key}"))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            failures.append(f"length:{path}")
        else:
            for index, expected_value in enumerate(expected):
                failures.extend(_compare(actual[index], expected_value, f"{path}[{index}]"))
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > FLOAT_TOLERANCE:
            failures.append(f"value:{path}")
    elif actual != expected:
        failures.append(f"value:{path}")
    return failures


def check(bundle_root: Path) -> dict[str, Any]:
    graph_root = bundle_root / "gfg"
    expected_path = bundle_root / "STRICT_LOGIT_LEVEL_AUDIT.json"
    archive, integrity_failures = _validate_archive(graph_root)
    actual = _compute(graph_root, archive) if not integrity_failures else {}
    expected = _read(expected_path) if expected_path.is_file() else {}
    comparison_failures = _compare(actual, expected) if actual and expected else ["expected_audit_missing"]
    checks = {
        "archive_integrity": not integrity_failures,
        "strict_logit_level_recomputation": bool(actual) and actual.get("status") == "PASS",
        "frozen_result_match": not comparison_failures,
    }
    return {
        "schema": "nanogpt-training-learning-inference-public-evidence-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "integrity_failures": integrity_failures,
        "comparison_failures": comparison_failures,
        "recomputed": actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.bundle_root.resolve())
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
