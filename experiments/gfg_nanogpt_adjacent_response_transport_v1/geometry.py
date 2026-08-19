from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .inventory import load_array, load_named_array, read_json, sha256_bytes, write_json


COMPONENT_PREFIXES = {
    "h0.attn": "transformer.h.0.attn.",
    "h0.mlp": "transformer.h.0.mlp.",
    "h1.attn": "transformer.h.1.attn.",
    "h1.mlp": "transformer.h.1.mlp.",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _rms(value: np.ndarray) -> float:
    child = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(child * child)))


def _load_ref(entry_root: Path, document: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(load_array(entry_root, document[key]), dtype=np.float64)


def _section_features(section: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    transition_path = Path(section["transition_path"])
    entry_root = transition_path.parents[3]
    transition = read_json(transition_path)
    update = load_named_array(entry_root, transition["step"]["parameter_update"])
    all_values = np.concatenate([value.reshape(-1) for value in update.values()]).astype(np.float64)
    total_norm = float(np.linalg.norm(all_values))
    component_norms: dict[str, float] = {}
    for component, prefix in COMPONENT_PREFIXES.items():
        values = [value.reshape(-1) for name, value in update.items() if name.startswith(prefix)]
        component_norms[component] = float(np.linalg.norm(np.concatenate(values).astype(np.float64)))

    probe = read_json(Path(section["receiver_probe_path"]))
    capability = float(probe["capability_accuracy"])
    support_arrays = {
        "necessity": _load_ref(entry_root, probe, "necessity"),
        "pair_backup": _load_ref(entry_root, probe, "pair_backup"),
        "single_failure_slack": _load_ref(entry_root, probe, "single_failure_slack"),
        "double_failure_slack": _load_ref(entry_root, probe, "double_failure_slack"),
        "support_concentration": _load_ref(entry_root, probe, "support_concentration"),
    }
    baseline_q10 = np.asarray(load_array(entry_root, probe["forwards"][0]["group_q10_margin"]), dtype=np.float64)
    feature_values: list[float] = [np.log1p(total_norm), capability]
    feature_values.extend(component_norms[name] / max(total_norm, np.finfo(np.float64).tiny) for name in COMPONENT_PREFIXES)
    for value in (*support_arrays.values(), baseline_q10):
        finite = value[np.isfinite(value)]
        feature_values.extend([float(np.mean(finite)), float(np.std(finite)), float(np.min(finite)), float(np.max(finite)), _rms(finite)])
    summary = {
        "section_id": section["section_id"],
        "entry_id_audit_only": section["entry_id_audit_only"],
        "pair_id": section["pair_id"],
        "update_l2_norm": total_norm,
        "component_update_l2_norms": component_norms,
        "receiver_capability": capability,
        "feature_sha256": sha256_bytes(np.asarray(feature_values, dtype=np.float64).tobytes(order="C")),
    }
    return np.asarray(feature_values, dtype=np.float64), summary


def build_geometry_controls(*, inventory_path: Path, output_path: Path) -> dict[str, Any]:
    inventory = read_json(inventory_path)
    _require(inventory["status"] == "PASS", "INVENTORY_NOT_PASS")
    sections = inventory["sections"]
    _require(len(sections) == 72, "SECTION_COUNT_INVALID")
    by_id = {row["section_id"]: row for row in sections}
    features: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for section in sections:
        feature, summary = _section_features(section)
        features[section["section_id"]] = feature
        summaries[section["section_id"]] = summary
    matrix = np.stack([features[row["section_id"]] for row in sections])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardized = {key: (value - mean) / scale for key, value in features.items()}

    pair_members: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        pair_members.setdefault(section["pair_id"], []).append(section)
    _require(all(len(rows) == 2 for rows in pair_members.values()), "PAIR_MEMBERSHIP_INVALID")
    treatment: list[dict[str, Any]] = []
    random_controls: list[dict[str, Any]] = []
    cross_run_controls: list[dict[str, Any]] = []
    for pair_id, rows in sorted(pair_members.items()):
        rows = sorted(rows, key=lambda row: row["section_ordinal_within_pair"])
        left, right = rows
        treatment.append({"pair_id": pair_id, "left_section_id": left["section_id"], "right_section_id": right["section_id"]})

        nonadjacent = [
            row
            for row in sections
            if row["entry_id_audit_only"] == left["entry_id_audit_only"]
            and row["pair_id"] != pair_id
            and abs(int(Path(row["receiver_state_path"]).stem.split("-")[-1]) - int(Path(left["receiver_state_path"]).stem.split("-")[-1])) > 1
        ]
        _require(bool(nonadjacent), f"NO_NONADJACENT_CONTROL:{pair_id}")
        nonadjacent.sort(key=lambda row: hashlib.sha256((pair_id + "\0" + row["section_id"]).encode()).hexdigest())
        random_controls.append(
            {
                "treatment_pair_id": pair_id,
                "left_section_id": left["section_id"],
                "right_section_id": nonadjacent[0]["section_id"],
                "selection": "deterministic_sha256_order; no curve values used",
            }
        )

        candidates = [row for row in sections if row["entry_id_audit_only"] != left["entry_id_audit_only"]]
        scored = sorted(
            (
                float(np.linalg.norm(standardized[left["section_id"]] - standardized[row["section_id"]])),
                row["section_id"],
                row,
            )
            for row in candidates
        )
        best_distance, _best_id, best = scored[0]
        cross_run_controls.append(
            {
                "treatment_pair_id": pair_id,
                "left_section_id": left["section_id"],
                "right_section_id": best["section_id"],
                "standardized_precurve_feature_distance": best_distance,
                "matching_inputs": "update magnitude/component fractions + receiver capability + CSRG/margin summaries",
                "target_level_identity_alignment": "REQUIRED_DURING_ANALYSIS; fixed target-group ordinal is forbidden",
            }
        )

    result = {
        "schema": "nanogpt-adjacent-response-geometry-controls-v1",
        "status": "FROZEN_BEFORE_FINITE_AMPLITUDE_CURVES",
        "curve_values_read": False,
        "global_unseen_entry_accessed": False,
        "section_count": 72,
        "treatment_pair_count": len(treatment),
        "random_nonadjacent_control_count": len(random_controls),
        "matched_cross_run_control_count": len(cross_run_controls),
        "feature_standardization": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "fit_population": "all 72 frozen development sections; no response-curve outcomes",
        },
        "section_geometry": [summaries[row["section_id"]] for row in sections],
        "treatment_pairs": treatment,
        "random_nonadjacent_controls": random_controls,
        "matched_cross_run_controls": cross_run_controls,
    }
    write_json(output_path, result)
    return result
