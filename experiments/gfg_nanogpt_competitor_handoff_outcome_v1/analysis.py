from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from experiments.gfg_nanogpt_adjacent_response_transport_v1.inventory import load_named_array
from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import (
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
)
from experiments.gfg_nanogpt_native_prebranch_left_history_v1.analysis import (
    IDENTITY_MATERIAL,
    RESPONSE_ROOT,
    STEPWISE_ROOT,
    RobustSpace,
    compile_dataset as compile_response_dataset,
)
from experiments.gfg_nanogpt_response_factor_analysis_v1.analysis import (
    _layout_slice,
    _load_locator,
)
from experiments.gfg_nanogpt_target_support_branch_v1.analysis import _remainder_mask


METHODS = (
    "f1_f3_f5_outcome",
    "all_competitor_gaps",
    "all_competitor_geometry",
    "all_competitor_combined",
)
K = 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _geometry_for_rows(
    logits: np.ndarray,
    groups: np.ndarray,
    current_wte: np.ndarray,
    update_wte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    require(logits.shape == (212, 24), f"CURRENT_LOGITS_SHAPE_INVALID:{logits.shape}")
    require(groups.shape == (212,), f"GROUP_SHAPE_INVALID:{groups.shape}")
    require(current_wte.shape == update_wte.shape == (24, 64), "WTE_SHAPE_INVALID")
    gaps = np.empty((212, 23), dtype=np.float64)
    geometry = np.empty((212, 23 * 5), dtype=np.float64)
    pseudo_inverse = np.linalg.pinv(current_wte)
    hidden_row_space = logits @ pseudo_inverse.T
    eps = np.finfo(np.float64).tiny
    for row_index, target_value in enumerate(groups):
        target = int(target_value)
        order = np.argsort(logits[row_index])[::-1]
        competitors = order[order != target]
        require(len(competitors) == 23, "COMPETITOR_COUNT_INVALID")
        gaps[row_index] = logits[row_index, target] - logits[row_index, competitors]
        boundary = current_wte[target][None, :] - current_wte[competitors]
        boundary_update = update_wte[target][None, :] - update_wte[competitors]
        boundary_norm = np.linalg.norm(boundary, axis=1)
        update_norm = np.linalg.norm(boundary_update, axis=1)
        dot = np.sum(boundary * boundary_update, axis=1)
        cosine = dot / np.maximum(boundary_norm * update_norm, eps)
        radial_ratio = dot / np.maximum(np.square(boundary_norm), eps)
        direct_gap_effect = boundary_update @ hidden_row_space[row_index]
        geometry[row_index] = np.stack(
            [boundary_norm, update_norm, cosine, radial_ratio, direct_gap_effect], axis=1
        ).reshape(-1)
    require(np.all(np.isfinite(gaps)) and np.all(np.isfinite(geometry)), "COMPETITOR_COORDINATE_NONFINITE")
    return gaps, geometry


def compile_competitor_coordinates(response: dict[str, Any]) -> dict[str, Any]:
    inventory = read_json(RESPONSE_ROOT / "RESOLVED_INVENTORY.json")
    inventory_by_section = {str(row["section_id"]): row for row in inventory["sections"]}
    identities = read_json(IDENTITY_MATERIAL)["entries"]
    records = response["records"]
    by_section: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_section[str(record["section_id"])].append(index)
    gaps = np.full((len(records), 23), np.nan, dtype=np.float64)
    geometry = np.full((len(records), 115), np.nan, dtype=np.float64)
    coordinate_rows: list[dict[str, Any]] = [{} for _ in records]
    source_rows: list[dict[str, Any]] = []
    for section_id, indices in sorted(by_section.items()):
        first = records[indices[0]]
        entry_id = str(first["entry_id"])
        resolved = inventory_by_section[section_id]
        section_path = Path(str(first["source_refs"]["section_npz"]))
        state_path = Path(str(resolved["receiver_state_path"]))
        transition_path = Path(str(resolved["transition_path"]))
        require(transition_path == Path(str(first["source_refs"]["transition"])), f"TRANSITION_PATH_MISMATCH:{section_id}")
        require(file_sha256(section_path) == str(first["source_refs"]["section_npz_sha256"]), f"SECTION_HASH_DRIFT:{section_id}")
        with np.load(section_path, allow_pickle=False) as data:
            alphas = np.asarray(data["alphas"], dtype=np.float64)
            alpha_matches = np.flatnonzero(np.isclose(alphas, 0.0, rtol=0.0, atol=1e-12))
            require(len(alpha_matches) == 1, f"ALPHA_ZERO_INVALID:{section_id}")
            alpha_index = int(alpha_matches[0])
            logits = np.asarray(data["all_logits"][alpha_index, 0], dtype=np.float64)
            groups = np.asarray(data["groups"], dtype=np.int64)
        state = read_json(state_path)
        current_named = load_named_array(state_path.parents[3], state["state"]["parameters"])
        current_wte = np.asarray(current_named["transformer.wte.weight"], dtype=np.float64)
        transition = read_json(transition_path)
        update_descriptor = transition["step"]["parameter_update"]
        update_flat = _load_locator(STEPWISE_ROOT / entry_id, update_descriptor).astype(np.float64, copy=False)
        update_wte = np.asarray(_layout_slice(update_flat, update_descriptor, "transformer.wte.weight"), dtype=np.float64)
        section_gaps, section_geometry = _geometry_for_rows(logits, groups, current_wte, update_wte)
        identity_index = {
            str(row["evaluation_unit_id"]): row_index
            for row_index, row in enumerate(identities[entry_id])
        }
        require(len(identity_index) == 212, f"IDENTITY_COUNT_INVALID:{entry_id}")
        for record_index in indices:
            record = records[record_index]
            row_index = identity_index[str(record["evaluation_unit_id"])]
            require(int(groups[row_index]) == int(record["target_group"]), f"TARGET_IDENTITY_MISMATCH:{record['record_id']}")
            gaps[record_index] = section_gaps[row_index]
            geometry[record_index] = section_geometry[row_index]
            coordinate_rows[record_index] = {
                "record_id": str(record["record_id"]),
                "entry_id": entry_id,
                "section_id": section_id,
                "evaluation_unit_id": str(record["evaluation_unit_id"]),
                "evaluation_row_index": row_index,
                "target_group": int(record["target_group"]),
            }
        source_rows.append(
            {
                "section_id": section_id,
                "entry_id": entry_id,
                "optimizer_step": int(first["optimizer_step"]),
                "section_npz_sha256": file_sha256(section_path),
                "receiver_state_sha256": file_sha256(state_path),
                "transition_sha256": file_sha256(transition_path),
                "current_parameter_commitment": str(state["state"]["parameters"]["raw_tensor_sha256"]),
                "actual_update_commitment": str(update_descriptor["raw_tensor_sha256"]),
                "alpha_index_used": alpha_index,
                "alpha_value_used": float(alphas[alpha_index]),
            }
        )
    require(np.all(np.isfinite(gaps)) and np.all(np.isfinite(geometry)), "COMPETITOR_COORDINATES_INCOMPLETE")
    return {
        "gaps": gaps,
        "geometry": geometry,
        "coordinate_rows": coordinate_rows,
        "source_rows": source_rows,
    }


def _block_distance(train: np.ndarray, test: np.ndarray, local_neighbors: np.ndarray) -> np.ndarray:
    scaler = RobustSpace().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return np.sqrt(
        np.sum(np.square(train_scaled[local_neighbors] - test_scaled[:, None, :]), axis=2)
        / train_scaled.shape[1]
    )


def _weights(distance: np.ndarray) -> np.ndarray:
    values = 1.0 / np.maximum(distance, 1e-9)
    return values / np.sum(values, axis=1, keepdims=True)


def _final_truth(response: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin0 = np.asarray(response["margin0"], dtype=np.float64)
    endpoint = np.asarray(response["curves"], dtype=np.float64)[:, -1]
    start_correct = margin0 > 0.0
    final_correct = margin0 + endpoint > 0.0
    boundary = np.empty(len(margin0), dtype=object)
    boundary[start_correct & final_correct] = "MAINTAIN_CORRECT"
    boundary[start_correct & ~final_correct] = "CORRECT_TO_WRONG"
    boundary[~start_correct & final_correct] = "WRONG_TO_CORRECT"
    boundary[~start_correct & ~final_correct] = "MAINTAIN_WRONG"
    return start_correct, final_correct, boundary


def _predict_fold(
    response: dict[str, Any], coordinates: dict[str, Any], labels: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    x = np.asarray(response["spaces"]["X0"], dtype=np.float64)
    base_scaler = RobustSpace().fit(x[train])
    train_x = base_scaler.transform(x[train])
    test_x = base_scaler.transform(x[test])
    count = min(K, len(train))
    base_distance, local = cKDTree(train_x).query(test_x, k=count, workers=-1)
    base_distance = np.asarray(base_distance, dtype=np.float64)
    local = np.asarray(local, dtype=np.int64)
    if count == 1:
        base_distance = base_distance[:, None]
        local = local[:, None]
    neighbors = train[local]
    base_distance /= math.sqrt(train_x.shape[1])
    gap_distance = _block_distance(coordinates["gaps"][train], coordinates["gaps"][test], local)
    geometry_distance = _block_distance(coordinates["geometry"][train], coordinates["geometry"][test], local)
    distances = {
        "f1_f3_f5_outcome": base_distance,
        "all_competitor_gaps": np.sqrt(np.square(base_distance) + np.square(gap_distance)),
        "all_competitor_geometry": np.sqrt(np.square(base_distance) + np.square(geometry_distance)),
        "all_competitor_combined": np.sqrt(
            np.square(base_distance) + np.square(gap_distance) + np.square(geometry_distance)
        ),
    }
    probabilities = {
        name: np.sum(labels[neighbors] * _weights(value), axis=1)
        for name, value in distances.items()
    }
    records = response["records"]
    ledger: list[dict[str, Any]] = []
    for position, index in enumerate(test):
        record = records[int(index)]
        ledger.append(
            {
                "record_id": str(record["record_id"]),
                "entry_id": str(record["entry_id"]),
                "optimizer_step": int(record["optimizer_step"]),
                "target_group": int(record["target_group"]),
                "neighbor_record_ids": [str(records[int(value)]["record_id"]) for value in neighbors[position]],
                "neighbor_entry_ids": [str(records[int(value)]["entry_id"]) for value in neighbors[position]],
                "probability_final_correct": {name: float(value[position]) for name, value in probabilities.items()},
            }
        )
    return probabilities, ledger


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    start_correct: np.ndarray,
    boundary: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    truth = truth[mask]
    prediction = prediction[mask]
    start_correct = start_correct[mask]
    boundary = boundary[mask]
    count = len(truth)
    if count == 0:
        return {"count": 0}
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    tn = int(np.sum(~prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "count": count,
        "correct_count": int(np.sum(prediction == truth)),
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "final_correct_precision": precision,
        "final_correct_recall": tpr,
        "final_correct_f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "transition_recall": {
            name: float(np.mean(prediction[boundary == name] == truth[boundary == name])) if np.any(boundary == name) else None
            for name in ("MAINTAIN_CORRECT", "CORRECT_TO_WRONG", "WRONG_TO_CORRECT", "MAINTAIN_WRONG")
        },
    }


def run_analysis() -> dict[str, Any]:
    response, response_audit, response_sources = compile_response_dataset()
    coordinates = compile_competitor_coordinates(response)
    remainder, _ = _remainder_mask(response)
    start_correct, truth, boundary = _final_truth(response)
    entries = np.asarray(response["entries"], dtype=object)
    development = np.isin(entries, np.asarray(DEVELOPMENT_RUNS, dtype=object))
    confirmation = np.isin(entries, np.asarray(CONFIRMATION_RUNS, dtype=object))
    probabilities = {name: np.full(len(truth), np.nan, dtype=np.float64) for name in METHODS}
    ledger: list[dict[str, Any]] = []
    for run in DEVELOPMENT_RUNS:
        test = np.flatnonzero(entries == run)
        train = np.flatnonzero(development & (entries != run))
        fold, rows = _predict_fold(response, coordinates, truth.astype(np.float64), train, test)
        for name, values in fold.items():
            probabilities[name][test] = values
        ledger.extend(rows)
    test = np.flatnonzero(confirmation)
    train = np.flatnonzero(development)
    fold, rows = _predict_fold(response, coordinates, truth.astype(np.float64), train, test)
    for name, values in fold.items():
        probabilities[name][test] = values
    ledger.extend(rows)
    predictions = {name: value >= 0.5 for name, value in probabilities.items()}
    for name in METHODS:
        require(np.all(np.isfinite(probabilities[name])), f"OUTCOME_PROBABILITY_INCOMPLETE:{name}")
    severe = np.asarray(response["labels"]["severe_conflict"], dtype=bool)
    by_record = {str(record["record_id"]): index for index, record in enumerate(response["records"])}
    for row in ledger:
        index = by_record[str(row["record_id"])]
        row["start_correct"] = bool(start_correct[index])
        row["truth_final_correct"] = bool(truth[index])
        row["truth_boundary"] = str(boundary[index])
        row["severe_conflict"] = bool(severe[index])
        row["group_level_remainder_311"] = bool(remainder[index])
        row["predicted_final_correct"] = {name: bool(predictions[name][index]) for name in METHODS}

    split_masks = {
        "development": development,
        "confirmation": confirmation,
        "all_runs": np.ones(len(truth), dtype=bool),
    }
    subset_masks = {
        "overall": np.ones(len(truth), dtype=bool),
        "severe_conflict": severe,
        "group_level_remainder_311": remainder,
    }
    metrics: dict[str, Any] = {}
    repairs: dict[str, Any] = {}
    baseline = predictions["f1_f3_f5_outcome"]
    for split_name, split_mask in split_masks.items():
        metrics[split_name] = {}
        repairs[split_name] = {}
        for name in METHODS:
            metrics[split_name][name] = {}
            repairs[split_name][name] = {}
            for subset_name, subset_mask in subset_masks.items():
                mask = split_mask & subset_mask
                metrics[split_name][name][subset_name] = _metrics(
                    truth, predictions[name], start_correct, boundary, mask
                )
                fixed = int(np.sum(mask & (baseline != truth) & (predictions[name] == truth)))
                broken = int(np.sum(mask & (baseline == truth) & (predictions[name] != truth)))
                repairs[split_name][name][subset_name] = {
                    "fixed_baseline_errors": fixed,
                    "newly_broken_baseline_answers": broken,
                    "net_repairs": fixed - broken,
                }
    main_all = repairs["all_runs"]["all_competitor_combined"]["group_level_remainder_311"]
    main_confirmation = repairs["confirmation"]["all_competitor_combined"]["group_level_remainder_311"]
    verdict = (
        "MULTI_COMPETITOR_HANDOFF_IMPROVES_FINAL_OUTCOME"
        if main_all["net_repairs"] > 0 and main_confirmation["net_repairs"] > 0
        else "MULTI_COMPETITOR_HANDOFF_DOES_NOT_MATERIALLY_IMPROVE_FINAL_OUTCOME"
    )
    decision = {
        "schema": "nanogpt-competitor-handoff-outcome-decision-v1",
        "status": "PASS",
        "verdict": verdict,
        "primary_method": "all_competitor_combined",
        "all_remainder_repairs": main_all,
        "confirmation_remainder_repairs": main_confirmation,
        "remainder_count": int(np.sum(remainder)),
        "curve_metrics_used_for_decision": False,
        "post_outcome_material_used_as_input": False,
    }
    return {
        "response": response,
        "response_audit": response_audit,
        "response_sources": response_sources,
        "coordinates": coordinates,
        "remainder": remainder,
        "truth": truth,
        "probabilities": probabilities,
        "prediction_ledger": ledger,
        "metrics": metrics,
        "repairs": repairs,
        "decision": decision,
        "feature_manifest": {
            "schema": "nanogpt-competitor-handoff-feature-manifest-v1",
            "status": "PASS",
            "all_competitor_gap_dimension": 23,
            "all_competitor_geometry_dimension": 115,
            "competitor_order": "descending alpha=0 incorrect-class logit",
            "geometry_fields_per_rank": [
                "current_boundary_row_l2", "boundary_update_l2", "boundary_update_cosine",
                "signed_radial_update_ratio", "row_space_direct_gap_effect_estimate",
            ],
            "raw_class_identity_used_as_numeric_coordinate": False,
            "allowed_alpha_values": [0.0],
            "post_outcome_input_fields": [],
        },
    }
