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

from experiments.gfg_nanogpt_local_branch_coordinate_v1.analysis import (
    CONFIRMATION_RUNS,
    DEVELOPMENT_RUNS,
)
from experiments.gfg_nanogpt_native_prebranch_left_history_v1.analysis import (
    IDENTITY_MATERIAL,
    RobustSpace,
    compile_dataset as compile_response_dataset,
)


REPORT_PARENT = Path(__file__).parents[1] / "gfg_nanogpt_cumulative_scientist_v1" / "reports"
PRIOR_BRANCH_LEDGER = REPORT_PARENT / "support_branch_explanation_v1" / "SUPPORT_BRANCH_LEDGER.jsonl.gz"
COMPONENTS = ("h0_attn", "h0_mlp", "h1_attn", "h1_mlp")
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
METHODS = ("f1_f3_f5", "target_support", "competitor_boundary", "target_support_competitor")
K = 64
DIAGNOSTIC_QUANTILE = 0.95


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


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _target_blocks(
    logits: np.ndarray, margins: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive pre-response target support and competitor-boundary coordinates."""
    require(logits.shape == (12, 212, 24), f"TARGET_LOGITS_SHAPE_INVALID:{logits.shape}")
    require(margins.shape == (12, 212), f"TARGET_MARGINS_SHAPE_INVALID:{margins.shape}")
    require(groups.shape == (212,), f"TARGET_GROUPS_SHAPE_INVALID:{groups.shape}")
    rows = np.arange(212)
    baseline_margin = margins[0]
    single_margin = margins[2:6]
    pair_margin = margins[6:12]
    necessity = np.maximum(0.0, baseline_margin[None, :] - single_margin)
    pair_backup = np.stack(
        [
            np.maximum(
                0.0,
                baseline_margin
                - pair_margin[index]
                - necessity[left]
                - necessity[right],
            )
            for index, (left, right) in enumerate(PAIRS)
        ]
    )
    total = np.sum(necessity, axis=0)
    defined = total > 0.0
    allocation = np.zeros_like(necessity)
    allocation[:, defined] = necessity[:, defined] / total[defined]
    concentration = np.sum(np.square(allocation), axis=0)
    effective = np.zeros(212, dtype=np.float64)
    effective[defined] = 1.0 / concentration[defined]
    support = np.concatenate(
        [
            necessity.T,
            pair_backup.T,
            allocation.T,
            concentration[:, None],
            effective[:, None],
            np.min(single_margin, axis=0)[:, None],
            np.min(pair_margin, axis=0)[:, None],
            defined.astype(np.float64)[:, None],
        ],
        axis=1,
    )

    baseline_logits = logits[0]
    masked = baseline_logits.copy()
    masked[rows, groups] = -np.inf
    competitor = np.argmax(masked, axis=1)
    gate_logits = logits[2:12]
    correct_displacement = gate_logits[:, rows, groups] - baseline_logits[rows, groups][None, :]
    fixed_competitor_displacement = (
        gate_logits[:, rows, competitor] - baseline_logits[rows, competitor][None, :]
    )
    gate_margin_displacement = margins[2:12] - baseline_margin[None, :]
    gate_masked = gate_logits.copy()
    gate_masked[:, rows, groups] = -np.inf
    gate_competitor = np.argmax(gate_masked, axis=2)
    competitor_switch = (gate_competitor != competitor[None, :]).astype(np.float64)
    boundary = np.concatenate(
        [
            gate_margin_displacement.T,
            correct_displacement.T,
            fixed_competitor_displacement.T,
            competitor_switch.T,
        ],
        axis=1,
    )
    require(support.shape == (212, 19), f"TARGET_SUPPORT_DIMENSION_INVALID:{support.shape}")
    require(boundary.shape == (212, 40), f"TARGET_BOUNDARY_DIMENSION_INVALID:{boundary.shape}")
    require(np.all(np.isfinite(support)), "TARGET_SUPPORT_NONFINITE")
    require(np.all(np.isfinite(boundary)), "TARGET_BOUNDARY_NONFINITE")
    return support, boundary


def compile_target_coordinates(response: dict[str, Any]) -> dict[str, Any]:
    identities = read_json(IDENTITY_MATERIAL)["entries"]
    records = response["records"]
    by_section: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_section[str(record["section_id"])].append(index)
    support = np.full((len(records), 19), np.nan, dtype=np.float64)
    boundary = np.full((len(records), 40), np.nan, dtype=np.float64)
    source_ledger: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = [{} for _ in records]
    for section_id, indices in sorted(by_section.items()):
        first = records[indices[0]]
        entry_id = str(first["entry_id"])
        path = Path(str(first["source_refs"]["section_npz"]))
        require(path.is_file(), f"SECTION_NPZ_MISSING:{path}")
        digest = file_sha256(path)
        require(digest == str(first["source_refs"]["section_npz_sha256"]), f"SECTION_NPZ_HASH_DRIFT:{section_id}")
        with np.load(path, allow_pickle=False) as data:
            alphas = np.asarray(data["alphas"], dtype=np.float64)
            matches = np.flatnonzero(np.isclose(alphas, 0.0, rtol=0.0, atol=1e-12))
            require(len(matches) == 1, f"ALPHA_ZERO_NOT_UNIQUE:{section_id}:{alphas.tolist()}")
            alpha_index = int(matches[0])
            groups = np.asarray(data["groups"], dtype=np.int64)
            section_support, section_boundary = _target_blocks(
                np.asarray(data["all_logits"][alpha_index], dtype=np.float64),
                np.asarray(data["all_margins"][alpha_index], dtype=np.float64),
                groups,
            )
        identity_rows = identities[entry_id]
        identity_index = {str(row["evaluation_unit_id"]): row_index for row_index, row in enumerate(identity_rows)}
        require(len(identity_index) == 212, f"IDENTITY_COUNT_INVALID:{entry_id}")
        for record_index in indices:
            record = records[record_index]
            row_index = identity_index[str(record["evaluation_unit_id"])]
            require(int(groups[row_index]) == int(record["target_group"]), f"TARGET_IDENTITY_MISMATCH:{record['record_id']}")
            support[record_index] = section_support[row_index]
            boundary[record_index] = section_boundary[row_index]
            coordinate_rows[record_index] = {
                "record_id": str(record["record_id"]),
                "entry_id": entry_id,
                "section_id": section_id,
                "evaluation_unit_id": str(record["evaluation_unit_id"]),
                "evaluation_row_index": row_index,
                "target_group": int(record["target_group"]),
            }
        source_ledger.append(
            {
                "section_id": section_id,
                "entry_id": entry_id,
                "optimizer_step": int(first["optimizer_step"]),
                "section_npz": str(path),
                "section_npz_sha256": digest,
                "alpha_index_used": alpha_index,
                "alpha_value_used": float(alphas[alpha_index]),
                "forbidden_alpha_values_present_but_not_used": [float(value) for value in alphas if not np.isclose(value, 0.0)],
            }
        )
    require(np.all(np.isfinite(support)) and np.all(np.isfinite(boundary)), "TARGET_COORDINATES_INCOMPLETE")
    return {
        "support": support,
        "boundary": boundary,
        "source_ledger": source_ledger,
        "coordinate_rows": coordinate_rows,
    }


def _block_distance(train: np.ndarray, test: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    scaler = RobustSpace().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    dimension = train_scaled.shape[1]
    return np.sqrt(
        np.sum(np.square(train_scaled[neighbors] - test_scaled[:, None, :]), axis=2) / dimension
    )


def _weights(distance: np.ndarray) -> np.ndarray:
    values = 1.0 / np.maximum(distance, 1e-9)
    return values / np.sum(values, axis=1, keepdims=True)


def _predict_fold(
    response: dict[str, Any], coordinates: dict[str, Any], train: np.ndarray, test: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    x = np.asarray(response["spaces"]["X0"], dtype=np.float64)
    curves = np.asarray(response["curves"], dtype=np.float64)
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
    support_distance = _block_distance(coordinates["support"][train], coordinates["support"][test], local)
    boundary_distance = _block_distance(coordinates["boundary"][train], coordinates["boundary"][test], local)
    distances = {
        "f1_f3_f5": base_distance,
        "target_support": np.sqrt(np.square(base_distance) + np.square(support_distance)),
        "competitor_boundary": np.sqrt(np.square(base_distance) + np.square(boundary_distance)),
        "target_support_competitor": np.sqrt(
            np.square(base_distance) + np.square(support_distance) + np.square(boundary_distance)
        ),
    }
    predictions = {
        name: np.sum(curves[neighbors] * _weights(value)[:, :, None], axis=1)
        for name, value in distances.items()
    }
    ledger: list[dict[str, Any]] = []
    records = response["records"]
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
                "base_distances": base_distance[position].tolist(),
                "target_support_distances": support_distance[position].tolist(),
                "competitor_boundary_distances": boundary_distance[position].tolist(),
                "true_curve": curves[int(index)].tolist(),
                "predicted_curves": {name: values[position].tolist() for name, values in predictions.items()},
            }
        )
    return predictions, ledger


def _metric_subset(curves: np.ndarray, prediction: np.ndarray, margin0: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    truth = curves[mask]
    estimate = prediction[mask]
    starts = margin0[mask]
    truth_endpoint = truth[:, -1]
    estimate_endpoint = estimate[:, -1]
    truth_start_correct = starts > 0.0
    truth_end_correct = starts + truth_endpoint > 0.0
    estimate_end_correct = starts + estimate_endpoint > 0.0
    unchanged = truth_start_correct == truth_end_correct
    wrong_to_correct = (~truth_start_correct) & truth_end_correct
    return {
        "count": count,
        "curve_rmse": float(np.sqrt(np.mean(np.square(estimate - truth)))),
        "endpoint_direction_accuracy": float(np.mean(np.sign(estimate_endpoint) == np.sign(truth_endpoint))),
        "boundary_accuracy": float(np.mean(estimate_end_correct == truth_end_correct)),
        "unchanged_false_crossing_rate": float(np.mean(estimate_end_correct[unchanged] != truth_start_correct[unchanged])) if np.any(unchanged) else None,
        "wrong_to_correct_recall": float(np.mean(estimate_end_correct[wrong_to_correct])) if np.any(wrong_to_correct) else None,
    }


def _remainder_mask(response: dict[str, Any]) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    prior = {str(row["record_id"]): row for row in read_rows(PRIOR_BRANCH_LEDGER)}
    record_ids = {str(row["record_id"]) for row in response["records"]}
    require(set(prior).issubset(record_ids), "PRIOR_BRANCH_LEDGER_HAS_UNKNOWN_RECORD")
    mask = np.asarray(
        [
            str(record["record_id"]) in prior
            and bool(prior[str(record["record_id"])]["severe_conflict"])
            and bool(prior[str(record["record_id"])]["m4_in_support"])
            and not bool(prior[str(record["record_id"])]["true_support_branch_divergent"])
            and not bool(prior[str(record["record_id"])]["primary_transition_mismatch"])
            for record in response["records"]
        ],
        dtype=bool,
    )
    require(int(np.sum(mask)) == 311, f"GROUP_LEVEL_REMAINDER_COUNT_DRIFT:{int(np.sum(mask))}")
    return mask, prior


def _pair_diagnostics(
    response: dict[str, Any], coordinates: dict[str, Any], remainder: np.ndarray, prior: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = response["records"]
    by_id = {str(row["record_id"]): index for index, row in enumerate(records)}
    development = np.isin(np.asarray(response["entries"], dtype=object), np.asarray(DEVELOPMENT_RUNS, dtype=object))
    support_scaler = RobustSpace().fit(coordinates["support"][development])
    boundary_scaler = RobustSpace().fit(coordinates["boundary"][development])
    support_scaled = support_scaler.transform(coordinates["support"])
    boundary_scaled = boundary_scaler.transform(coordinates["boundary"])
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        record_id = str(record["record_id"])
        if record_id not in prior:
            continue
        source = prior[record_id]
        nearest = by_id[str(source["nearest_record_id"])]
        support_distance = float(np.sqrt(np.mean(np.square(support_scaled[index] - support_scaled[nearest]))))
        boundary_distance = float(np.sqrt(np.mean(np.square(boundary_scaled[index] - boundary_scaled[nearest]))))
        full_distance = float(np.sqrt(np.square(support_distance) + np.square(boundary_distance)))
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "nearest_record_id": str(source["nearest_record_id"]),
                "entry_id": str(record["entry_id"]),
                "nearest_entry_id": str(records[nearest]["entry_id"]),
                "severe_conflict": bool(source["severe_conflict"]),
                "m4_in_support": bool(source["m4_in_support"]),
                "group_level_remainder_311": bool(remainder[index]),
                "target_support_distance": support_distance,
                "competitor_boundary_distance": boundary_distance,
                "combined_target_distance": full_distance,
            }
        )
    ordinary_development = [
        row for row in rows if row["entry_id"] in DEVELOPMENT_RUNS and row["m4_in_support"] and not row["severe_conflict"]
    ]
    require(bool(ordinary_development), "NO_DEVELOPMENT_ORDINARY_TARGET_PAIRS")
    thresholds = {
        name: float(np.quantile([row[name] for row in ordinary_development], DIAGNOSTIC_QUANTILE))
        for name in ("target_support_distance", "competitor_boundary_distance", "combined_target_distance")
    }
    residual = [row for row in rows if row["group_level_remainder_311"]]
    summary = {
        "schema": "nanogpt-target-support-pair-diagnostics-v1",
        "status": "PASS",
        "remainder_count": len(residual),
        "diagnostic_quantile": DIAGNOSTIC_QUANTILE,
        "thresholds_from_development_ordinary": thresholds,
        "remainder_above_target_support_threshold": sum(
            row["target_support_distance"] > thresholds["target_support_distance"] for row in residual
        ),
        "remainder_above_competitor_boundary_threshold": sum(
            row["competitor_boundary_distance"] > thresholds["competitor_boundary_distance"] for row in residual
        ),
        "remainder_above_combined_threshold": sum(
            row["combined_target_distance"] > thresholds["combined_target_distance"] for row in residual
        ),
    }
    return summary, rows


def run_analysis() -> dict[str, Any]:
    response, response_audit, response_sources = compile_response_dataset()
    coordinates = compile_target_coordinates(response)
    remainder, prior = _remainder_mask(response)
    entries = np.asarray(response["entries"], dtype=object)
    predictions = {name: np.full_like(response["curves"], np.nan) for name in METHODS}
    ledger: list[dict[str, Any]] = []
    for run in DEVELOPMENT_RUNS:
        test = np.flatnonzero(entries == run)
        train = np.flatnonzero(np.isin(entries, np.asarray(DEVELOPMENT_RUNS, dtype=object)) & (entries != run))
        fold, rows = _predict_fold(response, coordinates, train, test)
        for name, values in fold.items():
            predictions[name][test] = values
        ledger.extend(rows)
    confirmation = np.isin(entries, np.asarray(CONFIRMATION_RUNS, dtype=object))
    development = np.isin(entries, np.asarray(DEVELOPMENT_RUNS, dtype=object))
    test = np.flatnonzero(confirmation)
    train = np.flatnonzero(development)
    fold, rows = _predict_fold(response, coordinates, train, test)
    for name, values in fold.items():
        predictions[name][test] = values
    ledger.extend(rows)
    for name in METHODS:
        require(np.all(np.isfinite(predictions[name])), f"PREDICTION_INCOMPLETE:{name}")

    curves = np.asarray(response["curves"], dtype=np.float64)
    margin0 = np.asarray(response["margin0"], dtype=np.float64)
    severe = np.asarray(response["labels"]["severe_conflict"], dtype=bool)
    record_index = {str(record["record_id"]): index for index, record in enumerate(response["records"])}
    for row in ledger:
        index = record_index[str(row["record_id"])]
        row["margin0"] = float(margin0[index])
        row["severe_conflict"] = bool(severe[index])
        row["group_level_remainder_311"] = bool(remainder[index])
    split_masks = {
        "development": development,
        "confirmation": confirmation,
        "all_runs": np.ones(len(records := response["records"]), dtype=bool),
    }
    metrics: dict[str, Any] = {}
    for split_name, split_mask in split_masks.items():
        metrics[split_name] = {}
        for name in METHODS:
            metrics[split_name][name] = {
                "overall": _metric_subset(curves, predictions[name], margin0, split_mask),
                "severe_conflict": _metric_subset(curves, predictions[name], margin0, split_mask & severe),
                "group_level_remainder_311": _metric_subset(curves, predictions[name], margin0, split_mask & remainder),
            }
    diagnostic, pair_rows = _pair_diagnostics(response, coordinates, remainder, prior)
    base = metrics["all_runs"]["f1_f3_f5"]
    best_name = min(METHODS[1:], key=lambda name: metrics["all_runs"][name]["group_level_remainder_311"]["curve_rmse"])
    best = metrics["all_runs"][best_name]
    decision = {
        "schema": "nanogpt-target-support-branch-decision-v1",
        "status": "PASS",
        "best_target_method_by_remainder_curve_rmse": best_name,
        "remainder_curve_rmse_relative_improvement": (
            base["group_level_remainder_311"]["curve_rmse"] - best["group_level_remainder_311"]["curve_rmse"]
        ) / base["group_level_remainder_311"]["curve_rmse"],
        "remainder_boundary_accuracy_delta": (
            best["group_level_remainder_311"]["boundary_accuracy"] - base["group_level_remainder_311"]["boundary_accuracy"]
        ),
        "overall_curve_rmse_relative_improvement": (
            base["overall"]["curve_rmse"] - best["overall"]["curve_rmse"]
        ) / base["overall"]["curve_rmse"],
        "remainder_count": int(np.sum(remainder)),
        "verdict": "TARGET_LEVEL_COORDINATE_IMPROVES_REMAINDER" if (
            best["group_level_remainder_311"]["curve_rmse"] < base["group_level_remainder_311"]["curve_rmse"]
            and best["group_level_remainder_311"]["boundary_accuracy"] >= base["group_level_remainder_311"]["boundary_accuracy"]
        ) else "TARGET_LEVEL_COORDINATE_NOT_YET_JOINTLY_IMPROVING",
        "post_update_evidence_used_as_executable_input": False,
    }
    return {
        "response": response,
        "response_audit": response_audit,
        "response_sources": response_sources,
        "coordinates": coordinates,
        "remainder": remainder,
        "predictions": predictions,
        "prediction_ledger": ledger,
        "pair_diagnostics": diagnostic,
        "pair_ledger": pair_rows,
        "metrics": metrics,
        "decision": decision,
        "feature_manifest": {
            "schema": "nanogpt-target-support-feature-manifest-v1",
            "status": "PASS",
            "target_support_dimension": 19,
            "competitor_boundary_dimension": 40,
            "alpha_value_used": 0.0,
            "target_support_fields": [
                "necessity[4]", "pair_backup[6]", "allocation[4]", "concentration", "effective_support",
                "single_failure_slack", "double_failure_slack", "support_defined",
            ],
            "competitor_boundary_fields": [
                "gate_margin_displacement[10]", "correct_logit_displacement[10]",
                "fixed_competitor_logit_displacement[10]", "competitor_switch[10]",
            ],
            "future_response_fields_used": [],
        },
    }
