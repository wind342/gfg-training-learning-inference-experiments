from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_stepwise_support_transition_v1.one_step_index import (
    CAPABILITY_ZERO_BAND,
    DECLINE_DELTA_THRESHOLD,
)


SCHEMA = "nanogpt-one-step-500-csrg-development-pilot-v1"
HELD_OUT_ENTRY_ID = "entry-362ded584a953f360aec"
RIDGE_ALPHA = 1.0
STATE_NRMSE_TOLERANCE = 1.0
REQUIRED_SELECTION_COUNT = 500


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
            count += 1
    return count


def _finite_mapping(value: dict[str, Any], code: str) -> dict[str, float]:
    result = {str(name): float(child) for name, child in value.items()}
    require(all(math.isfinite(child) for child in result.values()), code)
    return result


def _tensor_rms(entry_root: Path, reference: dict[str, Any]) -> float:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "ONE_STEP_500_TENSOR_LOCATOR_INVALID")
    path = entry_root / locator
    require(path.is_file(), f"ONE_STEP_500_TENSOR_MISSING:{path}")
    require(file_sha256(path) == reference["file_sha256"], f"ONE_STEP_500_TENSOR_HASH_MISMATCH:{path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(array.shape) == list(reference["shape"]), "ONE_STEP_500_TENSOR_SHAPE_MISMATCH")
    require(str(array.dtype) == str(reference["dtype"]), "ONE_STEP_500_TENSOR_DTYPE_MISMATCH")
    values = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def _feature_cache_state_map(feature_cache: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for entry in feature_cache["entries"]:
        entry_id = str(entry["entry_id"])
        for window in entry["windows"]:
            window_id = str(window["window_id"])
            for state in window["states"]:
                key = (entry_id, window_id, int(state["optimizer_step"]))
                require(key not in result, f"ONE_STEP_500_DUPLICATE_STATE:{key}")
                result[key] = state
    return result


def _selected_index_rows(selection_path: Path, index_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = _read_jsonl(selection_path)
    require(len(selection) == REQUIRED_SELECTION_COUNT, "ONE_STEP_500_SELECTION_COUNT_INVALID")
    selected_ids = {str(row["sample_id"]) for row in selection}
    require(len(selected_ids) == REQUIRED_SELECTION_COUNT, "ONE_STEP_500_SELECTION_ID_DUPLICATE")
    index_rows = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in selected_ids:
                index_rows.append(row)
    require(len(index_rows) == REQUIRED_SELECTION_COUNT, "ONE_STEP_500_INDEX_RESOLUTION_INCOMPLETE")
    require(all(row["partition"] == "DEVELOPMENT" for row in index_rows), "ONE_STEP_500_HELD_OUT_LEAKAGE")
    selection_by_id = {str(row["sample_id"]): row for row in selection}
    return [selection_by_id[str(row["sample_id"])] for row in index_rows], index_rows


def _record_from_sources(
    *,
    selection_row: dict[str, Any],
    index_row: dict[str, Any],
    state_map: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    entry_id = str(index_row["entry_id"])
    window_id = str(index_row["window_id"])
    step = int(index_row["optimizer_step"])
    left = state_map[(entry_id, window_id, step)]
    right = state_map[(entry_id, window_id, step + 1)]
    entry_root = Path(index_row["references"]["entry_root"])
    action_refs = index_row["references"]["current_action"]["tensor_and_batch_refs"]
    update_rms = _tensor_rms(entry_root, action_refs["parameter_update"])
    preconditioned_rms = _tensor_rms(
        entry_root,
        action_refs["optimizer_deltas"]["post_preconditioned_direction"],
    )
    u_features = _finite_mapping(index_row["u_features"], "ONE_STEP_500_U_NONFINITE")
    u_features["actual_parameter_update_rms"] = update_rms
    u_features["actual_post_preconditioned_direction_rms"] = preconditioned_rms

    def state_payload(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "support_state": _finite_mapping(row["state"], "ONE_STEP_500_SUPPORT_STATE_NONFINITE"),
            "support_topology": _finite_mapping(row["topology"], "ONE_STEP_500_SUPPORT_TOPOLOGY_NONFINITE"),
            "optimizer_parameter_loads": _finite_mapping(row["loads"], "ONE_STEP_500_LOAD_NONFINITE"),
            "observables": _finite_mapping(row["observables"], "ONE_STEP_500_OBSERVABLE_NONFINITE"),
            "left_velocity": _finite_mapping(row["velocity"], "ONE_STEP_500_VELOCITY_NONFINITE"),
            "left_acceleration": _finite_mapping(row["acceleration"], "ONE_STEP_500_ACCELERATION_NONFINITE"),
            "left_velocity_mean_5": _finite_mapping(row["history_velocity_mean"], "ONE_STEP_500_HISTORY_NONFINITE"),
            "left_velocity_rms_5": _finite_mapping(row["history_velocity_rms"], "ONE_STEP_500_HISTORY_RMS_NONFINITE"),
            "topology_velocity": _finite_mapping(row["topology_velocity"], "ONE_STEP_500_TOPOLOGY_VELOCITY_NONFINITE"),
            "topology_acceleration": _finite_mapping(row["topology_acceleration"], "ONE_STEP_500_TOPOLOGY_ACCELERATION_NONFINITE"),
            "topology_velocity_mean_5": _finite_mapping(row["topology_history_velocity_mean"], "ONE_STEP_500_TOPOLOGY_HISTORY_NONFINITE"),
            "provenance": row["provenance"],
            "observed_state_id": row["observed_state_id"],
            "probe_observation_id": row["probe_observation_id"],
        }

    material = {
        "schema": SCHEMA,
        "sample_id": index_row["sample_id"],
        "entry_id": entry_id,
        "run_id": index_row["run_id"],
        "window_id": window_id,
        "optimizer_step": step,
        "partition": "DEVELOPMENT",
        "event_order": [
            "X_t established",
            "current batch forward and backward executed",
            "gradient, Adam increments and actual update formed",
            "prediction boundary",
            "target CSRG probe establishes X_t_plus_1",
        ],
        "X_t": state_payload(left),
        "U_t": u_features,
        "X_t_plus_1": state_payload(right),
        "source_index_references": index_row["references"],
        "screening_selection": selection_row,
        "representation_disclosure": {
            "field_coverage": "all registered compact CSRG state and topology summaries plus optimizer/parameter loads, observables and left-history summaries",
            "not_raw_csrg_arrays": True,
            "raw_csrg_access": "source_index_references and X_t/X_t_plus_1 provenance identify the original validated GFG objects",
            "tensor_payloads_copied": 0,
        },
    }
    return {**material, "record_sha256": payload_sha256(material)}


def _vector(record: dict[str, Any], *, target: bool = False) -> dict[str, float]:
    state = record["X_t_plus_1"] if target else record["X_t"]
    result = {f"s.{name}": float(value) for name, value in state["support_state"].items()}
    result.update({f"topology.{name}": float(value) for name, value in state["support_topology"].items()})
    result.update({f"load.{name}": float(value) for name, value in state["optimizer_parameter_loads"].items()})
    result.update({f"obs.{name}": float(value) for name, value in state["observables"].items()})
    if not target:
        result.update({f"u.{name}": float(value) for name, value in record["U_t"].items()})
    return result


def _ridge_fit_predict(
    train_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    input_names = sorted(set.intersection(*(set(_vector(row)) for row in train_records + target_records)))
    output_names = sorted(
        name
        for name in set.intersection(*(set(_vector(row, target=True)) for row in train_records + target_records))
        if name.startswith(("s.", "topology.", "obs.capability"))
    )
    require(bool(input_names) and bool(output_names), "ONE_STEP_500_EMPTY_MODEL_SCHEMA")

    def matrix(rows: list[dict[str, Any]], names: list[str], target: bool = False) -> np.ndarray:
        return np.asarray([[float(_vector(row, target=target)[name]) for name in names] for row in rows], dtype=np.float64)

    train_x = matrix(train_records, input_names)
    target_x = matrix(target_records, input_names)
    train_current = matrix(train_records, output_names)
    train_next = matrix(train_records, output_names, target=True)
    target_current = matrix(target_records, output_names)
    delta = train_next - train_current
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1.0e-15] = 1.0
    y_mean = delta.mean(axis=0)
    y_scale = delta.std(axis=0)
    y_scale[y_scale < 1.0e-12] = 1.0
    design = (train_x - x_mean) / x_scale
    targets = (delta - y_mean) / y_scale
    gram = design.T @ design + RIDGE_ALPHA * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ targets)
    predicted_delta = (((target_x - x_mean) / x_scale) @ weights) * y_scale + y_mean
    prediction = target_current + predicted_delta
    fit = {
        "input_names": input_names,
        "output_names": output_names,
        "output_delta_scales": {name: float(value) for name, value in zip(output_names, y_scale)},
        "ridge_alpha": RIDGE_ALPHA,
        "training_record_count": len(train_records),
    }
    return prediction, output_names, fit


def _direction(value: float) -> int:
    return 1 if value > CAPABILITY_ZERO_BAND else (-1 if value < -CAPABILITY_ZERO_BAND else 0)


def _adjudicate_fold(
    train_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    capability_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions, output_names, fit = _ridge_fit_predict(train_records, target_records)
    output_scales = np.asarray([fit["output_delta_scales"][name] for name in output_names], dtype=np.float64)
    state_indices = [index for index, name in enumerate(output_names) if name.startswith(("s.", "topology."))]
    capability_index = output_names.index("obs.capability")
    rows = []
    for record_index, record in enumerate(target_records):
        current = _vector(record)
        truth = _vector(record, target=True)
        predicted = {name: float(predictions[record_index, index]) for index, name in enumerate(output_names)}
        normalized = np.asarray(
            [(predicted[name] - truth[name]) / output_scales[index] for index, name in enumerate(output_names)],
            dtype=np.float64,
        )
        state_nrmse = float(np.sqrt(np.mean(np.square(normalized[state_indices]))))
        cap_truth = float(truth["obs.capability"])
        cap_current = float(current["obs.capability"])
        cap_prediction = predicted["obs.capability"]
        cap_absolute_error = abs(cap_prediction - cap_truth)
        actual_delta = cap_truth - cap_current
        predicted_delta = cap_prediction - cap_current
        state_error = state_nrmse > STATE_NRMSE_TOLERANCE
        capability_error = cap_absolute_error > capability_tolerance
        direction_error = _direction(actual_delta) != _direction(predicted_delta)
        decline_error = (actual_delta <= DECLINE_DELTA_THRESHOLD) != (predicted_delta <= DECLINE_DELTA_THRESHOLD)
        residuals = sorted(
            (
                {
                    "coordinate": name,
                    "normalized_absolute_error": float(abs(normalized[index])),
                    "prediction": predicted[name],
                    "truth": float(truth[name]),
                }
                for index, name in enumerate(output_names)
                if index != capability_index
            ),
            key=lambda row: (-row["normalized_absolute_error"], row["coordinate"]),
        )[:10]
        material = {
            "sample_id": record["sample_id"],
            "entry_id": record["entry_id"],
            "run_id": record["run_id"],
            "window_id": record["window_id"],
            "optimizer_step": record["optimizer_step"],
            "prediction": {
                "next_capability": cap_prediction,
                "next_functional_state": predicted,
            },
            "truth": {
                "next_capability": cap_truth,
            },
            "errors": {
                "state_nrmse": state_nrmse,
                "capability_absolute_error": cap_absolute_error,
                "state_error": state_error,
                "capability_error": capability_error,
                "capability_direction_error": direction_error,
                "decline_label_error": decline_error,
                "development_one_step_error": state_error or capability_error or direction_error or decline_error,
                "largest_normalized_state_residuals": residuals,
            },
        }
        rows.append({**material, "adjudication_sha256": payload_sha256(material)})
    return rows, fit


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(rows),
        "development_one_step_error_count": sum(row["errors"]["development_one_step_error"] for row in rows),
        "state_error_count": sum(row["errors"]["state_error"] for row in rows),
        "capability_error_count": sum(row["errors"]["capability_error"] for row in rows),
        "capability_direction_error_count": sum(row["errors"]["capability_direction_error"] for row in rows),
        "decline_label_error_count": sum(row["errors"]["decline_label_error"] for row in rows),
        "mean_state_nrmse": float(np.mean([row["errors"]["state_nrmse"] for row in rows])),
        "mean_capability_absolute_error": float(np.mean([row["errors"]["capability_absolute_error"] for row in rows])),
    }


def build_one_step_500_package(
    *,
    selection_path: Path,
    index_path: Path,
    screening_audit_path: Path,
    prior_submission_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    selection, index_rows = _selected_index_rows(selection_path, index_path)
    prior_manifest = read_json(prior_submission_root / "artifact_manifest.json")
    feature_descriptor = next(
        row for row in prior_manifest["stage_files"] if row["path"].endswith("stepwise_feature_cache.json")
    )
    feature_path = prior_submission_root / "stepwise_feature_cache.json"
    require(file_sha256(feature_path) == feature_descriptor["sha256"], "ONE_STEP_500_FEATURE_CACHE_HASH_MISMATCH")
    feature = read_json(feature_path)
    require(feature.get("status") == "COMPLETE", "ONE_STEP_500_FEATURE_CACHE_NOT_COMPLETE")
    state_map = _feature_cache_state_map(feature)
    records = [
        _record_from_sources(
            selection_row=selection_row,
            index_row=index_row,
            state_map=state_map,
        )
        for selection_row, index_row in zip(selection, index_rows)
    ]
    records.sort(key=lambda row: (row["entry_id"], row["window_id"], row["optimizer_step"], row["sample_id"]))
    require(len(records) == REQUIRED_SELECTION_COUNT, "ONE_STEP_500_RECORD_COUNT_INVALID")
    evidence_path = output_root / "selected_500_csrg_evidence.jsonl"
    _write_jsonl(evidence_path, records)

    screening = read_json(screening_audit_path)
    capability_tolerance = float(screening["development_frozen_absolute_error_tolerance"])
    entries = sorted({row["entry_id"] for row in records})
    require(HELD_OUT_ENTRY_ID not in entries and len(entries) == 12, "ONE_STEP_500_DEVELOPMENT_ENTRY_SET_INVALID")
    adjudications: list[dict[str, Any]] = []
    folds = []
    for held_out_entry in entries:
        training = [row for row in records if row["entry_id"] != held_out_entry]
        target = [row for row in records if row["entry_id"] == held_out_entry]
        fold_rows, fit = _adjudicate_fold(training, target, capability_tolerance)
        adjudications.extend(fold_rows)
        folds.append(
            {
                "held_out_entry_id": held_out_entry,
                "training_entry_count": len({row["entry_id"] for row in training}),
                "training_record_count": len(training),
                "target_record_count": len(target),
                "fit_schema_sha256": payload_sha256(fit),
                "metrics": _aggregate(fold_rows),
            }
        )
    adjudications.sort(key=lambda row: row["sample_id"])
    require(len(adjudications) == REQUIRED_SELECTION_COUNT, "ONE_STEP_500_ADJUDICATION_COUNT_INVALID")
    adjudication_path = output_root / "baseline_adjudications.jsonl"
    _write_jsonl(adjudication_path, adjudications)
    errors = [row for row in adjudications if row["errors"]["development_one_step_error"]]
    error_path = output_root / "development_one_step_errors.jsonl"
    _write_jsonl(error_path, errors)

    protocol = {
        "schema": "nanogpt-one-step-500-csrg-pilot-protocol-v1",
        "status": "FROZEN_EXECUTED",
        "population": "500 development records selected by the preceding capability-screening baseline",
        "outer_partition": "leave one of the twelve development entries out; global held-out entry never enters",
        "model": "fixed multioutput ridge over X_t and current formed U_t predicting adjacent compact CSRG state and capability deltas",
        "ridge_alpha": RIDGE_ALPHA,
        "state_nrmse_tolerance": STATE_NRMSE_TOLERANCE,
        "capability_absolute_error_tolerance": capability_tolerance,
        "capability_zero_band": CAPABILITY_ZERO_BAND,
        "decline_delta_threshold": DECLINE_DELTA_THRESHOLD,
        "error_rule": "state NRMSE exceeds tolerance OR capability absolute error exceeds tolerance OR capability direction is wrong OR decline label is wrong",
        "claim_limit": "development challenge errors, not an unbiased unseen-run scientific result and not proof of state irreducibility",
    }
    write_json(output_root / "PILOT_PROTOCOL.json", protocol)

    summary_material = {
        "schema": SCHEMA,
        "status": "PASS",
        "selected_record_count": len(records),
        "development_entry_count": len(entries),
        "global_held_out_entry_id": HELD_OUT_ENTRY_ID,
        "global_held_out_used": False,
        "aggregate": _aggregate(adjudications),
        "folds": folds,
        "error_record_count": len(errors),
        "raw_tensor_payloads_copied": 0,
        "new_gfg_constructed": False,
        "gpu_execution": False,
        "persistent_ai_session_used_during_evidence_build": False,
    }
    summary = {**summary_material, "summary_sha256": payload_sha256(summary_material)}
    write_json(output_root / "SUMMARY.json", summary)

    source_files = {
        "selection": selection_path,
        "lightweight_index": index_path,
        "screening_audit": screening_audit_path,
        "prior_feature_cache": feature_path,
        "prior_artifact_manifest": prior_submission_root / "artifact_manifest.json",
    }
    output_files = [
        evidence_path,
        adjudication_path,
        error_path,
        output_root / "PILOT_PROTOCOL.json",
        output_root / "SUMMARY.json",
    ]
    manifest_material = {
        "schema": "nanogpt-one-step-500-csrg-pilot-manifest-v1",
        "status": "PASS",
        "sources": {
            name: {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for name, path in source_files.items()
        },
        "outputs": {
            path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in output_files
        },
    }
    manifest = {**manifest_material, "manifest_sha256": payload_sha256(manifest_material)}
    write_json(output_root / "MANIFEST.json", manifest)
    return {"summary": summary, "manifest": manifest}


def validate_one_step_500_package(output_root: Path) -> dict[str, Any]:
    manifest = read_json(output_root / "MANIFEST.json")
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(material) == manifest["manifest_sha256"], "ONE_STEP_500_MANIFEST_HASH_INVALID")
    for name, descriptor in manifest["outputs"].items():
        path = output_root / name
        require(path.is_file(), f"ONE_STEP_500_OUTPUT_MISSING:{name}")
        require(path.stat().st_size == descriptor["size_bytes"], f"ONE_STEP_500_OUTPUT_SIZE_MISMATCH:{name}")
        require(file_sha256(path) == descriptor["sha256"], f"ONE_STEP_500_OUTPUT_HASH_MISMATCH:{name}")
    evidence = _read_jsonl(output_root / "selected_500_csrg_evidence.jsonl")
    adjudications = _read_jsonl(output_root / "baseline_adjudications.jsonl")
    errors = _read_jsonl(output_root / "development_one_step_errors.jsonl")
    require(len(evidence) == len(adjudications) == REQUIRED_SELECTION_COUNT, "ONE_STEP_500_VALIDATION_COUNT_MISMATCH")
    require(len({row["sample_id"] for row in evidence}) == REQUIRED_SELECTION_COUNT, "ONE_STEP_500_VALIDATION_ID_DUPLICATE")
    require(all(row["entry_id"] != HELD_OUT_ENTRY_ID for row in evidence), "ONE_STEP_500_VALIDATION_HELD_OUT_LEAKAGE")
    require(all(row["representation_disclosure"]["tensor_payloads_copied"] == 0 for row in evidence), "ONE_STEP_500_TENSOR_COPY_DECLARATION_INVALID")
    require(len(errors) == sum(row["errors"]["development_one_step_error"] for row in adjudications), "ONE_STEP_500_ERROR_LEDGER_MISMATCH")
    require(not list(output_root.rglob("*.npy")), "ONE_STEP_500_TENSOR_PAYLOAD_PRESENT")
    validation_material = {
        "schema": "nanogpt-one-step-500-csrg-pilot-validation-v1",
        "status": "PASS",
        "manifest_sha256": manifest["manifest_sha256"],
        "evidence_record_count": len(evidence),
        "adjudication_record_count": len(adjudications),
        "error_record_count": len(errors),
        "held_out_leakage": False,
        "tensor_payloads_present": False,
        "new_gfg_constructed": False,
    }
    validation = {**validation_material, "validation_sha256": payload_sha256(validation_material)}
    write_json(output_root / "VALIDATION.json", validation)
    return validation


__all__ = [
    "HELD_OUT_ENTRY_ID",
    "REQUIRED_SELECTION_COUNT",
    "RIDGE_ALPHA",
    "SCHEMA",
    "STATE_NRMSE_TOLERANCE",
    "build_one_step_500_package",
    "validate_one_step_500_package",
]
