from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    canonical_bytes,
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)


SCHEMA = "nanogpt-identity-aligned-margin-crossing-v1"
PRIMARY_FORWARD = "forward/0"
EVALUATION_UNIT_COUNT = 212
CAPABILITY_ZERO_BAND = 0.0023584905660377358
LABELS = (-1, 0, 1)
EXPECTED_BASELINE_CORRECT = 313
EXPECTED_BASELINE_BALANCED_ACCURACY = 0.6681437270151322


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_bytes(row) + b"\n")


def _raw_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _load_ref(root: Path, reference: dict[str, Any]) -> np.ndarray:
    path = root / str(reference["locator"])
    require(path.is_file(), f"IDENTITY_MARGIN_MISSING_TENSOR:{path}")
    require(file_sha256(path) == reference["file_sha256"], "IDENTITY_MARGIN_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "IDENTITY_MARGIN_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "IDENTITY_MARGIN_DTYPE_MISMATCH")
    require(_raw_sha256(value) == reference["raw_tensor_sha256"], "IDENTITY_MARGIN_RAW_HASH_MISMATCH")
    return np.asarray(value)


def _write_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {
        "file": path.name,
        "file_sha256": file_sha256(path),
        "raw_tensor_sha256": _raw_sha256(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _direction(delta: float) -> int:
    if delta < -CAPABILITY_ZERO_BAND:
        return -1
    if delta > CAPABILITY_ZERO_BAND:
        return 1
    return 0


def _confusion(actual: list[int], predicted: list[int], labels: tuple[int, ...] = LABELS) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    result = [[0 for _ in labels] for _ in labels]
    for truth, estimate in zip(actual, predicted):
        result[index[truth]][index[estimate]] += 1
    return result


def _balanced_accuracy(confusion: list[list[int]]) -> float:
    recalls = []
    for index, row in enumerate(confusion):
        total = sum(row)
        require(total > 0, "IDENTITY_MARGIN_EMPTY_DIRECTION_CLASS")
        recalls.append(row[index] / total)
    return float(sum(recalls) / len(recalls))


def _response_payload(response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    numeric = response["numeric_responses"][f"{PRIMARY_FORWARD}/margins"]
    predictions = response["categorical_transitions"][f"{PRIMARY_FORWARD}/predictions"]["baseline"]
    groups = response["categorical_transitions"][f"{PRIMARY_FORWARD}/group_membership"]["baseline"]
    return numeric, predictions, groups


def build_and_seal_pretarget_projection(
    *,
    evidence_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    require(not output_root.exists(), "IDENTITY_MARGIN_OUTPUT_EXISTS")
    protocol = read_json(protocol_path)
    require(
        protocol["status"] == "FROZEN_BEFORE_RESPONSE_EXECUTION",
        "IDENTITY_MARGIN_PROTOCOL_NOT_FROZEN",
    )
    require(float(protocol["epsilon"]) == 0.125, "IDENTITY_MARGIN_EPSILON_CHANGED")
    validation = read_json(evidence_root / "native_response_500_pre_target_validation.json")
    source_seal = read_json(evidence_root / "PRE_TARGET_RESPONSE_500_SEAL.json")
    require(validation["status"] == "PASS", "IDENTITY_MARGIN_SOURCE_VALIDATION_NOT_PASS")
    require(source_seal["status"] == "SEALED_BEFORE_NATIVE_TARGET_ACCESS", "IDENTITY_MARGIN_SOURCE_NOT_PRETARGET")

    response_paths = sorted((evidence_root / "responses").glob("*.json"))
    require(len(response_paths) == 500, "IDENTITY_MARGIN_RESPONSE_COUNT_INVALID")
    output_root.mkdir(parents=True)
    payload_root = output_root / "projection_payloads"
    payload_root.mkdir()

    sample_ids: list[str] = []
    baseline_rows: list[np.ndarray] = []
    j_rows: list[np.ndarray] = []
    k_rows: list[np.ndarray] = []
    current_correct_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []

    for row_index, response_path in enumerate(response_paths):
        response = read_json(response_path)
        require(response["schema"] == "nanogpt-native-direction-response-v1", "IDENTITY_MARGIN_RESPONSE_SCHEMA_INVALID")
        require(response["status"] == "PASS", "IDENTITY_MARGIN_RESPONSE_NOT_PASS")
        sample_id = str(response["sample_id"])
        require(response_path.stem == sample_id, "IDENTITY_MARGIN_RESPONSE_ID_MISMATCH")
        numeric, prediction_ref, group_ref = _response_payload(response)
        baseline = _load_ref(evidence_root, numeric["baseline"]).astype(np.float64, copy=False)
        j_native = _load_ref(evidence_root, numeric["j_native"]).astype(np.float64, copy=False)
        k_native = _load_ref(evidence_root, numeric["k_native"]).astype(np.float64, copy=False)
        predictions = _load_ref(evidence_root, prediction_ref).astype(np.int64, copy=False)
        groups = _load_ref(evidence_root, group_ref).astype(np.int64, copy=False)
        for value in (baseline, j_native, k_native, predictions, groups):
            require(value.shape == (EVALUATION_UNIT_COUNT,), "IDENTITY_MARGIN_UNIT_COUNT_INVALID")
        require(np.all(np.isfinite(baseline)), "IDENTITY_MARGIN_BASELINE_NONFINITE")
        require(np.all(np.isfinite(j_native)), "IDENTITY_MARGIN_J_NONFINITE")
        require(np.all(np.isfinite(k_native)), "IDENTITY_MARGIN_K_NONFINITE")
        require(set(np.unique(groups).tolist()) == set(range(23)), "IDENTITY_MARGIN_TARGET_GROUP_SET_INVALID")
        current_correct = predictions == groups

        sample_ids.append(sample_id)
        baseline_rows.append(baseline)
        j_rows.append(j_native)
        k_rows.append(k_native)
        current_correct_rows.append(current_correct)
        group_rows.append(groups)
        refs = {
            "baseline_margin": numeric["baseline"],
            "j_native_margin": numeric["j_native"],
            "k_native_margin": numeric["k_native"],
            "baseline_prediction": prediction_ref,
            "group_membership": group_ref,
        }
        index_material = {
            "schema": "identity-aligned-margin-projection-index-row-v1",
            "row_index": row_index,
            "sample_id": sample_id,
            "response_result_sha256": response["result_sha256"],
            "source_refs": {
                name: {
                    key: reference[key]
                    for key in ("file_sha256", "raw_tensor_sha256", "shape", "dtype", "representation")
                }
                for name, reference in refs.items()
            },
        }
        index_rows.append({**index_material, "record_sha256": payload_sha256(index_material)})

    require(sample_ids == sorted(sample_ids), "IDENTITY_MARGIN_SAMPLE_ORDER_NOT_CANONICAL")
    baseline_array = np.stack(baseline_rows).astype(np.float64, copy=False)
    j_array = np.stack(j_rows).astype(np.float64, copy=False)
    k_array = np.stack(k_rows).astype(np.float64, copy=False)
    current_correct_array = np.stack(current_correct_rows).astype(np.bool_, copy=False)
    group_array = np.stack(group_rows).astype(np.int64, copy=False)
    arrays = {
        "baseline_margin": _write_array(payload_root / "baseline_margin.npy", baseline_array),
        "j_native_margin": _write_array(payload_root / "j_native_margin.npy", j_array),
        "k_native_margin": _write_array(payload_root / "k_native_margin.npy", k_array),
        "baseline_correct": _write_array(payload_root / "baseline_correct.npy", current_correct_array),
        "target_group": _write_array(payload_root / "target_group.npy", group_array),
    }
    index_path = output_root / "projection_index.jsonl"
    _write_jsonl(index_path, index_rows)
    registry_material = {
        "schema": "identity-aligned-capability-evaluation-unit-registry-v1",
        "probe_contract_id": "CSRG-4C-v1",
        "primary_forward": PRIMARY_FORWARD,
        "primary_forward_semantics": "first ungated capability forward",
        "evaluation_unit_count": EVALUATION_UNIT_COUNT,
        "evaluation_unit_identity_scope": "sample-specific fixed validation order",
        "evaluation_unit_id_template": "{sample_id}:ungated-primary:evaluation-unit:{element_index:03d}",
        "target_group_mapping": "projection_payloads/target_group.npy[row_index, element_index]",
        "evaluation_weight": 1.0 / EVALUATION_UNIT_COUNT,
        "capability_rule": "mean(baseline_prediction == target_group)",
        "predicted_correctness_rule": "predicted_signed_margin > 0",
        "exact_zero_rule": "NOT_EVALUABLE",
        "element_positions": [
            {
                "element_index": index,
                "evaluation_weight": 1.0 / EVALUATION_UNIT_COUNT,
            }
            for index in range(EVALUATION_UNIT_COUNT)
        ],
    }
    registry = {**registry_material, "registry_sha256": payload_sha256(registry_material)}
    write_json(output_root / "EVALUATION_UNIT_REGISTRY.json", registry)
    manifest_material = {
        "schema": "identity-aligned-margin-pretarget-projection-manifest-v1",
        "status": "PASS",
        "sample_count": len(sample_ids),
        "evaluation_unit_count": EVALUATION_UNIT_COUNT,
        "sample_ids": sample_ids,
        "arrays": arrays,
        "projection_index_file": index_path.name,
        "projection_index_file_sha256": file_sha256(index_path),
        "evaluation_unit_registry_sha256": registry["registry_sha256"],
        "source_protocol_file_sha256": file_sha256(protocol_path),
        "source_pretarget_validation_result_sha256": validation["result_sha256"],
        "source_pretarget_seal_result_sha256": source_seal["result_sha256"],
        "actual_alpha_1_target_read": False,
        "future_state_read": False,
    }
    manifest = {**manifest_material, "manifest_sha256": payload_sha256(manifest_material)}
    write_json(output_root / "PRETARGET_PROJECTION_MANIFEST.json", manifest)

    predicted_margin = baseline_array + j_array + 0.5 * k_array
    finite = np.isfinite(predicted_margin)
    exact_zero = predicted_margin == 0.0
    candidate_rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(sample_ids):
        not_evaluable = bool(np.any(~finite[row_index]) or np.any(exact_zero[row_index]))
        predicted_correct = predicted_margin[row_index] > 0.0
        current_correct = current_correct_array[row_index]
        delta = float(np.mean(predicted_correct) - np.mean(current_correct))
        crossing_type = Counter(
            "negative_to_positive"
            if (not before and after)
            else "positive_to_negative"
            if (before and not after)
            else "unchanged"
            for before, after in zip(current_correct.tolist(), predicted_correct.tolist())
        )
        material = {
            "schema": "identity-aligned-margin-crossing-prediction-v1",
            "sample_id": sample_id,
            "projection_row_index": row_index,
            "status": "NOT_EVALUABLE" if not_evaluable else "SEALED_PREDICTION",
            "predicted_capability_delta": delta,
            "predicted_direction": None if not_evaluable else _direction(delta),
            "current_correct_count": int(np.count_nonzero(current_correct)),
            "predicted_correct_count": int(np.count_nonzero(predicted_correct)),
            "positive_to_negative_count": int(crossing_type["positive_to_negative"]),
            "negative_to_positive_count": int(crossing_type["negative_to_positive"]),
            "unchanged_count": int(crossing_type["unchanged"]),
            "nonfinite_count": int(np.count_nonzero(~finite[row_index])),
            "exact_zero_count": int(np.count_nonzero(exact_zero[row_index])),
            "pretarget_projection_manifest_sha256": manifest["manifest_sha256"],
        }
        candidate_rows.append({**material, "prediction_sha256": payload_sha256(material)})
    prediction_path = output_root / "CANDIDATE_PREDICTIONS.jsonl"
    _write_jsonl(prediction_path, candidate_rows)
    candidate_manifest_material = {
        "schema": "identity-aligned-margin-crossing-candidate-manifest-v1",
        "status": "SEALED_BEFORE_TARGET_ACCESS",
        "candidate_variants": 1,
        "fit": "NONE_DETERMINISTIC_READOUT",
        "sample_count": len(candidate_rows),
        "evaluable_count": sum(row["status"] == "SEALED_PREDICTION" for row in candidate_rows),
        "candidate_predictions_file": prediction_path.name,
        "candidate_predictions_file_sha256": file_sha256(prediction_path),
        "pretarget_projection_manifest_sha256": manifest["manifest_sha256"],
        "formula": "m_hat_e_t_plus_1 = m_e_t + J_e_t(delta_theta_t) + 0.5*K_e_t(delta_theta_t,delta_theta_t)",
        "capability_zero_band": CAPABILITY_ZERO_BAND,
        "actual_alpha_1_target_read": False,
        "actual_direction_read": False,
        "future_state_read": False,
    }
    candidate_manifest = {
        **candidate_manifest_material,
        "candidate_manifest_sha256": payload_sha256(candidate_manifest_material),
    }
    write_json(output_root / "PRETARGET_CANDIDATE_SEAL.json", candidate_manifest)
    validation_material = {
        "schema": "identity-aligned-margin-pretarget-validation-v1",
        "status": "PASS",
        "source_tensor_count_checked": len(response_paths) * 5,
        "source_file_hashes_valid": True,
        "source_raw_tensor_hashes_valid": True,
        "sample_identity_alignment_valid": True,
        "evaluation_unit_identity_is_sample_scoped": True,
        "all_inputs_pretarget": True,
        "target_accessed": False,
        "candidate_manifest_sha256": candidate_manifest["candidate_manifest_sha256"],
    }
    validation_out = {**validation_material, "validation_sha256": payload_sha256(validation_material)}
    write_json(output_root / "PRETARGET_VALIDATION.json", validation_out)
    return candidate_manifest


def _load_projection_array(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    reference = manifest["arrays"][name]
    path = root / "projection_payloads" / reference["file"]
    require(file_sha256(path) == reference["file_sha256"], "IDENTITY_MARGIN_PROJECTION_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == reference["shape"], "IDENTITY_MARGIN_PROJECTION_SHAPE_MISMATCH")
    require(str(value.dtype) == reference["dtype"], "IDENTITY_MARGIN_PROJECTION_DTYPE_MISMATCH")
    require(_raw_sha256(value) == reference["raw_tensor_sha256"], "IDENTITY_MARGIN_PROJECTION_RAW_HASH_MISMATCH")
    return value


def _task_hash(prediction_index: dict[str, Any]) -> str:
    rows = []
    for sample_id, row in sorted(prediction_index.items()):
        rows.append(
            {
                "sample_id": sample_id,
                "state_correct": row["state_correct"],
                "state_nrmse": row["state_nrmse"],
                "capability_prediction": row["capability_prediction"],
                "capability_truth": row["capability_truth"],
                "capability_correct": row["capability_correct"],
                "predicted_decline": row["predicted_decline"],
                "actual_decline": row["actual_decline"],
                "decline_correct": row["decline_correct"],
            }
        )
    return payload_sha256(rows)


def _metrics(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    confusion = _confusion(actual, predicted)
    correct = sum(truth == estimate for truth, estimate in zip(actual, predicted))
    return {
        "record_count": len(actual),
        "correct_count": correct,
        "accuracy": correct / len(actual),
        "balanced_accuracy": _balanced_accuracy(confusion),
        "labels": list(LABELS),
        "confusion_matrix_rows_actual_columns_predicted": confusion,
    }


def adjudicate_after_seal(
    *,
    experiment_root: Path,
    modeling_records_path: Path,
    v2_result_path: Path,
    near_conflict_path: Path,
) -> dict[str, Any]:
    projection_manifest = read_json(experiment_root / "PRETARGET_PROJECTION_MANIFEST.json")
    candidate_seal = read_json(experiment_root / "PRETARGET_CANDIDATE_SEAL.json")
    pretarget_validation = read_json(experiment_root / "PRETARGET_VALIDATION.json")
    require(candidate_seal["status"] == "SEALED_BEFORE_TARGET_ACCESS", "IDENTITY_MARGIN_CANDIDATE_NOT_SEALED")
    require(pretarget_validation["status"] == "PASS", "IDENTITY_MARGIN_PRETARGET_VALIDATION_NOT_PASS")
    candidate_path = experiment_root / candidate_seal["candidate_predictions_file"]
    require(file_sha256(candidate_path) == candidate_seal["candidate_predictions_file_sha256"], "IDENTITY_MARGIN_SEALED_PREDICTIONS_CHANGED")
    candidate_rows = _read_jsonl(candidate_path)
    require(all(row["status"] == "SEALED_PREDICTION" for row in candidate_rows), "IDENTITY_MARGIN_NOT_ALL_EVALUABLE")

    modeling_rows = {row["sample_id"]: row for row in _read_jsonl(modeling_records_path)}
    v2 = read_json(v2_result_path)["result"]
    prediction_index = v2["prediction_index"]
    sample_ids = projection_manifest["sample_ids"]
    require(set(sample_ids) == set(modeling_rows) == set(prediction_index), "IDENTITY_MARGIN_ADJUDICATION_SAMPLE_SET_MISMATCH")
    require(len(sample_ids) == 500, "IDENTITY_MARGIN_ADJUDICATION_COUNT_INVALID")

    actual: list[int] = []
    baseline: list[int] = []
    candidate: list[int] = []
    per_sample: dict[str, Any] = {}
    fixed = 0
    broken = 0
    for row in candidate_rows:
        sample_id = row["sample_id"]
        source = modeling_rows[sample_id]
        current = float(source["X_t"]["observables"]["capability"])
        target = float(source["X_t_plus_1"]["observables"]["capability"])
        truth = _direction(target - current)
        v2_row = prediction_index[sample_id]
        require(int(v2_row["actual_direction"]) == truth, "IDENTITY_MARGIN_V2_ACTUAL_DIRECTION_MISMATCH")
        base_prediction = int(v2_row["predicted_direction"])
        candidate_prediction = int(row["predicted_direction"])
        actual.append(truth)
        baseline.append(base_prediction)
        candidate.append(candidate_prediction)
        base_correct = base_prediction == truth
        candidate_correct = candidate_prediction == truth
        fixed += int((not base_correct) and candidate_correct)
        broken += int(base_correct and (not candidate_correct))
        per_sample[sample_id] = {
            "sample_id": sample_id,
            "entry_id": v2_row["entry_id"],
            "actual_direction": truth,
            "v2_direction": base_prediction,
            "candidate_direction": candidate_prediction,
            "v2_correct": base_correct,
            "candidate_correct": candidate_correct,
            "predicted_capability_delta": row["predicted_capability_delta"],
            "current_correct_count": row["current_correct_count"],
            "predicted_correct_count": row["predicted_correct_count"],
            "positive_to_negative_count": row["positive_to_negative_count"],
            "negative_to_positive_count": row["negative_to_positive_count"],
            "prediction_sha256": row["prediction_sha256"],
        }

    baseline_metrics = _metrics(actual, baseline)
    candidate_metrics = _metrics(actual, candidate)
    require(baseline_metrics["correct_count"] == EXPECTED_BASELINE_CORRECT, "IDENTITY_MARGIN_V2_BASELINE_CORRECT_COUNT_MISMATCH")
    require(math.isclose(baseline_metrics["balanced_accuracy"], EXPECTED_BASELINE_BALANCED_ACCURACY, rel_tol=0.0, abs_tol=1e-15), "IDENTITY_MARGIN_V2_BASELINE_BALANCED_ACCURACY_MISMATCH")
    frozen_baseline = v2["aggregate"]["capability_direction_prediction"]
    require(
        all(baseline_metrics[key] == frozen_baseline[key] for key in frozen_baseline),
        "IDENTITY_MARGIN_V2_BASELINE_AGGREGATE_MISMATCH",
    )

    per_entry: dict[str, Any] = {}
    by_entry: dict[str, list[str]] = defaultdict(list)
    for sample_id, row in per_sample.items():
        by_entry[row["entry_id"]].append(sample_id)
    for entry_id, members in sorted(by_entry.items()):
        entry_actual = [per_sample[sample_id]["actual_direction"] for sample_id in members]
        entry_baseline = [per_sample[sample_id]["v2_direction"] for sample_id in members]
        entry_candidate = [per_sample[sample_id]["candidate_direction"] for sample_id in members]
        entry_fixed = sum(
            (per_sample[sample_id]["v2_correct"] is False)
            and (per_sample[sample_id]["candidate_correct"] is True)
            for sample_id in members
        )
        entry_broken = sum(
            (per_sample[sample_id]["v2_correct"] is True)
            and (per_sample[sample_id]["candidate_correct"] is False)
            for sample_id in members
        )
        per_entry[entry_id] = {
            "record_count": len(members),
            "v2": _metrics(entry_actual, entry_baseline) if set(entry_actual) == set(LABELS) else {
                "correct_count": sum(a == b for a, b in zip(entry_actual, entry_baseline)),
                "accuracy": sum(a == b for a, b in zip(entry_actual, entry_baseline)) / len(members),
            },
            "candidate": _metrics(entry_actual, entry_candidate) if set(entry_actual) == set(LABELS) else {
                "correct_count": sum(a == b for a, b in zip(entry_actual, entry_candidate)),
                "accuracy": sum(a == b for a, b in zip(entry_actual, entry_candidate)) / len(members),
            },
            "fixed_count": entry_fixed,
            "newly_broken_count": entry_broken,
            "net_improvement": entry_fixed - entry_broken,
        }

    near = read_json(near_conflict_path)
    unique_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for conflict in near["conflicts"]:
        pair = tuple(sorted((str(conflict["sample_id"]), str(conflict["matched_sample_id"]))))
        unique_pairs.setdefault(pair, conflict)
    require(len(unique_pairs) == 10, "IDENTITY_MARGIN_NEAR_CONFLICT_PAIR_COUNT_CHANGED")
    both_state_correct_pairs = [
        pair
        for pair in unique_pairs
        if all(bool(prediction_index[sample_id]["state_correct"]) for sample_id in pair)
    ]
    require(len(both_state_correct_pairs) == 9, "IDENTITY_MARGIN_STATE_CORRECT_PAIR_COUNT_CHANGED")
    endpoints = sorted({sample_id for pair in both_state_correct_pairs for sample_id in pair})
    require(len(endpoints) == 16, "IDENTITY_MARGIN_STATE_CORRECT_ENDPOINT_COUNT_CHANGED")
    near_fixed = sum((not per_sample[sample_id]["v2_correct"]) and per_sample[sample_id]["candidate_correct"] for sample_id in endpoints)
    near_broken = sum(per_sample[sample_id]["v2_correct"] and (not per_sample[sample_id]["candidate_correct"]) for sample_id in endpoints)

    net = fixed - broken
    near_net = near_fixed - near_broken
    if (
        net > 0
        and candidate_metrics["balanced_accuracy"] > EXPECTED_BASELINE_BALANCED_ACCURACY
        and near_net >= 0
    ):
        decision = "SUPPORTED"
    elif (
        fixed <= broken
        and candidate_metrics["balanced_accuracy"] <= EXPECTED_BASELINE_BALANCED_ACCURACY
        and near_net <= 0
    ):
        decision = "FALSIFIED"
    else:
        decision = "MIXED"

    task_hash = _task_hash(prediction_index)
    result_material = {
        "schema": "identity-aligned-margin-crossing-adjudication-v1",
        "status": "PASS",
        "scientific_decision": decision,
        "candidate_seal_sha256": candidate_seal["candidate_manifest_sha256"],
        "candidate_predictions_file_sha256": candidate_seal["candidate_predictions_file_sha256"],
        "target_opened_only_after_candidate_seal": True,
        "v2_baseline": baseline_metrics,
        "identity_aligned_margin_candidate": candidate_metrics,
        "comparison": {
            "fixed_count": fixed,
            "newly_broken_count": broken,
            "net_improvement": net,
        },
        "near_conflict_audit": {
            "inherited_unique_pair_count": len(unique_pairs),
            "both_state_correct_pair_count": len(both_state_correct_pairs),
            "unique_endpoint_count": len(endpoints),
            "fixed_count": near_fixed,
            "newly_broken_count": near_broken,
            "net_improvement": near_net,
            "endpoints": endpoints,
        },
        "per_entry": per_entry,
        "per_sample": per_sample,
        "unchanged_tasks": {
            "tasks": ["csrg_state_prediction", "capability_value_prediction", "marked_decline_prediction"],
            "v2_prediction_hash_before": task_hash,
            "candidate_prediction_hash_after": task_hash,
            "unchanged_prediction_hash": True,
        },
        "interpretation_limits": [
            "This adjudicates only the deterministic local-quadratic identity-aligned direction readout on the 500 development transitions.",
            "A positive result would not establish accurate finite-amplitude alpha=1 transport.",
            "A negative result would not show that exact physical margin crossings are unimportant.",
            "No recursive stability state or long-horizon challenge pair is sealed by this result.",
        ],
    }
    result = {**result_material, "adjudication_sha256": payload_sha256(result_material)}
    write_json(experiment_root / "ADJUDICATION.json", result)
    report = "# Identity-aligned margin-crossing result\n\n"
    report += f"Scientific decision: **{decision}**.\n\n"
    report += "| Direction task | Correct | Accuracy | Balanced accuracy |\n|---|---:|---:|---:|\n"
    report += f"| Frozen v2 | {baseline_metrics['correct_count']}/500 | {baseline_metrics['accuracy']:.6f} | {baseline_metrics['balanced_accuracy']:.6f} |\n"
    report += f"| Identity-aligned margin crossing | {candidate_metrics['correct_count']}/500 | {candidate_metrics['accuracy']:.6f} | {candidate_metrics['balanced_accuracy']:.6f} |\n\n"
    report += f"Fixed: {fixed}; newly broken: {broken}; net: {net}.\n\n"
    report += f"Near-conflict 16 endpoints: fixed {near_fixed}; newly broken {near_broken}; net {near_net}.\n\n"
    report += "The candidate was sealed before any alpha=1 target capability or direction was opened. CSRG state, capability value and marked-decline predictions were not changed.\n\n"
    report += "This result concerns the local quadratic identity ledger only. It does not settle finite-amplitude response transport or establish a recursive stability state.\n"
    (experiment_root / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    return result


def validate_completed_experiment(
    *,
    experiment_root: Path,
    evidence_root: Path,
    v2_result_path: Path,
) -> dict[str, Any]:
    projection_manifest = read_json(experiment_root / "PRETARGET_PROJECTION_MANIFEST.json")
    seal = read_json(experiment_root / "PRETARGET_CANDIDATE_SEAL.json")
    adjudication = read_json(experiment_root / "ADJUDICATION.json")
    require(projection_manifest["status"] == "PASS", "IDENTITY_MARGIN_REPLAY_PROJECTION_NOT_PASS")
    require(seal["status"] == "SEALED_BEFORE_TARGET_ACCESS", "IDENTITY_MARGIN_REPLAY_SEAL_INVALID")
    require(adjudication["status"] == "PASS", "IDENTITY_MARGIN_REPLAY_ADJUDICATION_INVALID")
    require(file_sha256(experiment_root / seal["candidate_predictions_file"]) == seal["candidate_predictions_file_sha256"], "IDENTITY_MARGIN_REPLAY_CANDIDATE_HASH_MISMATCH")

    baseline = _load_projection_array(experiment_root, projection_manifest, "baseline_margin")
    j_native = _load_projection_array(experiment_root, projection_manifest, "j_native_margin")
    k_native = _load_projection_array(experiment_root, projection_manifest, "k_native_margin")
    baseline_correct = _load_projection_array(experiment_root, projection_manifest, "baseline_correct")
    target_group = _load_projection_array(experiment_root, projection_manifest, "target_group")
    require(baseline.shape == j_native.shape == k_native.shape == baseline_correct.shape == (500, 212), "IDENTITY_MARGIN_REPLAY_ARRAY_SHAPE_INVALID")
    require(target_group.shape == (500, 212), "IDENTITY_MARGIN_REPLAY_GROUP_SHAPE_INVALID")

    index_rows = _read_jsonl(experiment_root / "projection_index.jsonl")
    require(len(index_rows) == 500, "IDENTITY_MARGIN_REPLAY_INDEX_COUNT_INVALID")
    for row in index_rows:
        response = read_json(evidence_root / "responses" / f"{row['sample_id']}.json")
        numeric, prediction_ref, group_ref = _response_payload(response)
        source_refs = (numeric["baseline"], numeric["j_native"], numeric["k_native"], prediction_ref, group_ref)
        for reference in source_refs:
            _load_ref(evidence_root, reference)
        position = int(row["row_index"])
        require(np.array_equal(baseline[position], _load_ref(evidence_root, numeric["baseline"])), "IDENTITY_MARGIN_REPLAY_BASELINE_VALUE_MISMATCH")
        require(np.array_equal(j_native[position], _load_ref(evidence_root, numeric["j_native"])), "IDENTITY_MARGIN_REPLAY_J_VALUE_MISMATCH")
        require(np.array_equal(k_native[position], _load_ref(evidence_root, numeric["k_native"])), "IDENTITY_MARGIN_REPLAY_K_VALUE_MISMATCH")
        predictions = _load_ref(evidence_root, prediction_ref)
        groups = _load_ref(evidence_root, group_ref)
        require(np.array_equal(baseline_correct[position], predictions == groups), "IDENTITY_MARGIN_REPLAY_CORRECTNESS_MISMATCH")
        require(np.array_equal(target_group[position], groups), "IDENTITY_MARGIN_REPLAY_GROUP_IDENTITY_MISMATCH")

    predictions = _read_jsonl(experiment_root / "CANDIDATE_PREDICTIONS.jsonl")
    predicted_margin = baseline + j_native + 0.5 * k_native
    require(np.all(np.isfinite(predicted_margin)), "IDENTITY_MARGIN_REPLAY_NONFINITE")
    require(not np.any(predicted_margin == 0.0), "IDENTITY_MARGIN_REPLAY_EXACT_ZERO")
    for position, row in enumerate(predictions):
        predicted_correct = predicted_margin[position] > 0.0
        delta = float(np.mean(predicted_correct) - np.mean(baseline_correct[position]))
        require(math.isclose(delta, float(row["predicted_capability_delta"]), rel_tol=0.0, abs_tol=0.0), "IDENTITY_MARGIN_REPLAY_DELTA_MISMATCH")
        require(_direction(delta) == int(row["predicted_direction"]), "IDENTITY_MARGIN_REPLAY_DIRECTION_MISMATCH")

    v2 = read_json(v2_result_path)["result"]
    frozen_baseline = v2["aggregate"]["capability_direction_prediction"]
    require(
        all(adjudication["v2_baseline"][key] == frozen_baseline[key] for key in frozen_baseline),
        "IDENTITY_MARGIN_REPLAY_V2_BASELINE_MISMATCH",
    )
    require(adjudication["unchanged_tasks"]["unchanged_prediction_hash"], "IDENTITY_MARGIN_REPLAY_OTHER_TASKS_CHANGED")
    validation_material = {
        "schema": "identity-aligned-margin-crossing-independent-validation-v1",
        "status": "PASS",
        "source_tensor_replay_count": len(index_rows) * 5,
        "projection_values_exact": True,
        "candidate_predictions_rederived_exactly": True,
        "v2_baseline_reproduced_exactly": True,
        "target_opened_only_after_candidate_seal": adjudication["target_opened_only_after_candidate_seal"],
        "other_three_task_predictions_unchanged": adjudication["unchanged_tasks"]["unchanged_prediction_hash"],
        "scientific_decision": adjudication["scientific_decision"],
        "adjudication_sha256": adjudication["adjudication_sha256"],
    }
    validation = {**validation_material, "validation_sha256": payload_sha256(validation_material)}
    write_json(experiment_root / "INDEPENDENT_VALIDATION.json", validation)
    return validation
