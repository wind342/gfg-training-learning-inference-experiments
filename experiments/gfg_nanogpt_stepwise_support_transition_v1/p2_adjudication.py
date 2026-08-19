from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    read_json,
    require,
)

from .branches import _load_observation_arrays
from .execution import _checked_result, _read_checked
from .p2_response import P2_LABELS, P2_SCHEMA
from .p2_response_validator import _load_ref


SUPPORT_KEYS = (
    "necessity",
    "pair_backup",
    "single_failure_slack",
    "double_failure_slack",
    "effective_support",
    "support_concentration",
    "support_allocation",
)
PROJECTION_REPRODUCTION_ATOL = 1.0e-5


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _rms(values: Iterable[float]) -> float:
    rows = list(values)
    return math.sqrt(sum(value * value for value in rows) / len(rows)) if rows else 0.0


def _quantile(values: Iterable[float], fraction: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    position = fraction * (len(rows) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return rows[left]
    weight = position - left
    return rows[left] * (1.0 - weight) + rows[right] * weight


def _matrix_rows(value: np.ndarray) -> list[list[float]]:
    require(value.ndim == 2, "P2_PROJECTION_EXPECTED_MATRIX")
    return [[float(child) for child in row] for row in value]


def _raw_state_features(arrays: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, float]]:
    core: dict[str, float] = {}
    topology: dict[str, float] = {}
    necessity = _matrix_rows(arrays["necessity"])
    for index, row in enumerate(necessity):
        core[f"need_mean_c{index}"] = _mean(row)
        core[f"need_min_c{index}"] = min(row)
        core[f"need_rms_c{index}"] = _rms(row)
        topology[f"need_q10_c{index}"] = _quantile(row, 0.10)
        topology[f"need_q90_c{index}"] = _quantile(row, 0.90)
    backup = _matrix_rows(arrays["pair_backup"])
    for index, row in enumerate(backup):
        core[f"backup_mean_p{index}"] = _mean(row)
        core[f"backup_min_p{index}"] = min(row)
        topology[f"backup_q10_p{index}"] = _quantile(row, 0.10)
    for metric, prefix in (
        ("single_failure_slack", "single"),
        ("double_failure_slack", "double"),
        ("effective_support", "effective"),
    ):
        row = [float(value) for value in arrays[metric].reshape(-1)]
        core[f"{prefix}_mean"] = _mean(row)
        core[f"{prefix}_min"] = min(row)
        core[f"{prefix}_rms"] = _rms(row)
        topology[f"{prefix}_q10"] = _quantile(row, 0.10)
        topology[f"{prefix}_q50"] = _quantile(row, 0.50)
        topology[f"{prefix}_negative_fraction"] = sum(value < 0.0 for value in row) / len(row)
    concentration = [float(value) for value in arrays["support_concentration"].reshape(-1)]
    core["concentration_mean"] = _mean(concentration)
    core["concentration_max"] = max(concentration)
    core["concentration_rms"] = _rms(concentration)
    topology["concentration_q10"] = _quantile(concentration, 0.10)
    topology["concentration_q90"] = _quantile(concentration, 0.90)
    allocation = _matrix_rows(arrays["support_allocation"])
    for index, row in enumerate(allocation):
        core[f"allocation_mean_c{index}"] = _mean(row)
        topology[f"allocation_q10_c{index}"] = _quantile(row, 0.10)
        topology[f"allocation_q90_c{index}"] = _quantile(row, 0.90)
    dominance: list[float] = []
    entropy: list[float] = []
    for group in range(len(necessity[0])):
        magnitudes = [abs(necessity[component][group]) for component in range(len(necessity))]
        total = sum(magnitudes)
        if total <= 1e-30:
            dominance.append(0.0)
            entropy.append(0.0)
        else:
            probabilities = [value / total for value in magnitudes]
            dominance.append(max(probabilities))
            entropy.append(-sum(p * math.log(max(p, 1e-30)) for p in probabilities) / math.log(len(probabilities)))
    topology["necessity_dominance_mean"] = _mean(dominance)
    topology["necessity_entropy_mean"] = _mean(entropy)
    return core, topology


def project_state_150(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    core, topology = _raw_state_features(arrays)
    result: dict[str, float] = {}
    for prefix, values in (("s.", core), ("topology.", topology)):
        names = sorted(values)
        result.update({prefix + name: float(values[name]) if math.isfinite(float(values[name])) else 0.0 for name in names})
        result.update({prefix + "undefined__" + name: 0.0 if math.isfinite(float(values[name])) else 1.0 for name in names})
    require(len(result) == 150, f"P2_PROJECTION_COORDINATE_COUNT_INVALID:{len(result)}")
    return result


def _find_pilot_rows(path: Path, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("sample_id")) in sample_ids:
                result[str(row["sample_id"])] = row
    require(set(result) == sample_ids, "P2_PILOT_ROWS_MISSING")
    return result


def _projection_close(actual: dict[str, float], expected: dict[str, Any]) -> float:
    require(set(actual) == {f"s.{key}" for key in expected["support_state"]} | {f"topology.{key}" for key in expected["support_topology"]}, "P2_PROJECTION_SCHEMA_MISMATCH")
    maximum = 0.0
    for key, value in expected["support_state"].items():
        maximum = max(maximum, abs(actual[f"s.{key}"] - float(value)))
    for key, value in expected["support_topology"].items():
        maximum = max(maximum, abs(actual[f"topology.{key}"] - float(value)))
    return maximum


def _role_error(actual_delta: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    actual = actual_delta.astype(np.float64, copy=False).reshape(-1)
    predicted = prediction.astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    if not bool(np.any(finite)):
        return {"evaluable": False, "finite_coordinate_count": 0}
    actual = actual[finite]
    predicted = predicted[finite]
    residual = predicted - actual
    actual_rms = float(np.sqrt(np.mean(np.square(actual))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    return {
        "evaluable": True,
        "finite_coordinate_count": int(actual.size),
        "actual_delta_rms": actual_rms,
        "residual_rms": residual_rms,
        "nrmse": residual_rms / actual_rms if actual_rms > 0.0 else None,
        "actual_delta_exact_zero": actual_rms == 0.0,
    }


def _state_metrics(prediction: dict[str, float], truth: dict[str, float], scales: dict[str, float]) -> dict[str, Any]:
    names = sorted(truth)
    require(set(prediction) == set(truth), "P2_STATE_PREDICTION_SCHEMA_MISMATCH")
    require(all(name in scales and float(scales[name]) > 0.0 for name in names), "P2_STATE_SCALE_MISSING")
    residual = np.asarray([prediction[name] - truth[name] for name in names], dtype=np.float64)
    normalized = np.asarray([residual[index] / float(scales[name]) for index, name in enumerate(names)], dtype=np.float64)
    return {
        "coordinate_count": len(names),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "nrmse": float(np.sqrt(np.mean(np.square(normalized)))),
        "passes_frozen_state_tolerance": bool(np.sqrt(np.mean(np.square(normalized))) <= 1.0),
    }


def adjudicate_p2_response(
    *,
    formal_root: Path,
    response_root: Path,
    p2_protocol_path: Path,
    pilot_jsonl_path: Path,
    fitted_model_path: Path,
    proposal_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = read_json(p2_protocol_path)
    require(protocol["schema"] == P2_SCHEMA, "P2_ADJUDICATION_PROTOCOL_SCHEMA_INVALID")
    prevalidation = _read_checked(response_root / "p2_response_pre_target_validation.json", "nanogpt-p2-response-pre-target-validation-v1")
    replay = _read_checked(response_root / "p2_response_independent_replay_validation.json", "nanogpt-p2-independent-replay-validation-v1")
    require(prevalidation["status"] == replay["status"] == "PASS", "P2_ADJUDICATION_PREREQUISITE_NOT_PASS")
    proposal = read_json(proposal_path)
    require(proposal["package_name"] == protocol["protocol_id"], "P2_ADJUDICATION_PROPOSAL_ID_MISMATCH")
    model = read_json(fitted_model_path)
    scales = {str(key): float(value) for key, value in model["ridge"]["output_scales"].items() if str(key).startswith(("s.", "topology."))}
    require(len(scales) == 150, "P2_ADJUDICATION_SCALE_COUNT_INVALID")
    samples = {str(row["one_step_sample_id"]) for row in protocol["receivers"]}
    pilot_rows = _find_pilot_rows(pilot_jsonl_path, samples)

    baseline_arrays: dict[str, dict[str, np.ndarray]] = {}
    response_arrays: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for receiver_label in P2_LABELS:
        receiver_receipt = _read_checked(response_root / "receivers" / receiver_label / "receiver_receipt.json", "nanogpt-p2-receiver-receipt-v1")
        baseline_state_id = receiver_receipt["branches"]["baseline"]["state_id"]
        baseline_observation = _read_checked(response_root / "probe-observations" / "CSRG-4C-v1" / f"{baseline_state_id}.json", "nanogpt-stepwise-probe-observation-v1")
        baseline_arrays[receiver_label] = _load_observation_arrays(response_root, baseline_observation)
        for donor_label in P2_LABELS:
            response = _read_checked(response_root / "responses" / f"receiver-{receiver_label}-donor-{donor_label}.json", "nanogpt-p2-local-response-jk-v1")
            response_arrays[(receiver_label, donor_label)] = {
                "J": {key: _load_ref(response_root, row["j_first_order"]) for key, row in response["numeric_responses"].items()},
                "K": {key: _load_ref(response_root, row["k_curvature"]) for key, row in response["numeric_responses"].items()},
            }

    member_rows: list[dict[str, Any]] = []
    own_j_passes: list[bool] = []
    own_jk_passes: list[bool] = []
    receiver_local_majorities: list[bool] = []
    response_difference_detected = False
    for endpoint in protocol["receivers"]:
        label = str(endpoint["label"])
        other = "P2b" if label == "P2a" else "P2a"
        entry_root = formal_root / str(endpoint["entry_id"])
        target_path = entry_root / "probe-observations" / "CSRG-4C-v1" / f"{endpoint['sealed_native_target']['state_id']}.json"
        target = _read_checked(target_path, "nanogpt-stepwise-probe-observation-v1")
        require(target["probe_observation_id"] == endpoint["sealed_native_target"]["probe_observation_id"], f"P2_TARGET_PROBE_ID_MISMATCH:{label}")
        target_arrays = _load_observation_arrays(entry_root, target)
        baseline = baseline_arrays[label]
        own = response_arrays[(label, label)]
        cross = response_arrays[(other, label)]
        role_rows: list[dict[str, Any]] = []
        own_wins = 0
        cross_wins = 0
        ties = 0
        eligible = 0
        for key in sorted(own["J"]):
            actual_delta = target_arrays[key].astype(np.float64, copy=False) - baseline[key].astype(np.float64, copy=False)
            own_error = _role_error(actual_delta, own["J"][key])
            cross_error = _role_error(actual_delta, cross["J"][key])
            jk_error = _role_error(actual_delta, own["J"][key] + 0.5 * own["K"][key])
            decision = "NOT_EVALUABLE"
            if own_error.get("evaluable") and cross_error.get("evaluable") and not own_error.get("actual_delta_exact_zero"):
                eligible += 1
                if own_error["residual_rms"] < cross_error["residual_rms"]:
                    decision = "OWN_J_WIN"
                    own_wins += 1
                elif cross_error["residual_rms"] < own_error["residual_rms"]:
                    decision = "CROSS_J_WIN"
                    cross_wins += 1
                else:
                    decision = "TIE"
                    ties += 1
            elif own_error.get("actual_delta_exact_zero"):
                decision = "EXACT_ZERO_TARGET_ROLE_EXCLUDED"
            if np.any(np.not_equal(own["J"][key], cross["J"][key])):
                response_difference_detected = True
            role_rows.append({"role": key, "decision": decision, "own_j": own_error, "cross_j": cross_error, "own_j_plus_k_half": jk_error})
        receiver_local = own_wins > (eligible / 2.0)
        receiver_local_majorities.append(receiver_local)

        predicted_raw = {
            "own_j": {key: baseline[key] + own["J"][key] for key in SUPPORT_KEYS},
            "cross_j": {key: baseline[key] + cross["J"][key] for key in SUPPORT_KEYS},
            "own_j_plus_k_half": {key: baseline[key] + own["J"][key] + 0.5 * own["K"][key] for key in SUPPORT_KEYS},
        }
        projections = {name: project_state_150(value) for name, value in predicted_raw.items()}
        baseline_projection = project_state_150({key: baseline[key] for key in SUPPORT_KEYS})
        target_projection = project_state_150({key: target_arrays[key] for key in SUPPORT_KEYS})
        pilot = pilot_rows[str(endpoint["one_step_sample_id"])]
        baseline_projection_max_error = _projection_close(baseline_projection, pilot["X_t"])
        target_projection_max_error = _projection_close(target_projection, pilot["X_t_plus_1"])
        require(baseline_projection_max_error <= PROJECTION_REPRODUCTION_ATOL, f"P2_BASELINE_PROJECTION_DRIFT:{label}:{baseline_projection_max_error}")
        require(target_projection_max_error <= PROJECTION_REPRODUCTION_ATOL, f"P2_TARGET_PROJECTION_DRIFT:{label}:{target_projection_max_error}")
        state_metrics = {
            "persistence_baseline": _state_metrics(baseline_projection, target_projection, scales),
            **{name: _state_metrics(value, target_projection, scales) for name, value in projections.items()},
        }
        proposal_member = next(row for row in proposal["members"] if row["member_id"] == label)
        own_j_passes.append(bool(state_metrics["own_j"]["passes_frozen_state_tolerance"]))
        own_jk_passes.append(bool(state_metrics["own_j_plus_k_half"]["passes_frozen_state_tolerance"]))
        member_rows.append(
            {
                "member_id": label,
                "native_target_probe_id": target["probe_observation_id"],
                "native_target_probe_result_sha256": target["result_sha256"],
                "target_opened_only_after_seal": True,
                "role_comparison": {
                    "eligible_nonzero_numeric_role_count": eligible,
                    "own_j_win_count": own_wins,
                    "cross_j_win_count": cross_wins,
                    "tie_count": ties,
                    "own_j_strict_majority": receiver_local,
                    "roles": role_rows,
                },
                "state_metrics_150": state_metrics,
                "existing_frozen_pilot_state_nrmse": float(proposal_member["task_ledger"]["baseline_state_nrmse"]),
                "existing_selected_candidate_state_nrmse": float(proposal_member["task_ledger"]["candidate_state_nrmse"]),
                "projection_reproduction": {
                    "baseline_max_absolute_error": baseline_projection_max_error,
                    "target_max_absolute_error": target_projection_max_error,
                    "absolute_tolerance": PROJECTION_REPRODUCTION_ATOL,
                    "tolerance_semantics": "Numerical reproduction audit for float32 probe-summary aggregation only; not a scientific state-pass threshold.",
                    "coordinate_count": 150,
                },
                "secondary_unscored_capability": {
                    "baseline": float(pilot["X_t"]["observables"]["capability"]),
                    "native_target": float(pilot["X_t_plus_1"]["observables"]["capability"]),
                    "actual_delta": float(pilot["X_t_plus_1"]["observables"]["capability"] - pilot["X_t"]["observables"]["capability"]),
                },
            }
        )

    if all(receiver_local_majorities) and all(own_j_passes):
        outcome = "RECEIVER_LOCAL_FIRST_ORDER_SUPPORTED_FOR_P2_STATE"
    elif all(receiver_local_majorities) and not all(own_j_passes) and all(own_jk_passes):
        outcome = "REGISTERED_CURVATURE_REQUIRED_FOR_P2_STATE"
    elif all(receiver_local_majorities) and response_difference_detected and not (all(own_j_passes) or all(own_jk_passes)):
        outcome = "TWO_DIRECTION_RESPONSE_BASIS_INSUFFICIENT"
    elif not all(receiver_local_majorities) and response_difference_detected:
        outcome = "RAW_ACTION_DETAIL_OR_MODE_REMAINS_PRIMARY"
    else:
        outcome = "PRESENT_RESOLUTION_AMBIGUITY_REMAINS"

    return _checked_result(
        output_path,
        {
            "schema": "nanogpt-p2-native-target-adjudication-v1",
            "status": "PASS",
            "protocol_sha256": file_sha256(p2_protocol_path),
            "pretarget_validation_result_sha256": prevalidation["result_sha256"],
            "independent_replay_validation_result_sha256": replay["result_sha256"],
            "pilot_jsonl_file_sha256": file_sha256(pilot_jsonl_path),
            "fitted_model_file_sha256": file_sha256(fitted_model_path),
            "proposal_file_sha256": file_sha256(proposal_path),
            "primary_task": "CSRG_STATE_PREDICTION",
            "frozen_outcome": outcome,
            "members": member_rows,
            "receiver_local_majority_in_both_native_directions": all(receiver_local_majorities),
            "own_j_state_pass_in_both_members": all(own_j_passes),
            "own_j_plus_k_half_state_pass_in_both_members": all(own_jk_passes),
            "response_difference_detected": response_difference_detected,
            "capability_combined_with_primary_gate": False,
            "native_target_used_before_response_seal": False,
            "scientific_scope": "P2 local state adjudication only; not a 405-pair resolution count and not a complete stability transition law.",
        },
    )


__all__ = ["adjudicate_p2_response", "project_state_150"]
