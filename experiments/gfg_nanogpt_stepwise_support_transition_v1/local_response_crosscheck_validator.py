from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)

from .branches import _load_observation_arrays
from .execution import _read_checked
from .reciprocal_validator import _load_tensor


EXACT_H1_PARAMETER_BRANCH = {
    "A": {"A": "native_parameter_only", "B": "donor_parameter_delta"},
    "B": {"A": "donor_parameter_delta", "B": "native_parameter_only"},
}


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64), dtype=np.float64)))


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = prediction.astype(np.float64, copy=False).reshape(-1)
    target = target.astype(np.float64, copy=False).reshape(-1)
    require(prediction.shape == target.shape, "SST_LOCAL_RESPONSE_CROSSCHECK_SHAPE_MISMATCH")
    residual = prediction - target
    prediction_norm = float(np.linalg.norm(prediction))
    target_norm = float(np.linalg.norm(target))
    denominator = prediction_norm * target_norm
    return {
        "cosine": None if denominator == 0.0 else float(np.dot(prediction, target) / denominator),
        "prediction_rms": _rms(prediction),
        "target_rms": _rms(target),
        "residual_rms": _rms(residual),
        "nrmse": None if target_norm == 0.0 else float(np.linalg.norm(residual) / target_norm),
    }


def _exact_h1_arrays(
    reciprocal_root: Path,
    *,
    receiver_label: str,
    branch: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    receiver_root = reciprocal_root / f"recipient-{receiver_label}"
    horizon_root = receiver_root / "horizons" / "h-001"
    skip_state = _read_checked(
        horizon_root / "skip-state.json",
        "nanogpt-reciprocal-branch-state-v1",
    )
    effect_state = _read_checked(
        horizon_root / f"{branch}-state.json",
        "nanogpt-reciprocal-branch-state-v1",
    )
    require(skip_state["recipient_label"] == receiver_label, "SST_LOCAL_RESPONSE_CROSSCHECK_SKIP_RECEIVER_MISMATCH")
    require(effect_state["recipient_label"] == receiver_label, "SST_LOCAL_RESPONSE_CROSSCHECK_EFFECT_RECEIVER_MISMATCH")
    require(effect_state["branch"] == branch, "SST_LOCAL_RESPONSE_CROSSCHECK_EFFECT_BRANCH_MISMATCH")
    observations = []
    for state_record in (skip_state, effect_state):
        state_id = str(state_record["state"]["state_id"])
        observation = _read_checked(
            receiver_root / "probe-observations" / "CSRG-4C-v1" / f"{state_id}.json",
            "nanogpt-stepwise-probe-observation-v1",
        )
        observations.append(_load_observation_arrays(receiver_root, observation))
    return observations[0], observations[1]


def _local_response(
    local_root: Path,
    *,
    receiver_label: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    receiver_root = local_root / f"receiver-{receiver_label}"
    response = _read_checked(
        receiver_root / "local_response_jk.json",
        "nanogpt-local-response-jk-v1",
    )
    numeric = {
        key: _load_tensor(receiver_root, references["j_first_order"])
        for key, references in sorted(response["numeric_responses"].items())
    }
    return numeric, response["categorical_transitions"]


def validate_two_update_local_response_crosscheck(
    *,
    reciprocal_root: Path,
    a_update_root: Path,
    b_update_root: Path,
    a_update_protocol_path: Path,
    b_update_protocol_path: Path,
) -> dict[str, Any]:
    protocols = {
        "A": read_json(a_update_protocol_path),
        "B": read_json(b_update_protocol_path),
    }
    roots = {"A": a_update_root, "B": b_update_root}
    for donor_label in ("A", "B"):
        require(protocols[donor_label]["donor_update"]["label"] == donor_label, f"SST_LOCAL_RESPONSE_CROSSCHECK_DONOR_MISMATCH:{donor_label}")
        validation = read_json(roots[donor_label] / "local_response_jk_validation.json")
        require(validation["status"] == "PASS", f"SST_LOCAL_RESPONSE_CROSSCHECK_EVIDENCE_NOT_VALIDATED:{donor_label}")

    rows: dict[str, Any] = {}
    check_count = 0
    for donor_label in ("A", "B"):
        j_by_receiver = {
            receiver_label: _local_response(
                roots[donor_label],
                receiver_label=receiver_label,
            )
            for receiver_label in ("A", "B")
        }
        donor_rows: dict[str, Any] = {}
        for receiver_label in ("A", "B"):
            exact_branch = EXACT_H1_PARAMETER_BRANCH[donor_label][receiver_label]
            baseline_arrays, effect_arrays = _exact_h1_arrays(
                reciprocal_root,
                receiver_label=receiver_label,
                branch=exact_branch,
            )
            own_j, own_categorical = j_by_receiver[receiver_label]
            other_j, _ = j_by_receiver["B" if receiver_label == "A" else "A"]
            keys = sorted(set(own_j) & set(other_j) & set(baseline_arrays) & set(effect_arrays))
            numeric_rows: dict[str, Any] = {}
            own_better_cross = 0
            own_better_shared = 0
            jointly_better = 0
            for key in keys:
                baseline = baseline_arrays[key]
                effect = effect_arrays[key]
                if not np.issubdtype(baseline.dtype, np.floating):
                    continue
                target = effect.astype(np.float64, copy=False) - baseline.astype(np.float64, copy=False)
                require(own_j[key].shape == other_j[key].shape == target.shape, f"SST_LOCAL_RESPONSE_CROSSCHECK_RESPONSE_SHAPE_MISMATCH:{donor_label}:{receiver_label}:{key}")
                shared = (own_j[key].astype(np.float64, copy=False) + other_j[key].astype(np.float64, copy=False)) / 2.0
                own = _metrics(own_j[key], target)
                cross = _metrics(other_j[key], target)
                shared_metrics = _metrics(shared, target)
                better_cross = own["residual_rms"] < cross["residual_rms"]
                better_shared = own["residual_rms"] < shared_metrics["residual_rms"]
                own_better_cross += int(better_cross)
                own_better_shared += int(better_shared)
                jointly_better += int(better_cross and better_shared)
                numeric_rows[key] = {
                    "own_receiver_j": own,
                    "other_receiver_j": cross,
                    "shared_j": shared_metrics,
                    "own_better_than_other": better_cross,
                    "own_better_than_shared": better_shared,
                    "shape": list(target.shape),
                }
                check_count += 4

            categorical_rows: dict[str, Any] = {}
            for key, references in sorted(own_categorical.items()):
                require(key in baseline_arrays and key in effect_arrays, f"SST_LOCAL_RESPONSE_CROSSCHECK_CATEGORICAL_TARGET_MISSING:{donor_label}:{receiver_label}:{key}")
                baseline = baseline_arrays[key]
                full = effect_arrays[key]
                require(not np.issubdtype(baseline.dtype, np.floating), f"SST_LOCAL_RESPONSE_CROSSCHECK_CATEGORICAL_FLOATING:{donor_label}:{receiver_label}:{key}")
                plus_mask = _load_tensor(roots[donor_label] / f"receiver-{receiver_label}", references["plus_changed_mask"]).astype(bool, copy=False)
                full_mask = np.not_equal(full, baseline)
                require(plus_mask.shape == full_mask.shape, f"SST_LOCAL_RESPONSE_CROSSCHECK_CATEGORICAL_SHAPE_MISMATCH:{donor_label}:{receiver_label}:{key}")
                categorical_rows[key] = {
                    "epsilon_plus_changed_count": int(np.count_nonzero(plus_mask)),
                    "full_h1_changed_count": int(np.count_nonzero(full_mask)),
                    "overlap_count": int(np.count_nonzero(np.logical_and(plus_mask, full_mask))),
                    "categorical_values_subtracted": False,
                    "shape": list(full_mask.shape),
                }
                check_count += 2

            donor_rows[receiver_label] = {
                "exact_h1_branch": exact_branch,
                "numeric": numeric_rows,
                "categorical": categorical_rows,
                "numeric_role_count": len(numeric_rows),
                "own_better_than_other_count": own_better_cross,
                "own_better_than_shared_count": own_better_shared,
                "own_jointly_better_count": jointly_better,
            }
        rows[donor_label] = donor_rows

    material = {
        "schema": "nanogpt-two-update-local-response-crosscheck-v1",
        "status": "PASS",
        "a_update_protocol_sha256": file_sha256(a_update_protocol_path),
        "b_update_protocol_sha256": file_sha256(b_update_protocol_path),
        "a_update_validation_sha256": read_json(a_update_root / "local_response_jk_validation.json")["validation_sha256"],
        "b_update_validation_sha256": read_json(b_update_root / "local_response_jk_validation.json")["validation_sha256"],
        "reciprocal_validation_sha256": read_json(reciprocal_root / "reciprocal_pair_validation.json")["validation_sha256"],
        "rows": rows,
        "check_count": check_count,
        "categorical_values_subtracted": False,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(a_update_root / "local_response_two_update_crosscheck.json", result)
    return result
