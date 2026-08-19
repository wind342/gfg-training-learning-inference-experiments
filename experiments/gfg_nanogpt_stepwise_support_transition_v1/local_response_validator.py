from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_transition_v1.runtime import StateSnapshot

from .branches import _load_observation_arrays
from .execution import _read_checked
from .local_response import LOCAL_RESPONSE_BRANCHES, LOCAL_RESPONSE_TRANSPORT_BRANCHES
from .reciprocal_validator import _formal_prestate, _load_tensor, _state, _state_exact
from .storage import restorable_state_from_manifest


def _copy_state(state: StateSnapshot) -> StateSnapshot:
    return StateSnapshot(
        {name: value.clone() for name, value in state.parameters.items()},
        {
            name: {key: value.clone() for key, value in child.items()}
            for name, child in state.optimizer.items()
        },
    )


def _expected_state(
    receiver: StateSnapshot,
    donor_pre: StateSnapshot,
    donor_parameter_post: StateSnapshot,
    scale: float,
) -> StateSnapshot:
    result = _copy_state(receiver)
    require(set(donor_pre.parameters) == set(donor_parameter_post.parameters) == set(result.parameters), "SST_LOCAL_RESPONSE_VALIDATION_PARAMETER_SET_MISMATCH")
    for name in sorted(result.parameters):
        delta = donor_parameter_post.parameters[name] - donor_pre.parameters[name]
        result.parameters[name].add_(delta.mul(scale))
    return result


def _observation(root: Path, state_id: str) -> dict[str, Any]:
    return _read_checked(
        root / "probe-observations" / "CSRG-4C-v1" / f"{state_id}.json",
        "nanogpt-stepwise-probe-observation-v1",
    )


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64), dtype=np.float64)))


def validate_local_response_jk(
    *,
    root: Path,
    formal_root: Path,
    reciprocal_root: Path,
    local_response_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(local_response_protocol_path)
    require(protocol["schema"] == "nanogpt-local-response-jk-protocol-v1", "SST_LOCAL_RESPONSE_VALIDATION_PROTOCOL_SCHEMA_INVALID")
    receipt = _read_checked(root / "local_response_pair_receipt.json", "nanogpt-local-response-pair-receipt-v1")
    require(file_sha256(local_response_protocol_path) == receipt["local_response_protocol_sha256"], "SST_LOCAL_RESPONSE_VALIDATION_PROTOCOL_HASH_MISMATCH")
    epsilon = float(protocol["epsilon"])
    require(epsilon == float(receipt["epsilon"]) == 0.125, "SST_LOCAL_RESPONSE_VALIDATION_EPSILON_MISMATCH")
    endpoints = {str(row["label"]): row for row in protocol["receivers"]}
    branches = tuple(str(value) for value in protocol["branches"])
    require(
        branches in {LOCAL_RESPONSE_BRANCHES, LOCAL_RESPONSE_TRANSPORT_BRANCHES},
        "SST_LOCAL_RESPONSE_VALIDATION_BRANCHES_INVALID",
    )
    receiver_state_kind = str(protocol.get("receiver_state_kind", "skip"))
    require(
        receiver_state_kind in {"skip", "native_full"},
        "SST_LOCAL_RESPONSE_VALIDATION_RECEIVER_STATE_KIND_INVALID",
    )
    donor_label = str(protocol["donor_update"]["label"])
    require(donor_label in endpoints, "SST_LOCAL_RESPONSE_VALIDATION_DONOR_INVALID")
    donor_pre = _formal_prestate(formal_root, endpoints[donor_label])
    donor_parameter_post = _state(
        reciprocal_root,
        donor_label,
        1,
        "native_parameter_only",
    )
    require(
        all(
            torch.equal(donor_parameter_post.optimizer[name][key], donor_pre.optimizer[name][key])
            for name in donor_pre.optimizer
            for key in donor_pre.optimizer[name]
        ),
        "SST_LOCAL_RESPONSE_VALIDATION_DONOR_PARAMETER_STATE_CHANGED_OPTIMIZER",
    )
    checks = 0
    receiver_rows: dict[str, Any] = {}
    numeric_values: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for label in ("A", "B"):
        entry_root = root / f"receiver-{label}"
        receiver_receipt = _read_checked(entry_root / "local_response_receipt.json", "nanogpt-local-response-receiver-receipt-v1")
        require(receiver_receipt["receiver"]["entry_id"] == endpoints[label]["entry_id"], "SST_LOCAL_RESPONSE_VALIDATION_RECEIVER_ID_MISMATCH")
        seed = _read_checked(entry_root / "local_response_seed.json", "nanogpt-local-response-seed-v1")
        require(seed["donor_update_source_object_id"] == protocol["donor_update"]["source_object_id"], "SST_LOCAL_RESPONSE_VALIDATION_DONOR_OBJECT_MISMATCH")
        require(seed["adam_state_transplanted"] is False, "SST_LOCAL_RESPONSE_VALIDATION_ADAM_TRANSPLANTED")
        receiver_pre = (
            _formal_prestate(formal_root, endpoints[label])
            if receiver_state_kind == "skip"
            else _state(reciprocal_root, label, 1, "native_full")
        )
        scales = {
            "baseline": 0.0,
            "plus_epsilon": epsilon,
            "minus_epsilon": -epsilon,
            "plus_full": 1.0,
        }
        expected = {
            branch: _expected_state(
                receiver_pre,
                donor_pre,
                donor_parameter_post,
                scales[branch],
            )
            for branch in branches
        }
        observations: dict[str, dict[str, Any]] = {}
        for branch in branches:
            state_record = _read_checked(entry_root / "h-001" / f"{branch}-state.json", "nanogpt-local-response-state-v1")
            actual = restorable_state_from_manifest(entry_root, state_record["state"])
            require(_state_exact(actual, expected[branch]), f"SST_LOCAL_RESPONSE_VALIDATION_STATE_MISMATCH:{label}:{branch}")
            require(
                actual.commitment()["state_sha256"] == receiver_receipt["states"][branch]["state_sha256"],
                f"SST_LOCAL_RESPONSE_VALIDATION_STATE_RECEIPT_MISMATCH:{label}:{branch}",
            )
            observations[branch] = _observation(entry_root, state_record["state"]["state_id"])
            checks += 2
        for branch, observation in observations.items():
            arrays = _load_observation_arrays(entry_root, observation)
            for key in ("logits", "margins", "predictions", "group_membership", "group_q10_margin"):
                require(
                    np.array_equal(arrays[f"forward/0/{key}"], arrays[f"forward/1/{key}"]),
                    f"SST_LOCAL_RESPONSE_VALIDATION_BASELINE_REPEAT_MISMATCH:{label}:{branch}:{key}",
                )
                checks += 1
        response = _read_checked(entry_root / "local_response_jk.json", "nanogpt-local-response-jk-v1")
        require(response["categorical_values_subtracted"] is False, "SST_LOCAL_RESPONSE_VALIDATION_CATEGORICAL_SUBTRACTION")
        arrays_by_branch = {
            branch: _load_observation_arrays(entry_root, observation)
            for branch, observation in observations.items()
        }
        numeric_values[label] = {}
        numeric_summary: dict[str, Any] = {}
        categorical_summary: dict[str, Any] = {}
        for key, references in sorted(response["numeric_responses"].items()):
            baseline = arrays_by_branch["baseline"][key].astype(np.float64, copy=False)
            plus = arrays_by_branch["plus_epsilon"][key].astype(np.float64, copy=False)
            minus = arrays_by_branch["minus_epsilon"][key].astype(np.float64, copy=False)
            expected_j = (plus - minus) / (2.0 * epsilon)
            expected_k = (plus + minus - 2.0 * baseline) / (epsilon * epsilon)
            actual_j = _load_tensor(entry_root, references["j_first_order"])
            actual_k = _load_tensor(entry_root, references["k_curvature"])
            require(np.array_equal(actual_j, expected_j), f"SST_LOCAL_RESPONSE_VALIDATION_J_MISMATCH:{label}:{key}")
            require(np.array_equal(actual_k, expected_k), f"SST_LOCAL_RESPONSE_VALIDATION_K_MISMATCH:{label}:{key}")
            if "plus_full" in branches:
                full = arrays_by_branch["plus_full"][key].astype(
                    np.float64, copy=False
                )
                expected_full_delta = full - baseline
                require(
                    np.array_equal(
                        _load_tensor(entry_root, references["full_delta"]),
                        expected_full_delta,
                    ),
                    f"SST_LOCAL_RESPONSE_VALIDATION_FULL_DELTA_MISMATCH:{label}:{key}",
                )
                checks += 1
            numeric_values[label][key] = (expected_j, expected_k)
            numeric_summary[key] = {"j_rms": _rms(expected_j), "k_rms": _rms(expected_k), "shape": list(expected_j.shape)}
            checks += 2
        for key, references in sorted(response["categorical_transitions"].items()):
            baseline = arrays_by_branch["baseline"][key]
            plus = arrays_by_branch["plus_epsilon"][key]
            minus = arrays_by_branch["minus_epsilon"][key]
            plus_mask = np.not_equal(plus, baseline)
            minus_mask = np.not_equal(minus, baseline)
            require(np.array_equal(_load_tensor(entry_root, references["plus_changed_mask"]), plus_mask), f"SST_LOCAL_RESPONSE_VALIDATION_PLUS_MASK_MISMATCH:{label}:{key}")
            require(np.array_equal(_load_tensor(entry_root, references["minus_changed_mask"]), minus_mask), f"SST_LOCAL_RESPONSE_VALIDATION_MINUS_MASK_MISMATCH:{label}:{key}")
            require(int(references["plus_changed_count"]) == int(np.count_nonzero(plus_mask)), f"SST_LOCAL_RESPONSE_VALIDATION_PLUS_COUNT_MISMATCH:{label}:{key}")
            require(int(references["minus_changed_count"]) == int(np.count_nonzero(minus_mask)), f"SST_LOCAL_RESPONSE_VALIDATION_MINUS_COUNT_MISMATCH:{label}:{key}")
            if "plus_full" in branches:
                full = arrays_by_branch["plus_full"][key]
                full_mask = np.not_equal(full, baseline)
                require(
                    np.array_equal(
                        _load_tensor(entry_root, references["full_changed_mask"]),
                        full_mask,
                    ),
                    f"SST_LOCAL_RESPONSE_VALIDATION_FULL_MASK_MISMATCH:{label}:{key}",
                )
                require(
                    int(references["full_changed_count"])
                    == int(np.count_nonzero(full_mask)),
                    f"SST_LOCAL_RESPONSE_VALIDATION_FULL_COUNT_MISMATCH:{label}:{key}",
                )
                checks += 2
            categorical_summary[key] = {"plus_changed_count": int(np.count_nonzero(plus_mask)), "minus_changed_count": int(np.count_nonzero(minus_mask)), "shape": list(baseline.shape)}
            checks += 4
        receiver_rows[label] = {
            "receiver_receipt_sha256": receiver_receipt["result_sha256"],
            "numeric": numeric_summary,
            "categorical": categorical_summary,
            "capability_accuracy": {
                branch: receiver_receipt["states"][branch]["capability_accuracy"]
                for branch in branches
            },
        }
    shared_keys = sorted(set(numeric_values["A"]) & set(numeric_values["B"]))
    receiver_contrasts = {
        key: {
            "j_receiver_difference_rms": _rms(numeric_values["A"][key][0] - numeric_values["B"][key][0]),
            "k_receiver_difference_rms": _rms(numeric_values["A"][key][1] - numeric_values["B"][key][1]),
        }
        for key in shared_keys
        if numeric_values["A"][key][0].shape == numeric_values["B"][key][0].shape
    }
    material = {
        "schema": "nanogpt-local-response-jk-validation-v1",
        "status": "PASS",
        "local_response_protocol_sha256": receipt["local_response_protocol_sha256"],
        "pair_receipt_sha256": receipt["result_sha256"],
        "reciprocal_validation_sha256": read_json(reciprocal_root / "reciprocal_pair_validation.json")["validation_sha256"],
        "receiver_rows": receiver_rows,
        "receiver_contrasts": receiver_contrasts,
        "check_count": checks,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    if branches == LOCAL_RESPONSE_TRANSPORT_BRANCHES or receiver_state_kind != "skip":
        material.update(
            {
                "receiver_state_kind": receiver_state_kind,
                "branches": list(branches),
            }
        )
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(root / "local_response_jk_validation.json", result)
    return result
