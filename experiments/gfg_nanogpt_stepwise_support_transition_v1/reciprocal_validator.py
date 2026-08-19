from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import median
from typing import Any, Iterable

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

from .execution import _read_checked
from .reciprocal import RECIPROCAL_BRANCHES
from .storage import restorable_state_from_manifest


def _tensor_refs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"locator", "file_sha256", "raw_tensor_sha256", "shape", "dtype"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from _tensor_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _tensor_refs(child)


def _load_tensor(root: Path, reference: dict[str, Any]) -> np.ndarray:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "SST_RECIPROCAL_VALIDATION_TENSOR_LOCATOR_INVALID")
    path = root / locator
    require(path.is_file(), f"SST_RECIPROCAL_VALIDATION_TENSOR_MISSING:{path}")
    require(file_sha256(path) == reference["file_sha256"], f"SST_RECIPROCAL_VALIDATION_TENSOR_FILE_HASH_MISMATCH:{path}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "SST_RECIPROCAL_VALIDATION_TENSOR_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "SST_RECIPROCAL_VALIDATION_TENSOR_DTYPE_MISMATCH")
    raw = np.ascontiguousarray(value)
    require(hashlib.sha256(raw.tobytes(order="C")).hexdigest() == reference["raw_tensor_sha256"], "SST_RECIPROCAL_VALIDATION_TENSOR_RAW_HASH_MISMATCH")
    return raw


def _state(root: Path, label: str, horizon: int, branch: str) -> StateSnapshot:
    record = _read_checked(
        root / f"recipient-{label}" / "horizons" / f"h-{horizon:03d}" / f"{branch}-state.json",
        "nanogpt-reciprocal-branch-state-v1",
    )
    return restorable_state_from_manifest(root / f"recipient-{label}", record["state"])


def _state_exact(left: StateSnapshot, right: StateSnapshot) -> bool:
    return (
        set(left.parameters) == set(right.parameters)
        and set(left.optimizer) == set(right.optimizer)
        and all(torch.equal(left.parameters[name], right.parameters[name]) for name in left.parameters)
        and all(
            set(left.optimizer[name]) == set(right.optimizer[name])
            and all(torch.equal(left.optimizer[name][key], right.optimizer[name][key]) for key in left.optimizer[name])
            for name in left.optimizer
        )
    )


def _copy_state(state: StateSnapshot) -> StateSnapshot:
    return StateSnapshot(
        {name: value.clone() for name, value in state.parameters.items()},
        {name: {key: value.clone() for key, value in child.items()} for name, child in state.optimizer.items()},
    )


def _expected_transplant(
    recipient_pre: StateSnapshot,
    donor_pre: StateSnapshot,
    donor_post: StateSnapshot,
    *,
    parameter: bool,
    optimizer: bool,
    betas: tuple[float, float],
) -> StateSnapshot:
    result = _copy_state(recipient_pre)
    if parameter:
        for name in result.parameters:
            result.parameters[name].add_(donor_post.parameters[name] - donor_pre.parameters[name])
    if optimizer:
        for name in result.optimizer:
            result.optimizer[name]["step"].add_(donor_post.optimizer[name]["step"] - donor_pre.optimizer[name]["step"])
            for key, beta in (("exp_avg", betas[0]), ("exp_avg_sq", betas[1])):
                innovation = donor_post.optimizer[name][key] - beta * donor_pre.optimizer[name][key]
                result.optimizer[name][key].mul_(beta).add_(innovation)
            require(bool((result.optimizer[name]["exp_avg_sq"] >= 0).all()), "SST_RECIPROCAL_VALIDATION_NEGATIVE_SECOND_MOMENT")
    return result


def _formal_prestate(formal_root: Path, endpoint: dict[str, Any]) -> StateSnapshot:
    path = (
        formal_root
        / str(endpoint["entry_id"])
        / "windows"
        / str(endpoint["window_id"])
        / "states"
        / f"step-{int(endpoint['optimizer_step']):05d}.json"
    )
    record = _read_checked(path, "nanogpt-stepwise-state-v1")
    return restorable_state_from_manifest(formal_root / str(endpoint["entry_id"]), record["state"])


def _effect(root: Path, label: str, horizon: int, branch: str) -> dict[str, np.ndarray]:
    entry_root = root / f"recipient-{label}"
    value = _read_checked(
        entry_root / "horizons" / f"h-{horizon:03d}" / "effects.json",
        "nanogpt-reciprocal-branch-effects-v1",
    )
    return {
        key: _load_tensor(entry_root, branches[branch]).astype(np.float64, copy=False)
        for key, branches in value["numeric_effects"].items()
    }


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64), dtype=np.float64)))


def _signature_row(
    root: Path,
    *,
    recipient: str,
    donor: str,
    horizon: int,
    transferred_branch: str,
    native_branch: str,
) -> dict[str, Any]:
    transferred = _effect(root, recipient, horizon, transferred_branch)
    recipient_native = _effect(root, recipient, horizon, native_branch)
    donor_native = _effect(root, donor, horizon, native_branch)
    keys = sorted(set(transferred) & set(recipient_native) & set(donor_native))
    require(bool(keys), "SST_RECIPROCAL_VALIDATION_SIGNATURE_KEYS_EMPTY")
    rows: list[dict[str, Any]] = []
    for key in keys:
        require(transferred[key].shape == recipient_native[key].shape == donor_native[key].shape, "SST_RECIPROCAL_VALIDATION_SIGNATURE_SHAPE_MISMATCH")
        finite = np.isfinite(transferred[key]) & np.isfinite(recipient_native[key]) & np.isfinite(donor_native[key])
        if not bool(np.any(finite)):
            continue
        transferred_value = transferred[key][finite]
        recipient_value = recipient_native[key][finite]
        donor_value = donor_native[key][finite]
        separation = _rms(donor_value - recipient_value)
        if separation <= 1e-15:
            continue
        donor_distance = _rms(transferred_value - donor_value) / separation
        recipient_distance = _rms(transferred_value - recipient_value) / separation
        rows.append(
            {
                "key": key,
                "donor_distance": donor_distance,
                "recipient_distance": recipient_distance,
                "donor_closer": donor_distance < recipient_distance,
            }
        )
    if not rows:
        return {
            "recipient": recipient,
            "donor": donor,
            "horizon": horizon,
            "transferred_branch": transferred_branch,
            "native_signature_branch": native_branch,
            "informative_tensor_count": 0,
            "donor_closer_tensor_count": 0,
            "median_normalized_distance_to_donor": None,
            "median_normalized_distance_to_recipient": None,
            "aggregate_donor_closer": None,
            "adjudication_status": "UNINFORMATIVE_DONOR_AND_RECIPIENT_NATIVE_RESPONSES_IDENTICAL",
            "per_tensor": [],
        }
    donor_median = median(row["donor_distance"] for row in rows)
    recipient_median = median(row["recipient_distance"] for row in rows)
    return {
        "recipient": recipient,
        "donor": donor,
        "horizon": horizon,
        "transferred_branch": transferred_branch,
        "native_signature_branch": native_branch,
        "informative_tensor_count": len(rows),
        "donor_closer_tensor_count": sum(row["donor_closer"] for row in rows),
        "median_normalized_distance_to_donor": donor_median,
        "median_normalized_distance_to_recipient": recipient_median,
        "aggregate_donor_closer": donor_median < recipient_median,
        "adjudication_status": "INFORMATIVE",
        "per_tensor": rows,
    }


def _classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = {
        (row["recipient"], row["transferred_branch"], int(row["horizon"])): bool(row["aggregate_donor_closer"])
        for row in rows
    }

    def both(branch: str, horizon: int) -> bool:
        return index[("A", branch, horizon)] and index[("B", branch, horizon)]

    parameter_h1 = both("donor_parameter_delta", 1)
    optimizer_h1 = both("donor_optimizer_innovation", 1)
    joint_h1 = both("donor_joint_update", 1)
    optimizer_later = [h for h in (2, 5, 20, 100) if both("donor_optimizer_innovation", h) or both("donor_joint_update", h)]
    any_bilateral = any(both(branch, h) for branch in ("donor_parameter_delta", "donor_optimizer_innovation", "donor_joint_update") for h in (1, 2, 5, 20, 100))
    if parameter_h1 and optimizer_h1 and joint_h1:
        decision = "UPDATE_DOMINANT"
    elif parameter_h1 and not optimizer_h1 and bool(optimizer_later):
        decision = "DELAYED_OPTIMIZER_DOMINANT"
    elif not any_bilateral:
        decision = "RECEIVING_STATE_DOMINANT"
    else:
        decision = "NONSEPARABLE_INTERACTION_OR_OBSERVATION_FAILURE"
    return {
        "decision": decision,
        "parameter_donor_bilateral_at_h1": parameter_h1,
        "optimizer_donor_bilateral_at_h1": optimizer_h1,
        "joint_donor_bilateral_at_h1": joint_h1,
        "optimizer_or_joint_bilateral_later_horizons": optimizer_later,
        "decision_rule_posthoc_changed": False,
    }


def validate_reciprocal_pair(
    *,
    root: Path,
    formal_root: Path,
    reciprocal_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(reciprocal_protocol_path)
    require(protocol["schema"] == "nanogpt-reciprocal-matched-pair-protocol-v2", "SST_RECIPROCAL_VALIDATION_PROTOCOL_SCHEMA_INVALID")
    require(file_sha256(reciprocal_protocol_path) == _read_checked(root / "reciprocal_pair_receipt.json", "nanogpt-reciprocal-pair-receipt-v1")["reciprocal_protocol_sha256"], "SST_RECIPROCAL_VALIDATION_PROTOCOL_HASH_MISMATCH")
    endpoints = {str(row["label"]): row for row in protocol["endpoints"]}
    require(set(endpoints) == {"A", "B"}, "SST_RECIPROCAL_VALIDATION_ENDPOINTS_INVALID")
    json_count = 0
    tensor_paths: set[Path] = set()
    for path in sorted(root.rglob("*.json")):
        if path.name == "reciprocal_pair_validation.json":
            continue
        value = read_json(path)
        if "result_sha256" in value:
            material = {key: child for key, child in value.items() if key != "result_sha256"}
            require(payload_sha256(material) == value["result_sha256"], f"SST_RECIPROCAL_VALIDATION_RESULT_HASH_MISMATCH:{path}")
        json_count += 1
        entry_root = root / ("recipient-A" if "recipient-A" in path.parts else "recipient-B" if "recipient-B" in path.parts else "")
        if entry_root != root:
            for reference in _tensor_refs(value):
                tensor_path = entry_root / str(reference["locator"])
                if tensor_path not in tensor_paths:
                    _load_tensor(entry_root, reference)
                    tensor_paths.add(tensor_path)

    pre = {label: _formal_prestate(formal_root, endpoint) for label, endpoint in endpoints.items()}
    h1 = {label: {branch: _state(root, label, 1, branch) for branch in RECIPROCAL_BRANCHES} for label in endpoints}
    composition_checks: dict[str, Any] = {}
    for label, donor in (("A", "B"), ("B", "A")):
        require(_state_exact(h1[label]["skip"], pre[label]), f"SST_RECIPROCAL_VALIDATION_SKIP_DRIFT:{label}")
        native_parameter = StateSnapshot(h1[label]["native_full"].parameters, pre[label].optimizer)
        native_optimizer = StateSnapshot(pre[label].parameters, h1[label]["native_full"].optimizer)
        require(_state_exact(h1[label]["native_parameter_only"], native_parameter), f"SST_RECIPROCAL_VALIDATION_NATIVE_PARAMETER_COMPOSITION_INVALID:{label}")
        require(_state_exact(h1[label]["native_optimizer_only"], native_optimizer), f"SST_RECIPROCAL_VALIDATION_NATIVE_OPTIMIZER_COMPOSITION_INVALID:{label}")
        seed = _read_checked(root / f"recipient-{label}" / "seed_result.json", "nanogpt-reciprocal-seed-v1")
        betas = tuple(float(value) for value in seed["donor_native_full_step"]["optimizer_config"]["betas"])
        expected_parameter = _expected_transplant(pre[label], pre[donor], h1[donor]["native_full"], parameter=True, optimizer=False, betas=betas)
        expected_optimizer = _expected_transplant(pre[label], pre[donor], h1[donor]["native_full"], parameter=False, optimizer=True, betas=betas)
        expected_joint = _expected_transplant(pre[label], pre[donor], h1[donor]["native_full"], parameter=True, optimizer=True, betas=betas)
        require(_state_exact(h1[label]["donor_parameter_delta"], expected_parameter), f"SST_RECIPROCAL_VALIDATION_DONOR_PARAMETER_COMPOSITION_INVALID:{label}")
        require(_state_exact(h1[label]["donor_optimizer_innovation"], expected_optimizer), f"SST_RECIPROCAL_VALIDATION_DONOR_OPTIMIZER_COMPOSITION_INVALID:{label}")
        require(_state_exact(h1[label]["donor_joint_update"], expected_joint), f"SST_RECIPROCAL_VALIDATION_DONOR_JOINT_COMPOSITION_INVALID:{label}")
        composition_checks[label] = {"skip_exact": True, "native_compositions_exact": True, "donor_parameter_exact": True, "donor_optimizer_innovation_exact": True, "donor_joint_exact": True}

        receipt = _read_checked(root / f"recipient-{label}" / "reciprocal_receipt.json", "nanogpt-reciprocal-recipient-receipt-v1")
        require(tuple(receipt["branches"]) == RECIPROCAL_BRANCHES, "SST_RECIPROCAL_VALIDATION_BRANCH_SET_INVALID")
        require(receipt["horizons"] == protocol["horizons"], "SST_RECIPROCAL_VALIDATION_HORIZONS_INVALID")
        current = {branch: h1[label][branch].commitment()["state_sha256"] for branch in RECIPROCAL_BRANCHES}
        continuations = {int(row["physical_optimizer_step"]): row for row in receipt["continuation_results"]}
        start = int(endpoints[label]["optimizer_step"])
        require(set(continuations) == set(range(start + 1, start + 100)), "SST_RECIPROCAL_VALIDATION_CONTINUATION_COVERAGE_INVALID")
        for step in range(start + 1, start + 100):
            row = _read_checked(root / f"recipient-{label}" / "continuations" / f"step-{step:05d}-to-{step + 1:05d}.json", "nanogpt-reciprocal-continuation-v1")
            for branch in RECIPROCAL_BRANCHES:
                require(row["branches"][branch]["from_state_sha256"] == current[branch], f"SST_RECIPROCAL_VALIDATION_CONTINUATION_CHAIN_BROKEN:{label}:{step}:{branch}")
                current[branch] = row["branches"][branch]["to_state_sha256"]
            horizon = step + 1 - start
            if horizon in protocol["horizons"]:
                for branch in RECIPROCAL_BRANCHES:
                    require(_state(root, label, horizon, branch).commitment()["state_sha256"] == current[branch], f"SST_RECIPROCAL_VALIDATION_HORIZON_CHAIN_MISMATCH:{label}:{horizon}:{branch}")

    signature_rows: list[dict[str, Any]] = []
    for horizon in protocol["horizons"]:
        for label, donor in (("A", "B"), ("B", "A")):
            for transferred, native in (
                ("donor_parameter_delta", "native_parameter_only"),
                ("donor_optimizer_innovation", "native_optimizer_only"),
                ("donor_joint_update", "native_full"),
            ):
                signature_rows.append(
                    _signature_row(root, recipient=label, donor=donor, horizon=int(horizon), transferred_branch=transferred, native_branch=native)
                )
    adjudication = _classification(signature_rows)
    capability = {
        label: {
            str(horizon): {
                branch: _read_checked(
                    root / f"recipient-{label}" / "probe-observations" / "CSRG-4C-v1" / (
                        _read_checked(root / f"recipient-{label}" / "horizons" / f"h-{int(horizon):03d}" / f"{branch}-state.json", "nanogpt-reciprocal-branch-state-v1")["state"]["state_id"] + ".json"
                    ),
                    "nanogpt-stepwise-probe-observation-v1",
                )["capability_accuracy"]
                for branch in RECIPROCAL_BRANCHES
            }
            for horizon in protocol["horizons"]
        }
        for label in endpoints
    }
    material = {
        "schema": "nanogpt-reciprocal-pair-validation-v1",
        "status": "PASS",
        "protocol_sha256": file_sha256(reciprocal_protocol_path),
        "receipt_sha256": file_sha256(root / "reciprocal_pair_receipt.json"),
        "json_result_count": json_count,
        "unique_tensor_payload_count": len(tensor_paths),
        "composition_checks": composition_checks,
        "continuation_chain_exact": True,
        "future_information_used": False,
        "signature_rows": signature_rows,
        "adjudication": adjudication,
        "capability_accuracy": capability,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(root / "reciprocal_pair_validation.json", result)
    return result
