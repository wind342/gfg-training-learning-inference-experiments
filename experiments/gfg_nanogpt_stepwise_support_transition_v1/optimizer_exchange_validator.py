from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    relative_file_manifest,
    require,
    write_json,
)

from .branches import _load_observation_arrays
from .execution import _read_checked
from .optimizer_exchange import (
    EXCHANGE_BRANCHES,
    EXCHANGE_CONTINUATION_HORIZONS,
    EXCHANGE_OBSERVATION_HORIZONS,
    _branch_contract,
    _comparison_distance,
    compose_parameter_optimizer_state,
)
from .reciprocal_validator import _state_exact
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


def _validate_tensor(
    root: Path,
    reference: dict[str, Any],
    cache: dict[tuple[str, str], dict[str, Any]],
) -> np.ndarray:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "SST_OPTIMIZER_EXCHANGE_VALIDATION_TENSOR_LOCATOR_INVALID")
    path = root / locator
    require(path.is_file(), f"SST_OPTIMIZER_EXCHANGE_VALIDATION_TENSOR_MISSING:{path}")
    admitted = {
        name: reference[name]
        for name in ("file_sha256", "raw_tensor_sha256", "shape", "dtype")
    }
    key = (str(root.resolve()), locator)
    if key in cache:
        require(cache[key] == admitted, "SST_OPTIMIZER_EXCHANGE_VALIDATION_TENSOR_REFERENCE_CONFLICT")
    else:
        require(file_sha256(path) == reference["file_sha256"], "SST_OPTIMIZER_EXCHANGE_VALIDATION_TENSOR_FILE_HASH_MISMATCH")
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        require(list(value.shape) == list(reference["shape"]), "SST_OPTIMIZER_EXCHANGE_VALIDATION_TENSOR_SHAPE_MISMATCH")
        require(str(value.dtype) == str(reference["dtype"]), "SST_OPTIMIZER_EXCHANGE_VALIDATION_TENSOR_DTYPE_MISMATCH")
        raw_sha = hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
        require(raw_sha == reference["raw_tensor_sha256"], "SST_OPTIMIZER_EXCHANGE_VALIDATION_TENSOR_RAW_HASH_MISMATCH")
        cache[key] = admitted
    return np.asarray(np.load(path, allow_pickle=False, mmap_mode="r"))


def _validate_probe(
    root: Path,
    state_id: str,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    observation = _read_checked(
        root / "probe-observations" / "CSRG-4C-v1" / f"{state_id}.json",
        "nanogpt-stepwise-probe-observation-v1",
    )
    require(observation["observed_state_id"] == state_id, "SST_OPTIMIZER_EXCHANGE_VALIDATION_PROBE_STATE_MISMATCH")
    require(int(observation["actual_forward_count"]) == 12, "SST_OPTIMIZER_EXCHANGE_VALIDATION_PROBE_FORWARD_COUNT_MISMATCH")
    require(observation["baseline_byte_exact"] is True, "SST_OPTIMIZER_EXCHANGE_VALIDATION_BASELINE_NOT_EXACT")
    for reference in _tensor_refs(observation):
        _validate_tensor(root, reference, cache)
    return observation


def _observation_exact(
    left_root: Path,
    left: dict[str, Any],
    right_root: Path,
    right: dict[str, Any],
) -> bool:
    left_arrays = _load_observation_arrays(left_root, left)
    right_arrays = _load_observation_arrays(right_root, right)
    return (
        tuple(sorted(left_arrays)) == tuple(sorted(right_arrays))
        and all(np.array_equal(left_arrays[key], right_arrays[key], equal_nan=True) for key in left_arrays)
        and float(left["capability_accuracy"]) == float(right["capability_accuracy"])
    )


def validate_optimizer_exchange(
    *,
    root: Path,
    amplitude_root: Path,
    optimizer_exchange_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(optimizer_exchange_protocol_path)
    require(protocol["schema"] == "nanogpt-h20-reciprocal-optimizer-exchange-protocol-v1", "SST_OPTIMIZER_EXCHANGE_VALIDATION_PROTOCOL_SCHEMA_INVALID")
    branches = _branch_contract(protocol)
    receiver_contracts = {str(row["label"]): row for row in protocol["receivers"]}
    require(tuple(receiver_contracts) == ("A", "B"), "SST_OPTIMIZER_EXCHANGE_VALIDATION_RECEIVERS_INVALID")
    receipt = _read_checked(
        root / "optimizer_exchange_pair_receipt.json",
        "nanogpt-h20-optimizer-exchange-pair-receipt-v1",
    )
    require(receipt["optimizer_exchange_protocol_sha256"] == file_sha256(optimizer_exchange_protocol_path), "SST_OPTIMIZER_EXCHANGE_VALIDATION_PROTOCOL_HASH_MISMATCH")
    require(receipt["future_information_used"] is False, "SST_OPTIMIZER_EXCHANGE_VALIDATION_FUTURE_USED")
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    checks = 0
    receiver_rows: dict[str, Any] = {}
    for label in ("A", "B"):
        receiver = receiver_contracts[label]
        entry_root = root / f"receiver-{label}"
        amplitude_entry_root = amplitude_root / f"receiver-{label}"
        receiver_receipt = _read_checked(
            entry_root / "optimizer_exchange_receiver_receipt.json",
            "nanogpt-h20-optimizer-exchange-receiver-receipt-v1",
        )
        require(receiver_receipt["receiver"] == receiver, f"SST_OPTIMIZER_EXCHANGE_VALIDATION_RECEIVER_CONTRACT_MISMATCH:{label}")
        prior_h20: dict[float, Any] = {}
        prior_h100: dict[float, Any] = {}
        prior_h100_probe: dict[float, dict[str, Any]] = {}
        for scale in (0.0, 1.0):
            key = f"scale-{scale:.3f}".replace("-", "m").replace(".", "p")
            # Use the canonical helper's spelling rather than accepting an arbitrary file.
            from .amplitude_path import scale_key

            key = scale_key(scale)
            h20 = _read_checked(amplitude_entry_root / "horizons" / "h-020" / f"{key}-state.json", "nanogpt-amplitude-path-state-v1")
            h100 = _read_checked(amplitude_entry_root / "horizons" / "h-100" / f"{key}-state.json", "nanogpt-amplitude-path-state-v1")
            prior_h20[scale] = restorable_state_from_manifest(amplitude_entry_root, h20["state"])
            prior_h100[scale] = restorable_state_from_manifest(amplitude_entry_root, h100["state"])
            prior_h100_probe[scale] = _validate_probe(amplitude_entry_root, h100["state"]["state_id"], cache)

        h20_states: dict[str, Any] = {}
        observations: dict[int, dict[str, dict[str, Any]]] = {20: {}, 21: {}, 100: {}}
        horizon_rows = receiver_receipt["horizon_results"]
        require(tuple(sorted(int(value) for value in horizon_rows)) == EXCHANGE_OBSERVATION_HORIZONS, f"SST_OPTIMIZER_EXCHANGE_VALIDATION_HORIZON_SET_MISMATCH:{label}")
        for branch_id in EXCHANGE_BRANCHES:
            branch = branches[branch_id]
            record = _read_checked(entry_root / "horizons" / "h-020" / f"{branch_id}-state.json", "nanogpt-h20-optimizer-exchange-state-v1")
            require(record["branch_id"] == branch_id and int(record["horizon"]) == 20, "SST_OPTIMIZER_EXCHANGE_VALIDATION_H20_ID_MISMATCH")
            require(float(record["parameter_donor_scale"]) == float(branch["parameter_donor_scale"]), "SST_OPTIMIZER_EXCHANGE_VALIDATION_PARAMETER_DONOR_MISMATCH")
            require(float(record["optimizer_donor_scale"]) == float(branch["optimizer_donor_scale"]), "SST_OPTIMIZER_EXCHANGE_VALIDATION_OPTIMIZER_DONOR_MISMATCH")
            actual = restorable_state_from_manifest(entry_root, record["state"])
            expected = compose_parameter_optimizer_state(
                prior_h20[float(branch["parameter_donor_scale"])],
                prior_h20[float(branch["optimizer_donor_scale"])],
            )
            require(_state_exact(actual, expected), f"SST_OPTIMIZER_EXCHANGE_VALIDATION_H20_COMPOSITION_MISMATCH:{label}:{branch_id}")
            h20_states[branch_id] = actual
            observations[20][branch_id] = _validate_probe(entry_root, record["state"]["state_id"], cache)
            checks += 9

        chain = {branch_id: h20_states[branch_id].commitment()["state_sha256"] for branch_id in EXCHANGE_BRANCHES}
        continuation = receiver_receipt["continuation_results"]
        require(len(continuation) == len(EXCHANGE_CONTINUATION_HORIZONS), f"SST_OPTIMIZER_EXCHANGE_VALIDATION_CONTINUATION_COUNT_MISMATCH:{label}")
        for expected_horizon, row in zip(EXCHANGE_CONTINUATION_HORIZONS, continuation):
            require(int(row["horizon"]) == expected_horizon, "SST_OPTIMIZER_EXCHANGE_VALIDATION_CONTINUATION_HORIZON_MISMATCH")
            physical_step = int(row["physical_optimizer_step"])
            require(physical_step == int(receiver["base_optimizer_step"]) + expected_horizon - 1, "SST_OPTIMIZER_EXCHANGE_VALIDATION_PHYSICAL_STEP_MISMATCH")
            loaded: dict[str, dict[str, Any]] = {}
            for branch_id in EXCHANGE_BRANCHES:
                step = _read_checked(
                    entry_root / "continuations" / f"step-{physical_step:05d}-to-{physical_step + 1:05d}" / f"{branch_id}.json",
                    "nanogpt-h20-optimizer-exchange-continuation-step-v1",
                )
                require(step["from_state_sha256"] == chain[branch_id], f"SST_OPTIMIZER_EXCHANGE_VALIDATION_CHAIN_FROM_MISMATCH:{label}:{expected_horizon}:{branch_id}")
                require(step["to_state_sha256"] == row["steps"][branch_id]["to_state_sha256"], f"SST_OPTIMIZER_EXCHANGE_VALIDATION_CHAIN_TO_MISMATCH:{label}:{expected_horizon}:{branch_id}")
                require(step["step_evidence"]["execute_optimizer"] is True, "SST_OPTIMIZER_EXCHANGE_VALIDATION_OPTIMIZER_NOT_EXECUTED")
                require(step["future_information_used"] is False, "SST_OPTIMIZER_EXCHANGE_VALIDATION_STEP_FUTURE_USED")
                for reference in _tensor_refs(step):
                    _validate_tensor(entry_root, reference, cache)
                chain[branch_id] = str(step["to_state_sha256"])
                loaded[branch_id] = step
                checks += 6
            require(len({payload_sha256(value["same_batch_all_branches"]) for value in loaded.values()}) == 1, f"SST_OPTIMIZER_EXCHANGE_VALIDATION_BATCH_ALIGNMENT_MISMATCH:{label}:{expected_horizon}")
            require(len({int(value["same_external_rng_opportunity_all_branches"]) for value in loaded.values()}) == 1, f"SST_OPTIMIZER_EXCHANGE_VALIDATION_RNG_ALIGNMENT_MISMATCH:{label}:{expected_horizon}")
            checks += 2

        final_state_exact: dict[str, bool] = {}
        final_probe_exact: dict[str, bool] = {}
        for horizon in (21, 100):
            for branch_id in EXCHANGE_BRANCHES:
                record = _read_checked(entry_root / "horizons" / f"h-{horizon:03d}" / f"{branch_id}-state.json", "nanogpt-h20-optimizer-exchange-state-v1")
                require(record["state"]["commitment"]["state_sha256"] == horizon_rows[str(horizon)][branch_id]["state_sha256"], "SST_OPTIMIZER_EXCHANGE_VALIDATION_HORIZON_RECEIPT_MISMATCH")
                if horizon == 100:
                    require(record["state"]["commitment"]["state_sha256"] == chain[branch_id], "SST_OPTIMIZER_EXCHANGE_VALIDATION_FINAL_CHAIN_MISMATCH")
                observations[horizon][branch_id] = _validate_probe(entry_root, record["state"]["state_id"], cache)
                if horizon == 100 and branch_id in ("theta0_O0", "theta1_O1"):
                    scale = 0.0 if branch_id == "theta0_O0" else 1.0
                    state = restorable_state_from_manifest(entry_root, record["state"])
                    final_state_exact[branch_id] = _state_exact(state, prior_h100[scale])
                    final_probe_exact[branch_id] = _observation_exact(entry_root, observations[100][branch_id], amplitude_entry_root, prior_h100_probe[scale])
                    require(final_state_exact[branch_id], f"SST_OPTIMIZER_EXCHANGE_VALIDATION_NATIVE_CONTROL_STATE_NOT_EXACT:{label}:{branch_id}")
                    require(final_probe_exact[branch_id], f"SST_OPTIMIZER_EXCHANGE_VALIDATION_NATIVE_CONTROL_PROBE_NOT_EXACT:{label}:{branch_id}")
                checks += 5

        comparison = _read_checked(entry_root / "h100_frozen_comparison.json", "nanogpt-h20-optimizer-exchange-h100-comparison-v1")
        arrays = {branch_id: _load_observation_arrays(entry_root, observations[100][branch_id]) for branch_id in EXCHANGE_BRANCHES}
        capabilities = {branch_id: float(observations[100][branch_id]["capability_accuracy"]) for branch_id in EXCHANGE_BRANCHES}
        require(comparison["hybrid_comparisons"] == _comparison_distance(arrays, capabilities), f"SST_OPTIMIZER_EXCHANGE_VALIDATION_COMPARISON_MISMATCH:{label}")
        require(comparison["weights_fitted"] is False and comparison["thresholds_fitted"] is False, "SST_OPTIMIZER_EXCHANGE_VALIDATION_COMPARISON_FITTED")
        require(comparison["scientific_interpretation_performed"] is False, "SST_OPTIMIZER_EXCHANGE_VALIDATION_PREMATURE_INTERPRETATION")
        receiver_rows[label] = {
            "native_control_h100_state_byte_exact": final_state_exact,
            "native_control_h100_probe_byte_exact": final_probe_exact,
            "h100_capability_accuracy": capabilities,
            "h100_hybrid_comparisons": comparison["hybrid_comparisons"],
            "validated_continuation_step_count": len(continuation) * len(EXCHANGE_BRANCHES),
        }
        checks += 5

    material = {
        "schema": "nanogpt-h20-reciprocal-optimizer-exchange-validation-v1",
        "status": "PASS",
        "pair_receipt_sha256": receipt["result_sha256"],
        "optimizer_exchange_protocol_sha256": file_sha256(optimizer_exchange_protocol_path),
        "receiver_rows": receiver_rows,
        "validated_tensor_payload_count": len(cache),
        "check_count": checks,
        "native_controls_byte_exact": True,
        "branch_batch_and_rng_alignment_exact": True,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(root / "optimizer_exchange_validation.json", result)
    return result


def validate_optimizer_exchange_replay(*, primary_root: Path, replay_root: Path) -> dict[str, Any]:
    primary_receipt = read_json(primary_root / "optimizer_exchange_pair_receipt.json")
    replay_receipt = read_json(replay_root / "optimizer_exchange_pair_receipt.json")
    normalize_receipt = lambda value: {key: child for key, child in value.items() if key not in {"output_root", "result_sha256"}}
    require(normalize_receipt(primary_receipt) == normalize_receipt(replay_receipt), "SST_OPTIMIZER_EXCHANGE_REPLAY_RECEIPT_MISMATCH")
    manifests: dict[str, Any] = {}
    for label in ("A", "B"):
        primary = relative_file_manifest(primary_root / f"receiver-{label}")
        replay = relative_file_manifest(replay_root / f"receiver-{label}")
        require(primary == replay, f"SST_OPTIMIZER_EXCHANGE_REPLAY_RECEIVER_MISMATCH:{label}")
        manifests[label] = {"file_count": len(primary), "directory_sha256": payload_sha256(primary)}
    primary_validation = read_json(primary_root / "optimizer_exchange_validation.json")
    replay_validation = read_json(replay_root / "optimizer_exchange_validation.json")
    normalize_validation = lambda value: {key: child for key, child in value.items() if key not in {"pair_receipt_sha256", "validation_sha256"}}
    require(normalize_validation(primary_validation) == normalize_validation(replay_validation), "SST_OPTIMIZER_EXCHANGE_REPLAY_VALIDATION_MISMATCH")
    material = {
        "schema": "nanogpt-h20-reciprocal-optimizer-exchange-independent-replay-v1",
        "status": "PASS",
        "receiver_manifests": manifests,
        "primary_validation_sha256": primary_validation["validation_sha256"],
        "replay_validation_sha256": replay_validation["validation_sha256"],
        "normalized_validation_material_sha256": payload_sha256(normalize_validation(primary_validation)),
        "byte_exact_receiver_evidence": True,
        "future_information_used": False,
    }
    result = {**material, "independent_replay_sha256": payload_sha256(material)}
    write_json(primary_root / "optimizer_exchange_independent_replay.json", result)
    return result


__all__ = ["validate_optimizer_exchange", "validate_optimizer_exchange_replay"]
