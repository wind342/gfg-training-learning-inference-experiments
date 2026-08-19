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

from .amplitude_path import (
    AMPLITUDE_HORIZONS,
    AMPLITUDE_SCALES,
    RESPONSE_CENTERS,
    _state_at_scale,
    scale_key,
)
from .branches import _load_observation_arrays
from .execution import _read_checked
from .reciprocal_validator import _formal_prestate, _state, _state_exact
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
    require(locator.startswith("tensor-objects/"), "SST_AMPLITUDE_PATH_VALIDATION_TENSOR_LOCATOR_INVALID")
    key = (str(root), locator)
    path = root / locator
    require(path.is_file(), f"SST_AMPLITUDE_PATH_VALIDATION_TENSOR_MISSING:{path}")
    admitted_reference = {
        name: reference[name]
        for name in ("file_sha256", "raw_tensor_sha256", "shape", "dtype")
    }
    if key in cache:
        require(
            cache[key] == admitted_reference,
            "SST_AMPLITUDE_PATH_VALIDATION_TENSOR_REFERENCE_CONFLICT",
        )
    else:
        require(file_sha256(path) == reference["file_sha256"], "SST_AMPLITUDE_PATH_VALIDATION_TENSOR_FILE_HASH_MISMATCH")
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        require(list(value.shape) == list(reference["shape"]), "SST_AMPLITUDE_PATH_VALIDATION_TENSOR_SHAPE_MISMATCH")
        require(str(value.dtype) == str(reference["dtype"]), "SST_AMPLITUDE_PATH_VALIDATION_TENSOR_DTYPE_MISMATCH")
        require(
            hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
            == reference["raw_tensor_sha256"],
            "SST_AMPLITUDE_PATH_VALIDATION_TENSOR_RAW_HASH_MISMATCH",
        )
        cache[key] = admitted_reference
    return np.asarray(np.load(path, allow_pickle=False, mmap_mode="r"))


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def _rms(value: np.ndarray) -> float:
    finite = np.isfinite(value)
    if not bool(np.any(finite)):
        return 0.0
    selected = value[finite].astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(selected * selected, dtype=np.float64)))


def _validate_probe(
    root: Path,
    state_id: str,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    observation = _read_checked(
        root / "probe-observations" / "CSRG-4C-v1" / f"{state_id}.json",
        "nanogpt-stepwise-probe-observation-v1",
    )
    require(observation["observed_state_id"] == state_id, "SST_AMPLITUDE_PATH_VALIDATION_PROBE_STATE_MISMATCH")
    require(observation["actual_forward_count"] == 12, "SST_AMPLITUDE_PATH_VALIDATION_PROBE_FORWARD_COUNT_MISMATCH")
    require(observation["baseline_byte_exact"] is True, "SST_AMPLITUDE_PATH_VALIDATION_BASELINE_NOT_EXACT")
    for reference in _tensor_refs(observation):
        _validate_tensor(root, reference, cache)
    return observation


def validate_amplitude_path(
    *,
    root: Path,
    formal_root: Path,
    reciprocal_root: Path,
    amplitude_path_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(amplitude_path_protocol_path)
    require(protocol["schema"] == "nanogpt-b-update-amplitude-path-protocol-v1", "SST_AMPLITUDE_PATH_VALIDATION_PROTOCOL_SCHEMA_INVALID")
    receipt = _read_checked(root / "amplitude_path_pair_receipt.json", "nanogpt-amplitude-path-pair-receipt-v1")
    require(file_sha256(amplitude_path_protocol_path) == receipt["amplitude_path_protocol_sha256"], "SST_AMPLITUDE_PATH_VALIDATION_PROTOCOL_HASH_MISMATCH")
    require(tuple(float(v) for v in protocol["scales"]) == AMPLITUDE_SCALES, "SST_AMPLITUDE_PATH_VALIDATION_SCALES_INVALID")
    require(tuple(float(v) for v in protocol["response_centers"]) == RESPONSE_CENTERS, "SST_AMPLITUDE_PATH_VALIDATION_CENTERS_INVALID")
    require(tuple(int(v) for v in protocol["horizons"]) == AMPLITUDE_HORIZONS, "SST_AMPLITUDE_PATH_VALIDATION_HORIZONS_INVALID")
    endpoints = {str(row["label"]): row for row in protocol["receivers"]}
    donor_pre = _formal_prestate(formal_root, endpoints["B"])
    donor_post = _state(reciprocal_root, "B", 1, "native_parameter_only")
    donor_delta = {
        name: donor_post.parameters[name] - donor_pre.parameters[name]
        for name in donor_pre.parameters
    }
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    receiver_rows: dict[str, Any] = {}
    checks = 0
    for label in ("A", "B"):
        entry_root = root / f"receiver-{label}"
        endpoint = endpoints[label]
        receiver_pre = _formal_prestate(formal_root, endpoint)
        receiver_receipt = _read_checked(
            entry_root / "amplitude_path_receiver_receipt.json",
            "nanogpt-amplitude-path-receiver-receipt-v1",
        )
        seed = _read_checked(entry_root / "amplitude_path_seed.json", "nanogpt-amplitude-path-seed-v1")
        require(seed["receiver_skip_state_source_object_id"] == endpoint["skip_state_source_object_id"], "SST_AMPLITUDE_PATH_VALIDATION_RECEIVER_SOURCE_ID_MISMATCH")
        require(seed["donor_update_source_object_id"] == protocol["donor_update"]["source_object_id"], "SST_AMPLITUDE_PATH_VALIDATION_DONOR_SOURCE_ID_MISMATCH")
        for reference in _tensor_refs(seed):
            _validate_tensor(entry_root, reference, cache)
        observations: dict[float, dict[str, Any]] = {}
        arrays: dict[float, dict[str, np.ndarray]] = {}
        initial_hashes: dict[float, str] = {}
        for scale in AMPLITUDE_SCALES:
            key = scale_key(scale)
            record = _read_checked(
                entry_root / "h-001-path" / f"{key}-state.json",
                "nanogpt-amplitude-path-state-v1",
            )
            require(float(record["scale"]) == scale and int(record["horizon"]) == 1, "SST_AMPLITUDE_PATH_VALIDATION_INITIAL_SCALE_MISMATCH")
            require(record["receiver_optimizer_held_at_skip_state"] is True, "SST_AMPLITUDE_PATH_VALIDATION_INITIAL_OPTIMIZER_NOT_HELD")
            actual = restorable_state_from_manifest(entry_root, record["state"])
            expected = _state_at_scale(receiver_pre, donor_delta, scale)
            require(_state_exact(actual, expected), f"SST_AMPLITUDE_PATH_VALIDATION_INITIAL_STATE_MISMATCH:{label}:{key}")
            initial_hashes[scale] = actual.commitment()["state_sha256"]
            observations[scale] = _validate_probe(entry_root, record["state"]["state_id"], cache)
            arrays[scale] = _load_observation_arrays(entry_root, observations[scale])
            checks += 6

        response = _read_checked(entry_root / "h1_response_path.json", "nanogpt-amplitude-response-path-v1")
        require(response["computed_before_continuation"] is True, "SST_AMPLITUDE_PATH_VALIDATION_RESPONSE_NOT_FROZEN_FIRST")
        require(response["categorical_values_subtracted"] is False, "SST_AMPLITUDE_PATH_VALIDATION_CATEGORICAL_SUBTRACTION")
        numeric_rows: dict[str, Any] = {}
        epsilon = float(protocol["epsilon"])
        for role, encoded in sorted(response["numeric_responses"].items()):
            values = {
                scale: arrays[scale][role].astype(np.float64, copy=False)
                for scale in AMPLITUDE_SCALES
            }
            for scale in AMPLITUDE_SCALES:
                encoded_scale = encoded["scale_values"][scale_key(scale)]
                require(
                    float(encoded_scale["scale"]) == scale,
                    f"SST_AMPLITUDE_PATH_VALIDATION_SCALE_LABEL_MISMATCH:{label}:{role}:{scale}",
                )
                require(
                    _equal(
                        _validate_tensor(
                            entry_root,
                            encoded_scale["value"],
                            cache,
                        ),
                        values[scale],
                    ),
                    f"SST_AMPLITUDE_PATH_VALIDATION_SCALE_VALUE_MISMATCH:{label}:{role}:{scale}",
                )
            derivatives: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            for center in RESPONSE_CENTERS:
                j_value = (values[center + epsilon] - values[center - epsilon]) / (2.0 * epsilon)
                k_value = (values[center + epsilon] + values[center - epsilon] - 2.0 * values[center]) / (epsilon * epsilon)
                derivatives[center] = (j_value, k_value)
                center_row = encoded["centers"][scale_key(center)]
                require(_equal(_validate_tensor(entry_root, center_row["j_first_order"], cache), j_value), f"SST_AMPLITUDE_PATH_VALIDATION_J_MISMATCH:{label}:{role}:{center}")
                require(_equal(_validate_tensor(entry_root, center_row["k_curvature"], cache), k_value), f"SST_AMPLITUDE_PATH_VALIDATION_K_MISMATCH:{label}:{role}:{center}")
            simpson = (derivatives[0.0][0] + 4.0 * derivatives[0.25][0] + 2.0 * derivatives[0.5][0] + 4.0 * derivatives[0.75][0] + derivatives[1.0][0]) / 12.0
            exact = values[1.0] - values[0.0]
            predictions = {
                "simpson": simpson,
                "start_j": derivatives[0.0][0],
                "start_jk": derivatives[0.0][0] + 0.5 * derivatives[0.0][1],
                "end_j": derivatives[1.0][0],
                "end_jk": derivatives[1.0][0] - 0.5 * derivatives[1.0][1],
            }
            references = {
                "simpson": "simpson_delta_prediction",
                "start_j": "start_endpoint_j_prediction",
                "start_jk": "start_endpoint_jk_prediction",
                "end_j": "end_endpoint_j_prediction",
                "end_jk": "end_endpoint_jk_prediction",
            }
            require(_equal(_validate_tensor(entry_root, encoded["exact_scale_zero_to_one_delta"], cache), exact), f"SST_AMPLITUDE_PATH_VALIDATION_EXACT_DELTA_MISMATCH:{label}:{role}")
            residuals: dict[str, float] = {}
            for name, prediction in predictions.items():
                require(_equal(_validate_tensor(entry_root, encoded[references[name]], cache), prediction), f"SST_AMPLITUDE_PATH_VALIDATION_PREDICTION_MISMATCH:{label}:{role}:{name}")
                residuals[name] = _rms(prediction - exact)
            numeric_rows[role] = {
                "shape": list(exact.shape),
                "exact_delta_rms": _rms(exact),
                "residual_rms": residuals,
                "simpson_strictly_best": residuals["simpson"] < min(
                    residuals[name] for name in ("start_j", "start_jk", "end_j", "end_jk")
                ),
            }
            checks += 13 + len(AMPLITUDE_SCALES)

        categorical_rows: dict[str, Any] = {}
        for role, encoded in sorted(response["categorical_paths"].items()):
            for scale in AMPLITUDE_SCALES:
                encoded_scale = encoded["scale_values"][scale_key(scale)]
                require(
                    float(encoded_scale["scale"]) == scale,
                    f"SST_AMPLITUDE_PATH_VALIDATION_CATEGORY_SCALE_LABEL_MISMATCH:{label}:{role}:{scale}",
                )
                require(
                    _equal(
                        _validate_tensor(entry_root, encoded_scale["value"], cache),
                        arrays[scale][role],
                    ),
                    f"SST_AMPLITUDE_PATH_VALIDATION_CATEGORY_SCALE_VALUE_MISMATCH:{label}:{role}:{scale}",
                )
            endpoint_mask = np.not_equal(arrays[1.0][role], arrays[0.0][role])
            require(_equal(_validate_tensor(entry_root, encoded["endpoint_changed_mask"], cache), endpoint_mask), f"SST_AMPLITUDE_PATH_VALIDATION_CATEGORY_ENDPOINT_MISMATCH:{label}:{role}")
            ordered = list(AMPLITUDE_SCALES)
            composed = arrays[0.0][role].copy()
            event_count = 0
            for left, right in zip(ordered, ordered[1:]):
                transition = encoded["adjacent_transitions"][f"{scale_key(left)}_to_{scale_key(right)}"]
                mask = np.not_equal(arrays[right][role], arrays[left][role])
                require(_equal(_validate_tensor(entry_root, transition["changed_mask"], cache), mask), f"SST_AMPLITUDE_PATH_VALIDATION_CATEGORY_PATH_MISMATCH:{label}:{role}:{right}")
                composed[mask] = arrays[right][role][mask]
                event_count += int(np.count_nonzero(mask))
            require(_equal(composed, arrays[1.0][role]), f"SST_AMPLITUDE_PATH_VALIDATION_CATEGORY_COMPOSITION_MISMATCH:{label}:{role}")
            categorical_rows[role] = {
                "endpoint_changed_count": int(np.count_nonzero(endpoint_mask)),
                "adjacent_event_count": event_count,
                "path_composes_to_exact_endpoint": True,
            }
            checks += 12 + 2 * len(AMPLITUDE_SCALES)

        chain = {scale: initial_hashes[scale] for scale in RESPONSE_CENTERS}
        horizon_capabilities: dict[str, Any] = {
            "1": {
                scale_key(scale): observations[scale]["capability_accuracy"]
                for scale in RESPONSE_CENTERS
            }
        }
        continuation = receiver_receipt["continuation_results"]
        require(len(continuation) == 99, "SST_AMPLITUDE_PATH_VALIDATION_CONTINUATION_COUNT_MISMATCH")
        for expected_horizon, row in enumerate(continuation, start=2):
            require(int(row["horizon"]) == expected_horizon, "SST_AMPLITUDE_PATH_VALIDATION_CONTINUATION_HORIZON_MISMATCH")
            step_results = {}
            for scale in RESPONSE_CENTERS:
                key = scale_key(scale)
                path = entry_root / "continuations" / f"step-{int(row['physical_optimizer_step']):05d}-to-{int(row['physical_optimizer_step']) + 1:05d}" / f"{key}.json"
                step_result = _read_checked(path, "nanogpt-amplitude-path-continuation-step-v1")
                require(step_result["from_state_sha256"] == chain[scale], f"SST_AMPLITUDE_PATH_VALIDATION_CHAIN_FROM_MISMATCH:{label}:{expected_horizon}:{key}")
                require(step_result["to_state_sha256"] == row["steps"][key]["to_state_sha256"], f"SST_AMPLITUDE_PATH_VALIDATION_CHAIN_TO_MISMATCH:{label}:{expected_horizon}:{key}")
                require(step_result["step_evidence"]["execute_optimizer"] is True, "SST_AMPLITUDE_PATH_VALIDATION_OPTIMIZER_NOT_EXECUTED")
                require(step_result["future_information_used"] is False, "SST_AMPLITUDE_PATH_VALIDATION_FUTURE_USED")
                for reference in _tensor_refs(step_result):
                    _validate_tensor(entry_root, reference, cache)
                chain[scale] = step_result["to_state_sha256"]
                step_results[key] = step_result
                checks += 7
            batch_hashes = {payload_sha256(value["same_batch_all_scales"]) for value in step_results.values()}
            rng_values = {int(value["same_external_rng_opportunity_all_scales"]) for value in step_results.values()}
            require(len(batch_hashes) == 1 and len(rng_values) == 1, f"SST_AMPLITUDE_PATH_VALIDATION_BRANCH_ALIGNMENT_MISMATCH:{label}:{expected_horizon}")

        horizon_index = {int(row["horizon"]): row for row in receiver_receipt["horizon_results"]}
        require(tuple(sorted(horizon_index)) == AMPLITUDE_HORIZONS, "SST_AMPLITUDE_PATH_VALIDATION_REGISTERED_HORIZONS_MISMATCH")
        for horizon in AMPLITUDE_HORIZONS[1:]:
            capabilities: dict[str, float] = {}
            for scale in RESPONSE_CENTERS:
                key = scale_key(scale)
                record = _read_checked(
                    entry_root / "horizons" / f"h-{horizon:03d}" / f"{key}-state.json",
                    "nanogpt-amplitude-path-state-v1",
                )
                require(record["state"]["commitment"]["state_sha256"] == horizon_index[horizon]["states"][key]["state_sha256"], "SST_AMPLITUDE_PATH_VALIDATION_HORIZON_RECEIPT_MISMATCH")
                if horizon == max(AMPLITUDE_HORIZONS):
                    require(record["state"]["commitment"]["state_sha256"] == chain[scale], "SST_AMPLITUDE_PATH_VALIDATION_FINAL_CHAIN_MISMATCH")
                observation = _validate_probe(entry_root, record["state"]["state_id"], cache)
                capabilities[key] = float(observation["capability_accuracy"])
                checks += 4
            horizon_capabilities[str(horizon)] = capabilities

        receiver_rows[label] = {
            "numeric_roles": numeric_rows,
            "categorical_roles": categorical_rows,
            "simpson_strictly_best_role_count": sum(
                bool(row["simpson_strictly_best"]) for row in numeric_rows.values()
            ),
            "numeric_role_count": len(numeric_rows),
            "categorical_role_count": len(categorical_rows),
            "horizon_capability_accuracy": horizon_capabilities,
            "tensor_payload_validation_count": sum(
                1 for path_root, _locator in cache if path_root == str(entry_root)
            ),
        }

    material = {
        "schema": "nanogpt-b-update-amplitude-path-validation-v1",
        "status": "PASS",
        "pair_receipt_sha256": receipt["result_sha256"],
        "amplitude_path_protocol_sha256": file_sha256(amplitude_path_protocol_path),
        "receiver_rows": receiver_rows,
        "validated_tensor_payload_count": len(cache),
        "check_count": checks,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(root / "amplitude_path_validation.json", result)
    return result


def validate_amplitude_path_replay(
    *,
    primary_root: Path,
    replay_root: Path,
) -> dict[str, Any]:
    primary_receipt = read_json(primary_root / "amplitude_path_pair_receipt.json")
    replay_receipt = read_json(replay_root / "amplitude_path_pair_receipt.json")
    normalize_receipt = lambda value: {
        key: child
        for key, child in value.items()
        if key not in {"output_root", "result_sha256"}
    }
    require(normalize_receipt(primary_receipt) == normalize_receipt(replay_receipt), "SST_AMPLITUDE_PATH_REPLAY_RECEIPT_MISMATCH")
    manifests: dict[str, Any] = {}
    for label in ("A", "B"):
        primary = relative_file_manifest(primary_root / f"receiver-{label}")
        replay = relative_file_manifest(replay_root / f"receiver-{label}")
        require(primary == replay, f"SST_AMPLITUDE_PATH_REPLAY_RECEIVER_MISMATCH:{label}")
        manifests[label] = {
            "file_count": len(primary),
            "directory_sha256": payload_sha256(primary),
        }
    primary_validation = read_json(primary_root / "amplitude_path_validation.json")
    replay_validation = read_json(replay_root / "amplitude_path_validation.json")
    normalize_validation = lambda value: {
        key: child
        for key, child in value.items()
        if key not in {"pair_receipt_sha256", "validation_sha256"}
    }
    require(normalize_validation(primary_validation) == normalize_validation(replay_validation), "SST_AMPLITUDE_PATH_REPLAY_VALIDATION_MISMATCH")
    material = {
        "schema": "nanogpt-b-update-amplitude-path-independent-replay-v1",
        "status": "PASS",
        "receiver_manifests": manifests,
        "primary_validation_sha256": primary_validation["validation_sha256"],
        "replay_validation_sha256": replay_validation["validation_sha256"],
        "normalized_validation_material_sha256": payload_sha256(
            normalize_validation(primary_validation)
        ),
        "byte_exact_receiver_evidence": True,
        "future_information_used": False,
    }
    result = {**material, "independent_replay_sha256": payload_sha256(material)}
    write_json(primary_root / "amplitude_path_independent_replay.json", result)
    return result
