from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
)
from experiments.gfg_nanogpt_support_transition_v1.runtime import (
    StateSnapshot,
    assert_snapshot_isolation,
)

from .branches import _load_observation_arrays, _probe_at_state
from .contracts import ComponentRegistry, ProbeContract
from .execution import _checked_result, _encode_batch, _encode_state, _encode_step
from .local_response import _categorical_transition, _scaled_delta
from .reciprocal import _seed_execution, _transplant_state, _window
from .runtime import StepwiseTrainingRuntime
from .storage import TensorStore


AMPLITUDE_SCALES = (
    -0.125,
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
)
RESPONSE_CENTERS = (0.0, 0.25, 0.5, 0.75, 1.0)
AMPLITUDE_HORIZONS = (1, 2, 5, 20, 100)


def scale_key(scale: float) -> str:
    sign = "m" if scale < 0 else "p"
    magnitude = f"{abs(scale):.3f}".replace(".", "p")
    return sign + magnitude


def _state_at_scale(
    receiver: StateSnapshot,
    donor_delta: Mapping[str, Any],
    scale: float,
) -> StateSnapshot:
    return _transplant_state(
        receiver,
        parameter_delta=_scaled_delta(dict(donor_delta), scale),
    )


def _numeric_path(
    store: TensorStore,
    *,
    label: str,
    arrays: dict[float, dict[str, np.ndarray]],
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = tuple(sorted(arrays[0.0]))
    require(
        all(tuple(sorted(value)) == keys for value in arrays.values()),
        "SST_AMPLITUDE_PATH_PROBE_OUTPUT_SET_MISMATCH",
    )
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for key in keys:
        values = {scale: arrays[scale][key] for scale in AMPLITUDE_SCALES}
        shapes = {tuple(value.shape) for value in values.values()}
        require(len(shapes) == 1, f"SST_AMPLITUDE_PATH_SHAPE_MISMATCH:{key}")
        prefix = f"amplitude-path:{label}:h1:{key}"
        if np.issubdtype(values[0.0].dtype, np.floating):
            values64 = {
                scale: value.astype(np.float64, copy=False)
                for scale, value in values.items()
            }
            centers: dict[str, Any] = {}
            derivatives: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            for center in RESPONSE_CENTERS:
                plus = values64[center + epsilon]
                base = values64[center]
                minus = values64[center - epsilon]
                j_value = (plus - minus) / (2.0 * epsilon)
                k_value = (plus + minus - 2.0 * base) / (epsilon * epsilon)
                derivatives[center] = (j_value, k_value)
                center_key = scale_key(center)
                centers[center_key] = {
                    "scale": center,
                    "j_first_order": store.put(
                        j_value,
                        representation=f"{prefix}:{center_key}:central_first_order_response",
                    ),
                    "k_curvature": store.put(
                        k_value,
                        representation=f"{prefix}:{center_key}:central_second_order_curvature",
                    ),
                }
            j0, k0 = derivatives[0.0]
            j1, k1 = derivatives[1.0]
            simpson = (
                derivatives[0.0][0]
                + 4.0 * derivatives[0.25][0]
                + 2.0 * derivatives[0.5][0]
                + 4.0 * derivatives[0.75][0]
                + derivatives[1.0][0]
            ) / 12.0
            exact = values64[1.0] - values64[0.0]
            numeric[key] = {
                "scale_values": {
                    scale_key(scale): {
                        "scale": scale,
                        "value": store.put(
                            value,
                            representation=f"{prefix}:{scale_key(scale)}:numeric_value_float64",
                        ),
                    }
                    for scale, value in values64.items()
                },
                "centers": centers,
                "simpson_delta_prediction": store.put(
                    simpson,
                    representation=f"{prefix}:fixed_five_center_simpson_delta_prediction",
                ),
                "exact_scale_zero_to_one_delta": store.put(
                    exact,
                    representation=f"{prefix}:exact_scale_zero_to_one_delta",
                ),
                "start_endpoint_j_prediction": store.put(
                    j0,
                    representation=f"{prefix}:start_endpoint_first_order_prediction",
                ),
                "start_endpoint_jk_prediction": store.put(
                    j0 + 0.5 * k0,
                    representation=f"{prefix}:start_endpoint_signed_second_order_prediction",
                ),
                "end_endpoint_j_prediction": store.put(
                    j1,
                    representation=f"{prefix}:end_endpoint_first_order_prediction",
                ),
                "end_endpoint_jk_prediction": store.put(
                    j1 - 0.5 * k1,
                    representation=f"{prefix}:end_endpoint_signed_second_order_prediction",
                ),
            }
        else:
            baseline = values[0.0]
            endpoint = values[1.0]
            scale_values = {
                scale_key(scale): {
                    "scale": scale,
                    "value": store.put(
                        value,
                        representation=f"{prefix}:{scale_key(scale)}:categorical_value",
                    ),
                }
                for scale, value in values.items()
            }
            adjacent = {}
            ordered = list(AMPLITUDE_SCALES)
            for left, right in zip(ordered, ordered[1:]):
                mask = np.not_equal(values[right], values[left])
                adjacent[f"{scale_key(left)}_to_{scale_key(right)}"] = {
                    "from_scale": left,
                    "to_scale": right,
                    "changed_mask": store.put(
                        mask,
                        representation=(
                            f"{prefix}:{scale_key(left)}_to_{scale_key(right)}:"
                            "categorical_transition_mask"
                        ),
                    ),
                    "changed_count": int(np.count_nonzero(mask)),
                }
            categorical[key] = {
                "scale_values": scale_values,
                "adjacent_transitions": adjacent,
                "endpoint_changed_mask": store.put(
                    np.not_equal(endpoint, baseline),
                    representation=f"{prefix}:scale_zero_to_one_categorical_transition_mask",
                ),
                "endpoint_changed_count": int(np.count_nonzero(endpoint != baseline)),
            }
    return numeric, categorical


def _execute_receiver_path(
    *,
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    output_root: Path,
    receiver: dict[str, Any],
    donor: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha256: str,
    main_protocol_sha256: str,
) -> dict[str, Any]:
    label = str(receiver["endpoint"]["label"])
    endpoint = receiver["endpoint"]
    root = output_root / f"receiver-{label}"
    initial = receiver["prestate"]
    donor_delta = donor["parameter_delta"]
    states = {
        scale: _state_at_scale(initial, donor_delta, scale)
        for scale in AMPLITUDE_SCALES
    }
    assert_snapshot_isolation(states.values())
    experiment_id = "b-update-amplitude-path-" + payload_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "receiver": endpoint,
            "donor": donor["endpoint"],
        }
    )[:32]
    seed = _checked_result(
        root / "amplitude_path_seed.json",
        {
            "schema": "nanogpt-amplitude-path-seed-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver": endpoint,
            "donor": donor["endpoint"],
            "receiver_skip_state_source_object_id": endpoint[
                "skip_state_source_object_id"
            ],
            "donor_update_source_object_id": protocol["donor_update"][
                "source_object_id"
            ],
            "receiver_prestate_id": receiver["source_record"]["state"]["state_id"],
            "receiver_prestate_sha256": initial.commitment()["state_sha256"],
            "donor_native_full_step": _encode_step(
                store,
                donor["full_evidence"],
                transition_id=f"{experiment_id}:donor-native-full",
            ),
            "scales": list(AMPLITUDE_SCALES),
            "response_centers": list(RESPONSE_CENTERS),
            "continuation_horizons": list(AMPLITUDE_HORIZONS),
            "adam_state_transplanted": False,
            "future_information_used": False,
        },
    )

    observations: dict[float, dict[str, Any]] = {}
    state_rows: dict[str, Any] = {}
    optimizer_step = int(endpoint["optimizer_step"])
    for scale in AMPLITUDE_SCALES:
        key = scale_key(scale)
        manifest = _encode_state(
            store,
            states[scale],
            entry_id=str(endpoint["entry_id"]),
            window_id=f"{experiment_id}:scale:{key}",
            optimizer_step=optimizer_step + 1,
            protocol_sha256=main_protocol_sha256,
        )
        state_result = _checked_result(
            root / "h-001-path" / f"{key}-state.json",
            {
                "schema": "nanogpt-amplitude-path-state-v1",
                "status": "PASS",
                "experiment_id": experiment_id,
                "receiver_label": label,
                "scale": scale,
                "horizon": 1,
                "state": manifest,
                "state_summary": runtime.state_summary(states[scale]),
                "receiver_optimizer_held_at_skip_state": True,
            },
        )
        observation = _probe_at_state(
            runtime,
            store,
            branch_entry_root=root,
            state=states[scale],
            state_manifest=manifest,
        )
        observations[scale] = observation
        state_rows[key] = {
            "scale": scale,
            "state_id": manifest["state_id"],
            "state_sha256": manifest["commitment"]["state_sha256"],
            "state_result_sha256": state_result["result_sha256"],
            "probe_observation_id": observation["probe_observation_id"],
            "probe_result_sha256": observation["result_sha256"],
            "capability_accuracy": observation["capability_accuracy"],
        }

    arrays = {
        scale: _load_observation_arrays(root, observation)
        for scale, observation in observations.items()
    }
    numeric, categorical = _numeric_path(
        store,
        label=label,
        arrays=arrays,
        epsilon=float(protocol["epsilon"]),
    )
    path_result = _checked_result(
        root / "h1_response_path.json",
        {
            "schema": "nanogpt-amplitude-response-path-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver_label": label,
            "donor_label": str(donor["endpoint"]["label"]),
            "epsilon": protocol["epsilon"],
            "scales": list(AMPLITUDE_SCALES),
            "response_centers": list(RESPONSE_CENTERS),
            "simpson_contract": protocol["simpson_contract"],
            "numeric_responses": numeric,
            "categorical_paths": categorical,
            "categorical_values_subtracted": False,
            "computed_before_continuation": True,
            "future_information_used": False,
        },
    )

    continuation_states = {
        scale: states[scale].clone() for scale in RESPONSE_CENTERS
    }
    horizon_rows: list[dict[str, Any]] = [
        {
            "horizon": 1,
            "states": {
                scale_key(scale): state_rows[scale_key(scale)]
                for scale in RESPONSE_CENTERS
            },
        }
    ]
    continuation_rows: list[dict[str, Any]] = []
    for horizon in range(2, max(AMPLITUDE_HORIZONS) + 1):
        physical_step = optimizer_step + horizon - 1
        batch = runtime.load_batch(physical_step)
        opportunity_seed = runtime.derive_seed(
            protocol_sha256,
            str(endpoint["entry_id"]),
            physical_step,
            horizon,
        )
        step_rows: dict[str, Any] = {}
        for scale in RESPONSE_CENTERS:
            key = scale_key(scale)
            runtime.restore(continuation_states[scale])
            from_sha = continuation_states[scale].commitment()["state_sha256"]
            evidence = runtime.train_actual_step(
                batch,
                execute_optimizer=True,
                seed=opportunity_seed,
            )
            continuation_states[scale] = runtime.snapshot()
            encoded = _encode_step(
                store,
                evidence,
                transition_id=f"{experiment_id}:{key}:h{horizon:03d}",
            )
            step_result = _checked_result(
                root
                / "continuations"
                / f"step-{physical_step:05d}-to-{physical_step + 1:05d}"
                / f"{key}.json",
                {
                    "schema": "nanogpt-amplitude-path-continuation-step-v1",
                    "status": "PASS",
                    "experiment_id": experiment_id,
                    "receiver_label": label,
                    "scale": scale,
                    "horizon": horizon,
                    "physical_optimizer_step": physical_step,
                    "from_state_sha256": from_sha,
                    "to_state_sha256": continuation_states[scale].commitment()[
                        "state_sha256"
                    ],
                    "same_batch_all_scales": _encode_batch(batch),
                    "same_external_rng_opportunity_all_scales": opportunity_seed,
                    "step_evidence": encoded,
                    "post_state_summary": runtime.state_summary(
                        continuation_states[scale]
                    ),
                    "future_information_used": False,
                },
            )
            step_rows[key] = {
                "result_sha256": step_result["result_sha256"],
                "from_state_sha256": from_sha,
                "to_state_sha256": continuation_states[scale].commitment()[
                    "state_sha256"
                ],
            }
        continuation_rows.append(
            {
                "physical_optimizer_step": physical_step,
                "horizon": horizon,
                "steps": step_rows,
            }
        )
        if horizon in AMPLITUDE_HORIZONS:
            registered: dict[str, Any] = {}
            for scale in RESPONSE_CENTERS:
                key = scale_key(scale)
                manifest = _encode_state(
                    store,
                    continuation_states[scale],
                    entry_id=str(endpoint["entry_id"]),
                    window_id=f"{experiment_id}:continuation:{key}",
                    optimizer_step=optimizer_step + horizon,
                    protocol_sha256=main_protocol_sha256,
                )
                state_result = _checked_result(
                    root / "horizons" / f"h-{horizon:03d}" / f"{key}-state.json",
                    {
                        "schema": "nanogpt-amplitude-path-state-v1",
                        "status": "PASS",
                        "experiment_id": experiment_id,
                        "receiver_label": label,
                        "scale": scale,
                        "horizon": horizon,
                        "state": manifest,
                        "state_summary": runtime.state_summary(
                            continuation_states[scale]
                        ),
                        "receiver_optimizer_held_at_skip_state": False,
                    },
                )
                observation = _probe_at_state(
                    runtime,
                    store,
                    branch_entry_root=root,
                    state=continuation_states[scale],
                    state_manifest=manifest,
                )
                registered[key] = {
                    "scale": scale,
                    "state_id": manifest["state_id"],
                    "state_sha256": manifest["commitment"]["state_sha256"],
                    "state_result_sha256": state_result["result_sha256"],
                    "probe_observation_id": observation["probe_observation_id"],
                    "probe_result_sha256": observation["result_sha256"],
                    "capability_accuracy": observation["capability_accuracy"],
                }
            horizon_rows.append({"horizon": horizon, "states": registered})
        if horizon % 10 == 0:
            print(
                {
                    "event": "SST_AMPLITUDE_PATH_CONTINUATION_PROGRESS",
                    "receiver": label,
                    "horizon": horizon,
                },
                flush=True,
            )

    return _checked_result(
        root / "amplitude_path_receiver_receipt.json",
        {
            "schema": "nanogpt-amplitude-path-receiver-receipt-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver": endpoint,
            "donor": donor["endpoint"],
            "seed_result_sha256": seed["result_sha256"],
            "h1_state_rows": state_rows,
            "h1_response_path_result_sha256": path_result["result_sha256"],
            "horizon_results": horizon_rows,
            "continuation_results": continuation_rows,
            "future_information_used": False,
        },
    )


def execute_amplitude_path(
    *,
    formal_root: Path,
    source_root: Path,
    trainer_root: Path,
    output_root: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    main_protocol_path: Path,
    amplitude_path_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(amplitude_path_protocol_path)
    require(
        protocol["schema"] == "nanogpt-b-update-amplitude-path-protocol-v1",
        "SST_AMPLITUDE_PATH_PROTOCOL_SCHEMA_INVALID",
    )
    require(tuple(float(v) for v in protocol["scales"]) == AMPLITUDE_SCALES, "SST_AMPLITUDE_PATH_SCALES_INVALID")
    require(tuple(float(v) for v in protocol["response_centers"]) == RESPONSE_CENTERS, "SST_AMPLITUDE_PATH_CENTERS_INVALID")
    require(tuple(float(v) for v in protocol["continuation_scales"]) == RESPONSE_CENTERS, "SST_AMPLITUDE_PATH_CONTINUATION_SCALES_INVALID")
    require(tuple(int(v) for v in protocol["horizons"]) == AMPLITUDE_HORIZONS, "SST_AMPLITUDE_PATH_HORIZONS_INVALID")
    require(float(protocol["epsilon"]) == 0.125, "SST_AMPLITUDE_PATH_EPSILON_INVALID")
    endpoints = protocol["receivers"]
    require(len(endpoints) == 2 and {str(v["label"]) for v in endpoints} == {"A", "B"}, "SST_AMPLITUDE_PATH_RECEIVERS_INVALID")
    require(str(protocol["donor_update"]["label"]) == "B", "SST_AMPLITUDE_PATH_DONOR_INVALID")
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    require(probe_contract.probe_contract_id == protocol["probe_contract_id"], "SST_AMPLITUDE_PATH_PROBE_CONTRACT_ID_MISMATCH")
    protocol_sha = file_sha256(amplitude_path_protocol_path)
    main_protocol_sha = file_sha256(main_protocol_path)
    executions: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        window = _window(formal_root, endpoint)
        runtime = StepwiseTrainingRuntime(
            source_root / str(endpoint["source_bundle_id"]),
            trainer_root,
            registry,
            probe_contract,
        )
        try:
            executions[str(endpoint["label"])] = _seed_execution(
                runtime,
                formal_root=formal_root,
                endpoint=endpoint,
                window=window,
                protocol_sha256=protocol_sha,
            )
        finally:
            runtime.close()
    donor = executions["B"]
    completed: list[dict[str, Any]] = []
    for label in ("A", "B"):
        endpoint = executions[label]["endpoint"]
        store = TensorStore(output_root / f"receiver-{label}" / "tensor-objects")
        runtime = StepwiseTrainingRuntime(
            source_root / str(endpoint["source_bundle_id"]),
            trainer_root,
            registry,
            probe_contract,
        )
        try:
            receipt = _execute_receiver_path(
                runtime=runtime,
                store=store,
                output_root=output_root,
                receiver=executions[label],
                donor=donor,
                protocol=protocol,
                protocol_sha256=protocol_sha,
                main_protocol_sha256=main_protocol_sha,
            )
            completed.append(
                {"receiver_label": label, "receipt_sha256": receipt["result_sha256"]}
            )
            print(
                {"event": "SST_AMPLITUDE_PATH_RECEIVER_COMPLETE", "receiver": label},
                flush=True,
            )
        finally:
            runtime.close()
    return _checked_result(
        output_root / "amplitude_path_pair_receipt.json",
        {
            "schema": "nanogpt-amplitude-path-pair-receipt-v1",
            "status": "PASS",
            "formal_root": str(formal_root.resolve()),
            "source_root": str(source_root.resolve()),
            "trainer_root": str(trainer_root.resolve()),
            "output_root": str(output_root.resolve()),
            "amplitude_path_protocol_sha256": protocol_sha,
            "main_protocol_sha256": main_protocol_sha,
            "component_registry_sha256": registry.source_sha256,
            "probe_contract_sha256": probe_contract.source_sha256,
            "completed": completed,
            "receiver_count": 2,
            "scale_count": len(AMPLITUDE_SCALES),
            "continuation_scale_count": len(RESPONSE_CENTERS),
            "horizons": list(AMPLITUDE_HORIZONS),
            "epsilon": protocol["epsilon"],
            "h1_path_computed_before_continuation": True,
            "future_information_used": False,
        },
    )
