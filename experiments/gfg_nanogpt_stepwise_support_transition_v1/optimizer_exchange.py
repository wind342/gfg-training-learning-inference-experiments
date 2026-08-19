from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

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

from .amplitude_path import scale_key
from .branches import _load_observation_arrays, _probe_at_state
from .contracts import ComponentRegistry, ProbeContract
from .execution import _checked_result, _encode_batch, _encode_state, _encode_step, _read_checked
from .reciprocal_gfg_validator import _main_object_index
from .runtime import StepwiseTrainingRuntime
from .storage import TensorStore, restorable_state_from_manifest


EXCHANGE_BRANCHES = (
    "theta0_O0",
    "theta1_O1",
    "theta0_O1",
    "theta1_O0",
)
EXCHANGE_OBSERVATION_HORIZONS = (20, 21, 100)
EXCHANGE_CONTINUATION_HORIZONS = tuple(range(21, 101))


def _clone_parameters(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().contiguous().cpu().clone()
        for name, value in values.items()
    }


def _clone_optimizer(
    values: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            key: value.detach().contiguous().cpu().clone()
            for key, value in child.items()
        }
        for name, child in values.items()
    }


def compose_parameter_optimizer_state(
    parameter_donor: StateSnapshot,
    optimizer_donor: StateSnapshot,
) -> StateSnapshot:
    require(
        set(parameter_donor.parameters) == set(optimizer_donor.parameters),
        "SST_OPTIMIZER_EXCHANGE_PARAMETER_SET_MISMATCH",
    )
    require(
        set(parameter_donor.optimizer) == set(optimizer_donor.optimizer),
        "SST_OPTIMIZER_EXCHANGE_OPTIMIZER_SET_MISMATCH",
    )
    for name in sorted(parameter_donor.parameters):
        require(
            parameter_donor.parameters[name].shape
            == optimizer_donor.parameters[name].shape,
            f"SST_OPTIMIZER_EXCHANGE_PARAMETER_SHAPE_MISMATCH:{name}",
        )
        require(
            set(optimizer_donor.optimizer[name]) == {"exp_avg", "exp_avg_sq", "step"},
            f"SST_OPTIMIZER_EXCHANGE_OPTIMIZER_KEYS_INVALID:{name}",
        )
        for key in ("exp_avg", "exp_avg_sq"):
            require(
                optimizer_donor.optimizer[name][key].shape
                == parameter_donor.parameters[name].shape,
                f"SST_OPTIMIZER_EXCHANGE_OPTIMIZER_SHAPE_MISMATCH:{name}:{key}",
            )
    return StateSnapshot(
        parameters=_clone_parameters(parameter_donor.parameters),
        optimizer=_clone_optimizer(optimizer_donor.optimizer),
    )


def _branch_contract(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["branch_id"]): row for row in protocol["branches"]}
    require(tuple(rows) == EXCHANGE_BRANCHES, "SST_OPTIMIZER_EXCHANGE_BRANCH_ORDER_INVALID")
    expected = {
        "theta0_O0": (0.0, 0.0, "native_control"),
        "theta1_O1": (1.0, 1.0, "native_control"),
        "theta0_O1": (0.0, 1.0, "reciprocal_optimizer_exchange"),
        "theta1_O0": (1.0, 0.0, "reciprocal_optimizer_exchange"),
    }
    for branch_id, (parameter_scale, optimizer_scale, kind) in expected.items():
        row = rows[branch_id]
        require(float(row["parameter_donor_scale"]) == parameter_scale, "SST_OPTIMIZER_EXCHANGE_PARAMETER_DONOR_INVALID")
        require(float(row["optimizer_donor_scale"]) == optimizer_scale, "SST_OPTIMIZER_EXCHANGE_OPTIMIZER_DONOR_INVALID")
        require(str(row["kind"]) == kind, "SST_OPTIMIZER_EXCHANGE_BRANCH_KIND_INVALID")
    return rows


def _prior_amplitude_preconditions(
    *,
    protocol: dict[str, Any],
    amplitude_root: Path,
    amplitude_graph_root: Path,
) -> dict[str, dict[float, dict[str, Any]]]:
    manifest = read_json(amplitude_graph_root / "amplitude_path_gfg_manifest.json")
    frozen = protocol["source_amplitude_path"]
    require(manifest["schema"] == frozen["graph_schema"], "SST_OPTIMIZER_EXCHANGE_PRIOR_GRAPH_SCHEMA_MISMATCH")
    require(manifest["manifest_sha256"] == frozen["graph_manifest_sha256"], "SST_OPTIMIZER_EXCHANGE_PRIOR_GRAPH_MANIFEST_MISMATCH")
    require(manifest["database_sha256"] == frozen["graph_database_sha256"], "SST_OPTIMIZER_EXCHANGE_PRIOR_GRAPH_DATABASE_MISMATCH")
    require(manifest["evidence_validation_sha256"] == frozen["evidence_validation_sha256"], "SST_OPTIMIZER_EXCHANGE_PRIOR_VALIDATION_MISMATCH")
    require(manifest["amplitude_path_protocol_sha256"] == frozen["protocol_sha256"], "SST_OPTIMIZER_EXCHANGE_PRIOR_PROTOCOL_MISMATCH")
    database = amplitude_graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "SST_OPTIMIZER_EXCHANGE_PRIOR_DATABASE_FILE_MISMATCH")
    prior_objects = _main_object_index(database)
    result: dict[str, dict[float, dict[str, Any]]] = {}
    for receiver in protocol["receivers"]:
        label = str(receiver["label"])
        entry_root = amplitude_root / f"receiver-{label}"
        result[label] = {}
        for scale in (0.0, 1.0):
            endpoint = receiver["endpoints"][str(int(scale))]
            key = scale_key(scale)
            h20_record = _read_checked(
                entry_root / "horizons" / "h-020" / f"{key}-state.json",
                "nanogpt-amplitude-path-state-v1",
            )
            h100_record = _read_checked(
                entry_root / "horizons" / "h-100" / f"{key}-state.json",
                "nanogpt-amplitude-path-state-v1",
            )
            h20_object_id = str(endpoint["h20_source_object_id"])
            h100_object_id = str(endpoint["h100_source_object_id"])
            require(manifest["state_catalog"][f"{label}:20:{key}"] == h20_object_id, "SST_OPTIMIZER_EXCHANGE_H20_OBJECT_ID_MISMATCH")
            require(manifest["state_catalog"][f"{label}:100:{key}"] == h100_object_id, "SST_OPTIMIZER_EXCHANGE_H100_OBJECT_ID_MISMATCH")
            require(prior_objects[h20_object_id]["content_sha256"] == endpoint["h20_source_content_sha256"], "SST_OPTIMIZER_EXCHANGE_H20_OBJECT_CONTENT_MISMATCH")
            require(prior_objects[h100_object_id]["content_sha256"] == endpoint["h100_source_content_sha256"], "SST_OPTIMIZER_EXCHANGE_H100_OBJECT_CONTENT_MISMATCH")
            require(h20_record["state"]["commitment"]["state_sha256"] == endpoint["h20_state_sha256"], "SST_OPTIMIZER_EXCHANGE_H20_STATE_MISMATCH")
            require(h100_record["state"]["commitment"]["state_sha256"] == endpoint["h100_state_sha256"], "SST_OPTIMIZER_EXCHANGE_H100_STATE_MISMATCH")
            result[label][scale] = {
                "h20_record": h20_record,
                "h20_state": restorable_state_from_manifest(entry_root, h20_record["state"]),
                "h100_record": h100_record,
                "h20_object": prior_objects[h20_object_id],
                "h100_object": prior_objects[h100_object_id],
            }
    return result


def _comparison_distance(
    branch_arrays: dict[str, dict[str, np.ndarray]],
    capability: dict[str, float],
) -> dict[str, Any]:
    keys = tuple(sorted(branch_arrays[EXCHANGE_BRANCHES[0]]))
    require(
        all(tuple(sorted(branch_arrays[name])) == keys for name in EXCHANGE_BRANCHES),
        "SST_OPTIMIZER_EXCHANGE_COMPARISON_ROLE_SET_MISMATCH",
    )
    controls = ("theta0_O0", "theta1_O1")
    hybrids = ("theta0_O1", "theta1_O0")
    result: dict[str, Any] = {}
    for hybrid in hybrids:
        distances: dict[str, Any] = {}
        for control in controls:
            numeric_sum = 0.0
            numeric_count = 0
            finite_status_mismatch_count = 0
            categorical_mismatch_count = 0
            categorical_coordinate_count = 0
            per_role: dict[str, Any] = {}
            for role in keys:
                left = branch_arrays[hybrid][role]
                right = branch_arrays[control][role]
                require(left.shape == right.shape, f"SST_OPTIMIZER_EXCHANGE_COMPARISON_SHAPE_MISMATCH:{role}")
                if np.issubdtype(left.dtype, np.floating):
                    require(np.issubdtype(right.dtype, np.floating), "SST_OPTIMIZER_EXCHANGE_COMPARISON_DTYPE_FAMILY_MISMATCH")
                    left64 = left.astype(np.float64, copy=False)
                    right64 = right.astype(np.float64, copy=False)
                    left_finite = np.isfinite(left64)
                    right_finite = np.isfinite(right64)
                    finite = left_finite & right_finite
                    mismatch = int(np.count_nonzero(left_finite ^ right_finite))
                    delta = left64[finite] - right64[finite]
                    squared_sum = float(np.sum(delta * delta, dtype=np.float64))
                    count = int(delta.size)
                    numeric_sum += squared_sum
                    numeric_count += count
                    finite_status_mismatch_count += mismatch
                    per_role[role] = {
                        "kind": "numeric",
                        "finite_coordinate_count": count,
                        "finite_status_mismatch_count": mismatch,
                        "rms": math.sqrt(squared_sum / count) if count else 0.0,
                    }
                else:
                    require(not np.issubdtype(right.dtype, np.floating), "SST_OPTIMIZER_EXCHANGE_COMPARISON_DTYPE_FAMILY_MISMATCH")
                    mismatch = int(np.count_nonzero(left != right))
                    count = int(left.size)
                    categorical_mismatch_count += mismatch
                    categorical_coordinate_count += count
                    per_role[role] = {
                        "kind": "categorical",
                        "coordinate_count": count,
                        "mismatch_count": mismatch,
                    }
            distances[control] = {
                "numeric_unweighted_finite_coordinate_rms": (
                    math.sqrt(numeric_sum / numeric_count) if numeric_count else 0.0
                ),
                "numeric_finite_coordinate_count": numeric_count,
                "numeric_finite_status_mismatch_count": finite_status_mismatch_count,
                "categorical_exact_mismatch_count": categorical_mismatch_count,
                "categorical_coordinate_count": categorical_coordinate_count,
                "capability_absolute_difference": abs(capability[hybrid] - capability[control]),
                "per_role": per_role,
            }
        result[hybrid] = {
            "parameter_donor_control": "theta0_O0" if hybrid == "theta0_O1" else "theta1_O1",
            "optimizer_donor_control": "theta1_O1" if hybrid == "theta0_O1" else "theta0_O0",
            "hybrid_capability_accuracy": capability[hybrid],
            "distances": distances,
        }
    return result


def _execute_receiver(
    *,
    runtime: StepwiseTrainingRuntime,
    store: TensorStore,
    output_root: Path,
    receiver: dict[str, Any],
    source_states: dict[float, dict[str, Any]],
    branches: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    protocol_sha256: str,
    main_protocol_sha256: str,
) -> dict[str, Any]:
    label = str(receiver["label"])
    entry_root = output_root / f"receiver-{label}"
    endpoint_states = {
        scale: source_states[scale]["h20_state"] for scale in (0.0, 1.0)
    }
    states = {
        branch_id: compose_parameter_optimizer_state(
            endpoint_states[float(row["parameter_donor_scale"])],
            endpoint_states[float(row["optimizer_donor_scale"])],
        )
        for branch_id, row in branches.items()
    }
    assert_snapshot_isolation(states.values())
    experiment_id = "h20-reciprocal-optimizer-exchange-" + payload_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "receiver": receiver,
            "source_manifest_sha256": protocol["source_amplitude_path"]["graph_manifest_sha256"],
        }
    )[:32]
    h20_rows: dict[str, Any] = {}
    h20_observations: dict[str, dict[str, Any]] = {}
    h20_step = int(receiver["h20_optimizer_step"])
    for branch_id in EXCHANGE_BRANCHES:
        row = branches[branch_id]
        manifest = _encode_state(
            store,
            states[branch_id],
            entry_id=str(receiver["entry_id"]),
            window_id=f"{experiment_id}:h20:{branch_id}",
            optimizer_step=h20_step,
            protocol_sha256=main_protocol_sha256,
        )
        state_result = _checked_result(
            entry_root / "horizons" / "h-020" / f"{branch_id}-state.json",
            {
                "schema": "nanogpt-h20-optimizer-exchange-state-v1",
                "status": "PASS",
                "experiment_id": experiment_id,
                "receiver_label": label,
                "branch_id": branch_id,
                "branch_kind": row["kind"],
                "horizon": 20,
                "parameter_donor_scale": row["parameter_donor_scale"],
                "optimizer_donor_scale": row["optimizer_donor_scale"],
                "parameter_source_object_id": receiver["endpoints"][str(int(row["parameter_donor_scale"]))]["h20_source_object_id"],
                "optimizer_source_object_id": receiver["endpoints"][str(int(row["optimizer_donor_scale"]))]["h20_source_object_id"],
                "state": manifest,
                "state_summary": runtime.state_summary(states[branch_id]),
                "future_information_used": False,
            },
        )
        observation = _probe_at_state(
            runtime,
            store,
            branch_entry_root=entry_root,
            state=states[branch_id],
            state_manifest=manifest,
        )
        h20_observations[branch_id] = observation
        h20_rows[branch_id] = {
            "state_result_sha256": state_result["result_sha256"],
            "state_id": manifest["state_id"],
            "state_sha256": manifest["commitment"]["state_sha256"],
            "probe_result_sha256": observation["result_sha256"],
            "capability_accuracy": observation["capability_accuracy"],
        }

    current_states = {name: state.clone() for name, state in states.items()}
    chain = {name: state.commitment()["state_sha256"] for name, state in current_states.items()}
    continuation_rows: list[dict[str, Any]] = []
    horizon_rows: dict[str, Any] = {"20": h20_rows}
    observation_index: dict[int, dict[str, dict[str, Any]]] = {20: h20_observations}
    for horizon in EXCHANGE_CONTINUATION_HORIZONS:
        physical_step = int(receiver["base_optimizer_step"]) + horizon - 1
        batch = runtime.load_batch(physical_step)
        opportunity_seed = runtime.derive_seed(
            str(protocol["source_amplitude_path"]["protocol_sha256"]),
            str(receiver["entry_id"]),
            physical_step,
            horizon,
        )
        step_rows: dict[str, Any] = {}
        for branch_id in EXCHANGE_BRANCHES:
            runtime.restore(current_states[branch_id])
            from_sha256 = current_states[branch_id].commitment()["state_sha256"]
            evidence = runtime.train_actual_step(
                batch,
                execute_optimizer=True,
                seed=opportunity_seed,
            )
            current_states[branch_id] = runtime.snapshot()
            encoded = _encode_step(
                store,
                evidence,
                transition_id=f"{experiment_id}:{branch_id}:h{horizon:03d}",
            )
            step_result = _checked_result(
                entry_root
                / "continuations"
                / f"step-{physical_step:05d}-to-{physical_step + 1:05d}"
                / f"{branch_id}.json",
                {
                    "schema": "nanogpt-h20-optimizer-exchange-continuation-step-v1",
                    "status": "PASS",
                    "experiment_id": experiment_id,
                    "receiver_label": label,
                    "branch_id": branch_id,
                    "horizon": horizon,
                    "physical_optimizer_step": physical_step,
                    "from_state_sha256": from_sha256,
                    "to_state_sha256": current_states[branch_id].commitment()["state_sha256"],
                    "same_batch_all_branches": _encode_batch(batch),
                    "same_external_rng_opportunity_all_branches": opportunity_seed,
                    "step_evidence": encoded,
                    "post_state_summary": runtime.state_summary(current_states[branch_id]),
                    "future_information_used": False,
                },
            )
            chain[branch_id] = current_states[branch_id].commitment()["state_sha256"]
            step_rows[branch_id] = {
                "result_sha256": step_result["result_sha256"],
                "from_state_sha256": from_sha256,
                "to_state_sha256": chain[branch_id],
            }
        continuation_rows.append(
            {
                "horizon": horizon,
                "physical_optimizer_step": physical_step,
                "steps": step_rows,
            }
        )
        if horizon in (21, 100):
            registered: dict[str, Any] = {}
            registered_observations: dict[str, dict[str, Any]] = {}
            for branch_id in EXCHANGE_BRANCHES:
                row = branches[branch_id]
                manifest = _encode_state(
                    store,
                    current_states[branch_id],
                    entry_id=str(receiver["entry_id"]),
                    window_id=f"{experiment_id}:h{horizon}:{branch_id}",
                    optimizer_step=int(receiver["base_optimizer_step"]) + horizon,
                    protocol_sha256=main_protocol_sha256,
                )
                state_result = _checked_result(
                    entry_root / "horizons" / f"h-{horizon:03d}" / f"{branch_id}-state.json",
                    {
                        "schema": "nanogpt-h20-optimizer-exchange-state-v1",
                        "status": "PASS",
                        "experiment_id": experiment_id,
                        "receiver_label": label,
                        "branch_id": branch_id,
                        "branch_kind": row["kind"],
                        "horizon": horizon,
                        "parameter_donor_scale": row["parameter_donor_scale"],
                        "optimizer_donor_scale": row["optimizer_donor_scale"],
                        "state": manifest,
                        "state_summary": runtime.state_summary(current_states[branch_id]),
                        "future_information_used": False,
                    },
                )
                observation = _probe_at_state(
                    runtime,
                    store,
                    branch_entry_root=entry_root,
                    state=current_states[branch_id],
                    state_manifest=manifest,
                )
                registered_observations[branch_id] = observation
                registered[branch_id] = {
                    "state_result_sha256": state_result["result_sha256"],
                    "state_id": manifest["state_id"],
                    "state_sha256": manifest["commitment"]["state_sha256"],
                    "probe_result_sha256": observation["result_sha256"],
                    "capability_accuracy": observation["capability_accuracy"],
                }
            observation_index[horizon] = registered_observations
            horizon_rows[str(horizon)] = registered
        if horizon % 10 == 0:
            print(
                {
                    "event": "SST_OPTIMIZER_EXCHANGE_CONTINUATION_PROGRESS",
                    "receiver": label,
                    "horizon": horizon,
                },
                flush=True,
            )

    h100_arrays = {
        branch_id: _load_observation_arrays(entry_root, observation_index[100][branch_id])
        for branch_id in EXCHANGE_BRANCHES
    }
    h100_capability = {
        branch_id: float(observation_index[100][branch_id]["capability_accuracy"])
        for branch_id in EXCHANGE_BRANCHES
    }
    comparison = _comparison_distance(h100_arrays, h100_capability)
    comparison_result = _checked_result(
        entry_root / "h100_frozen_comparison.json",
        {
            "schema": "nanogpt-h20-optimizer-exchange-h100-comparison-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver_label": label,
            "comparison_contract": protocol["comparison_contract"],
            "control_capability_accuracy": {
                name: h100_capability[name] for name in ("theta0_O0", "theta1_O1")
            },
            "hybrid_comparisons": comparison,
            "weights_fitted": False,
            "thresholds_fitted": False,
            "scientific_interpretation_performed": False,
            "future_information_used_for_execution_or_selection": False,
        },
    )
    return _checked_result(
        entry_root / "optimizer_exchange_receiver_receipt.json",
        {
            "schema": "nanogpt-h20-optimizer-exchange-receiver-receipt-v1",
            "status": "PASS",
            "experiment_id": experiment_id,
            "receiver": receiver,
            "horizon_results": horizon_rows,
            "continuation_results": continuation_rows,
            "h100_comparison_result_sha256": comparison_result["result_sha256"],
            "future_information_used": False,
        },
    )


def execute_optimizer_exchange(
    *,
    amplitude_root: Path,
    amplitude_graph_root: Path,
    source_root: Path,
    trainer_root: Path,
    output_root: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    main_protocol_path: Path,
    optimizer_exchange_protocol_path: Path,
) -> dict[str, Any]:
    require(not output_root.exists(), "SST_OPTIMIZER_EXCHANGE_OUTPUT_ROOT_EXISTS")
    protocol = read_json(optimizer_exchange_protocol_path)
    require(protocol["schema"] == "nanogpt-h20-reciprocal-optimizer-exchange-protocol-v1", "SST_OPTIMIZER_EXCHANGE_PROTOCOL_SCHEMA_INVALID")
    require(int(protocol["exchange_horizon"]) == 20, "SST_OPTIMIZER_EXCHANGE_HORIZON_INVALID")
    require(tuple(int(value) for value in protocol["observation_horizons"]) == EXCHANGE_OBSERVATION_HORIZONS, "SST_OPTIMIZER_EXCHANGE_OBSERVATION_HORIZONS_INVALID")
    require(int(protocol["continuation_final_horizon"]) == 100, "SST_OPTIMIZER_EXCHANGE_FINAL_HORIZON_INVALID")
    require(tuple(protocol["optimizer_keys"]) == ("exp_avg", "exp_avg_sq", "step"), "SST_OPTIMIZER_EXCHANGE_OPTIMIZER_KEYS_INVALID")
    branches = _branch_contract(protocol)
    receivers = {str(row["label"]): row for row in protocol["receivers"]}
    require(tuple(receivers) == ("A", "B"), "SST_OPTIMIZER_EXCHANGE_RECEIVERS_INVALID")
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    require(probe_contract.probe_contract_id == protocol["probe_contract_id"], "SST_OPTIMIZER_EXCHANGE_PROBE_CONTRACT_MISMATCH")
    protocol_sha256 = file_sha256(optimizer_exchange_protocol_path)
    main_protocol_sha256 = file_sha256(main_protocol_path)
    source_states = _prior_amplitude_preconditions(
        protocol=protocol,
        amplitude_root=amplitude_root,
        amplitude_graph_root=amplitude_graph_root,
    )
    completed: list[dict[str, Any]] = []
    for label in ("A", "B"):
        receiver = receivers[label]
        store = TensorStore(output_root / f"receiver-{label}" / "tensor-objects")
        runtime = StepwiseTrainingRuntime(
            source_root / str(receiver["source_bundle_id"]),
            trainer_root,
            registry,
            probe_contract,
        )
        try:
            receipt = _execute_receiver(
                runtime=runtime,
                store=store,
                output_root=output_root,
                receiver=receiver,
                source_states=source_states[label],
                branches=branches,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                main_protocol_sha256=main_protocol_sha256,
            )
            completed.append(
                {"receiver_label": label, "receipt_sha256": receipt["result_sha256"]}
            )
            print(
                {"event": "SST_OPTIMIZER_EXCHANGE_RECEIVER_COMPLETE", "receiver": label},
                flush=True,
            )
        finally:
            runtime.close()
    return _checked_result(
        output_root / "optimizer_exchange_pair_receipt.json",
        {
            "schema": "nanogpt-h20-optimizer-exchange-pair-receipt-v1",
            "status": "PASS",
            "amplitude_root": str(amplitude_root.resolve()),
            "amplitude_graph_root": str(amplitude_graph_root.resolve()),
            "source_root": str(source_root.resolve()),
            "trainer_root": str(trainer_root.resolve()),
            "output_root": str(output_root.resolve()),
            "optimizer_exchange_protocol_sha256": protocol_sha256,
            "main_protocol_sha256": main_protocol_sha256,
            "component_registry_sha256": registry.source_sha256,
            "probe_contract_sha256": probe_contract.source_sha256,
            "source_amplitude_manifest_sha256": protocol["source_amplitude_path"]["graph_manifest_sha256"],
            "completed": completed,
            "receiver_count": 2,
            "branch_count": len(EXCHANGE_BRANCHES),
            "continuation_step_count_per_branch": len(EXCHANGE_CONTINUATION_HORIZONS),
            "observation_horizons": list(EXCHANGE_OBSERVATION_HORIZONS),
            "future_information_used": False,
            "scientific_interpretation_performed": False,
        },
    )


__all__ = [
    "EXCHANGE_BRANCHES",
    "EXCHANGE_CONTINUATION_HORIZONS",
    "EXCHANGE_OBSERVATION_HORIZONS",
    "compose_parameter_optimizer_state",
    "execute_optimizer_exchange",
]
