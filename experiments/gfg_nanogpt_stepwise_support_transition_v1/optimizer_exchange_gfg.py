from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import GraphRef, SupportGFGWriter

from .branch_gfg import _branch_batch_sources
from .contracts import ComponentRegistry, ProbeContract
from .execution import _read_checked
from .local_response_gfg import _prior_origin
from .optimizer_exchange import EXCHANGE_BRANCHES, EXCHANGE_CONTINUATION_HORIZONS, _branch_contract
from .reciprocal_gfg_validator import _main_object_index
from .stepwise_gfg import _bind_all, _emit_probe, _external_tensor, _object, _occurrence


GRAPH_SCHEMA = "nanogpt-h20-reciprocal-optimizer-exchange-gfg-v1"
BLOCK_SCHEMA = "nanogpt-h20-reciprocal-optimizer-exchange-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-h20-reciprocal-optimizer-exchange-gfg-manifest-v1"


def _tensor_refs(value: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if {"locator", "file_sha256", "raw_tensor_sha256", "shape", "dtype"} <= set(value):
            yield prefix, value
        else:
            for key, child in sorted(value.items()):
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                yield from _tensor_refs(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            yield from _tensor_refs(child, child_prefix)


def _stage_tensors(evidence_root: Path, graph_root: Path) -> int:
    target = graph_root / "tensor-objects"
    target.mkdir(parents=True, exist_ok=True)
    staged: set[str] = set()
    for label in ("A", "B"):
        for source in sorted((evidence_root / f"receiver-{label}" / "tensor-objects").glob("*.npy")):
            destination = target / source.name
            source_sha = file_sha256(source)
            if destination.exists():
                require(file_sha256(destination) == source_sha, "SST_OPTIMIZER_EXCHANGE_GFG_TENSOR_COLLISION")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            staged.add(source.name)
    return len(staged)


def _emit_training_step(
    writer: SupportGFGWriter,
    *,
    step: dict[str, Any],
    receiver_label: str,
    branch_id: str,
    prestate_ref: GraphRef,
    batch_sources: list[tuple[GraphRef, str]],
    contract_ref: GraphRef,
) -> GraphRef:
    physical_step = int(step["physical_optimizer_step"])
    occurrence = _occurrence(
        writer,
        occurrence_type="optimizer_exchange_native_training_step_occurrence",
        optimizer_step=physical_step,
        operation="native_training_step_after_h20_optimizer_exchange",
        contract_id="H20-RECIPROCAL-OPTIMIZER-STATE-EXCHANGE-v1",
        payload={
            "receiver_label": receiver_label,
            "branch_id": branch_id,
            "horizon": step["horizon"],
            "from_state_sha256": step["from_state_sha256"],
            "to_state_sha256": step["to_state_sha256"],
            "same_external_rng_opportunity_all_branches": step["same_external_rng_opportunity_all_branches"],
            "future_information_used": False,
        },
    )
    sources = [(prestate_ref, "complete_pretraining_state"), *batch_sources, (contract_ref, "frozen_optimizer_exchange_continuation_contract")]
    tensors: list[tuple[GraphRef, str]] = []
    for name, reference in _tensor_refs(step["step_evidence"]):
        outcome = _external_tensor(
            writer,
            reference=reference,
            semantic_key=f"optimizer-exchange:{receiver_label}:{branch_id}:step:{physical_step}:{name}",
            role=f"actual_training_step_{name.replace('/', '_')}",
            optimizer_step=physical_step,
        )
        tensors.append((outcome, name))
    _bind_all(writer, occurrence, sources, tensors)
    state = _object(
        writer,
        semantic_key=f"optimizer-exchange:{receiver_label}:{branch_id}:post-step:{physical_step + 1}:state",
        role="optimizer_exchange_post_training_state",
        optimizer_step=physical_step + 1,
        payload={
            "receiver_label": receiver_label,
            "branch_id": branch_id,
            "horizon": step["horizon"],
            "from_state_sha256": step["from_state_sha256"],
            "to_state_sha256": step["to_state_sha256"],
            "post_state_summary": step["post_state_summary"],
            "step_result_sha256": step["result_sha256"],
        },
        object_kind="training_state_transition_result",
    )
    writer.bind(occurrence, [*sources, *tensors], state, payload={"outcome_kind": "complete_post_training_state_identity"})
    return state


def build_optimizer_exchange_gfg(
    *,
    evidence_root: Path,
    amplitude_graph_root: Path,
    source_root: Path,
    graph_root: Path,
    optimizer_exchange_protocol_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
) -> dict[str, Any]:
    require(not graph_root.exists(), "SST_OPTIMIZER_EXCHANGE_GFG_ROOT_EXISTS")
    graph_root.mkdir(parents=True)
    protocol = read_json(optimizer_exchange_protocol_path)
    branches = _branch_contract(protocol)
    validation = read_json(evidence_root / "optimizer_exchange_validation.json")
    require(validation["status"] == "PASS", "SST_OPTIMIZER_EXCHANGE_GFG_EVIDENCE_NOT_VALIDATED")
    prior_manifest = read_json(amplitude_graph_root / "amplitude_path_gfg_manifest.json")
    require(prior_manifest["manifest_sha256"] == protocol["source_amplitude_path"]["graph_manifest_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_PRIOR_MANIFEST_MISMATCH")
    prior_objects = _main_object_index(amplitude_graph_root / str(prior_manifest["database"]))
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    source_manifests = {
        str(row["label"]): read_json(source_root / str(row["source_bundle_id"]) / "manifest.json")
        for row in protocol["receivers"]
    }
    staged_count = _stage_tensors(evidence_root, graph_root)
    writer = SupportGFGWriter(
        graph_root / "optimizer_exchange_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id="nanogpt-h20-reciprocal-optimizer-exchange-v1",
        source_bundle_id=str(prior_manifest["manifest_sha256"]),
        contract_sha256=file_sha256(optimizer_exchange_protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    writer.start_block("optimizer_exchange_contract", 0)
    contract_ref = _object(
        writer,
        semantic_key="optimizer-exchange:frozen-protocol",
        role="frozen_h20_reciprocal_optimizer_exchange_protocol",
        optimizer_step=0,
        payload={"protocol": protocol, "protocol_sha256": file_sha256(optimizer_exchange_protocol_path), "evidence_validation_sha256": validation["validation_sha256"], "prior_amplitude_manifest_sha256": prior_manifest["manifest_sha256"]},
        object_kind="frozen_contract",
    )
    registry_ref = _object(writer, semantic_key="optimizer-exchange:component-registry", role="versioned_component_registry", optimizer_step=0, payload={"registry": read_json(component_registry_path), "registry_sha256": registry.source_sha256}, object_kind="frozen_contract")
    probe_ref = _object(writer, semantic_key="optimizer-exchange:probe-contract", role="versioned_probe_contract", optimizer_step=0, payload={"probe_contract": read_json(probe_contract_path), "probe_contract_sha256": probe_contract.source_sha256}, object_kind="frozen_contract")
    writer.flush_block()

    state_catalog: dict[str, str] = {}
    continuation_catalog: dict[str, str] = {}
    probe_catalog: dict[str, str] = {}
    comparison_catalog: dict[str, str] = {}
    exactness_catalog: dict[str, str] = {}
    for label in ("A", "B"):
        receiver = next(row for row in protocol["receivers"] if row["label"] == label)
        entry_root = evidence_root / f"receiver-{label}"
        source_bundle_sha = str(source_manifests[label]["bundle_manifest_sha256"])
        h20_origins: dict[float, GraphRef] = {}
        h100_origins: dict[float, GraphRef] = {}
        for scale in (0.0, 1.0):
            endpoint = receiver["endpoints"][str(int(scale))]
            h20_origins[scale] = _prior_origin(writer, object_index=prior_objects, object_id=str(endpoint["h20_source_object_id"]), prior_manifest=prior_manifest)
            h100_origins[scale] = _prior_origin(writer, object_index=prior_objects, object_id=str(endpoint["h100_source_object_id"]), prior_manifest=prior_manifest)

        writer.start_block(f"optimizer_exchange_receiver_{label}_h20", int(receiver["h20_optimizer_step"]))
        current_refs: dict[str, GraphRef] = {}
        for branch_id in EXCHANGE_BRANCHES:
            branch = branches[branch_id]
            record = _read_checked(entry_root / "horizons" / "h-020" / f"{branch_id}-state.json", "nanogpt-h20-optimizer-exchange-state-v1")
            occurrence = _occurrence(
                writer,
                occurrence_type="h20_parameter_optimizer_state_composition_occurrence",
                optimizer_step=int(receiver["h20_optimizer_step"]),
                operation="compose_complete_parameter_map_with_complete_adam_state_map",
                contract_id=str(protocol["protocol_id"]),
                payload={"receiver_label": label, "branch_id": branch_id, "branch_kind": branch["kind"], "parameter_donor_scale": branch["parameter_donor_scale"], "optimizer_donor_scale": branch["optimizer_donor_scale"], "interpolation_used": False},
            )
            state = _object(
                writer,
                semantic_key=f"optimizer-exchange:{label}:h20:{branch_id}:state",
                role="h20_parameter_optimizer_composed_state",
                optimizer_step=int(receiver["h20_optimizer_step"]),
                payload={"state_record": record, "receiver_label": label, "branch_id": branch_id},
                object_kind="restorable_optimizer_exchange_state",
            )
            parameter_scale = float(branch["parameter_donor_scale"])
            optimizer_scale = float(branch["optimizer_donor_scale"])
            sources: list[tuple[GraphRef, str]]
            if parameter_scale == optimizer_scale:
                sources = [(h20_origins[parameter_scale], "complete_native_endpoint_parameter_and_optimizer_state")]
            else:
                sources = [(h20_origins[parameter_scale], "complete_parameter_map_donor"), (h20_origins[optimizer_scale], "complete_adam_state_map_donor")]
            sources.append((contract_ref, "frozen_state_composition_contract"))
            writer.bind(occurrence, sources, state, payload={"outcome_kind": "restorable_h20_composed_state"})
            current_refs[branch_id] = state
            state_catalog[f"{label}:20:{branch_id}"] = state.object_id
            observation = _read_checked(entry_root / "probe-observations" / probe_contract.probe_contract_id / f"{record['state']['state_id']}.json", "nanogpt-stepwise-probe-observation-v1")
            probe_result = _emit_probe(writer, observation=observation, state_origin=state, validation_sources=[], contract_ref=probe_ref, registry_ref=registry_ref, semantic_prefix=f"optimizer-exchange:{label}:h20:{branch_id}", optimizer_step=int(receiver["h20_optimizer_step"]), probe_contract=probe_contract)
            probe_catalog[f"{label}:20:{branch_id}"] = probe_result.object_id
        writer.flush_block()

        receiver_receipt = _read_checked(entry_root / "optimizer_exchange_receiver_receipt.json", "nanogpt-h20-optimizer-exchange-receiver-receipt-v1")
        for row in receiver_receipt["continuation_results"]:
            horizon = int(row["horizon"])
            physical_step = int(row["physical_optimizer_step"])
            writer.start_block(f"optimizer_exchange_receiver_{label}_continuation", physical_step)
            first = _read_checked(entry_root / "continuations" / f"step-{physical_step:05d}-to-{physical_step + 1:05d}" / f"{EXCHANGE_BRANCHES[0]}.json", "nanogpt-h20-optimizer-exchange-continuation-step-v1")
            batch_sources = _branch_batch_sources(writer, batch=first["same_batch_all_branches"], source_bundle_id=source_bundle_sha, branch_contract_ref=contract_ref, semantic_prefix=f"optimizer-exchange:{label}:continuation:{physical_step}", optimizer_step=physical_step)
            next_refs: dict[str, GraphRef] = {}
            for branch_id in EXCHANGE_BRANCHES:
                step = _read_checked(entry_root / "continuations" / f"step-{physical_step:05d}-to-{physical_step + 1:05d}" / f"{branch_id}.json", "nanogpt-h20-optimizer-exchange-continuation-step-v1")
                next_refs[branch_id] = _emit_training_step(writer, step=step, receiver_label=label, branch_id=branch_id, prestate_ref=current_refs[branch_id], batch_sources=batch_sources, contract_ref=contract_ref)
                continuation_catalog[f"{label}:{horizon}:{branch_id}"] = next_refs[branch_id].object_id
            current_refs = next_refs
            writer.flush_block()
            if horizon not in (21, 100):
                continue
            writer.start_block(f"optimizer_exchange_receiver_{label}_horizon", int(receiver["base_optimizer_step"]) + horizon)
            materialized: dict[str, GraphRef] = {}
            for branch_id in EXCHANGE_BRANCHES:
                record = _read_checked(entry_root / "horizons" / f"h-{horizon:03d}" / f"{branch_id}-state.json", "nanogpt-h20-optimizer-exchange-state-v1")
                occurrence = _occurrence(writer, occurrence_type="optimizer_exchange_horizon_state_materialization_occurrence", optimizer_step=int(receiver["base_optimizer_step"]) + horizon, operation="materialize_registered_optimizer_exchange_horizon_state", contract_id=str(protocol["protocol_id"]), payload={"receiver_label": label, "branch_id": branch_id, "horizon": horizon})
                state = _object(writer, semantic_key=f"optimizer-exchange:{label}:h{horizon}:{branch_id}:restorable-state", role="optimizer_exchange_registered_horizon_state", optimizer_step=int(receiver["base_optimizer_step"]) + horizon, payload={"state_record": record, "receiver_label": label, "branch_id": branch_id, "horizon": horizon}, object_kind="restorable_optimizer_exchange_state")
                writer.bind(occurrence, [(current_refs[branch_id], "actual_continuation_state"), (contract_ref, "registered_horizon_contract")], state, payload={"outcome_kind": "restorable_optimizer_exchange_horizon_state"})
                materialized[branch_id] = state
                state_catalog[f"{label}:{horizon}:{branch_id}"] = state.object_id
                observation = _read_checked(entry_root / "probe-observations" / probe_contract.probe_contract_id / f"{record['state']['state_id']}.json", "nanogpt-stepwise-probe-observation-v1")
                probe_result = _emit_probe(writer, observation=observation, state_origin=state, validation_sources=[], contract_ref=probe_ref, registry_ref=registry_ref, semantic_prefix=f"optimizer-exchange:{label}:h{horizon}:{branch_id}", optimizer_step=int(receiver["base_optimizer_step"]) + horizon, probe_contract=probe_contract)
                probe_catalog[f"{label}:{horizon}:{branch_id}"] = probe_result.object_id
            current_refs = materialized
            writer.flush_block()

        writer.start_block(f"optimizer_exchange_receiver_{label}_h100_validation", int(receiver["base_optimizer_step"]) + 100)
        for branch_id, scale in (("theta0_O0", 0.0), ("theta1_O1", 1.0)):
            occurrence = _occurrence(writer, occurrence_type="native_control_exactness_validation_occurrence", optimizer_step=int(receiver["base_optimizer_step"]) + 100, operation="validate_reexecuted_native_control_against_prior_amplitude_endpoint", contract_id=str(protocol["protocol_id"]), payload={"receiver_label": label, "branch_id": branch_id, "horizon": 100})
            result = _object(writer, semantic_key=f"optimizer-exchange:{label}:h100:{branch_id}:native-control-exactness", role="validated_native_control_exactness", optimizer_step=int(receiver["base_optimizer_step"]) + 100, payload={"receiver_label": label, "branch_id": branch_id, "state_byte_exact": validation["receiver_rows"][label]["native_control_h100_state_byte_exact"][branch_id], "probe_byte_exact": validation["receiver_rows"][label]["native_control_h100_probe_byte_exact"][branch_id], "evidence_validation_sha256": validation["validation_sha256"]}, object_kind="validated_analysis_result")
            writer.bind(occurrence, [(current_refs[branch_id], "reexecuted_native_control_state"), (h100_origins[scale], "prior_amplitude_endpoint_state"), (contract_ref, "frozen_native_control_exactness_contract")], result, payload={"outcome_kind": "validated_native_control_exactness"})
            exactness_catalog[f"{label}:{branch_id}"] = result.object_id
        comparison = _read_checked(entry_root / "h100_frozen_comparison.json", "nanogpt-h20-optimizer-exchange-h100-comparison-v1")
        occurrence = _occurrence(writer, occurrence_type="frozen_h100_optimizer_exchange_comparison_occurrence", optimizer_step=int(receiver["base_optimizer_step"]) + 100, operation="compute_frozen_unweighted_csrg_and_capability_distances", contract_id=str(protocol["protocol_id"]), payload={"receiver_label": label, "weights_fitted": False, "thresholds_fitted": False, "scientific_interpretation_performed": False})
        result = _object(writer, semantic_key=f"optimizer-exchange:{label}:h100:frozen-comparison", role="validated_h100_optimizer_exchange_comparison", optimizer_step=int(receiver["base_optimizer_step"]) + 100, payload={"comparison": comparison, "validation_receiver_row": validation["receiver_rows"][label]}, object_kind="validated_analysis_result")
        writer.bind(occurrence, [*[(current_refs[branch_id], f"h100_state_for_{branch_id}") for branch_id in EXCHANGE_BRANCHES], (contract_ref, "frozen_h100_comparison_contract")], result, payload={"outcome_kind": "validated_h100_comparison"})
        comparison_catalog[label] = result.object_id
        writer.flush_block()

    closed = writer.close()
    material = {
        **closed,
        "status": "PASS",
        "optimizer_exchange_protocol_sha256": file_sha256(optimizer_exchange_protocol_path),
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "evidence_validation_sha256": validation["validation_sha256"],
        "prior_amplitude_manifest_sha256": prior_manifest["manifest_sha256"],
        "source_bundle_manifest_sha256": {label: source_manifests[label]["bundle_manifest_sha256"] for label in ("A", "B")},
        "staged_tensor_payload_count": staged_count,
        "state_catalog": state_catalog,
        "continuation_catalog": continuation_catalog,
        "probe_catalog": probe_catalog,
        "native_control_exactness_catalog": exactness_catalog,
        "comparison_catalog": comparison_catalog,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    manifest = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(graph_root / "optimizer_exchange_gfg_manifest.json", manifest)
    return manifest


__all__ = ["build_optimizer_exchange_gfg", "GRAPH_SCHEMA"]
