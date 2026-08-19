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
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import (
    GraphRef,
    SupportGFGWriter,
)

from .amplitude_path import AMPLITUDE_HORIZONS, AMPLITUDE_SCALES, RESPONSE_CENTERS, scale_key
from .branch_gfg import _branch_batch_sources
from .contracts import ComponentRegistry, ProbeContract
from .execution import _read_checked
from .local_response_gfg import _prior_origin
from .reciprocal_gfg_validator import _main_object_index
from .stepwise_gfg import _bind_all, _emit_probe, _external_tensor, _object, _occurrence


GRAPH_SCHEMA = "nanogpt-b-update-amplitude-path-gfg-v1"
BLOCK_SCHEMA = "nanogpt-b-update-amplitude-path-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-b-update-amplitude-path-gfg-manifest-v1"


def _stage_tensors(evidence_root: Path, graph_root: Path) -> int:
    target = graph_root / "tensor-objects"
    target.mkdir(parents=True, exist_ok=True)
    staged: set[str] = set()
    for label in ("A", "B"):
        for source in sorted((evidence_root / f"receiver-{label}" / "tensor-objects").glob("*.npy")):
            destination = target / source.name
            source_sha = file_sha256(source)
            if destination.exists():
                require(file_sha256(destination) == source_sha, "SST_AMPLITUDE_PATH_GFG_TENSOR_COLLISION")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            staged.add(source.name)
    return len(staged)


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


def _emit_step(
    writer: SupportGFGWriter,
    *,
    step: dict[str, Any],
    receiver_label: str,
    scale: float,
    prestate_ref: GraphRef,
    batch_sources: list[tuple[GraphRef, str]],
    contract_ref: GraphRef,
) -> GraphRef:
    physical_step = int(step["physical_optimizer_step"])
    key = scale_key(scale)
    occurrence = _occurrence(
        writer,
        occurrence_type="amplitude_path_native_training_step_occurrence",
        optimizer_step=physical_step,
        operation="native_training_step",
        contract_id="B-UPDATE-AMPLITUDE-PATH-CONTINUATION-v1",
        payload={
            "receiver_label": receiver_label,
            "scale": scale,
            "horizon": step["horizon"],
            "execute_optimizer": True,
            "from_state_sha256": step["from_state_sha256"],
            "to_state_sha256": step["to_state_sha256"],
            "step_evidence_sha256": step["step_evidence"]["step_evidence_sha256"],
        },
    )
    sources = [
        (prestate_ref, "complete_pretraining_state"),
        *batch_sources,
        (contract_ref, "frozen_amplitude_path_continuation_contract"),
    ]
    tensors: list[tuple[GraphRef, str]] = []
    for name, reference in _tensor_refs(step["step_evidence"]):
        outcome = _external_tensor(
            writer,
            reference=reference,
            semantic_key=(
                f"amplitude-path:{receiver_label}:{key}:"
                f"step:{physical_step}:{name}"
            ),
            role=f"actual_training_step_{name.replace('/', '_')}",
            optimizer_step=physical_step,
        )
        tensors.append((outcome, name))
    _bind_all(writer, occurrence, sources, tensors)
    state = _object(
        writer,
        semantic_key=(
            f"amplitude-path:{receiver_label}:{key}:"
            f"post-step:{physical_step + 1}:state"
        ),
        role="amplitude_path_post_training_state",
        optimizer_step=physical_step + 1,
        payload={
            "receiver_label": receiver_label,
            "scale": scale,
            "horizon": step["horizon"],
            "from_state_sha256": step["from_state_sha256"],
            "to_state_sha256": step["to_state_sha256"],
            "post_state_summary": step["post_state_summary"],
            "step_result_sha256": step["result_sha256"],
        },
        object_kind="training_state_transition_result",
    )
    writer.bind(
        occurrence,
        [*sources, *[(ref, role) for ref, role in tensors]],
        state,
        payload={"outcome_kind": "complete_post_training_state_identity"},
    )
    return state


def build_amplitude_path_gfg(
    *,
    evidence_root: Path,
    reciprocal_graph_root: Path,
    source_root: Path,
    graph_root: Path,
    amplitude_path_protocol_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
) -> dict[str, Any]:
    require(not graph_root.exists(), "SST_AMPLITUDE_PATH_GFG_ROOT_EXISTS")
    graph_root.mkdir(parents=True)
    protocol = read_json(amplitude_path_protocol_path)
    validation = read_json(evidence_root / "amplitude_path_validation.json")
    require(validation["status"] == "PASS", "SST_AMPLITUDE_PATH_GFG_EVIDENCE_NOT_VALIDATED")
    prior_manifest = read_json(reciprocal_graph_root / "reciprocal_matched_pair_gfg_manifest.json")
    prior_objects = _main_object_index(reciprocal_graph_root / str(prior_manifest["database"]))
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    source_manifests = {
        str(row["label"]): read_json(source_root / str(row["source_bundle_id"]) / "manifest.json")
        for row in protocol["receivers"]
    }
    staged_count = _stage_tensors(evidence_root, graph_root)
    writer = SupportGFGWriter(
        graph_root / "amplitude_path_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id="nanogpt-b-update-amplitude-path-v1",
        source_bundle_id=str(prior_manifest["manifest_sha256"]),
        contract_sha256=file_sha256(amplitude_path_protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    writer.start_block("amplitude_path_contract", 0)
    contract_ref = _object(
        writer,
        semantic_key="amplitude-path:frozen-protocol",
        role="frozen_b_update_amplitude_path_protocol",
        optimizer_step=0,
        payload={
            "protocol": protocol,
            "protocol_sha256": file_sha256(amplitude_path_protocol_path),
            "evidence_validation_sha256": validation["validation_sha256"],
            "prior_reciprocal_manifest_sha256": prior_manifest["manifest_sha256"],
        },
        object_kind="frozen_contract",
    )
    registry_ref = _object(
        writer,
        semantic_key="amplitude-path:component-registry",
        role="versioned_component_registry",
        optimizer_step=0,
        payload={"registry": read_json(component_registry_path), "registry_sha256": registry.source_sha256},
        object_kind="frozen_contract",
    )
    probe_ref = _object(
        writer,
        semantic_key="amplitude-path:probe-contract",
        role="versioned_probe_contract",
        optimizer_step=0,
        payload={"probe_contract": read_json(probe_contract_path), "probe_contract_sha256": probe_contract.source_sha256},
        object_kind="frozen_contract",
    )
    writer.flush_block()
    donor_update = _prior_origin(
        writer,
        object_index=prior_objects,
        object_id=str(protocol["donor_update"]["source_object_id"]),
        prior_manifest=prior_manifest,
    )
    state_catalog: dict[str, str] = {}
    response_catalog: dict[str, str] = {}
    continuation_catalog: dict[str, str] = {}
    for label in ("A", "B"):
        endpoint = next(row for row in protocol["receivers"] if row["label"] == label)
        entry_root = evidence_root / f"receiver-{label}"
        optimizer_step = int(endpoint["optimizer_step"])
        source_bundle_id = str(source_manifests[label]["bundle_manifest_sha256"])
        prior_state = _prior_origin(
            writer,
            object_index=prior_objects,
            object_id=str(endpoint["skip_state_source_object_id"]),
            prior_manifest=prior_manifest,
        )
        writer.start_block(f"amplitude_path_receiver_{label}_h1", optimizer_step + 1)
        state_refs: dict[float, GraphRef] = {}
        probe_refs: dict[float, GraphRef] = {}
        for scale in AMPLITUDE_SCALES:
            key = scale_key(scale)
            state_record = _read_checked(
                entry_root / "h-001-path" / f"{key}-state.json",
                "nanogpt-amplitude-path-state-v1",
            )
            occurrence = _occurrence(
                writer,
                occurrence_type="amplitude_path_parameter_displacement_occurrence",
                optimizer_step=optimizer_step + 1,
                operation="apply_scaled_realized_parameter_delta",
                contract_id=str(protocol["protocol_id"]),
                payload={
                    "receiver_label": label,
                    "scale": scale,
                    "adam_state_transplanted": False,
                    "receiver_optimizer_held_at_skip_state": True,
                },
            )
            state = _object(
                writer,
                semantic_key=f"amplitude-path:{label}:h1:{key}:state",
                role="amplitude_path_restorable_state",
                optimizer_step=optimizer_step + 1,
                payload={"state_record": state_record, "receiver_label": label, "scale": scale},
                object_kind="restorable_amplitude_path_state",
            )
            sources = [
                (prior_state, "immutable_receiver_skip_state"),
                (contract_ref, "frozen_amplitude_path_contract"),
            ]
            if scale != 0.0:
                sources.append((donor_update, "scaled_exact_B_parameter_update"))
            writer.bind(occurrence, sources, state, payload={"outcome_kind": "restorable_amplitude_path_state"})
            state_refs[scale] = state
            state_catalog[f"{label}:1:{key}"] = state.object_id
            observation = _read_checked(
                entry_root / "probe-observations" / probe_contract.probe_contract_id / f"{state_record['state']['state_id']}.json",
                "nanogpt-stepwise-probe-observation-v1",
            )
            probe_refs[scale] = _emit_probe(
                writer,
                observation=observation,
                state_origin=state,
                validation_sources=[],
                contract_ref=probe_ref,
                registry_ref=registry_ref,
                semantic_prefix=f"amplitude-path:{label}:h1:{key}",
                optimizer_step=optimizer_step + 1,
                probe_contract=probe_contract,
            )
        response = _read_checked(entry_root / "h1_response_path.json", "nanogpt-amplitude-response-path-v1")
        center_refs: dict[float, GraphRef] = {}
        for center in RESPONSE_CENTERS:
            center_key = scale_key(center)
            occurrence = _occurrence(
                writer,
                occurrence_type="amplitude_path_central_response_occurrence",
                optimizer_step=optimizer_step + 1,
                operation="compute_registered_central_finite_difference_responses",
                contract_id=str(protocol["protocol_id"]),
                payload={"receiver_label": label, "center_scale": center, "epsilon": protocol["epsilon"]},
            )
            sources = [
                (probe_refs[center - 0.125], "minus_epsilon_probe_result"),
                (probe_refs[center], "center_probe_result"),
                (probe_refs[center + 0.125], "plus_epsilon_probe_result"),
                (contract_ref, "frozen_central_difference_contract"),
            ]
            outputs: list[tuple[GraphRef, str]] = []
            for role, encoded in sorted(response["numeric_responses"].items()):
                center_row = encoded["centers"][center_key]
                for field in ("j_first_order", "k_curvature"):
                    output = _external_tensor(
                        writer,
                        reference=center_row[field],
                        semantic_key=f"amplitude-path:{label}:h1:{center_key}:{role}:{field}",
                        role=f"{field}_of:{role}",
                        optimizer_step=optimizer_step + 1,
                    )
                    outputs.append((output, f"{field}:{role}"))
            _bind_all(writer, occurrence, sources, outputs)
            summary = _object(
                writer,
                semantic_key=f"amplitude-path:{label}:h1:{center_key}:response-summary",
                role="amplitude_path_center_response_summary",
                optimizer_step=optimizer_step + 1,
                payload={"receiver_label": label, "center_scale": center, "numeric_role_count": len(response["numeric_responses"])},
                object_kind="validated_analysis_result",
            )
            writer.bind(occurrence, [*sources, *outputs], summary, payload={"outcome_kind": "center_response_summary"})
            center_refs[center] = summary
        path_occurrence = _occurrence(
            writer,
            occurrence_type="fixed_simpson_response_path_occurrence",
            optimizer_step=optimizer_step + 1,
            operation="compute_fixed_five_center_composite_simpson_response_path",
            contract_id=str(protocol["protocol_id"]),
            payload={"receiver_label": label, "simpson_contract": protocol["simpson_contract"], "categorical_values_subtracted": False},
        )
        path_sources = [
            *[(center_refs[center], f"center_response_at_scale_{center}") for center in RESPONSE_CENTERS],
            (contract_ref, "frozen_simpson_path_contract"),
        ]
        path_outputs: list[tuple[GraphRef, str]] = []
        for role, encoded in sorted(response["numeric_responses"].items()):
            for field in (
                "simpson_delta_prediction",
                "exact_scale_zero_to_one_delta",
                "start_endpoint_j_prediction",
                "start_endpoint_jk_prediction",
                "end_endpoint_j_prediction",
                "end_endpoint_jk_prediction",
            ):
                output = _external_tensor(
                    writer,
                    reference=encoded[field],
                    semantic_key=f"amplitude-path:{label}:h1:{role}:{field}",
                    role=f"{field}_of:{role}",
                    optimizer_step=optimizer_step + 1,
                )
                path_outputs.append((output, f"{field}:{role}"))
        for role, encoded in sorted(response["categorical_paths"].items()):
            endpoint_output = _external_tensor(
                writer,
                reference=encoded["endpoint_changed_mask"],
                semantic_key=f"amplitude-path:{label}:h1:{role}:endpoint-categorical-mask",
                role=f"endpoint_categorical_transition_mask_of:{role}",
                optimizer_step=optimizer_step + 1,
            )
            path_outputs.append((endpoint_output, f"categorical_endpoint_mask:{role}"))
            for transition_key, transition in sorted(encoded["adjacent_transitions"].items()):
                output = _external_tensor(
                    writer,
                    reference=transition["changed_mask"],
                    semantic_key=f"amplitude-path:{label}:h1:{role}:{transition_key}:categorical-mask",
                    role=f"adjacent_categorical_transition_mask_of:{role}",
                    optimizer_step=optimizer_step + 1,
                )
                path_outputs.append((output, f"categorical_path_mask:{role}:{transition_key}"))
        _bind_all(writer, path_occurrence, path_sources, path_outputs)
        response_summary = _object(
            writer,
            semantic_key=f"amplitude-path:{label}:h1:validated-response-path-summary",
            role="validated_amplitude_response_path_summary",
            optimizer_step=optimizer_step + 1,
            payload={
                "receiver_label": label,
                "response_result_sha256": response["result_sha256"],
                "validation": validation["receiver_rows"][label],
                "computed_before_continuation": True,
                "scientific_interpretation_performed": False,
            },
            object_kind="validated_analysis_result",
        )
        writer.bind(path_occurrence, [*path_sources, *path_outputs], response_summary, payload={"outcome_kind": "validated_amplitude_response_path_summary"})
        response_catalog[label] = response_summary.object_id
        writer.flush_block()

        current_refs = {scale: state_refs[scale] for scale in RESPONSE_CENTERS}
        receiver_receipt = _read_checked(
            entry_root / "amplitude_path_receiver_receipt.json",
            "nanogpt-amplitude-path-receiver-receipt-v1",
        )
        horizon_rows = {int(row["horizon"]): row for row in receiver_receipt["horizon_results"]}
        for row in receiver_receipt["continuation_results"]:
            physical_step = int(row["physical_optimizer_step"])
            writer.start_block(f"amplitude_path_receiver_{label}_continuation", physical_step)
            first_key = scale_key(RESPONSE_CENTERS[0])
            first_step = _read_checked(
                entry_root / "continuations" / f"step-{physical_step:05d}-to-{physical_step + 1:05d}" / f"{first_key}.json",
                "nanogpt-amplitude-path-continuation-step-v1",
            )
            batch_sources = _branch_batch_sources(
                writer,
                batch=first_step["same_batch_all_scales"],
                source_bundle_id=source_bundle_id,
                branch_contract_ref=contract_ref,
                semantic_prefix=f"amplitude-path:{label}:continuation:{physical_step}",
                optimizer_step=physical_step,
            )
            next_refs: dict[float, GraphRef] = {}
            for scale in RESPONSE_CENTERS:
                key = scale_key(scale)
                step = _read_checked(
                    entry_root / "continuations" / f"step-{physical_step:05d}-to-{physical_step + 1:05d}" / f"{key}.json",
                    "nanogpt-amplitude-path-continuation-step-v1",
                )
                next_refs[scale] = _emit_step(
                    writer,
                    step=step,
                    receiver_label=label,
                    scale=scale,
                    prestate_ref=current_refs[scale],
                    batch_sources=batch_sources,
                    contract_ref=contract_ref,
                )
                continuation_catalog[f"{label}:{int(row['horizon'])}:{key}"] = next_refs[scale].object_id
            current_refs = next_refs
            writer.flush_block()
            horizon = int(row["horizon"])
            if horizon not in AMPLITUDE_HORIZONS[1:]:
                continue
            writer.start_block(f"amplitude_path_receiver_{label}_horizon", optimizer_step + horizon)
            horizon_state_refs: dict[float, GraphRef] = {}
            for scale in RESPONSE_CENTERS:
                key = scale_key(scale)
                state_record = _read_checked(
                    entry_root / "horizons" / f"h-{horizon:03d}" / f"{key}-state.json",
                    "nanogpt-amplitude-path-state-v1",
                )
                occurrence = _occurrence(
                    writer,
                    occurrence_type="amplitude_path_horizon_state_materialization_occurrence",
                    optimizer_step=optimizer_step + horizon,
                    operation="materialize_registered_amplitude_path_horizon_state",
                    contract_id=str(protocol["protocol_id"]),
                    payload={"receiver_label": label, "scale": scale, "horizon": horizon},
                )
                state = _object(
                    writer,
                    semantic_key=f"amplitude-path:{label}:h{horizon}:{key}:restorable-state",
                    role="amplitude_path_registered_horizon_state",
                    optimizer_step=optimizer_step + horizon,
                    payload={"state_record": state_record, "receiver_label": label, "scale": scale, "horizon": horizon},
                    object_kind="restorable_amplitude_path_state",
                )
                writer.bind(
                    occurrence,
                    [(current_refs[scale], "actual_continuation_state"), (contract_ref, "registered_horizon_contract")],
                    state,
                    payload={"outcome_kind": "restorable_amplitude_path_horizon_state"},
                )
                horizon_state_refs[scale] = state
                state_catalog[f"{label}:{horizon}:{key}"] = state.object_id
                observation = _read_checked(
                    entry_root / "probe-observations" / probe_contract.probe_contract_id / f"{state_record['state']['state_id']}.json",
                    "nanogpt-stepwise-probe-observation-v1",
                )
                _emit_probe(
                    writer,
                    observation=observation,
                    state_origin=state,
                    validation_sources=[],
                    contract_ref=probe_ref,
                    registry_ref=registry_ref,
                    semantic_prefix=f"amplitude-path:{label}:h{horizon}:{key}",
                    optimizer_step=optimizer_step + horizon,
                    probe_contract=probe_contract,
                )
            current_refs = horizon_state_refs
            writer.flush_block()
    closed = writer.close()
    material = {
        **closed,
        "status": "PASS",
        "amplitude_path_protocol_sha256": file_sha256(amplitude_path_protocol_path),
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "evidence_validation_sha256": validation["validation_sha256"],
        "prior_reciprocal_manifest_sha256": prior_manifest["manifest_sha256"],
        "source_bundle_manifest_sha256": {
            label: source_manifests[label]["bundle_manifest_sha256"]
            for label in ("A", "B")
        },
        "staged_tensor_payload_count": staged_count,
        "state_catalog": state_catalog,
        "response_summary_catalog": response_catalog,
        "continuation_catalog": continuation_catalog,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    manifest = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(graph_root / "amplitude_path_gfg_manifest.json", manifest)
    return manifest
