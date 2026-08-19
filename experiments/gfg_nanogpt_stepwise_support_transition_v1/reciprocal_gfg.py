from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import TrainingGFG
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import GraphRef, SupportGFGWriter

from .branch_gfg import _main_object_index, _main_state_origin, _tensor_refs
from .contracts import ComponentRegistry, ProbeContract
from .reciprocal import RECIPROCAL_BRANCHES
from .stepwise_gfg import _bind_all, _emit_probe, _external_tensor, _object, _occurrence, _source_row, _validation_probe_sources


GRAPH_SCHEMA = "nanogpt-reciprocal-matched-pair-gfg-v1"
BLOCK_SCHEMA = "nanogpt-reciprocal-matched-pair-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-reciprocal-matched-pair-gfg-manifest-v1"


def _stage_tensors(evidence_root: Path, graph_root: Path) -> int:
    target = graph_root / "tensor-objects"
    target.mkdir(parents=True, exist_ok=True)
    staged: set[str] = set()
    for label in ("A", "B"):
        source_root = evidence_root / f"recipient-{label}"
        # Probe records contain tensor references inside nested component/result
        # structures that are intentionally not flattened by ``_tensor_refs``.
        # Stage the already independently validated content-addressed payload
        # store itself so every reference admitted by the GFG compiler remains
        # resolvable.  Equal names must denote byte-identical payloads.
        for source in sorted((source_root / "tensor-objects").glob("*.npy")):
            name = source.name
            destination = target / name
            source_sha256 = file_sha256(source)
            if destination.exists():
                require(file_sha256(destination) == source_sha256, "SST_RECIPROCAL_GFG_TENSOR_COLLISION")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            staged.add(name)
    return len(staged)


def _batch_sources(
    writer: SupportGFGWriter,
    *,
    batch: dict[str, Any],
    source_bundle_id: str,
    contract_ref: GraphRef,
    semantic_prefix: str,
    optimizer_step: int,
) -> list[tuple[GraphRef, str]]:
    sources = [
        (writer.origin(_source_row(reference), source_bundle_id=source_bundle_id), role)
        for role, reference in sorted(batch["source_training_gfg_objects"].items())
    ]
    availability = batch.get("batch_selection_order_availability")
    if isinstance(availability, dict) and availability.get("outcome_kind") == "ExplicitDisposition":
        occurrence = _occurrence(
            writer,
            occurrence_type="batch_selection_order_availability_occurrence",
            optimizer_step=optimizer_step,
            operation="adjudicate_source_batch_selection_order_availability",
            contract_id="RECIPROCAL-MATCHED-PAIR-v2",
            payload={"semantic_prefix": semantic_prefix, "reconstruction_or_guess_used": False},
        )
        disposition = _object(
            writer,
            semantic_key=f"{semantic_prefix}:batch-selection-order-disposition",
            role="explicit_disposition",
            optimizer_step=optimizer_step,
            payload=availability,
            object_kind="ExplicitDisposition",
        )
        writer.bind(occurrence, [*sources, (contract_ref, "frozen_batch_identity_contract")], disposition, payload={"outcome_kind": "ExplicitDisposition"})
    return sources


def _native_step(
    writer: SupportGFGWriter,
    *,
    label: str,
    seed: dict[str, Any],
    prestate: GraphRef,
    batch_sources: list[tuple[GraphRef, str]],
    contract_ref: GraphRef,
) -> GraphRef:
    step = int(seed["recipient"]["optimizer_step"])
    evidence = seed["recipient_native_full_step"]
    occurrence = _occurrence(
        writer,
        occurrence_type="actual_native_training_step_occurrence",
        optimizer_step=step,
        operation="native_nanogpt_adamw_training_step",
        contract_id="RECIPROCAL-MATCHED-PAIR-v2",
        payload={
            "recipient_label": label,
            "seed_id": seed["seed_id"],
            "loss": evidence["loss"],
            "total_gradient_norm": evidence["total_gradient_norm"],
            "execute_optimizer": evidence["execute_optimizer"],
        },
    )
    sources = [(prestate, "actual_recipient_prestate"), *batch_sources, (contract_ref, "frozen_reciprocal_contract")]
    outcomes: list[tuple[GraphRef, str]] = []
    for name, reference in _tensor_refs(evidence):
        outcome = _external_tensor(
            writer,
            reference=reference,
            semantic_key=f"reciprocal:{label}:native-step:{step}:{name}",
            role=f"native_training_step_result:{name}",
            optimizer_step=step,
        )
        outcomes.append((outcome, name))
    _bind_all(writer, occurrence, sources, outcomes)
    summary = _object(
        writer,
        semantic_key=f"reciprocal:{label}:native-step:{step}:summary",
        role="actual_native_training_update_summary",
        optimizer_step=step,
        payload={
            "recipient_label": label,
            "seed_id": seed["seed_id"],
            "optimizer_step": step,
            "parameter_update": evidence["parameter_update"],
            "optimizer_deltas": evidence["optimizer_deltas"],
            "optimizer_config": evidence["optimizer_config"],
        },
    )
    writer.bind(occurrence, [*sources, *outcomes], summary, payload={"outcome_kind": "actual_native_training_update"})
    return summary


def _branch_seed(
    writer: SupportGFGWriter,
    *,
    label: str,
    donor_label: str,
    branch: str,
    optimizer_step: int,
    recipient_prestate: GraphRef,
    recipient_native: GraphRef,
    donor_native: GraphRef,
    contract_ref: GraphRef,
) -> GraphRef:
    source_plan: dict[str, list[tuple[GraphRef, str]]] = {
        "skip": [(recipient_prestate, "unchanged_recipient_prestate")],
        "native_full": [(recipient_native, "recipient_native_parameter_and_optimizer_update")],
        "native_parameter_only": [(recipient_prestate, "recipient_optimizer_prestate"), (recipient_native, "recipient_native_parameter_delta")],
        "native_optimizer_only": [(recipient_prestate, "recipient_parameter_prestate"), (recipient_native, "recipient_native_optimizer_update")],
        "donor_parameter_delta": [(recipient_prestate, "recipient_prestate"), (donor_native, "donor_native_parameter_delta")],
        "donor_optimizer_innovation": [(recipient_prestate, "recipient_adam_memory_prestate"), (donor_native, "donor_realized_adam_innovation")],
        "donor_joint_update": [(recipient_prestate, "recipient_prestate"), (donor_native, "donor_parameter_delta_and_adam_innovation")],
    }
    occurrence = _occurrence(
        writer,
        occurrence_type="reciprocal_branch_state_establishment_occurrence",
        optimizer_step=optimizer_step,
        operation="establish_reciprocal_counterfactual_branch_state",
        contract_id="RECIPROCAL-MATCHED-PAIR-v2",
        payload={"recipient_label": label, "donor_label": donor_label, "branch": branch},
    )
    outcome = _object(
        writer,
        semantic_key=f"reciprocal:{label}:seed-branch:{branch}",
        role=f"{branch}_established_branch_state",
        optimizer_step=optimizer_step + 1,
        payload={
            "recipient_label": label,
            "donor_label": donor_label,
            "branch": branch,
            "native_training_occurrence_claimed": branch == "native_full",
            "counterfactual_composition_claimed": branch not in {"skip", "native_full"},
        },
        object_kind="reciprocal_branch_state",
    )
    writer.bind(occurrence, [*source_plan[branch], (contract_ref, "frozen_branch_composition_contract")], outcome, payload={"outcome_kind": "established_reciprocal_branch_state"})
    return outcome


def _horizon_state(
    writer: SupportGFGWriter,
    *,
    label: str,
    branch: str,
    horizon: int,
    record: dict[str, Any],
    prior: GraphRef,
    contract_ref: GraphRef,
) -> GraphRef:
    step = int(record["physical_optimizer_opportunity"])
    occurrence = _occurrence(
        writer,
        occurrence_type="reciprocal_branch_horizon_state_materialization_occurrence",
        optimizer_step=step,
        operation="materialize_reciprocal_branch_restorable_state",
        contract_id="RECIPROCAL-MATCHED-PAIR-v2",
        payload={"recipient_label": label, "branch": branch, "horizon": horizon},
    )
    outcome = _object(
        writer,
        semantic_key=f"reciprocal:{label}:horizon:{horizon}:{branch}:complete-state",
        role=f"{branch}_restorable_state_at_horizon",
        optimizer_step=step,
        payload={"recipient_label": label, "branch": branch, "horizon": horizon, "state": record["state"], "state_summary": record["state_summary"], "state_result_sha256": record["result_sha256"]},
        object_kind="restorable_branch_state",
    )
    writer.bind(occurrence, [(prior, "actual_branch_state_chain"), (contract_ref, "frozen_horizon_contract")], outcome, payload={"outcome_kind": "restorable_branch_state"})
    return outcome


def _effect_block(
    writer: SupportGFGWriter,
    *,
    label: str,
    horizon: int,
    effect: dict[str, Any],
    probes: dict[str, GraphRef],
    contract_ref: GraphRef,
    optimizer_step: int,
) -> GraphRef:
    occurrence = _occurrence(
        writer,
        occurrence_type="reciprocal_branch_response_contrast_occurrence",
        optimizer_step=optimizer_step,
        operation="compute_branch_minus_recipient_skip_responses",
        contract_id="RECIPROCAL-MATCHED-PAIR-v2",
        payload={"recipient_label": label, "horizon": horizon, "response_semantics": effect["response_semantics"]},
    )
    sources = [(probes[branch], f"{branch}_support_observation") for branch in RECIPROCAL_BRANCHES] + [(contract_ref, "frozen_response_contrast_contract")]
    outcomes: list[tuple[GraphRef, str]] = []
    for name, reference in _tensor_refs(effect["numeric_effects"], "numeric_effects"):
        outcome = _external_tensor(
            writer,
            reference=reference,
            semantic_key=f"reciprocal:{label}:horizon:{horizon}:response:{name}",
            role=f"reciprocal_branch_response:{name}",
            optimizer_step=optimizer_step,
        )
        outcomes.append((outcome, name))
    _bind_all(writer, occurrence, sources, outcomes)
    summary = _object(
        writer,
        semantic_key=f"reciprocal:{label}:horizon:{horizon}:response-summary",
        role="reciprocal_branch_response_summary",
        optimizer_step=optimizer_step,
        payload={"recipient_label": label, "horizon": horizon, "effect_result_sha256": effect["result_sha256"], "tensor_count": len(outcomes)},
    )
    writer.bind(occurrence, [*sources, *outcomes], summary, payload={"outcome_kind": "reciprocal_response_summary"})
    return summary


def _continuation(
    writer: SupportGFGWriter,
    *,
    label: str,
    branch: str,
    row: dict[str, Any],
    prior: GraphRef,
    batch_sources: list[tuple[GraphRef, str]],
    contract_ref: GraphRef,
) -> GraphRef:
    step = int(row["physical_optimizer_step"])
    branch_row = row["branches"][branch]
    occurrence = _occurrence(
        writer,
        occurrence_type="actual_reciprocal_branch_continuation_step_occurrence",
        optimizer_step=step,
        operation="native_training_step_under_reciprocal_branch_continuation",
        contract_id="RECIPROCAL-MATCHED-PAIR-v2",
        payload={"recipient_label": label, "branch": branch, **branch_row},
    )
    outcome = _object(
        writer,
        semantic_key=f"reciprocal:{label}:continuation:{step}:{branch}:state-commitment",
        role=f"{branch}_actual_continuation_state_commitment",
        optimizer_step=step + 1,
        payload={"recipient_label": label, "branch": branch, **branch_row},
        object_kind="branch_state_commitment",
    )
    writer.bind(occurrence, [(prior, "actual_branch_prestate"), *batch_sources, (contract_ref, "frozen_aligned_continuation_contract")], outcome, payload={"outcome_kind": "branch_state_commitment"})
    return outcome


def build_reciprocal_gfg(
    *,
    evidence_root: Path,
    formal_root: Path,
    source_root: Path,
    graph_root: Path,
    reciprocal_protocol_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
) -> dict[str, Any]:
    require(read_json(evidence_root / "reciprocal_pair_validation.json")["status"] == "PASS", "SST_RECIPROCAL_GFG_EVIDENCE_NOT_VALIDATED")
    protocol = read_json(reciprocal_protocol_path)
    endpoints = {str(row["label"]): row for row in protocol["endpoints"]}
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    require(not graph_root.exists(), f"SST_RECIPROCAL_GFG_ROOT_ALREADY_EXISTS:{graph_root}")
    graph_root.mkdir(parents=True)
    staged_count = _stage_tensors(evidence_root, graph_root)
    combined_source_id = payload_sha256({label: row["source_bundle_id"] for label, row in endpoints.items()})
    database = graph_root / "reciprocal_matched_pair_gfg.sqlite3"
    writer = SupportGFGWriter(
        database,
        graph_root / "tensor-objects",
        scope_id="nanogpt-reciprocal-matched-pair-v1",
        source_bundle_id=combined_source_id,
        contract_sha256=file_sha256(reciprocal_protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    source_graphs: dict[str, TrainingGFG] = {}
    main_manifests: dict[str, dict[str, Any]] = {}
    main_objects: dict[str, dict[str, dict[str, Any]]] = {}
    for label, endpoint in endpoints.items():
        source_graphs[label] = TrainingGFG(source_root / str(endpoint["source_bundle_id"]) / "participant_gfg.sqlite3")
        manifest = read_json(formal_root / str(endpoint["entry_id"]) / "stepwise_support_transition_gfg_manifest.json")
        main_manifests[label] = manifest
        state_record = read_json(formal_root / str(endpoint["entry_id"]) / "windows" / str(endpoint["window_id"]) / "states" / f"step-{int(endpoint['optimizer_step']):05d}.json")
        key = f"{endpoint['window_id']}:{state_record['state']['state_id']}"
        required = {str(manifest["state_catalog"][key])}
        main_objects[label] = _main_object_index(formal_root / str(endpoint["entry_id"]) / str(manifest["database"]), required)
    state_catalog: dict[str, str] = {}
    response_summaries: list[GraphRef] = []
    try:
        writer.start_block("reciprocal_graph_contracts", 0)
        contract_ref = _object(writer, semantic_key="contract:reciprocal-matched-pair-v2", role="frozen_reciprocal_matched_pair_contract", optimizer_step=0, payload={"contract": protocol, "file_sha256": file_sha256(reciprocal_protocol_path)}, object_kind="frozen_contract")
        registry_ref = _object(writer, semantic_key=f"component-registry:{registry.registry_id}", role="versioned_component_registry", optimizer_step=0, payload={"registry": read_json(component_registry_path), "file_sha256": registry.source_sha256}, object_kind="frozen_contract")
        probe_contract_ref = _object(writer, semantic_key=f"probe-contract:{probe_contract.probe_contract_id}", role="versioned_probe_contract", optimizer_step=0, payload={"contract": read_json(probe_contract_path), "file_sha256": probe_contract.source_sha256}, object_kind="frozen_contract")
        validation_sources = {
            label: _validation_probe_sources(
                writer,
                source_graph=source_graphs[label],
                source_bundle=source_root / str(endpoint["source_bundle_id"]),
                source_bundle_id=str(endpoint["source_bundle_id"]),
                protocol_ref=contract_ref,
            )
            for label, endpoint in endpoints.items()
        }
        writer.flush_block()

        prestate_refs: dict[str, GraphRef] = {}
        native_refs: dict[str, GraphRef] = {}
        seeds: dict[str, dict[str, Any]] = {}
        for label, endpoint in endpoints.items():
            writer._last_occurrence_id = None
            writer.start_block(f"reciprocal:{label}:native-seed", int(endpoint["optimizer_step"]))
            seed = read_json(evidence_root / f"recipient-{label}" / "seed_result.json")
            seeds[label] = seed
            prestate_refs[label] = _main_state_origin(
                writer,
                main_manifest=main_manifests[label],
                main_objects=main_objects[label],
                window_id=str(endpoint["window_id"]),
                state_id=str(seed["recipient_prestate_id"]),
                optimizer_step=int(endpoint["optimizer_step"]),
            )
            batch_sources = _batch_sources(writer, batch=seed["recipient_batch"], source_bundle_id=str(endpoint["source_bundle_id"]), contract_ref=contract_ref, semantic_prefix=f"reciprocal:{label}:native-seed", optimizer_step=int(endpoint["optimizer_step"]))
            native_refs[label] = _native_step(writer, label=label, seed=seed, prestate=prestate_refs[label], batch_sources=batch_sources, contract_ref=contract_ref)
            writer.flush_block()

        for label, donor_label in (("A", "B"), ("B", "A")):
            endpoint = endpoints[label]
            receipt = read_json(evidence_root / f"recipient-{label}" / "reciprocal_receipt.json")
            seed_id = str(receipt["seed_id"])
            writer._last_occurrence_id = None
            writer.start_block(f"reciprocal:{label}:branch-establishment", int(endpoint["optimizer_step"]))
            current = {
                branch: _branch_seed(
                    writer,
                    label=label,
                    donor_label=donor_label,
                    branch=branch,
                    optimizer_step=int(endpoint["optimizer_step"]),
                    recipient_prestate=prestate_refs[label],
                    recipient_native=native_refs[label],
                    donor_native=native_refs[donor_label],
                    contract_ref=contract_ref,
                )
                for branch in RECIPROCAL_BRANCHES
            }
            writer.flush_block()
            horizons = {int(row["horizon"]): row for row in receipt["horizon_results"]}
            continuations = {
                int(row["physical_optimizer_step"]): read_json(evidence_root / f"recipient-{label}" / "continuations" / f"step-{int(row['physical_optimizer_step']):05d}-to-{int(row['physical_optimizer_step']) + 1:05d}.json")
                for row in receipt["continuation_results"]
            }
            max_horizon = max(horizons)
            for horizon in range(1, max_horizon + 1):
                if horizon in horizons:
                    writer.start_block(f"reciprocal:{label}:horizon", int(endpoint["optimizer_step"]) + horizon)
                    horizon_refs: dict[str, GraphRef] = {}
                    probes: dict[str, GraphRef] = {}
                    for branch in RECIPROCAL_BRANCHES:
                        record = read_json(evidence_root / f"recipient-{label}" / "horizons" / f"h-{horizon:03d}" / f"{branch}-state.json")
                        state_ref = _horizon_state(writer, label=label, branch=branch, horizon=horizon, record=record, prior=current[branch], contract_ref=contract_ref)
                        horizon_refs[branch] = state_ref
                        observation = read_json(evidence_root / f"recipient-{label}" / "probe-observations" / probe_contract.probe_contract_id / f"{record['state']['state_id']}.json")
                        probes[branch] = _emit_probe(writer, observation=observation, state_origin=state_ref, validation_sources=validation_sources[label], contract_ref=probe_contract_ref, registry_ref=registry_ref, semantic_prefix=f"reciprocal:{label}:horizon:{horizon}:{branch}", optimizer_step=int(endpoint["optimizer_step"]) + horizon, probe_contract=probe_contract)
                        state_catalog[f"{label}:{horizon}:{branch}"] = state_ref.object_id
                    current = horizon_refs
                    effect = read_json(evidence_root / f"recipient-{label}" / "horizons" / f"h-{horizon:03d}" / "effects.json")
                    response_summaries.append(_effect_block(writer, label=label, horizon=horizon, effect=effect, probes=probes, contract_ref=contract_ref, optimizer_step=int(endpoint["optimizer_step"]) + horizon))
                    writer.flush_block()
                if horizon == max_horizon:
                    continue
                step = int(endpoint["optimizer_step"]) + horizon
                row = continuations[step]
                writer.start_block(f"reciprocal:{label}:continuation", step)
                batch_sources = _batch_sources(writer, batch=row["same_batch_all_branches"], source_bundle_id=str(endpoint["source_bundle_id"]), contract_ref=contract_ref, semantic_prefix=f"reciprocal:{label}:continuation:{step}", optimizer_step=step)
                current = {branch: _continuation(writer, label=label, branch=branch, row=row, prior=current[branch], batch_sources=batch_sources, contract_ref=contract_ref) for branch in RECIPROCAL_BRANCHES}
                writer.flush_block()

        validation = read_json(evidence_root / "reciprocal_pair_validation.json")
        writer._last_occurrence_id = None
        writer.start_block("reciprocal:cross-run-adjudication", max(int(row["optimizer_step"]) for row in endpoints.values()) + 100)
        occurrence = _occurrence(writer, occurrence_type="reciprocal_cross_run_adjudication_occurrence", optimizer_step=max(int(row["optimizer_step"]) for row in endpoints.values()) + 100, operation="adjudicate_reciprocal_response_signatures", contract_id="RECIPROCAL-MATCHED-PAIR-v2", payload={"decision": validation["adjudication"]["decision"], "validation_sha256": validation["validation_sha256"]})
        outcome = _object(writer, semantic_key="reciprocal:cross-run-adjudication:result", role="reciprocal_matched_pair_adjudication", optimizer_step=max(int(row["optimizer_step"]) for row in endpoints.values()) + 100, payload={"adjudication": validation["adjudication"], "signature_rows": validation["signature_rows"], "capability_accuracy": validation["capability_accuracy"], "validation_sha256": validation["validation_sha256"]})
        writer.bind(occurrence, [(value, "validated_reciprocal_response_summary") for value in response_summaries] + [(contract_ref, "frozen_adjudication_contract")], outcome, payload={"outcome_kind": "reciprocal_matched_pair_adjudication"})
        writer.flush_block()
        manifest = writer.close()
    finally:
        for graph in source_graphs.values():
            graph.close()
    material = {
        **manifest,
        "evidence_validation_sha256": read_json(evidence_root / "reciprocal_pair_validation.json")["validation_sha256"],
        "reciprocal_protocol_sha256": file_sha256(reciprocal_protocol_path),
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "source_bundle_ids": {label: endpoint["source_bundle_id"] for label, endpoint in endpoints.items()},
        "staged_tensor_payload_count": staged_count,
        "state_catalog": state_catalog,
        "recipient_count": 2,
        "branch_count_per_recipient": len(RECIPROCAL_BRANCHES),
        "horizons": protocol["horizons"],
    }
    result = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(graph_root / "reciprocal_matched_pair_gfg_manifest.json", result)
    return result
