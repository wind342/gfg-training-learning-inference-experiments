from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any
import zlib

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

from .contracts import ComponentRegistry, ProbeContract
from .execution import _read_checked
from .stepwise_gfg import _bind_all, _emit_probe, _external_tensor, _object, _occurrence


GRAPH_SCHEMA = "nanogpt-p2-reciprocal-local-response-gfg-v1"
BLOCK_SCHEMA = "nanogpt-p2-reciprocal-local-response-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-p2-reciprocal-local-response-gfg-manifest-v1"
BRANCH_PATTERN = re.compile(r"^update_(P2[ab])_(minus|plus)_0\.125$")


def _stage_tensors(evidence_root: Path, graph_root: Path) -> int:
    source_root = evidence_root / "tensor-objects"
    require(source_root.is_dir(), "P2_GFG_TENSOR_SOURCE_MISSING")
    target_root = graph_root / "tensor-objects"
    target_root.mkdir(parents=True, exist_ok=True)
    staged = 0
    for source in sorted(source_root.glob("*.npy")):
        destination = target_root / source.name
        if destination.exists():
            require(file_sha256(destination) == file_sha256(source), "P2_GFG_TENSOR_COLLISION")
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        staged += 1
    return staged


def _objects_from_steps(database: Path, steps: set[int]) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    result: dict[str, dict[str, Any]] = {}
    try:
        placeholders = ",".join("?" for _ in steps)
        query = (
            "SELECT payload_zlib FROM graph_blocks WHERE optimizer_step IN ("
            + placeholders
            + ") ORDER BY block_ordinal"
        )
        for row in connection.execute(query, tuple(sorted(steps))):
            block = json.loads(zlib.decompress(row["payload_zlib"]))
            for value in block["objects"]:
                result[str(value["object_id"])] = value
    finally:
        connection.close()
    return result


def _source_material(formal_root: Path, endpoint: dict[str, Any]) -> dict[str, Any]:
    entry_root = formal_root / str(endpoint["entry_id"])
    manifest_path = entry_root / "stepwise_support_transition_gfg_manifest.json"
    manifest = read_json(manifest_path)
    database = entry_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "P2_GFG_SOURCE_DATABASE_HASH_MISMATCH")
    optimizer_step = int(endpoint["optimizer_step"])
    # The formal prestate for update t is the poststate formed by update t-1.
    objects = _objects_from_steps(database, {optimizer_step - 1, optimizer_step, optimizer_step + 1})

    prestate_id = str(endpoint["prestate"]["source_object_id"])
    target_state_id = str(
        manifest["state_catalog"][
            f"{endpoint['window_id']}:{endpoint['sealed_native_target']['state_id']}"
        ]
    )
    target_probe_id = str(endpoint["sealed_native_target"]["probe_observation_id"])
    target_probe_matches = [
        value
        for value in objects.values()
        if value["role"] == "complete_probe_observation"
        and value.get("payload", {}).get("probe_observation_id") == target_probe_id
    ]
    update_matches = [
        value
        for value in objects.values()
        if value["role"] == "complete_named_parameter_update"
        and endpoint["native_update"]["raw_tensor_sha256"]
        in json.dumps(value.get("payload", {}), ensure_ascii=False, sort_keys=True)
    ]
    require(prestate_id in objects, "P2_GFG_SOURCE_PRESTATE_OBJECT_MISSING")
    require(target_state_id in objects, "P2_GFG_SOURCE_TARGET_STATE_OBJECT_MISSING")
    require(len(target_probe_matches) == 1, "P2_GFG_SOURCE_TARGET_PROBE_NOT_UNIQUE")
    require(len(update_matches) == 1, "P2_GFG_SOURCE_NATIVE_UPDATE_NOT_UNIQUE")
    return {
        "database": database,
        "entry_root": entry_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "objects": {
            "prestate": objects[prestate_id],
            "native_update": update_matches[0],
            "target_state": objects[target_state_id],
            "target_probe": target_probe_matches[0],
        },
    }


def _origin(
    writer: SupportGFGWriter,
    source: dict[str, Any],
    source_object: dict[str, Any],
) -> GraphRef:
    manifest = source["manifest"]
    return writer.origin(
        source_object,
        source_bundle_id=str(manifest["manifest_sha256"]),
        source_graph_schema=str(manifest["schema"]),
    )


def _branch_donor(branch: str) -> str | None:
    if branch == "baseline":
        return None
    match = BRANCH_PATTERN.fullmatch(branch)
    require(match is not None, f"P2_GFG_BRANCH_INVALID:{branch}")
    return str(match.group(1))


def _branch_scale(branch: str, epsilon: float) -> float:
    if branch == "baseline":
        return 0.0
    match = BRANCH_PATTERN.fullmatch(branch)
    require(match is not None, f"P2_GFG_BRANCH_INVALID:{branch}")
    return epsilon if match.group(2) == "plus" else -epsilon


def build_p2_response_gfg(
    *,
    evidence_root: Path,
    formal_root: Path,
    graph_root: Path,
    p2_protocol_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
) -> dict[str, Any]:
    require(not graph_root.exists(), "P2_GFG_ROOT_EXISTS")
    graph_root.mkdir(parents=True)
    protocol = read_json(p2_protocol_path)
    require(protocol["status"] == "FROZEN_BEFORE_RESPONSE_EXECUTION_AND_BEFORE_NATIVE_TARGET_ACCESS", "P2_GFG_PROTOCOL_NOT_FROZEN")
    pretarget_validation = read_json(evidence_root / "p2_response_pre_target_validation.json")
    replay_validation = read_json(evidence_root / "p2_response_independent_replay_validation.json")
    seal = read_json(evidence_root / "PRE_TARGET_RESPONSE_SEAL.json")
    pair_receipt = read_json(evidence_root / "p2_response_pair_receipt.json")
    adjudication = read_json(evidence_root / "p2_native_target_adjudication.json")
    require(pretarget_validation["status"] == "PASS", "P2_GFG_PRETARGET_VALIDATION_NOT_PASS")
    require(replay_validation["status"] == "PASS", "P2_GFG_REPLAY_VALIDATION_NOT_PASS")
    require(seal["status"] == "SEALED_BEFORE_NATIVE_TARGET_ACCESS", "P2_GFG_RESPONSE_NOT_SEALED")
    require(adjudication["status"] == "PASS", "P2_GFG_ADJUDICATION_NOT_PASS")
    require(adjudication["frozen_outcome"] == "TWO_DIRECTION_RESPONSE_BASIS_INSUFFICIENT", "P2_GFG_FROZEN_OUTCOME_CHANGED")

    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    sources = {
        str(endpoint["label"]): _source_material(formal_root, endpoint)
        for endpoint in protocol["receivers"]
    }
    staged_count = _stage_tensors(evidence_root, graph_root)
    source_manifest_shas = [
        str(sources[label]["manifest"]["manifest_sha256"])
        for label in sorted(sources)
    ]
    composite_source_id = payload_sha256({"source_manifests": source_manifest_shas})
    writer = SupportGFGWriter(
        graph_root / "p2_reciprocal_local_response_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id=str(protocol["protocol_id"]),
        source_bundle_id=composite_source_id,
        contract_sha256=file_sha256(p2_protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )

    writer.start_block("p2_contract_and_pretarget_evidence", 0)
    protocol_ref = _object(
        writer,
        semantic_key="p2:frozen-protocol",
        role="frozen_p2_response_protocol",
        optimizer_step=0,
        payload={"protocol": protocol, "protocol_sha256": file_sha256(p2_protocol_path)},
        object_kind="frozen_contract",
    )
    registry_ref = _object(
        writer,
        semantic_key="p2:component-registry",
        role="versioned_component_registry",
        optimizer_step=0,
        payload={"registry": read_json(component_registry_path), "registry_sha256": registry.source_sha256},
        object_kind="frozen_contract",
    )
    probe_ref = _object(
        writer,
        semantic_key="p2:probe-contract",
        role="versioned_probe_contract",
        optimizer_step=0,
        payload={"probe_contract": read_json(probe_contract_path), "probe_contract_sha256": probe_contract.source_sha256},
        object_kind="frozen_contract",
    )
    receipt_ref = _object(
        writer,
        semantic_key="p2:pair-receipt",
        role="validated_pretarget_pair_receipt",
        optimizer_step=0,
        payload={"receipt": pair_receipt, "file_sha256": file_sha256(evidence_root / "p2_response_pair_receipt.json")},
        object_kind="validated_execution_record",
    )
    seal_ref = _object(
        writer,
        semantic_key="p2:pretarget-seal",
        role="pretarget_response_seal",
        optimizer_step=0,
        payload={"seal": seal, "file_sha256": file_sha256(evidence_root / "PRE_TARGET_RESPONSE_SEAL.json")},
        object_kind="validated_execution_record",
    )
    validation_ref = _object(
        writer,
        semantic_key="p2:pretarget-validation",
        role="independent_pretarget_validation",
        optimizer_step=0,
        payload={"validation": pretarget_validation, "file_sha256": file_sha256(evidence_root / "p2_response_pre_target_validation.json")},
        object_kind="validated_execution_record",
    )
    replay_ref = _object(
        writer,
        semantic_key="p2:independent-replay-validation",
        role="independent_replay_validation",
        optimizer_step=0,
        payload={"validation": replay_validation, "file_sha256": file_sha256(evidence_root / "p2_response_independent_replay_validation.json")},
        object_kind="validated_execution_record",
    )
    writer.flush_block()

    prestate_origins: dict[str, GraphRef] = {}
    update_origins: dict[str, GraphRef] = {}
    target_state_origins: dict[str, GraphRef] = {}
    target_probe_origins: dict[str, GraphRef] = {}
    for label, source in sources.items():
        prestate_origins[label] = _origin(writer, source, source["objects"]["prestate"])
        update_origins[label] = _origin(writer, source, source["objects"]["native_update"])
        target_state_origins[label] = _origin(writer, source, source["objects"]["target_state"])
        target_probe_origins[label] = _origin(writer, source, source["objects"]["target_probe"])

    state_catalog: dict[str, str] = {}
    probe_catalog: dict[str, str] = {}
    probe_summaries: dict[tuple[str, str], GraphRef] = {}
    for endpoint in protocol["receivers"]:
        receiver = str(endpoint["label"])
        optimizer_step = int(endpoint["optimizer_step"])
        receipt = _read_checked(
            evidence_root / "receivers" / receiver / "receiver_receipt.json",
            "nanogpt-p2-receiver-receipt-v1",
        )
        writer.start_block(f"p2_receiver_{receiver}_states_and_probes", optimizer_step)
        for branch in protocol["branches_per_receiver"]:
            state_record = _read_checked(
                evidence_root / "receivers" / receiver / "states" / f"{branch}.json",
                "nanogpt-p2-analysis-state-v1",
            )
            donor = _branch_donor(str(branch))
            occurrence = _occurrence(
                writer,
                occurrence_type="p2_scaled_parameter_displacement_occurrence",
                optimizer_step=optimizer_step,
                operation="apply_exact_scaled_native_parameter_update",
                contract_id=str(protocol["protocol_id"]),
                payload={
                    "receiver_label": receiver,
                    "donor_label": donor,
                    "branch": branch,
                    "scale": _branch_scale(str(branch), float(protocol["epsilon"])),
                    "optimizer_identical_to_receiver_prestate": True,
                },
            )
            state_outcome = _object(
                writer,
                semantic_key=f"p2:{receiver}:{branch}:analysis-state",
                role=f"p2_{branch}_analysis_state",
                optimizer_step=optimizer_step,
                payload={"state_record": state_record},
                object_kind="restorable_p2_analysis_state",
            )
            state_sources = [
                (prestate_origins[receiver], "exact_receiver_prestate"),
                (protocol_ref, "frozen_response_protocol"),
            ]
            if donor is not None:
                state_sources.append((update_origins[donor], "exact_scaled_native_parameter_update"))
            writer.bind(
                occurrence,
                state_sources,
                state_outcome,
                payload={"outcome_kind": "restorable_p2_analysis_state"},
            )
            state_catalog[f"{receiver}:{branch}"] = state_outcome.object_id
            observed_state_id = str(receipt["branches"][branch]["state_id"])
            observation = _read_checked(
                evidence_root / "probe-observations" / probe_contract.probe_contract_id / f"{observed_state_id}.json",
                "nanogpt-stepwise-probe-observation-v1",
            )
            probe_summary = _emit_probe(
                writer,
                observation=observation,
                state_origin=state_outcome,
                validation_sources=[(receipt_ref, "validated_execution_receipt")],
                contract_ref=probe_ref,
                registry_ref=registry_ref,
                semantic_prefix=f"p2:{receiver}:{branch}",
                optimizer_step=optimizer_step,
                probe_contract=probe_contract,
            )
            probe_summaries[(receiver, str(branch))] = probe_summary
            probe_catalog[f"{receiver}:{branch}"] = probe_summary.object_id
        writer.flush_block()

    response_summaries: dict[str, GraphRef] = {}
    for receiver in ("P2a", "P2b"):
        writer.start_block(f"p2_receiver_{receiver}_central_responses", int(next(e["optimizer_step"] for e in protocol["receivers"] if e["label"] == receiver)))
        for donor in ("P2a", "P2b"):
            response = _read_checked(
                evidence_root / "responses" / f"receiver-{receiver}-donor-{donor}.json",
                "nanogpt-p2-local-response-jk-v1",
            )
            minus_branch = f"update_{donor}_minus_0.125"
            plus_branch = f"update_{donor}_plus_0.125"
            occurrence = _occurrence(
                writer,
                occurrence_type="p2_central_response_derivation_occurrence",
                optimizer_step=int(next(e["optimizer_step"] for e in protocol["receivers"] if e["label"] == receiver)),
                operation="derive_registered_central_J_and_K_response",
                contract_id=str(protocol["protocol_id"]),
                payload={"receiver_label": receiver, "donor_label": donor, "epsilon": protocol["epsilon"]},
            )
            response_sources = [
                (probe_summaries[(receiver, "baseline")], "baseline_probe_observation"),
                (probe_summaries[(receiver, minus_branch)], "minus_epsilon_probe_observation"),
                (probe_summaries[(receiver, plus_branch)], "plus_epsilon_probe_observation"),
                (update_origins[donor], "exact_native_update_direction"),
                (protocol_ref, "frozen_response_definition"),
            ]
            response_outputs: list[tuple[GraphRef, str]] = []
            for key, references in sorted(response["numeric_responses"].items()):
                for field in ("j_first_order", "k_curvature"):
                    output = _external_tensor(
                        writer,
                        reference=references[field],
                        semantic_key=f"p2:{receiver}:{donor}:{key}:{field}",
                        role=f"{field}_of:{key}",
                        optimizer_step=int(next(e["optimizer_step"] for e in protocol["receivers"] if e["label"] == receiver)),
                    )
                    response_outputs.append((output, f"{field}:{key}"))
            for key, references in sorted(response["categorical_transitions"].items()):
                for field in ("baseline", "plus", "minus", "plus_changed_mask", "minus_changed_mask"):
                    output = _external_tensor(
                        writer,
                        reference=references[field],
                        semantic_key=f"p2:{receiver}:{donor}:{key}:categorical-{field}",
                        role=f"categorical_{field}_of:{key}",
                        optimizer_step=int(next(e["optimizer_step"] for e in protocol["receivers"] if e["label"] == receiver)),
                    )
                    response_outputs.append((output, f"categorical_{field}:{key}"))
            _bind_all(writer, occurrence, response_sources, response_outputs)
            summary = _object(
                writer,
                semantic_key=f"p2:{receiver}:{donor}:validated-response-summary",
                role="validated_p2_local_response_summary",
                optimizer_step=int(next(e["optimizer_step"] for e in protocol["receivers"] if e["label"] == receiver)),
                payload={
                    "receiver_label": receiver,
                    "donor_label": donor,
                    "response_result_sha256": response["result_sha256"],
                    "numeric_response_count": len(response["numeric_responses"]),
                    "categorical_response_count": len(response["categorical_transitions"]),
                    "pretarget_validation_result_sha256": pretarget_validation["result_sha256"],
                },
                object_kind="validated_analysis_result",
            )
            writer.bind(
                occurrence,
                [*response_sources, *response_outputs, (validation_ref, "independent_response_validation")],
                summary,
                payload={"outcome_kind": "validated_p2_local_response_summary"},
            )
            response_summaries[f"{receiver}:{donor}"] = summary
        writer.flush_block()

    writer.start_block("p2_sealed_receiver_response_comparison", 0)
    comparison_occurrence = _occurrence(
        writer,
        occurrence_type="p2_receiver_local_response_comparison_occurrence",
        optimizer_step=0,
        operation="compare_receiver_conditioned_responses_after_pretarget_seal",
        contract_id=str(protocol["protocol_id"]),
        payload={"native_target_content_opened": False, "response_basis_count": 4},
    )
    sealed_package = _object(
        writer,
        semantic_key="p2:sealed-pretarget-response-package",
        role="sealed_validated_p2_response_package",
        optimizer_step=0,
        payload={
            "pair_receipt_result_sha256": pair_receipt["result_sha256"],
            "seal_result_sha256": seal["result_sha256"],
            "pretarget_validation_result_sha256": pretarget_validation["result_sha256"],
            "independent_replay_validation_result_sha256": replay_validation["result_sha256"],
            "future_information_used": False,
            "native_target_content_opened": False,
            "response_result_sha256": {
                key: read_json(evidence_root / "responses" / f"receiver-{key.split(':')[0]}-donor-{key.split(':')[1]}.json")["result_sha256"]
                for key in sorted(response_summaries)
            },
        },
        object_kind="validated_analysis_result",
    )
    writer.bind(
        comparison_occurrence,
        [
            *((response_summaries[key], "sealed_local_response_basis") for key in sorted(response_summaries)),
            (receipt_ref, "validated_execution_receipt"),
            (seal_ref, "pretarget_seal"),
            (validation_ref, "independent_pretarget_validation"),
            (replay_ref, "independent_exact_replay_validation"),
            (protocol_ref, "frozen_comparison_contract"),
        ],
        sealed_package,
        payload={"outcome_kind": "sealed_validated_p2_response_package"},
    )
    writer.flush_block()

    writer.start_block("p2_native_target_adjudication", 0)
    adjudication_occurrence = _occurrence(
        writer,
        occurrence_type="p2_native_target_adjudication_occurrence",
        optimizer_step=0,
        operation="adjudicate_sealed_response_basis_against_native_next_state",
        contract_id=str(protocol["protocol_id"]),
        payload={"target_access_after_pretarget_seal": True, "scientific_scope": adjudication["scientific_scope"]},
    )
    adjudication_input_ref = _object(
        writer,
        semantic_key="p2:adjudication-input-attestation",
        role="frozen_adjudication_input_attestation",
        optimizer_step=0,
        payload={
            "pilot_jsonl_file_sha256": adjudication["pilot_jsonl_file_sha256"],
            "fitted_model_file_sha256": adjudication["fitted_model_file_sha256"],
            "proposal_file_sha256": adjudication["proposal_file_sha256"],
            "native_target_used_before_response_seal": False,
        },
        object_kind="validated_execution_record",
    )
    adjudication_outcome = _object(
        writer,
        semantic_key="p2:native-target-adjudication",
        role="p2_native_target_adjudication",
        optimizer_step=0,
        payload={"adjudication": adjudication, "file_sha256": file_sha256(evidence_root / "p2_native_target_adjudication.json")},
        object_kind="validated_scientific_result",
    )
    adjudication_sources: list[tuple[GraphRef, str]] = [
        (sealed_package, "sealed_pretarget_response_basis"),
        (adjudication_input_ref, "frozen_adjudication_inputs"),
        (protocol_ref, "frozen_adjudication_contract"),
    ]
    for label in ("P2a", "P2b"):
        adjudication_sources.extend(
            [
                (target_state_origins[label], f"{label}_native_next_state"),
                (target_probe_origins[label], f"{label}_native_next_probe_observation"),
            ]
        )
    writer.bind(
        adjudication_occurrence,
        adjudication_sources,
        adjudication_outcome,
        payload={"outcome_kind": adjudication["frozen_outcome"]},
    )
    writer.flush_block()
    closed = writer.close()

    source_graph_rows = []
    for label in sorted(sources):
        source = sources[label]
        source_graph_rows.append(
            {
                "label": label,
                "entry_id": source["manifest"]["entry_id"],
                "manifest_sha256": source["manifest"]["manifest_sha256"],
                "database_sha256": source["manifest"]["database_sha256"],
                "prestate_object_id": source["objects"]["prestate"]["object_id"],
                "native_update_object_id": source["objects"]["native_update"]["object_id"],
                "target_state_object_id": source["objects"]["target_state"]["object_id"],
                "target_probe_object_id": source["objects"]["target_probe"]["object_id"],
            }
        )
    material = {
        **closed,
        "status": "PASS",
        "p2_protocol_sha256": file_sha256(p2_protocol_path),
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "pretarget_validation_result_sha256": pretarget_validation["result_sha256"],
        "independent_replay_validation_result_sha256": replay_validation["result_sha256"],
        "pretarget_seal_result_sha256": seal["result_sha256"],
        "adjudication_result_sha256": adjudication["result_sha256"],
        "frozen_outcome": adjudication["frozen_outcome"],
        "source_graphs": source_graph_rows,
        "staged_tensor_payload_count": staged_count,
        "state_catalog": state_catalog,
        "probe_catalog": probe_catalog,
        "response_summary_catalog": {key: value.object_id for key, value in sorted(response_summaries.items())},
        "sealed_response_package_object_id": sealed_package.object_id,
        "adjudication_object_id": adjudication_outcome.object_id,
        "target_information_used_before_seal": False,
        "long_horizon_challenge_pairs_resolved": 0,
    }
    manifest = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(graph_root / "p2_reciprocal_local_response_gfg_manifest.json", manifest)
    return manifest


__all__ = ["build_p2_response_gfg"]
