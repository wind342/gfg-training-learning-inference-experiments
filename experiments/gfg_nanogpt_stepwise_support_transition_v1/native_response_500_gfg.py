from __future__ import annotations

import json
import os
from pathlib import Path
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

from .execution import _read_checked
from .stepwise_gfg import _external_tensor, _object, _occurrence


GRAPH_SCHEMA = "nanogpt-native-direction-response-500-gfg-v1"
BLOCK_SCHEMA = "nanogpt-native-direction-response-500-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-native-direction-response-500-gfg-manifest-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stage_tensors(evidence_root: Path, graph_root: Path) -> int:
    source_root = evidence_root / "tensor-objects"
    require(source_root.is_dir(), "NATIVE_RESPONSE_500_GFG_TENSOR_SOURCE_MISSING")
    target_root = graph_root / "tensor-objects"
    target_root.mkdir(parents=True, exist_ok=True)
    staged = 0
    for source in sorted(source_root.glob("*.npy")):
        destination = target_root / source.name
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        require(file_sha256(destination) == file_sha256(source), "NATIVE_RESPONSE_500_GFG_TENSOR_STAGE_MISMATCH")
        staged += 1
    return staged


def _objects_from_steps(database: Path, steps: set[int]) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    result: dict[str, dict[str, Any]] = {}
    try:
        values = sorted(steps)
        for offset in range(0, len(values), 500):
            chunk = values[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            query = "SELECT payload_zlib FROM graph_blocks WHERE optimizer_step IN (" + placeholders + ") ORDER BY block_ordinal"
            for row in connection.execute(query, tuple(chunk)):
                block = json.loads(zlib.decompress(row["payload_zlib"]))
                for value in block["objects"]:
                    result[str(value["object_id"])] = value
    finally:
        connection.close()
    return result


def _source_catalog(formal_root: Path, endpoints: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for endpoint in endpoints:
        grouped.setdefault(str(endpoint["entry_id"]), []).append(endpoint)
    result: dict[str, dict[str, Any]] = {}
    for entry_id, rows in sorted(grouped.items()):
        entry_root = formal_root / entry_id
        manifest_path = entry_root / "stepwise_support_transition_gfg_manifest.json"
        manifest = read_json(manifest_path)
        database = entry_root / str(manifest["database"])
        require(file_sha256(database) == manifest["database_sha256"], f"NATIVE_RESPONSE_500_GFG_SOURCE_DATABASE_HASH_MISMATCH:{entry_id}")
        steps = {int(row["optimizer_step"]) + delta for row in rows for delta in (-1, 0, 1)}
        objects = _objects_from_steps(database, steps)
        for endpoint in rows:
            sample_id = str(endpoint["sample_id"])
            prestate_id = str(endpoint["prestate"]["source_object_id"])
            target_state_id = str(endpoint["sealed_native_target_identity"]["source_object_id"])
            update_hash = str(endpoint["native_update"]["raw_tensor_sha256"])
            target_probe_id = str(endpoint["sealed_native_target_identity"]["probe_observation_id"])
            update_matches = [
                value for value in objects.values()
                if value["role"] == "complete_named_parameter_update"
                and update_hash in json.dumps(value.get("payload", {}), sort_keys=True)
            ]
            target_probe_matches = [
                value for value in objects.values()
                if value["role"] == "complete_probe_observation"
                and value.get("payload", {}).get("probe_observation_id") == target_probe_id
            ]
            require(prestate_id in objects, f"NATIVE_RESPONSE_500_GFG_PRESTATE_SOURCE_MISSING:{sample_id}")
            require(target_state_id in objects, f"NATIVE_RESPONSE_500_GFG_TARGET_STATE_SOURCE_MISSING:{sample_id}")
            require(len(update_matches) == 1, f"NATIVE_RESPONSE_500_GFG_UPDATE_SOURCE_NOT_UNIQUE:{sample_id}")
            require(len(target_probe_matches) == 1, f"NATIVE_RESPONSE_500_GFG_TARGET_PROBE_SOURCE_NOT_UNIQUE:{sample_id}")
            result[sample_id] = {
                "manifest": manifest,
                "prestate": objects[prestate_id],
                "native_update": update_matches[0],
                "target_state": objects[target_state_id],
                "target_probe": target_probe_matches[0],
            }
    require(len(result) == 500, "NATIVE_RESPONSE_500_GFG_SOURCE_CATALOG_COUNT_INVALID")
    return result


def _origin(writer: SupportGFGWriter, source: dict[str, Any], key: str) -> GraphRef:
    manifest = source["manifest"]
    return writer.origin(
        source[key],
        source_bundle_id=str(manifest["manifest_sha256"]),
        source_graph_schema=str(manifest["schema"]),
    )


def build_native_response_500_gfg(
    *,
    evidence_root: Path,
    formal_root: Path,
    modeling_root: Path,
    graph_root: Path,
    response_protocol_path: Path,
) -> dict[str, Any]:
    require(not graph_root.exists(), "NATIVE_RESPONSE_500_GFG_ROOT_EXISTS")
    protocol = read_json(response_protocol_path)
    validation = read_json(evidence_root / "native_response_500_pre_target_validation.json")
    seal = read_json(evidence_root / "PRE_TARGET_RESPONSE_500_SEAL.json")
    modeling_manifest = read_json(modeling_root / "MANIFEST.json")
    modeling_validation = read_json(modeling_root / "VALIDATION.json")
    require(validation["status"] == "PASS", "NATIVE_RESPONSE_500_GFG_RESPONSE_VALIDATION_NOT_PASS")
    require(seal["status"] == "SEALED_BEFORE_NATIVE_TARGET_ACCESS", "NATIVE_RESPONSE_500_GFG_RESPONSE_NOT_SEALED")
    require(modeling_manifest["status"] == modeling_validation["status"] == "PASS", "NATIVE_RESPONSE_500_GFG_MODELING_PACKAGE_NOT_PASS")
    endpoints = list(protocol["receivers"])
    require(len(endpoints) == 500, "NATIVE_RESPONSE_500_GFG_PROTOCOL_COUNT_INVALID")
    modeling_rows = _read_jsonl(modeling_root / modeling_manifest["records_file"])
    modeling_by_id = {row["sample_id"]: row for row in modeling_rows}
    require(len(modeling_by_id) == 500, "NATIVE_RESPONSE_500_GFG_MODELING_COUNT_INVALID")
    sources = _source_catalog(formal_root, endpoints)

    graph_root.mkdir(parents=True)
    staged_count = _stage_tensors(evidence_root, graph_root)
    composite_source_id = payload_sha256(
        {"source_manifests": sorted({source["manifest"]["manifest_sha256"] for source in sources.values()})}
    )
    writer = SupportGFGWriter(
        graph_root / "native_response_500_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id=str(protocol["protocol_id"]),
        source_bundle_id=composite_source_id,
        contract_sha256=file_sha256(response_protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    writer.start_block("native_response_500_contracts", 0)
    protocol_ref = _object(
        writer,
        semantic_key="native-response-500:frozen-protocol",
        role="frozen_native_response_500_protocol",
        optimizer_step=0,
        payload={"protocol": protocol, "protocol_sha256": file_sha256(response_protocol_path)},
        object_kind="frozen_contract",
    )
    validation_ref = _object(
        writer,
        semantic_key="native-response-500:pretarget-validation",
        role="validated_native_response_500_execution",
        optimizer_step=0,
        payload={"validation": validation, "file_sha256": file_sha256(evidence_root / "native_response_500_pre_target_validation.json")},
        object_kind="validated_execution_record",
    )
    seal_ref = _object(
        writer,
        semantic_key="native-response-500:pretarget-seal",
        role="pretarget_native_response_500_seal",
        optimizer_step=0,
        payload={"seal": seal, "file_sha256": file_sha256(evidence_root / "PRE_TARGET_RESPONSE_500_SEAL.json")},
        object_kind="validated_execution_record",
    )
    modeling_ref = _object(
        writer,
        semantic_key="native-response-500:modeling-view-manifest",
        role="validated_response_500_modeling_projection",
        optimizer_step=0,
        payload={"manifest": modeling_manifest, "validation": modeling_validation},
        object_kind="validated_analysis_result",
    )
    writer.flush_block()

    sample_catalog: dict[str, str] = {}
    response_catalog: dict[str, str] = {}
    for index, endpoint in enumerate(sorted(endpoints, key=lambda row: row["sample_id"]), 1):
        sample_id = str(endpoint["sample_id"])
        step = int(endpoint["optimizer_step"])
        source = sources[sample_id]
        prestate_origin = _origin(writer, source, "prestate")
        update_origin = _origin(writer, source, "native_update")
        target_state_origin = _origin(writer, source, "target_state")
        target_probe_origin = _origin(writer, source, "target_probe")
        receipt = _read_checked(
            evidence_root / "samples" / sample_id / "sample_receipt.json",
            "nanogpt-native-direction-response-sample-receipt-v1",
        )
        response = _read_checked(
            evidence_root / "responses" / f"{sample_id}.json",
            "nanogpt-native-direction-response-v1",
        )
        writer._last_occurrence_id = None
        writer.start_block(f"native_response_500_sample_{index:04d}", step)
        probe_refs: dict[str, GraphRef] = {}
        for branch in protocol["branches_per_receiver"]:
            state_record = _read_checked(
                evidence_root / "samples" / sample_id / "states" / f"{branch}.json",
                "nanogpt-native-direction-analysis-state-v1",
            )
            scale = 0.0 if branch == "baseline" else (float(protocol["epsilon"]) if branch.startswith("native_plus") else -float(protocol["epsilon"]))
            state_occurrence = _occurrence(
                writer,
                occurrence_type="native_direction_scaled_parameter_displacement_occurrence",
                optimizer_step=step,
                operation="apply_exact_scaled_native_parameter_update",
                contract_id=str(protocol["protocol_id"]),
                payload={"sample_id": sample_id, "branch": branch, "scale": scale, "optimizer_unchanged": True},
            )
            state_ref = _object(
                writer,
                semantic_key=f"native-response-500:{sample_id}:{branch}:state",
                role=f"{branch}_restorable_analysis_state",
                optimizer_step=step,
                payload={"state_record": state_record},
                object_kind="restorable_analysis_state",
            )
            state_sources = [(prestate_origin, "exact_receiver_prestate"), (protocol_ref, "frozen_response_protocol")]
            if branch != "baseline":
                state_sources.append((update_origin, "exact_scaled_native_parameter_update"))
            writer.bind(state_occurrence, state_sources, state_ref, payload={"outcome_kind": "restorable_analysis_state"})
            observation = _read_checked(
                evidence_root / "probe-observations" / receipt["probe_contract_id"] / f"{state_record['state']['state_id']}.json",
                "nanogpt-stepwise-probe-observation-v1",
            )
            probe_occurrence = _occurrence(
                writer,
                occurrence_type="csrg_probe_execution_occurrence",
                optimizer_step=step,
                operation="execute_frozen_csrg_4c_probe",
                contract_id=receipt["probe_contract_id"],
                payload={"sample_id": sample_id, "branch": branch, "actual_forward_count": observation["actual_forward_count"]},
            )
            probe_ref = _object(
                writer,
                semantic_key=f"native-response-500:{sample_id}:{branch}:complete-probe",
                role="complete_probe_observation",
                optimizer_step=step,
                payload={"observation": observation},
                object_kind="content_addressed_probe_observation",
            )
            writer.bind(
                probe_occurrence,
                [(state_ref, "probed_analysis_state"), (protocol_ref, "frozen_probe_contract_context")],
                probe_ref,
                payload={"outcome_kind": "complete_probe_observation"},
            )
            probe_refs[str(branch)] = probe_ref

        response_occurrence = _occurrence(
            writer,
            occurrence_type="native_direction_central_response_derivation_occurrence",
            optimizer_step=step,
            operation="derive_central_J_native_and_K_native",
            contract_id=str(protocol["protocol_id"]),
            payload={"sample_id": sample_id, "epsilon": protocol["epsilon"]},
        )
        response_sources = [
            (probe_refs["baseline"], "baseline_probe_observation"),
            (probe_refs["native_minus_0.125"], "minus_epsilon_probe_observation"),
            (probe_refs["native_plus_0.125"], "plus_epsilon_probe_observation"),
            (update_origin, "exact_native_update_direction"),
            (protocol_ref, "frozen_response_definition"),
        ]
        response_outputs: list[tuple[GraphRef, str]] = []
        for key, references in sorted(response["numeric_responses"].items()):
            for field in ("j_native", "k_native"):
                ref = _external_tensor(
                    writer,
                    reference=references[field],
                    semantic_key=f"native-response-500:{sample_id}:{key}:{field}",
                    role=f"{field}_of:{key}",
                    optimizer_step=step,
                )
                writer.bind(response_occurrence, response_sources, ref, payload={"outcome_kind": f"{field}:{key}"})
                response_outputs.append((ref, f"{field}:{key}"))
        for key, references in sorted(response["categorical_transitions"].items()):
            for field in ("plus_changed_mask", "minus_changed_mask"):
                ref = _external_tensor(
                    writer,
                    reference=references[field],
                    semantic_key=f"native-response-500:{sample_id}:{key}:{field}",
                    role=f"{field}_of:{key}",
                    optimizer_step=step,
                )
                writer.bind(response_occurrence, response_sources, ref, payload={"outcome_kind": f"{field}:{key}"})
                response_outputs.append((ref, f"{field}:{key}"))
        response_summary = _object(
            writer,
            semantic_key=f"native-response-500:{sample_id}:validated-response-summary",
            role="validated_receiver_conditioned_native_response",
            optimizer_step=step,
            payload={
                "sample_id": sample_id,
                "response_result_sha256": response["result_sha256"],
                "sample_receipt_result_sha256": receipt["result_sha256"],
                "numeric_response_count": len(response["numeric_responses"]),
                "categorical_response_count": len(response["categorical_transitions"]),
                "response_tensor_outcome_count": len(response_outputs),
            },
            object_kind="validated_analysis_result",
        )
        writer.bind(
            response_occurrence,
            [*response_sources, *response_outputs, (validation_ref, "independent_response_validation"), (seal_ref, "pretarget_seal")],
            response_summary,
            payload={"outcome_kind": "validated_receiver_conditioned_native_response"},
        )
        response_catalog[sample_id] = response_summary.object_id

        sample_occurrence = _occurrence(
            writer,
            occurrence_type="response_augmented_one_step_record_assembly_occurrence",
            optimizer_step=step,
            operation="join_sealed_receiver_response_with_validated_one_step_record",
            contract_id=str(protocol["protocol_id"]),
            payload={"sample_id": sample_id, "performed_after_pretarget_seal": True},
        )
        sample_ref = _object(
            writer,
            semantic_key=f"native-response-500:{sample_id}:modeling-record",
            role="response_augmented_one_step_modeling_record",
            optimizer_step=step,
            payload={"record": modeling_by_id[sample_id]},
            object_kind="validated_analysis_result",
        )
        writer.bind(
            sample_occurrence,
            [
                (prestate_origin, "current_prestate"),
                (update_origin, "current_formed_native_update"),
                (response_summary, "receiver_conditioned_response_state"),
                (target_state_origin, "next_native_state"),
                (target_probe_origin, "next_native_probe_observation"),
                (modeling_ref, "validated_modeling_projection"),
            ],
            sample_ref,
            payload={"outcome_kind": "response_augmented_one_step_modeling_record"},
        )
        sample_catalog[sample_id] = sample_ref.object_id
        writer.flush_block()
        if index % 25 == 0:
            print({"event": "NATIVE_RESPONSE_500_GFG_PROGRESS", "built": index}, flush=True)

    manifest = writer.close()
    manifest.update(
        {
            "protocol_sha256": file_sha256(response_protocol_path),
            "pretarget_validation_result_sha256": validation["result_sha256"],
            "pretarget_seal_result_sha256": seal["result_sha256"],
            "modeling_manifest_sha256": modeling_manifest["manifest_sha256"],
            "sample_count": 500,
            "response_count": 500,
            "staged_tensor_count": staged_count,
            "sample_catalog": sample_catalog,
            "response_catalog": response_catalog,
        }
    )
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = payload_sha256(material)
    write_json(graph_root / "native_response_500_gfg_manifest.json", manifest)
    return manifest


__all__ = ["build_native_response_500_gfg"]
