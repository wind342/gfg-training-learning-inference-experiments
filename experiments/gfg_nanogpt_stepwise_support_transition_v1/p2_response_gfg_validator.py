from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import zlib

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    canonical_bytes,
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import SupportGFG

from .p2_response_gfg import GRAPH_SCHEMA


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


def _validate_tensor(graph_root: Path, reference: dict[str, Any]) -> None:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "P2_GFG_VALIDATION_TENSOR_LOCATOR_INVALID")
    path = graph_root / locator
    require(path.is_file(), "P2_GFG_VALIDATION_TENSOR_MISSING")
    require(file_sha256(path) == reference["file_sha256"], "P2_GFG_VALIDATION_TENSOR_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "P2_GFG_VALIDATION_TENSOR_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "P2_GFG_VALIDATION_TENSOR_DTYPE_MISMATCH")
    require(hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest() == reference["raw_tensor_sha256"], "P2_GFG_VALIDATION_TENSOR_RAW_HASH_MISMATCH")


def _source_objects(
    database: Path,
    object_ids: set[str],
    optimizer_steps: set[int],
) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    result: dict[str, dict[str, Any]] = {}
    try:
        placeholders = ",".join("?" for _ in optimizer_steps)
        query = (
            "SELECT payload_zlib FROM graph_blocks WHERE optimizer_step IN ("
            + placeholders
            + ") ORDER BY block_ordinal"
        )
        for row in connection.execute(query, tuple(sorted(optimizer_steps))):
            block = json.loads(zlib.decompress(row["payload_zlib"]))
            for value in block["objects"]:
                object_id = str(value["object_id"])
                if object_id in object_ids:
                    result[object_id] = value
            if set(result) == object_ids:
                break
    finally:
        connection.close()
    return result


def validate_p2_response_gfg(
    *,
    graph_root: Path,
    evidence_root: Path,
    formal_root: Path,
    p2_protocol_path: Path,
) -> dict[str, Any]:
    manifest = read_json(graph_root / "p2_reciprocal_local_response_gfg_manifest.json")
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(material) == manifest["manifest_sha256"], "P2_GFG_VALIDATION_MANIFEST_HASH_MISMATCH")
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "P2_GFG_VALIDATION_DATABASE_HASH_MISMATCH")
    require(file_sha256(p2_protocol_path) == manifest["p2_protocol_sha256"], "P2_GFG_VALIDATION_PROTOCOL_HASH_MISMATCH")
    protocol = read_json(p2_protocol_path)
    pretarget = read_json(evidence_root / "p2_response_pre_target_validation.json")
    replay = read_json(evidence_root / "p2_response_independent_replay_validation.json")
    seal = read_json(evidence_root / "PRE_TARGET_RESPONSE_SEAL.json")
    adjudication = read_json(evidence_root / "p2_native_target_adjudication.json")
    require(pretarget["result_sha256"] == manifest["pretarget_validation_result_sha256"], "P2_GFG_VALIDATION_PRETARGET_MISMATCH")
    require(replay["result_sha256"] == manifest["independent_replay_validation_result_sha256"], "P2_GFG_VALIDATION_REPLAY_MISMATCH")
    require(seal["result_sha256"] == manifest["pretarget_seal_result_sha256"], "P2_GFG_VALIDATION_SEAL_MISMATCH")
    require(adjudication["result_sha256"] == manifest["adjudication_result_sha256"], "P2_GFG_VALIDATION_ADJUDICATION_MISMATCH")
    require(adjudication["frozen_outcome"] == manifest["frozen_outcome"], "P2_GFG_VALIDATION_OUTCOME_MISMATCH")
    require(manifest["target_information_used_before_seal"] is False, "P2_GFG_VALIDATION_TARGET_PRESEAL_LEAKAGE")
    require(manifest["long_horizon_challenge_pairs_resolved"] == 0, "P2_GFG_VALIDATION_FALSE_RESOLUTION_CLAIM")

    source_by_manifest: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    endpoint_by_label = {str(value["label"]): value for value in protocol["receivers"]}
    for row in manifest["source_graphs"]:
        source_root = formal_root / str(row["entry_id"])
        source_manifest = read_json(source_root / "stepwise_support_transition_gfg_manifest.json")
        require(source_manifest["manifest_sha256"] == row["manifest_sha256"], "P2_GFG_VALIDATION_SOURCE_MANIFEST_MISMATCH")
        source_database = source_root / str(source_manifest["database"])
        require(file_sha256(source_database) == row["database_sha256"], "P2_GFG_VALIDATION_SOURCE_DATABASE_MISMATCH")
        object_ids = {
            str(row["prestate_object_id"]),
            str(row["native_update_object_id"]),
            str(row["target_state_object_id"]),
            str(row["target_probe_object_id"]),
        }
        step = int(endpoint_by_label[str(row["label"])]["optimizer_step"])
        objects = _source_objects(source_database, object_ids, {step - 1, step, step + 1})
        require(set(objects) == object_ids, "P2_GFG_VALIDATION_SOURCE_OBJECT_MISSING")
        source_by_manifest[str(row["manifest_sha256"])] = (source_manifest, objects)

    graph = SupportGFG(database)
    known_objects: dict[str, dict[str, Any]] = {}
    known_occurrences: dict[str, dict[str, Any]] = {}
    known_facts: dict[str, dict[str, Any]] = {}
    object_roles: Counter[str] = Counter()
    occurrence_types: Counter[str] = Counter()
    relations: list[dict[str, Any]] = []
    fact_stage: dict[str, str] = {}
    target_origin_ids: set[str] = set()
    checks = 0
    tensor_ref_count = 0
    stages: list[str] = []
    try:
        require(graph.metadata("schema") == GRAPH_SCHEMA, "P2_GFG_VALIDATION_SCHEMA_INVALID")
        origins = {str(row["origin_id"]): dict(row) for row in graph.connection.execute("SELECT * FROM origin_catalog")}
        require(len(origins) == 8, "P2_GFG_VALIDATION_ORIGIN_COUNT_MISMATCH")
        target_source_ids = {
            str(row["target_state_object_id"])
            for row in manifest["source_graphs"]
        } | {
            str(row["target_probe_object_id"])
            for row in manifest["source_graphs"]
        }
        for origin_id, origin in origins.items():
            bundle_id = str(origin["source_bundle_id"])
            require(bundle_id in source_by_manifest, "P2_GFG_VALIDATION_ORIGIN_BUNDLE_UNKNOWN")
            source = source_by_manifest[bundle_id][1].get(str(origin["source_object_id"]))
            require(source is not None, "P2_GFG_VALIDATION_ORIGIN_SOURCE_MISSING")
            require(source["content_sha256"] == origin["source_content_sha256"], "P2_GFG_VALIDATION_ORIGIN_CONTENT_MISMATCH")
            require(source["role"] == origin["source_role"], "P2_GFG_VALIDATION_ORIGIN_ROLE_MISMATCH")
            if str(origin["source_object_id"]) in target_source_ids:
                target_origin_ids.add(origin_id)
            checks += 4

        prior_block_sha: str | None = None
        for row, block in graph.blocks():
            stage = str(block["stage"])
            stages.append(stage)
            require(hashlib.sha256(canonical_bytes(block)).hexdigest() == row["payload_sha256"], "P2_GFG_VALIDATION_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_block_sha, "P2_GFG_VALIDATION_BLOCK_CHAIN_MISMATCH")
            require(len(block["objects"]) == row["object_count"], "P2_GFG_VALIDATION_OBJECT_COUNT_MISMATCH")
            require(len(block["occurrences"]) == row["occurrence_count"], "P2_GFG_VALIDATION_OCCURRENCE_COUNT_MISMATCH")
            require(sum(len(value["sources"]) for value in block["fact_blocks"]) == row["fact_count"], "P2_GFG_VALIDATION_FACT_COUNT_MISMATCH")
            require(len(block["relations"]) == row["relation_count"], "P2_GFG_VALIDATION_RELATION_COUNT_MISMATCH")
            for value in block["objects"]:
                object_id = str(value["object_id"])
                require(object_id not in known_objects, "P2_GFG_VALIDATION_DUPLICATE_OBJECT")
                require(payload_sha256(value["payload"]) == value["content_sha256"], "P2_GFG_VALIDATION_OBJECT_CONTENT_MISMATCH")
                for reference in _tensor_refs(value["payload"]):
                    _validate_tensor(graph_root, reference)
                    tensor_ref_count += 1
                known_objects[object_id] = value
                object_roles[str(value["role"])] += 1
            for value in block["occurrences"]:
                occurrence_id = str(value["occurrence_id"])
                require(occurrence_id not in known_occurrences, "P2_GFG_VALIDATION_DUPLICATE_OCCURRENCE")
                require(bool(value["transform_reference"]), "P2_GFG_VALIDATION_TRANSFORM_REFERENCE_EMPTY")
                known_occurrences[occurrence_id] = value
                occurrence_types[str(value["occurrence_type"])] += 1
            for value in block["fact_blocks"]:
                fact_id = str(value["fact_block_id"])
                require(fact_id not in known_facts, "P2_GFG_VALIDATION_DUPLICATE_FACT")
                require(value["occurrence_id"] in known_occurrences, "P2_GFG_VALIDATION_FACT_OCCURRENCE_MISSING")
                require(value["outcome"]["object_id"] in known_objects, "P2_GFG_VALIDATION_FACT_OUTCOME_MISSING")
                for source in value["sources"]:
                    require(source["source_id"] in known_objects or source["source_id"] in origins, "P2_GFG_VALIDATION_FACT_SOURCE_MISSING")
                    if source["source_id"] in target_origin_ids:
                        require(stage == "p2_native_target_adjudication", "P2_GFG_VALIDATION_TARGET_USED_BEFORE_ADJUDICATION")
                known_facts[fact_id] = value
                fact_stage[fact_id] = stage
            relations.extend(block["relations"])
            prior_block_sha = row["block_sha256"]
            checks += 6

        require(prior_block_sha == manifest["final_block_sha256"], "P2_GFG_VALIDATION_FINAL_BLOCK_MISMATCH")
        require(stages[-2:] == ["p2_sealed_receiver_response_comparison", "p2_native_target_adjudication"], "P2_GFG_VALIDATION_FINAL_STAGE_ORDER_MISMATCH")
        endpoint_ids = set(known_objects) | set(known_occurrences) | set(known_facts) | set(origins)
        for relation in relations:
            require(relation["source_id"] in endpoint_ids, "P2_GFG_VALIDATION_RELATION_SOURCE_MISSING")
            require(relation["target_id"] in endpoint_ids, "P2_GFG_VALIDATION_RELATION_TARGET_MISSING")
            checks += 2
        incidence = Counter(str(r["target_id"]) for r in relations if r["relation_type"] == "realizes_fact")
        require(set(incidence) == set(known_facts), "P2_GFG_VALIDATION_INCIDENCE_COVERAGE_MISMATCH")
        require(all(count == 1 for count in incidence.values()), "P2_GFG_VALIDATION_INCIDENCE_NOT_EXACT")

        require(occurrence_types["p2_scaled_parameter_displacement_occurrence"] == 10, "P2_GFG_VALIDATION_STATE_OCCURRENCE_COUNT_MISMATCH")
        require(occurrence_types["baseline_probe_occurrence"] == 20, "P2_GFG_VALIDATION_BASELINE_PROBE_COUNT_MISMATCH")
        require(occurrence_types["gated_support_probe_occurrence"] == 100, "P2_GFG_VALIDATION_GATED_PROBE_COUNT_MISMATCH")
        require(occurrence_types["support_metric_derivation_occurrence"] == 10, "P2_GFG_VALIDATION_SUPPORT_DERIVATION_COUNT_MISMATCH")
        require(occurrence_types["p2_central_response_derivation_occurrence"] == 4, "P2_GFG_VALIDATION_RESPONSE_DERIVATION_COUNT_MISMATCH")
        require(occurrence_types["p2_receiver_local_response_comparison_occurrence"] == 1, "P2_GFG_VALIDATION_COMPARISON_COUNT_MISMATCH")
        require(occurrence_types["p2_native_target_adjudication_occurrence"] == 1, "P2_GFG_VALIDATION_ADJUDICATION_COUNT_MISMATCH")
        require(object_roles["validated_p2_local_response_summary"] == 4, "P2_GFG_VALIDATION_RESPONSE_SUMMARY_COUNT_MISMATCH")
        require(object_roles["sealed_validated_p2_response_package"] == 1, "P2_GFG_VALIDATION_SEALED_PACKAGE_COUNT_MISMATCH")
        require(object_roles["p2_native_target_adjudication"] == 1, "P2_GFG_VALIDATION_ADJUDICATION_OBJECT_COUNT_MISMATCH")
        require(len(manifest["state_catalog"]) == 10, "P2_GFG_VALIDATION_STATE_CATALOG_MISMATCH")
        require(len(manifest["probe_catalog"]) == 10, "P2_GFG_VALIDATION_PROBE_CATALOG_MISMATCH")
        require(len(manifest["response_summary_catalog"]) == 4, "P2_GFG_VALIDATION_RESPONSE_CATALOG_MISMATCH")
        adjudication_object = known_objects[str(manifest["adjudication_object_id"])]
        require(adjudication_object["payload"]["adjudication"] == adjudication, "P2_GFG_VALIDATION_ADJUDICATION_PAYLOAD_MISMATCH")
        require(adjudication_object["payload"]["adjudication"]["frozen_outcome"] == "TWO_DIRECTION_RESPONSE_BASIS_INSUFFICIENT", "P2_GFG_VALIDATION_FALSE_SCIENTIFIC_OUTCOME")
        checks += 15
    finally:
        graph.close()

    result_material = {
        "schema": "nanogpt-p2-reciprocal-local-response-gfg-validation-v1",
        "status": "PASS",
        "manifest_sha256": manifest["manifest_sha256"],
        "database_sha256": manifest["database_sha256"],
        "p2_protocol_sha256": manifest["p2_protocol_sha256"],
        "source_graph_manifest_sha256s": sorted(source_by_manifest),
        "pretarget_validation_result_sha256": pretarget["result_sha256"],
        "independent_replay_validation_result_sha256": replay["result_sha256"],
        "adjudication_result_sha256": adjudication["result_sha256"],
        "frozen_outcome": adjudication["frozen_outcome"],
        "counts": manifest["counts"],
        "occurrence_types": dict(sorted(occurrence_types.items())),
        "object_roles": dict(sorted(object_roles.items())),
        "tensor_reference_validation_count": tensor_ref_count,
        "target_origin_count": len(target_origin_ids),
        "target_information_used_before_seal": False,
        "long_horizon_challenge_pairs_resolved": 0,
        "check_count": checks,
    }
    result = {**result_material, "validation_sha256": payload_sha256(result_material)}
    write_json(graph_root / "p2_reciprocal_local_response_gfg_validation.json", result)
    return result


__all__ = ["validate_p2_response_gfg"]
