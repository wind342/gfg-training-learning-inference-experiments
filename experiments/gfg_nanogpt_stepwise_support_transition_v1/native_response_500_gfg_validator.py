from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
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

from .native_response_500_gfg import GRAPH_SCHEMA, _source_catalog


def validate_native_response_500_gfg(
    *,
    graph_root: Path,
    formal_root: Path,
    response_protocol_path: Path,
) -> dict[str, Any]:
    protocol = read_json(response_protocol_path)
    manifest_path = graph_root / "native_response_500_gfg_manifest.json"
    manifest = read_json(manifest_path)
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(material) == manifest["manifest_sha256"], "NATIVE_RESPONSE_500_GFG_MANIFEST_HASH_INVALID")
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "NATIVE_RESPONSE_500_GFG_DATABASE_HASH_MISMATCH")
    require(manifest["sample_count"] == manifest["response_count"] == 500, "NATIVE_RESPONSE_500_GFG_MANIFEST_COUNT_INVALID")
    sources = _source_catalog(formal_root, list(protocol["receivers"]))
    expected_source_objects: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sources.values():
        bundle_id = str(source["manifest"]["manifest_sha256"])
        for key in ("prestate", "native_update", "target_state", "target_probe"):
            value = source[key]
            expected_source_objects[(bundle_id, str(value["object_id"]))] = value

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    checks = 0
    object_roles = Counter()
    occurrence_types = Counter()
    relation_types = Counter()
    known_objects: dict[str, str] = {}
    known_occurrences: set[str] = set()
    known_facts: set[str] = set()
    prior_block = None
    tensor_count = 0
    origins = {row["origin_id"]: dict(row) for row in connection.execute("SELECT * FROM origin_catalog")}
    try:
        metadata = {row["key"]: json.loads(row["value_json"]) for row in connection.execute("SELECT key,value_json FROM metadata")}
        require(metadata["schema"] == GRAPH_SCHEMA, "NATIVE_RESPONSE_500_GFG_SCHEMA_INVALID")
        require(metadata["scope_id"] == protocol["protocol_id"], "NATIVE_RESPONSE_500_GFG_SCOPE_INVALID")
        checks += 2
        for origin in origins.values():
            key = (str(origin["source_bundle_id"]), str(origin["source_object_id"]))
            source = expected_source_objects.get(key)
            require(source is not None, "NATIVE_RESPONSE_500_GFG_ORIGIN_SOURCE_MISSING")
            require(source["content_sha256"] == origin["source_content_sha256"], "NATIVE_RESPONSE_500_GFG_ORIGIN_CONTENT_MISMATCH")
            require(source["role"] == origin["source_role"], "NATIVE_RESPONSE_500_GFG_ORIGIN_ROLE_MISMATCH")
            checks += 3

        for row in connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal"):
            block = json.loads(zlib.decompress(row["payload_zlib"]))
            raw = canonical_bytes(block)
            require(hashlib.sha256(raw).hexdigest() == row["payload_sha256"], "NATIVE_RESPONSE_500_GFG_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_block, "NATIVE_RESPONSE_500_GFG_BLOCK_CHAIN_MISMATCH")
            block_material = {
                "block_ordinal": row["block_ordinal"],
                "optimizer_step": row["optimizer_step"],
                "payload_sha256": row["payload_sha256"],
                "prior_block_sha256": row["prior_block_sha256"],
                "scope_id": metadata["scope_id"],
                "stage": row["stage"],
            }
            require(payload_sha256(block_material) == row["block_sha256"], "NATIVE_RESPONSE_500_GFG_BLOCK_HASH_MISMATCH")
            require(row["block_id"] == "block_" + row["block_sha256"], "NATIVE_RESPONSE_500_GFG_BLOCK_ID_MISMATCH")
            require(len(block["objects"]) == row["object_count"], "NATIVE_RESPONSE_500_GFG_OBJECT_COUNT_MISMATCH")
            require(len(block["occurrences"]) == row["occurrence_count"], "NATIVE_RESPONSE_500_GFG_OCCURRENCE_COUNT_MISMATCH")
            require(len(block["relations"]) == row["relation_count"], "NATIVE_RESPONSE_500_GFG_RELATION_COUNT_MISMATCH")
            require(sum(len(fact["sources"]) for fact in block["fact_blocks"]) == row["fact_count"], "NATIVE_RESPONSE_500_GFG_FACT_COUNT_MISMATCH")
            prior_block = row["block_sha256"]
            checks += 8

            block_objects: dict[str, str] = {}
            for value in block["objects"]:
                object_id = str(value["object_id"])
                require(object_id not in known_objects and object_id not in block_objects, "NATIVE_RESPONSE_500_GFG_DUPLICATE_OBJECT")
                require(payload_sha256(value["payload"]) == value["content_sha256"], "NATIVE_RESPONSE_500_GFG_OBJECT_CONTENT_INVALID")
                block_objects[object_id] = str(value["content_sha256"])
                object_roles[str(value["role"])] += 1
                if value["object_kind"] == "content_addressed_tensor":
                    reference = value["payload"]
                    path = graph_root / str(reference["locator"])
                    require(path.is_file(), "NATIVE_RESPONSE_500_GFG_TENSOR_MISSING")
                    require(file_sha256(path) == reference["file_sha256"], "NATIVE_RESPONSE_500_GFG_TENSOR_FILE_HASH_MISMATCH")
                    array = np.load(path, allow_pickle=False, mmap_mode="r")
                    require(list(array.shape) == list(reference["shape"]), "NATIVE_RESPONSE_500_GFG_TENSOR_SHAPE_MISMATCH")
                    require(str(array.dtype) == str(reference["dtype"]), "NATIVE_RESPONSE_500_GFG_TENSOR_DTYPE_MISMATCH")
                    require(hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest() == reference["raw_tensor_sha256"], "NATIVE_RESPONSE_500_GFG_TENSOR_RAW_HASH_MISMATCH")
                    tensor_count += 1
                    checks += 5

            block_occurrences = set()
            for value in block["occurrences"]:
                occurrence_id = str(value["occurrence_id"])
                require(occurrence_id not in known_occurrences and occurrence_id not in block_occurrences, "NATIVE_RESPONSE_500_GFG_DUPLICATE_OCCURRENCE")
                block_occurrences.add(occurrence_id)
                occurrence_types[str(value["occurrence_type"])] += 1
                checks += 1

            available_objects = {**known_objects, **block_objects}
            available_occurrences = known_occurrences | block_occurrences
            available_sources = set(available_objects) | set(origins)
            block_facts = set()
            for fact in block["fact_blocks"]:
                fact_id = str(fact["fact_block_id"])
                require(fact_id not in known_facts and fact_id not in block_facts, "NATIVE_RESPONSE_500_GFG_DUPLICATE_FACT")
                require(str(fact["occurrence_id"]) in available_occurrences, "NATIVE_RESPONSE_500_GFG_FACT_OCCURRENCE_MISSING")
                outcome = fact["outcome"]
                require(str(outcome["object_id"]) in available_objects, "NATIVE_RESPONSE_500_GFG_FACT_OUTCOME_MISSING")
                require(available_objects[str(outcome["object_id"])] == outcome["content_sha256"], "NATIVE_RESPONSE_500_GFG_FACT_OUTCOME_HASH_MISMATCH")
                for source in fact["sources"]:
                    source_id = str(source["source_id"])
                    require(source_id in available_sources, "NATIVE_RESPONSE_500_GFG_FACT_SOURCE_MISSING")
                    expected_sha = (
                        payload_sha256(json.loads(origins[source_id]["payload_json"]))
                        if source_id in origins
                        else available_objects[source_id]
                    )
                    require(expected_sha == source["content_sha256"], "NATIVE_RESPONSE_500_GFG_FACT_SOURCE_HASH_MISMATCH")
                block_facts.add(fact_id)
                checks += 4 + 2 * len(fact["sources"])

            relation_endpoint_ids = available_sources | available_occurrences | known_facts | block_facts
            for relation in block["relations"]:
                require(str(relation["source_id"]) in relation_endpoint_ids, "NATIVE_RESPONSE_500_GFG_RELATION_SOURCE_MISSING")
                require(str(relation["target_id"]) in relation_endpoint_ids, "NATIVE_RESPONSE_500_GFG_RELATION_TARGET_MISSING")
                relation_types[str(relation["relation_type"])] += 1
                checks += 2
            known_objects.update(block_objects)
            known_occurrences.update(block_occurrences)
            known_facts.update(block_facts)
    finally:
        connection.close()

    require(prior_block == manifest["final_block_sha256"], "NATIVE_RESPONSE_500_GFG_FINAL_BLOCK_MISMATCH")
    require(object_roles["response_augmented_one_step_modeling_record"] == 500, "NATIVE_RESPONSE_500_GFG_SAMPLE_ROLE_COUNT_INVALID")
    require(object_roles["validated_receiver_conditioned_native_response"] == 500, "NATIVE_RESPONSE_500_GFG_RESPONSE_ROLE_COUNT_INVALID")
    require(occurrence_types["native_direction_central_response_derivation_occurrence"] == 500, "NATIVE_RESPONSE_500_GFG_RESPONSE_OCCURRENCE_COUNT_INVALID")
    require(occurrence_types["response_augmented_one_step_record_assembly_occurrence"] == 500, "NATIVE_RESPONSE_500_GFG_SAMPLE_OCCURRENCE_COUNT_INVALID")
    require(set(manifest["sample_catalog"]) == set(manifest["response_catalog"]) == {row["sample_id"] for row in protocol["receivers"]}, "NATIVE_RESPONSE_500_GFG_CATALOG_IDENTITY_INVALID")
    require(all(value in known_objects for value in manifest["sample_catalog"].values()), "NATIVE_RESPONSE_500_GFG_SAMPLE_CATALOG_TARGET_MISSING")
    require(all(value in known_objects for value in manifest["response_catalog"].values()), "NATIVE_RESPONSE_500_GFG_RESPONSE_CATALOG_TARGET_MISSING")
    result_material = {
        "schema": "nanogpt-native-direction-response-500-gfg-validation-v1",
        "status": "PASS",
        "graph_schema": GRAPH_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "database_sha256": manifest["database_sha256"],
        "counts": manifest["counts"],
        "sample_count": 500,
        "response_count": 500,
        "origin_count": len(origins),
        "tensor_reference_validation_count": tensor_count,
        "object_roles": dict(sorted(object_roles.items())),
        "occurrence_types": dict(sorted(occurrence_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
        "all_source_origins_exact": True,
        "all_blocks_and_relations_exact": True,
        "all_tensor_references_exact": True,
        "pretarget_response_seal_preserved": True,
        "check_count": checks,
    }
    result = {**result_material, "validation_sha256": payload_sha256(result_material)}
    write_json(graph_root / "native_response_500_gfg_validation.json", result)
    return result


__all__ = ["validate_native_response_500_gfg"]
