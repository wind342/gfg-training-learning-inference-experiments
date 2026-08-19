from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    canonical_bytes,
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import TrainingGFG
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import SupportGFG, _source_objects_for_step

from .reciprocal import RECIPROCAL_BRANCHES
from .reciprocal_gfg import GRAPH_SCHEMA


def _main_object_index(database: Path) -> dict[str, dict[str, Any]]:
    graph = SupportGFG(database)
    result: dict[str, dict[str, Any]] = {}
    try:
        for _row, block in graph.blocks():
            for value in block["objects"]:
                result[str(value["object_id"])] = value
    finally:
        graph.close()
    return result


def validate_reciprocal_gfg(
    *,
    graph_root: Path,
    evidence_root: Path,
    formal_root: Path,
    source_root: Path,
    reciprocal_protocol_path: Path,
) -> dict[str, Any]:
    manifest_path = graph_root / "reciprocal_matched_pair_gfg_manifest.json"
    manifest = read_json(manifest_path)
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(material) == manifest["manifest_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_MANIFEST_HASH_MISMATCH")
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_DATABASE_HASH_MISMATCH")
    protocol = read_json(reciprocal_protocol_path)
    require(file_sha256(reciprocal_protocol_path) == manifest["reciprocal_protocol_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_PROTOCOL_HASH_MISMATCH")
    endpoints = {str(row["label"]): row for row in protocol["endpoints"]}
    evidence_validation = read_json(evidence_root / "reciprocal_pair_validation.json")
    require(evidence_validation["validation_sha256"] == manifest["evidence_validation_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_EVIDENCE_REFERENCE_MISMATCH")
    expected_catalog = {
        f"{label}:{horizon}:{branch}"
        for label in ("A", "B")
        for horizon in protocol["horizons"]
        for branch in RECIPROCAL_BRANCHES
    }
    require(set(manifest["state_catalog"]) == expected_catalog, "SST_RECIPROCAL_GFG_VALIDATION_STATE_CATALOG_INVALID")

    source_graphs = {
        str(endpoint["source_bundle_id"]): TrainingGFG(source_root / str(endpoint["source_bundle_id"]) / "participant_gfg.sqlite3")
        for endpoint in endpoints.values()
    }
    main_graphs: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    for endpoint in endpoints.values():
        entry_id = str(endpoint["entry_id"])
        main_manifest = read_json(formal_root / entry_id / "stepwise_support_transition_gfg_manifest.json")
        main_graphs[str(main_manifest["manifest_sha256"])] = (
            main_manifest,
            _main_object_index(formal_root / entry_id / str(main_manifest["database"])),
        )

    graph = SupportGFG(database)
    checks = 0
    occurrence_types: Counter[str] = Counter()
    object_roles: Counter[str] = Counter()
    known_objects: dict[str, dict[str, Any]] = {}
    known_occurrences: dict[str, dict[str, Any]] = {}
    known_facts: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []
    try:
        require(graph.metadata("schema") == GRAPH_SCHEMA, "SST_RECIPROCAL_GFG_VALIDATION_SCHEMA_INVALID")
        origins = {str(row["origin_id"]): dict(row) for row in graph.connection.execute("SELECT * FROM origin_catalog")}
        source_cache: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for origin in origins.values():
            bundle_id = str(origin["source_bundle_id"])
            require(bundle_id in source_graphs, f"SST_RECIPROCAL_GFG_VALIDATION_ORIGIN_BUNDLE_UNKNOWN:{bundle_id}")
            cache_key = (bundle_id, int(origin["source_optimizer_step"]))
            if cache_key not in source_cache:
                source_cache[cache_key] = _source_objects_for_step(source_graphs[bundle_id], cache_key[1])
            source = source_cache[cache_key].get(str(origin["source_object_id"]))
            require(source is not None, "SST_RECIPROCAL_GFG_VALIDATION_ORIGIN_SOURCE_MISSING")
            require(source["content_sha256"] == origin["source_content_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_ORIGIN_HASH_MISMATCH")
            require(source["role"] == origin["source_role"], "SST_RECIPROCAL_GFG_VALIDATION_ORIGIN_ROLE_MISMATCH")
            checks += 3

        prior_block: str | None = None
        for row, block in graph.blocks():
            require(hashlib.sha256(canonical_bytes(block)).hexdigest() == row["payload_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_block, "SST_RECIPROCAL_GFG_VALIDATION_BLOCK_CHAIN_MISMATCH")
            require(len(block["objects"]) == row["object_count"], "SST_RECIPROCAL_GFG_VALIDATION_OBJECT_COUNT_MISMATCH")
            require(len(block["occurrences"]) == row["occurrence_count"], "SST_RECIPROCAL_GFG_VALIDATION_OCCURRENCE_COUNT_MISMATCH")
            require(len(block["relations"]) == row["relation_count"], "SST_RECIPROCAL_GFG_VALIDATION_RELATION_COUNT_MISMATCH")
            require(sum(len(value["sources"]) for value in block["fact_blocks"]) == row["fact_count"], "SST_RECIPROCAL_GFG_VALIDATION_FACT_COUNT_MISMATCH")
            for value in block["objects"]:
                object_id = str(value["object_id"])
                require(object_id not in known_objects, "SST_RECIPROCAL_GFG_VALIDATION_DUPLICATE_OBJECT")
                require(payload_sha256(value["payload"]) == value["content_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_OBJECT_HASH_MISMATCH")
                known_objects[object_id] = value
                object_roles[str(value["role"])] += 1
                if value["object_kind"] == "content_addressed_tensor":
                    payload = value["payload"]
                    path = graph_root / str(payload["locator"])
                    require(path.is_file(), "SST_RECIPROCAL_GFG_VALIDATION_TENSOR_MISSING")
                    require(file_sha256(path) == payload["file_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_TENSOR_FILE_HASH_MISMATCH")
                    array = np.load(path, allow_pickle=False, mmap_mode="r")
                    require(list(array.shape) == payload["shape"], "SST_RECIPROCAL_GFG_VALIDATION_TENSOR_SHAPE_MISMATCH")
                    require(str(array.dtype) == payload["dtype"], "SST_RECIPROCAL_GFG_VALIDATION_TENSOR_DTYPE_MISMATCH")
                    require(hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest() == payload["raw_tensor_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_TENSOR_RAW_HASH_MISMATCH")
                    checks += 5
                if value["object_kind"] == "GeneratedOrigin" and value["role"] == "generated_main_training_state_origin":
                    payload = value["payload"]
                    manifest_sha = str(payload["source_graph_manifest_sha256"])
                    require(manifest_sha in main_graphs, "SST_RECIPROCAL_GFG_VALIDATION_MAIN_ORIGIN_GRAPH_UNKNOWN")
                    source = main_graphs[manifest_sha][1].get(str(payload["source_object_id"]))
                    require(source is not None, "SST_RECIPROCAL_GFG_VALIDATION_MAIN_ORIGIN_SOURCE_MISSING")
                    require(source["content_sha256"] == payload["source_content_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_MAIN_ORIGIN_HASH_MISMATCH")
                    checks += 2
            for occurrence in block["occurrences"]:
                occurrence_id = str(occurrence["occurrence_id"])
                require(occurrence_id not in known_occurrences, "SST_RECIPROCAL_GFG_VALIDATION_DUPLICATE_OCCURRENCE")
                require(bool(occurrence["transform_reference"]), "SST_RECIPROCAL_GFG_VALIDATION_TRANSFORM_MISSING")
                known_occurrences[occurrence_id] = occurrence
                occurrence_types[str(occurrence["occurrence_type"])] += 1
            for fact in block["fact_blocks"]:
                fact_id = str(fact["fact_block_id"])
                require(fact_id not in known_facts, "SST_RECIPROCAL_GFG_VALIDATION_DUPLICATE_FACT")
                require(str(fact["occurrence_id"]) in known_occurrences, "SST_RECIPROCAL_GFG_VALIDATION_FACT_OCCURRENCE_MISSING")
                outcome_id = str(fact["outcome"]["object_id"])
                require(outcome_id in known_objects, "SST_RECIPROCAL_GFG_VALIDATION_FACT_OUTCOME_MISSING")
                require(known_objects[outcome_id]["content_sha256"] == fact["outcome"]["content_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_FACT_OUTCOME_HASH_MISMATCH")
                require(bool(fact["sources"]), "SST_RECIPROCAL_GFG_VALIDATION_FACT_SOURCES_EMPTY")
                for source in fact["sources"]:
                    source_id = str(source["source_id"])
                    source_hash = known_objects.get(source_id, {}).get("content_sha256")
                    if source_hash is None and source_id in origins:
                        source_hash = payload_sha256(json.loads(origins[source_id]["payload_json"]))
                    require(source_hash == source["content_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_FACT_SOURCE_HASH_MISMATCH")
                    require(bool(source["relation_role"]), "SST_RECIPROCAL_GFG_VALIDATION_RELATION_ROLE_EMPTY")
                known_facts[fact_id] = fact
            relations.extend(block["relations"])
            prior_block = str(row["block_sha256"])
            checks += 7

        require(prior_block == manifest["final_block_sha256"], "SST_RECIPROCAL_GFG_VALIDATION_FINAL_BLOCK_MISMATCH")
        valid_types = {"generated_origin_dependency", "program_order", "reads_from", "realizes_fact"}
        realizes: Counter[str] = Counter()
        reads: Counter[str] = Counter()
        for relation in relations:
            relation_type = str(relation["relation_type"])
            require(relation_type in valid_types, "SST_RECIPROCAL_GFG_VALIDATION_RELATION_TYPE_INVALID")
            if relation_type == "realizes_fact":
                require(str(relation["source_id"]) in known_occurrences, "SST_RECIPROCAL_GFG_VALIDATION_INCIDENCE_SOURCE_MISSING")
                require(str(relation["target_id"]) in known_facts, "SST_RECIPROCAL_GFG_VALIDATION_INCIDENCE_TARGET_MISSING")
                realizes[str(relation["target_id"])] += 1
            elif relation_type == "reads_from":
                require(str(relation["target_id"]) in known_occurrences, "SST_RECIPROCAL_GFG_VALIDATION_READ_TARGET_MISSING")
                reads[str(relation["target_id"])] += 1
            elif relation_type == "program_order":
                require(str(relation["source_id"]) in known_occurrences and str(relation["target_id"]) in known_occurrences, "SST_RECIPROCAL_GFG_VALIDATION_ORDER_ENDPOINT_MISSING")
        require(set(realizes) == set(known_facts) and all(value == 1 for value in realizes.values()), "SST_RECIPROCAL_GFG_VALIDATION_INCIDENCE_NOT_EXACT")
        facts_by_occurrence: Counter[str] = Counter(str(value["occurrence_id"]) for value in known_facts.values())
        require(set(facts_by_occurrence) == set(known_occurrences), "SST_RECIPROCAL_GFG_VALIDATION_ZERO_FACT_OCCURRENCE")
        require(occurrence_types["actual_native_training_step_occurrence"] == 2, "SST_RECIPROCAL_GFG_VALIDATION_NATIVE_STEP_COUNT_INVALID")
        require(occurrence_types["reciprocal_branch_state_establishment_occurrence"] == 14, "SST_RECIPROCAL_GFG_VALIDATION_BRANCH_COUNT_INVALID")
        require(occurrence_types["reciprocal_branch_horizon_state_materialization_occurrence"] == 70, "SST_RECIPROCAL_GFG_VALIDATION_HORIZON_STATE_COUNT_INVALID")
        require(occurrence_types["actual_reciprocal_branch_continuation_step_occurrence"] == 1386, "SST_RECIPROCAL_GFG_VALIDATION_CONTINUATION_COUNT_INVALID")
        require(occurrence_types["reciprocal_branch_response_contrast_occurrence"] == 10, "SST_RECIPROCAL_GFG_VALIDATION_RESPONSE_COUNT_INVALID")
        require(occurrence_types["reciprocal_cross_run_adjudication_occurrence"] == 1, "SST_RECIPROCAL_GFG_VALIDATION_ADJUDICATION_COUNT_INVALID")
        require(object_roles["reciprocal_matched_pair_adjudication"] == 1, "SST_RECIPROCAL_GFG_VALIDATION_ADJUDICATION_OBJECT_INVALID")
        require(manifest["counts"]["blocks"] == graph.connection.execute("SELECT COUNT(*) FROM graph_blocks").fetchone()[0], "SST_RECIPROCAL_GFG_VALIDATION_MANIFEST_BLOCK_COUNT_INVALID")
        result_material = {
            "schema": "nanogpt-reciprocal-matched-pair-gfg-validation-v1",
            "status": "PASS",
            "checks": checks,
            "database_sha256": file_sha256(database),
            "manifest_sha256": manifest["manifest_sha256"],
            "evidence_validation_sha256": evidence_validation["validation_sha256"],
            "counts": {
                "blocks": manifest["counts"]["blocks"],
                "objects": len(known_objects),
                "occurrences": len(known_occurrences),
                "atomic_fact_blocks": len(known_facts),
                "source_incidences": manifest["counts"]["facts"],
                "origins": len(origins),
                "relations": len(relations),
                "tensor_payloads": manifest["counts"]["tensor_payloads"],
            },
            "occurrence_type_counts": dict(sorted(occurrence_types.items())),
            "state_catalog_exact": True,
            "realizes_fact_exact": True,
            "source_origin_replay_exact": True,
            "main_stepwise_origin_replay_exact": True,
            "future_information_used": False,
            "adjudication": evidence_validation["adjudication"],
        }
        result = {**result_material, "validation_sha256": payload_sha256(result_material)}
        write_json(graph_root / "reciprocal_matched_pair_gfg_validation.json", result)
        return result
    finally:
        graph.close()
        for source_graph in source_graphs.values():
            source_graph.close()
