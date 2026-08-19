from __future__ import annotations

from collections import Counter
import hashlib
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
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import SupportGFG

from .local_response_gfg_validator import _tensor_refs
from .reciprocal_gfg_validator import _main_object_index
from .response_transport_gfg import GRAPH_SCHEMA


def _validate_tensor(graph_root: Path, reference: dict[str, Any]) -> None:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "SST_RESPONSE_TRANSPORT_GFG_TENSOR_LOCATOR_INVALID")
    path = graph_root / locator
    require(path.is_file(), f"SST_RESPONSE_TRANSPORT_GFG_TENSOR_MISSING:{path}")
    require(file_sha256(path) == reference["file_sha256"], "SST_RESPONSE_TRANSPORT_GFG_TENSOR_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "SST_RESPONSE_TRANSPORT_GFG_TENSOR_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "SST_RESPONSE_TRANSPORT_GFG_TENSOR_DTYPE_MISMATCH")
    require(
        hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
        == reference["raw_tensor_sha256"],
        "SST_RESPONSE_TRANSPORT_GFG_TENSOR_RAW_HASH_MISMATCH",
    )


def validate_response_transport_gfg(
    *,
    evidence_root: Path,
    graph_root: Path,
    protocol_path: Path,
    a_skip_graph_root: Path,
    b_skip_graph_root: Path,
    a_native_full_graph_root: Path,
    b_native_full_graph_root: Path,
) -> dict[str, Any]:
    manifest = read_json(graph_root / "response_transport_cross_gfg_manifest.json")
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(material) == manifest["manifest_sha256"], "SST_RESPONSE_TRANSPORT_GFG_MANIFEST_HASH_MISMATCH")
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "SST_RESPONSE_TRANSPORT_GFG_DATABASE_HASH_MISMATCH")
    require(file_sha256(protocol_path) == manifest["response_transport_protocol_sha256"], "SST_RESPONSE_TRANSPORT_GFG_PROTOCOL_HASH_MISMATCH")
    evidence = read_json(evidence_root / "response_transport_cross_validation.json")
    require(evidence["validation_sha256"] == manifest["evidence_validation_sha256"], "SST_RESPONSE_TRANSPORT_GFG_EVIDENCE_HASH_MISMATCH")
    graph_roots = {
        "A:skip": a_skip_graph_root,
        "A:native_full": a_native_full_graph_root,
        "B:skip": b_skip_graph_root,
        "B:native_full": b_native_full_graph_root,
    }
    source_graphs: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    for label, root in graph_roots.items():
        source_manifest = read_json(root / "local_response_jk_gfg_manifest.json")
        source_validation = read_json(root / "local_response_jk_gfg_validation.json")
        require(source_validation["status"] == "PASS", f"SST_RESPONSE_TRANSPORT_GFG_SOURCE_NOT_VALIDATED:{label}")
        require(
            manifest["input_graph_manifests"][label] == source_manifest["manifest_sha256"],
            f"SST_RESPONSE_TRANSPORT_GFG_SOURCE_MANIFEST_MISMATCH:{label}",
        )
        source_graphs[source_manifest["manifest_sha256"]] = (
            source_manifest,
            _main_object_index(root / str(source_manifest["database"])),
        )

    graph = SupportGFG(database)
    known_objects: dict[str, dict[str, Any]] = {}
    known_occurrences: dict[str, dict[str, Any]] = {}
    known_facts: dict[str, dict[str, Any]] = {}
    occurrence_types: Counter[str] = Counter()
    object_roles: Counter[str] = Counter()
    relations: list[dict[str, Any]] = []
    tensor_ref_count = 0
    checks = 0
    try:
        require(graph.metadata("schema") == GRAPH_SCHEMA, "SST_RESPONSE_TRANSPORT_GFG_SCHEMA_INVALID")
        origins = {str(row["origin_id"]): dict(row) for row in graph.connection.execute("SELECT * FROM origin_catalog")}
        require(len(origins) == 8, "SST_RESPONSE_TRANSPORT_GFG_ORIGIN_COUNT_MISMATCH")
        for origin in origins.values():
            bundle_id = str(origin["source_bundle_id"])
            require(bundle_id in source_graphs, "SST_RESPONSE_TRANSPORT_GFG_ORIGIN_BUNDLE_UNKNOWN")
            source = source_graphs[bundle_id][1].get(str(origin["source_object_id"]))
            require(source is not None, "SST_RESPONSE_TRANSPORT_GFG_ORIGIN_SOURCE_MISSING")
            require(source["content_sha256"] == origin["source_content_sha256"], "SST_RESPONSE_TRANSPORT_GFG_ORIGIN_CONTENT_MISMATCH")
            require(source["role"] == origin["source_role"] == "validated_local_response_summary", "SST_RESPONSE_TRANSPORT_GFG_ORIGIN_ROLE_MISMATCH")
            checks += 4

        prior_block_sha: str | None = None
        for row, block in graph.blocks():
            require(hashlib.sha256(canonical_bytes(block)).hexdigest() == row["payload_sha256"], "SST_RESPONSE_TRANSPORT_GFG_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_block_sha, "SST_RESPONSE_TRANSPORT_GFG_BLOCK_CHAIN_MISMATCH")
            require(len(block["objects"]) == row["object_count"], "SST_RESPONSE_TRANSPORT_GFG_OBJECT_COUNT_MISMATCH")
            require(len(block["occurrences"]) == row["occurrence_count"], "SST_RESPONSE_TRANSPORT_GFG_OCCURRENCE_COUNT_MISMATCH")
            require(sum(len(value["sources"]) for value in block["fact_blocks"]) == row["fact_count"], "SST_RESPONSE_TRANSPORT_GFG_FACT_COUNT_MISMATCH")
            require(len(block["relations"]) == row["relation_count"], "SST_RESPONSE_TRANSPORT_GFG_RELATION_COUNT_MISMATCH")
            for value in block["objects"]:
                object_id = str(value["object_id"])
                require(object_id not in known_objects, "SST_RESPONSE_TRANSPORT_GFG_DUPLICATE_OBJECT")
                require(payload_sha256(value["payload"]) == value["content_sha256"], "SST_RESPONSE_TRANSPORT_GFG_OBJECT_CONTENT_MISMATCH")
                for reference in _tensor_refs(value["payload"]):
                    _validate_tensor(graph_root, reference)
                    tensor_ref_count += 1
                known_objects[object_id] = value
                object_roles[str(value["role"])] += 1
            for value in block["occurrences"]:
                occurrence_id = str(value["occurrence_id"])
                require(occurrence_id not in known_occurrences, "SST_RESPONSE_TRANSPORT_GFG_DUPLICATE_OCCURRENCE")
                known_occurrences[occurrence_id] = value
                occurrence_types[str(value["occurrence_type"])] += 1
            for value in block["fact_blocks"]:
                fact_id = str(value["fact_block_id"])
                require(fact_id not in known_facts, "SST_RESPONSE_TRANSPORT_GFG_DUPLICATE_FACT")
                require(value["occurrence_id"] in known_occurrences, "SST_RESPONSE_TRANSPORT_GFG_FACT_OCCURRENCE_MISSING")
                require(value["outcome"]["object_id"] in known_objects, "SST_RESPONSE_TRANSPORT_GFG_FACT_OUTCOME_MISSING")
                for source in value["sources"]:
                    require(source["source_id"] in known_objects or source["source_id"] in origins, "SST_RESPONSE_TRANSPORT_GFG_FACT_SOURCE_MISSING")
                known_facts[fact_id] = value
            relations.extend(block["relations"])
            checks += 6
            prior_block_sha = row["block_sha256"]

        require(prior_block_sha == manifest["final_block_sha256"], "SST_RESPONSE_TRANSPORT_GFG_FINAL_BLOCK_MISMATCH")
        endpoints = set(known_objects) | set(known_occurrences) | set(known_facts) | set(origins)
        for relation in relations:
            require(relation["source_id"] in endpoints, "SST_RESPONSE_TRANSPORT_GFG_RELATION_SOURCE_MISSING")
            require(relation["target_id"] in endpoints, "SST_RESPONSE_TRANSPORT_GFG_RELATION_TARGET_MISSING")
            checks += 2
        incidence = Counter(
            str(relation["target_id"])
            for relation in relations
            if relation["relation_type"] == "realizes_fact"
        )
        require(set(incidence) == set(known_facts), "SST_RESPONSE_TRANSPORT_GFG_INCIDENCE_COVERAGE_MISMATCH")
        require(all(count == 1 for count in incidence.values()), "SST_RESPONSE_TRANSPORT_GFG_INCIDENCE_NOT_EXACT")
        require(occurrence_types["response_state_transport_computation_occurrence"] == 172, "SST_RESPONSE_TRANSPORT_GFG_TRANSPORT_OCCURRENCE_COUNT_MISMATCH")
        require(occurrence_types["response_transport_context_adjudication_occurrence"] == 4, "SST_RESPONSE_TRANSPORT_GFG_CONTEXT_OCCURRENCE_COUNT_MISMATCH")
        require(occurrence_types["response_transport_overall_adjudication_occurrence"] == 1, "SST_RESPONSE_TRANSPORT_GFG_OVERALL_OCCURRENCE_COUNT_MISMATCH")
        require(object_roles["validated_response_transport_context_adjudication"] == 4, "SST_RESPONSE_TRANSPORT_GFG_CONTEXT_OBJECT_COUNT_MISMATCH")
        require(object_roles["validated_response_transport_overall_adjudication"] == 1, "SST_RESPONSE_TRANSPORT_GFG_OVERALL_OBJECT_COUNT_MISMATCH")
        require(len(manifest["transport_catalog"]) == 172, "SST_RESPONSE_TRANSPORT_GFG_TRANSPORT_CATALOG_COUNT_MISMATCH")
        require(set(manifest["context_catalog"]) == {"A:A", "A:B", "B:A", "B:B"}, "SST_RESPONSE_TRANSPORT_GFG_CONTEXT_CATALOG_MISMATCH")
        require(manifest["overall_adjudication_object_id"] in known_objects, "SST_RESPONSE_TRANSPORT_GFG_OVERALL_OBJECT_MISSING")
        require(tensor_ref_count == 172, "SST_RESPONSE_TRANSPORT_GFG_TENSOR_REFERENCE_COUNT_MISMATCH")
        checks += 10
    finally:
        graph.close()

    validation_material = {
        "schema": "nanogpt-response-transport-cross-gfg-validation-v1",
        "status": "PASS",
        "manifest_sha256": manifest["manifest_sha256"],
        "database_sha256": manifest["database_sha256"],
        "evidence_validation_sha256": evidence["validation_sha256"],
        "counts": manifest["counts"],
        "occurrence_types": dict(sorted(occurrence_types.items())),
        "object_roles": dict(sorted(object_roles.items())),
        "tensor_reference_validation_count": tensor_ref_count,
        "check_count": checks,
        "mechanical_scientific_outcome": evidence["mechanical_scientific_outcome"],
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    result = {**validation_material, "validation_sha256": payload_sha256(validation_material)}
    write_json(graph_root / "response_transport_cross_gfg_validation.json", result)
    return result


__all__ = ["validate_response_transport_gfg"]
