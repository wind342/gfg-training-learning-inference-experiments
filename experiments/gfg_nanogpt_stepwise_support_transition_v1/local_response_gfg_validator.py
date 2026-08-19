from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Iterable

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

from .local_response_gfg import GRAPH_SCHEMA
from .local_response import LOCAL_RESPONSE_BRANCHES, LOCAL_RESPONSE_TRANSPORT_BRANCHES
from .reciprocal_gfg_validator import _main_object_index


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
    require(locator.startswith("tensor-objects/"), "SST_LOCAL_RESPONSE_GFG_VALIDATION_TENSOR_LOCATOR_INVALID")
    path = graph_root / locator
    require(path.is_file(), f"SST_LOCAL_RESPONSE_GFG_VALIDATION_TENSOR_MISSING:{path}")
    require(file_sha256(path) == reference["file_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_TENSOR_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "SST_LOCAL_RESPONSE_GFG_VALIDATION_TENSOR_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "SST_LOCAL_RESPONSE_GFG_VALIDATION_TENSOR_DTYPE_MISMATCH")
    require(hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest() == reference["raw_tensor_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_TENSOR_RAW_HASH_MISMATCH")


def validate_local_response_gfg(
    *,
    graph_root: Path,
    evidence_root: Path,
    reciprocal_graph_root: Path,
    local_response_protocol_path: Path,
) -> dict[str, Any]:
    manifest_path = graph_root / "local_response_jk_gfg_manifest.json"
    manifest = read_json(manifest_path)
    protocol = read_json(local_response_protocol_path)
    branches = tuple(str(value) for value in protocol["branches"])
    require(
        branches in {LOCAL_RESPONSE_BRANCHES, LOCAL_RESPONSE_TRANSPORT_BRANCHES},
        "SST_LOCAL_RESPONSE_GFG_VALIDATION_BRANCHES_INVALID",
    )
    manifest_material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(manifest_material) == manifest["manifest_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_MANIFEST_HASH_MISMATCH")
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_DATABASE_HASH_MISMATCH")
    require(file_sha256(local_response_protocol_path) == manifest["local_response_protocol_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_PROTOCOL_HASH_MISMATCH")
    evidence_validation = read_json(evidence_root / "local_response_jk_validation.json")
    require(evidence_validation["validation_sha256"] == manifest["evidence_validation_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_EVIDENCE_HASH_MISMATCH")
    prior_manifest = read_json(reciprocal_graph_root / "reciprocal_matched_pair_gfg_manifest.json")
    require(prior_manifest["manifest_sha256"] == manifest["prior_reciprocal_manifest_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_PRIOR_MANIFEST_MISMATCH")
    prior_objects = _main_object_index(reciprocal_graph_root / str(prior_manifest["database"]))

    graph = SupportGFG(database)
    known_objects: dict[str, dict[str, Any]] = {}
    known_occurrences: dict[str, dict[str, Any]] = {}
    known_facts: dict[str, dict[str, Any]] = {}
    occurrence_types: Counter[str] = Counter()
    object_roles: Counter[str] = Counter()
    relations: list[dict[str, Any]] = []
    checks = 0
    tensor_ref_count = 0
    try:
        require(graph.metadata("schema") == GRAPH_SCHEMA, "SST_LOCAL_RESPONSE_GFG_VALIDATION_SCHEMA_INVALID")
        origins = {str(row["origin_id"]): dict(row) for row in graph.connection.execute("SELECT * FROM origin_catalog")}
        for origin in origins.values():
            require(origin["source_bundle_id"] == prior_manifest["manifest_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_ORIGIN_BUNDLE_MISMATCH")
            source = prior_objects.get(str(origin["source_object_id"]))
            require(source is not None, "SST_LOCAL_RESPONSE_GFG_VALIDATION_ORIGIN_SOURCE_MISSING")
            require(source["content_sha256"] == origin["source_content_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_ORIGIN_CONTENT_MISMATCH")
            require(source["role"] == origin["source_role"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_ORIGIN_ROLE_MISMATCH")
            checks += 4

        prior_block_sha: str | None = None
        for row, block in graph.blocks():
            require(hashlib.sha256(canonical_bytes(block)).hexdigest() == row["payload_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_block_sha, "SST_LOCAL_RESPONSE_GFG_VALIDATION_BLOCK_CHAIN_MISMATCH")
            require(len(block["objects"]) == row["object_count"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_OBJECT_COUNT_MISMATCH")
            require(len(block["occurrences"]) == row["occurrence_count"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_OCCURRENCE_COUNT_MISMATCH")
            require(sum(len(value["sources"]) for value in block["fact_blocks"]) == row["fact_count"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_FACT_COUNT_MISMATCH")
            require(len(block["relations"]) == row["relation_count"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_RELATION_COUNT_MISMATCH")
            for value in block["objects"]:
                object_id = str(value["object_id"])
                require(object_id not in known_objects, "SST_LOCAL_RESPONSE_GFG_VALIDATION_DUPLICATE_OBJECT")
                require(payload_sha256(value["payload"]) == value["content_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_OBJECT_CONTENT_MISMATCH")
                for reference in _tensor_refs(value["payload"]):
                    _validate_tensor(graph_root, reference)
                    tensor_ref_count += 1
                known_objects[object_id] = value
                object_roles[str(value["role"])] += 1
            for value in block["occurrences"]:
                occurrence_id = str(value["occurrence_id"])
                require(occurrence_id not in known_occurrences, "SST_LOCAL_RESPONSE_GFG_VALIDATION_DUPLICATE_OCCURRENCE")
                known_occurrences[occurrence_id] = value
                occurrence_types[str(value["occurrence_type"])] += 1
            for value in block["fact_blocks"]:
                fact_id = str(value["fact_block_id"])
                require(fact_id not in known_facts, "SST_LOCAL_RESPONSE_GFG_VALIDATION_DUPLICATE_FACT")
                require(value["occurrence_id"] in known_occurrences, "SST_LOCAL_RESPONSE_GFG_VALIDATION_FACT_OCCURRENCE_MISSING")
                require(value["outcome"]["object_id"] in known_objects, "SST_LOCAL_RESPONSE_GFG_VALIDATION_FACT_OUTCOME_MISSING")
                for source in value["sources"]:
                    require(source["source_id"] in known_objects or source["source_id"] in origins, "SST_LOCAL_RESPONSE_GFG_VALIDATION_FACT_SOURCE_MISSING")
                known_facts[fact_id] = value
            relations.extend(block["relations"])
            checks += 6
            prior_block_sha = row["block_sha256"]

        require(prior_block_sha == manifest["final_block_sha256"], "SST_LOCAL_RESPONSE_GFG_VALIDATION_FINAL_BLOCK_MISMATCH")
        endpoint_ids = set(known_objects) | set(known_occurrences) | set(known_facts) | set(origins)
        for relation in relations:
            require(relation["source_id"] in endpoint_ids, "SST_LOCAL_RESPONSE_GFG_VALIDATION_RELATION_SOURCE_MISSING")
            require(relation["target_id"] in endpoint_ids, "SST_LOCAL_RESPONSE_GFG_VALIDATION_RELATION_TARGET_MISSING")
            checks += 2
        incidence = Counter(
            str(relation["target_id"])
            for relation in relations
            if relation["relation_type"] == "realizes_fact"
        )
        require(set(incidence) == set(known_facts), "SST_LOCAL_RESPONSE_GFG_VALIDATION_INCIDENCE_COVERAGE_MISMATCH")
        require(all(count == 1 for count in incidence.values()), "SST_LOCAL_RESPONSE_GFG_VALIDATION_INCIDENCE_NOT_EXACT")
        state_count = 2 * len(branches)
        require(occurrence_types["local_parameter_displacement_application_occurrence"] == state_count, "SST_LOCAL_RESPONSE_GFG_VALIDATION_APPLICATION_COUNT_MISMATCH")
        require(occurrence_types["baseline_probe_occurrence"] == 2 * state_count, "SST_LOCAL_RESPONSE_GFG_VALIDATION_BASELINE_PROBE_COUNT_MISMATCH")
        require(occurrence_types["gated_support_probe_occurrence"] == 10 * state_count, "SST_LOCAL_RESPONSE_GFG_VALIDATION_GATED_PROBE_COUNT_MISMATCH")
        require(occurrence_types["support_metric_derivation_occurrence"] == state_count, "SST_LOCAL_RESPONSE_GFG_VALIDATION_METRIC_DERIVATION_COUNT_MISMATCH")
        require(occurrence_types["central_finite_difference_response_occurrence"] == 2, "SST_LOCAL_RESPONSE_GFG_VALIDATION_FINITE_DIFFERENCE_COUNT_MISMATCH")
        require(occurrence_types["receiver_local_response_comparison_occurrence"] == 1, "SST_LOCAL_RESPONSE_GFG_VALIDATION_COMPARISON_COUNT_MISMATCH")
        require(object_roles["validated_local_response_summary"] == 2, "SST_LOCAL_RESPONSE_GFG_VALIDATION_RESPONSE_SUMMARY_COUNT_MISMATCH")
        require(object_roles["receiver_conditioned_local_response_contrasts"] == 1, "SST_LOCAL_RESPONSE_GFG_VALIDATION_CONTRAST_COUNT_MISMATCH")
        require(set(manifest["state_catalog"]) == {f"{label}:{branch}" for label in ("A", "B") for branch in branches}, "SST_LOCAL_RESPONSE_GFG_VALIDATION_STATE_CATALOG_MISMATCH")
        if branches == LOCAL_RESPONSE_TRANSPORT_BRANCHES:
            require(tuple(manifest["branches"]) == branches, "SST_LOCAL_RESPONSE_GFG_VALIDATION_MANIFEST_BRANCHES_MISMATCH")
            require(
                str(manifest["receiver_state_kind"])
                == str(protocol.get("receiver_state_kind", "skip")),
                "SST_LOCAL_RESPONSE_GFG_VALIDATION_RECEIVER_STATE_KIND_MISMATCH",
            )
        checks += 11
    finally:
        graph.close()

    material = {
        "schema": "nanogpt-local-response-jk-gfg-validation-v1",
        "status": "PASS",
        "manifest_sha256": manifest["manifest_sha256"],
        "database_sha256": manifest["database_sha256"],
        "evidence_validation_sha256": evidence_validation["validation_sha256"],
        "prior_reciprocal_manifest_sha256": prior_manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "occurrence_types": dict(sorted(occurrence_types.items())),
        "object_roles": dict(sorted(object_roles.items())),
        "tensor_reference_validation_count": tensor_ref_count,
        "check_count": checks,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(graph_root / "local_response_jk_gfg_validation.json", result)
    return result
