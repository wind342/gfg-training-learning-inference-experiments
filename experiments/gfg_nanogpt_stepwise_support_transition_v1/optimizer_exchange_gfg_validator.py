from __future__ import annotations

from collections import Counter
import hashlib
import json
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
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import TrainingGFG
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import SupportGFG, _source_objects_for_step

from .optimizer_exchange import EXCHANGE_BRANCHES, EXCHANGE_CONTINUATION_HORIZONS, _branch_contract
from .optimizer_exchange_gfg import GRAPH_SCHEMA
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


def _validate_tensor(graph_root: Path, reference: dict[str, Any], cache: dict[str, dict[str, Any]]) -> None:
    locator = str(reference["locator"])
    require(locator.startswith("tensor-objects/"), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_LOCATOR_INVALID")
    admitted = {name: reference[name] for name in ("file_sha256", "raw_tensor_sha256", "shape", "dtype")}
    if locator in cache:
        require(cache[locator] == admitted, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_REFERENCE_CONFLICT")
        return
    path = graph_root / locator
    require(path.is_file(), f"SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_MISSING:{path}")
    require(file_sha256(path) == reference["file_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_FILE_HASH_MISMATCH")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(reference["shape"]), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_SHAPE_MISMATCH")
    require(str(value.dtype) == str(reference["dtype"]), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_DTYPE_MISMATCH")
    require(hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest() == reference["raw_tensor_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_RAW_HASH_MISMATCH")
    cache[locator] = admitted


def validate_optimizer_exchange_gfg(
    *,
    graph_root: Path,
    evidence_root: Path,
    amplitude_graph_root: Path,
    source_root: Path,
    optimizer_exchange_protocol_path: Path,
) -> dict[str, Any]:
    manifest = read_json(graph_root / "optimizer_exchange_gfg_manifest.json")
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(payload_sha256(material) == manifest["manifest_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_MANIFEST_HASH_MISMATCH")
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_DATABASE_HASH_MISMATCH")
    protocol = read_json(optimizer_exchange_protocol_path)
    _branch_contract(protocol)
    require(file_sha256(optimizer_exchange_protocol_path) == manifest["optimizer_exchange_protocol_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_PROTOCOL_HASH_MISMATCH")
    evidence = read_json(evidence_root / "optimizer_exchange_validation.json")
    require(evidence["validation_sha256"] == manifest["evidence_validation_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_EVIDENCE_HASH_MISMATCH")
    prior_manifest = read_json(amplitude_graph_root / "amplitude_path_gfg_manifest.json")
    require(prior_manifest["manifest_sha256"] == manifest["prior_amplitude_manifest_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_PRIOR_MANIFEST_MISMATCH")
    prior_objects = _main_object_index(amplitude_graph_root / str(prior_manifest["database"]))

    source_manifests = {
        str(row["label"]): read_json(source_root / str(row["source_bundle_id"]) / "manifest.json")
        for row in protocol["receivers"]
    }
    source_graphs = {
        str(source_manifests[label]["bundle_manifest_sha256"]): TrainingGFG(source_root / str(next(row["source_bundle_id"] for row in protocol["receivers"] if row["label"] == label)) / "participant_gfg.sqlite3")
        for label in ("A", "B")
    }
    graph = SupportGFG(database)
    known_objects: dict[str, dict[str, Any]] = {}
    known_occurrences: dict[str, dict[str, Any]] = {}
    known_facts: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []
    occurrence_types: Counter[str] = Counter()
    object_roles: Counter[str] = Counter()
    tensor_cache: dict[str, dict[str, Any]] = {}
    checks = 0
    try:
        require(graph.metadata("schema") == GRAPH_SCHEMA, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_SCHEMA_INVALID")
        require(graph.metadata("contract_sha256") == manifest["optimizer_exchange_protocol_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_METADATA_CONTRACT_MISMATCH")
        origins = {str(row["origin_id"]): dict(row) for row in graph.connection.execute("SELECT * FROM origin_catalog")}
        source_cache: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        prior_origin_ids: set[str] = set()
        for origin in origins.values():
            bundle_id = str(origin["source_bundle_id"])
            source_id = str(origin["source_object_id"])
            if bundle_id == prior_manifest["manifest_sha256"]:
                source = prior_objects.get(source_id)
                require(source is not None, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_PRIOR_ORIGIN_MISSING")
                prior_origin_ids.add(source_id)
            else:
                require(bundle_id in source_graphs, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_SOURCE_BUNDLE_UNKNOWN")
                cache_key = (bundle_id, int(origin["source_optimizer_step"]))
                if cache_key not in source_cache:
                    source_cache[cache_key] = _source_objects_for_step(source_graphs[bundle_id], cache_key[1])
                source = source_cache[cache_key].get(source_id)
                require(source is not None, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_SOURCE_ORIGIN_MISSING")
            require(source["content_sha256"] == origin["source_content_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_ORIGIN_CONTENT_MISMATCH")
            require(source["role"] == origin["source_role"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_ORIGIN_ROLE_MISMATCH")
            checks += 3
        required_prior_ids = {
            str(endpoint[field])
            for receiver in protocol["receivers"]
            for endpoint in receiver["endpoints"].values()
            for field in ("h20_source_object_id", "h100_source_object_id")
        }
        require(required_prior_ids <= prior_origin_ids, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_REQUIRED_PRIOR_ORIGIN_MISSING")

        prior_block_sha: str | None = None
        for row, block in graph.blocks():
            require(hashlib.sha256(canonical_bytes(block)).hexdigest() == row["payload_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_block_sha, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_BLOCK_CHAIN_MISMATCH")
            require(len(block["objects"]) == row["object_count"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_OBJECT_COUNT_MISMATCH")
            require(len(block["occurrences"]) == row["occurrence_count"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_OCCURRENCE_COUNT_MISMATCH")
            require(sum(len(value["sources"]) for value in block["fact_blocks"]) == row["fact_count"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_FACT_COUNT_MISMATCH")
            require(len(block["relations"]) == row["relation_count"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_RELATION_COUNT_MISMATCH")
            for value in block["objects"]:
                object_id = str(value["object_id"])
                require(object_id not in known_objects, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_DUPLICATE_OBJECT")
                require(payload_sha256(value["payload"]) == value["content_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_OBJECT_CONTENT_MISMATCH")
                for reference in _tensor_refs(value["payload"]):
                    _validate_tensor(graph_root, reference, tensor_cache)
                known_objects[object_id] = value
                object_roles[str(value["role"])] += 1
            for value in block["occurrences"]:
                occurrence_id = str(value["occurrence_id"])
                require(occurrence_id not in known_occurrences, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_DUPLICATE_OCCURRENCE")
                require(bool(value["transform_reference"]), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TRANSFORM_MISSING")
                known_occurrences[occurrence_id] = value
                occurrence_types[str(value["occurrence_type"])] += 1
            for value in block["fact_blocks"]:
                fact_id = str(value["fact_block_id"])
                require(fact_id not in known_facts, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_DUPLICATE_FACT")
                require(value["occurrence_id"] in known_occurrences, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_FACT_OCCURRENCE_MISSING")
                outcome_id = str(value["outcome"]["object_id"])
                require(outcome_id in known_objects, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_FACT_OUTCOME_MISSING")
                require(known_objects[outcome_id]["content_sha256"] == value["outcome"]["content_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_FACT_OUTCOME_CONTENT_MISMATCH")
                require(bool(value["sources"]), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_FACT_SOURCES_EMPTY")
                for source in value["sources"]:
                    source_id = str(source["source_id"])
                    source_hash = known_objects.get(source_id, {}).get("content_sha256")
                    if source_hash is None and source_id in origins:
                        source_hash = payload_sha256(json.loads(origins[source_id]["payload_json"]))
                    require(source_hash == source["content_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_FACT_SOURCE_CONTENT_MISMATCH")
                    require(bool(source["relation_role"]), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_RELATION_ROLE_EMPTY")
                known_facts[fact_id] = value
            relations.extend(block["relations"])
            prior_block_sha = str(row["block_sha256"])
            checks += 7
        require(prior_block_sha == manifest["final_block_sha256"], "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_FINAL_BLOCK_MISMATCH")

        endpoint_ids = set(known_objects) | set(known_occurrences) | set(known_facts) | set(origins)
        incidence: Counter[str] = Counter()
        for relation in relations:
            require(relation["source_id"] in endpoint_ids, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_RELATION_SOURCE_MISSING")
            require(relation["target_id"] in endpoint_ids, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_RELATION_TARGET_MISSING")
            if relation["relation_type"] == "realizes_fact":
                require(relation["source_id"] in known_occurrences, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_INCIDENCE_SOURCE_INVALID")
                require(relation["target_id"] in known_facts, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_INCIDENCE_TARGET_INVALID")
                incidence[str(relation["target_id"])] += 1
            checks += 2
        require(set(incidence) == set(known_facts), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_INCIDENCE_COVERAGE_MISMATCH")
        require(all(count == 1 for count in incidence.values()), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_INCIDENCE_NOT_EXACT")

        expected_states = {f"{label}:{horizon}:{branch}" for label in ("A", "B") for horizon in (20, 21, 100) for branch in EXCHANGE_BRANCHES}
        expected_continuations = {f"{label}:{horizon}:{branch}" for label in ("A", "B") for horizon in EXCHANGE_CONTINUATION_HORIZONS for branch in EXCHANGE_BRANCHES}
        require(set(manifest["state_catalog"]) == expected_states, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_STATE_CATALOG_MISMATCH")
        require(set(manifest["probe_catalog"]) == expected_states, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_PROBE_CATALOG_MISMATCH")
        require(set(manifest["continuation_catalog"]) == expected_continuations, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_CONTINUATION_CATALOG_MISMATCH")
        require(set(manifest["native_control_exactness_catalog"]) == {f"{label}:{branch}" for label in ("A", "B") for branch in ("theta0_O0", "theta1_O1")}, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_EXACTNESS_CATALOG_MISMATCH")
        require(set(manifest["comparison_catalog"]) == {"A", "B"}, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_COMPARISON_CATALOG_MISMATCH")
        for catalog_name in ("state_catalog", "probe_catalog", "continuation_catalog", "native_control_exactness_catalog", "comparison_catalog"):
            require(set(manifest[catalog_name].values()) <= set(known_objects), f"SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_CATALOG_OBJECT_MISSING:{catalog_name}")

        require(occurrence_types["h20_parameter_optimizer_state_composition_occurrence"] == 8, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_COMPOSITION_COUNT_MISMATCH")
        require(occurrence_types["optimizer_exchange_native_training_step_occurrence"] == 2 * len(EXCHANGE_CONTINUATION_HORIZONS) * len(EXCHANGE_BRANCHES), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_STEP_COUNT_MISMATCH")
        require(occurrence_types["optimizer_exchange_horizon_state_materialization_occurrence"] == 16, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_HORIZON_COUNT_MISMATCH")
        require(occurrence_types["native_control_exactness_validation_occurrence"] == 4, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_EXACTNESS_COUNT_MISMATCH")
        require(occurrence_types["frozen_h100_optimizer_exchange_comparison_occurrence"] == 2, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_COMPARISON_COUNT_MISMATCH")
        require(occurrence_types["baseline_probe_occurrence"] == 48, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_BASELINE_PROBE_COUNT_MISMATCH")
        require(occurrence_types["gated_support_probe_occurrence"] == 240, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_GATED_PROBE_COUNT_MISMATCH")
        require(occurrence_types["support_metric_derivation_occurrence"] == 24, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_SUPPORT_METRIC_COUNT_MISMATCH")
        require(object_roles["validated_native_control_exactness"] == 4, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_EXACTNESS_OBJECT_COUNT_MISMATCH")
        require(object_roles["validated_h100_optimizer_exchange_comparison"] == 2, "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_COMPARISON_OBJECT_COUNT_MISMATCH")
        require(manifest["counts"]["objects"] == len(known_objects), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_MANIFEST_OBJECT_COUNT_MISMATCH")
        require(manifest["counts"]["occurrences"] == len(known_occurrences), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_MANIFEST_OCCURRENCE_COUNT_MISMATCH")
        require(manifest["counts"]["relations"] == len(relations), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_MANIFEST_RELATION_COUNT_MISMATCH")
        require(manifest["counts"]["tensor_payloads"] == len(list((graph_root / "tensor-objects").glob("*.npy"))), "SST_OPTIMIZER_EXCHANGE_GFG_VALIDATION_TENSOR_DIRECTORY_COUNT_MISMATCH")
        checks += 24
    finally:
        graph.close()
        for source_graph in source_graphs.values():
            source_graph.close()

    result_material = {
        "schema": "nanogpt-h20-reciprocal-optimizer-exchange-gfg-validation-v1",
        "status": "PASS",
        "manifest_sha256": manifest["manifest_sha256"],
        "database_sha256": manifest["database_sha256"],
        "evidence_validation_sha256": evidence["validation_sha256"],
        "prior_amplitude_manifest_sha256": prior_manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "occurrence_type_counts": dict(sorted(occurrence_types.items())),
        "object_role_counts": dict(sorted(object_roles.items())),
        "unique_tensor_reference_count": len(tensor_cache),
        "check_count": checks,
        "realizes_fact_exact": True,
        "origin_replay_exact": True,
        "future_information_used": False,
        "scientific_interpretation_performed": False,
    }
    result = {**result_material, "validation_sha256": payload_sha256(result_material)}
    write_json(graph_root / "optimizer_exchange_gfg_validation.json", result)
    return result


__all__ = ["validate_optimizer_exchange_gfg"]
