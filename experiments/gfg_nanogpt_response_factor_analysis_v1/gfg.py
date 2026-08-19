from __future__ import annotations

import gzip
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import GraphRef, SupportGFGWriter

from .analysis import canonical_json, file_sha256, read_json, require, sha256_bytes, write_json


GRAPH_SCHEMA = "nanogpt-response-factor-analysis-gfg-v1"
BLOCK_SCHEMA = "nanogpt-response-factor-analysis-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-response-factor-analysis-gfg-manifest-v1"
SCOPE_ID = "nanogpt-response-factor-analysis-v1"


PHASE_OUTPUTS = {
    "execution_audit": (
        "RECOVERY_AUDIT.json",
        "BOUNDARY_VIOLATION.json",
    ),
    "feature_formation": (
        "PRETARGET_FEATURE_AVAILABILITY.json",
        "PRETARGET_FACTOR_RECORDS.jsonl.gz",
        "FACTOR_SCHEMA.json",
    ),
    "matching": (
        "MATCHING_PROTOCOL.json",
        "MATCH_LEDGER.jsonl.gz",
        "MATCH_LEDGER_MANIFEST.json",
    ),
    "factor_adjudication": (
        "SINGLE_FACTOR_RESULTS.json",
        "INCREMENTAL_CONDITIONING_RESULTS.json",
        "LEAVE_ONE_FACTOR_OUT_RESULTS.json",
        "HISTORY_INCREMENT_RESULTS.json",
        "RESPONSE_TYPE_CONDITIONING.json",
    ),
    "counterexample_adjudication": (
        "UNCHANGED_TARGET_ANALYSIS.json",
        "IDENTITY_RESIDUAL_RESULTS.json",
        "SURVIVING_CONDITIONAL_COUNTEREXAMPLES.json",
        "ROBUSTNESS_AND_SENSITIVITY.json",
    ),
    "scientific_assessment": ("SCIENTIFIC_ASSESSMENT.md",),
}


def _file_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file_name": path.name,
        "file_sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".json":
        value = read_json(path)
        payload["document_schema"] = value.get("schema")
        payload["document_status"] = value.get("status")
    return payload


def _object(writer: SupportGFGWriter, path: Path, role: str, optimizer_step: int = 0) -> GraphRef:
    return writer.object(
        semantic_key=f"response-factor-analysis:file:{path.name}",
        role=role,
        optimizer_step=optimizer_step,
        payload=_file_payload(path),
        object_kind="content_addressed_analysis_artifact",
    )


def _occurrence(writer: SupportGFGWriter, phase: str, ordinal: int) -> str:
    return writer.occurrence(
        occurrence_type=f"response_factor_{phase}_occurrence",
        optimizer_step=ordinal,
        transform_reference={
            "transform_id": f"response-factor-analysis:{phase}:v1",
            "implementation": "experiments.gfg_nanogpt_response_factor_analysis_v1",
        },
        payload={
            "phase": phase,
            "ordinal": ordinal,
            "current_step_alpha_positive_used_as_condition": False,
            "prediction_model_trained": False,
        },
    )


def build_factor_analysis_gfg(*, analysis_root: Path, response_root: Path, graph_root: Path) -> dict[str, Any]:
    require(not graph_root.exists(), f"GFG_ROOT_EXISTS:{graph_root}")
    manifest = read_json(analysis_root / "MANIFEST.json")
    require(
        manifest["status"]
        in {
            "ANALYSIS_COMPLETE_PENDING_GFG_VALIDATION",
            "ANALYSIS_COMPLETE_PENDING_GFG_VALIDATION_WITH_DISCLOSED_BOUNDARY_VIOLATION",
        },
        "ANALYSIS_NOT_READY_FOR_GFG",
    )
    unseen_diagnostic_accessed = bool(manifest["global_unseen_entry_accessed"])
    contract_path = analysis_root / "FACTOR_ANALYSIS_CONTRACT.md"
    source_names = (
        "SELECTION_MANIFEST.json",
        "IDENTITY_MATERIAL.json",
        "RESOLVED_INVENTORY.json",
        "UPDATE_GEOMETRY_CONTROL_MANIFEST.json",
        "FINITE_AMPLITUDE_CURVES_MANIFEST.json",
        "VALIDATION.json",
        "INDEPENDENT_REPLAY.json",
    )
    source_material = {name: file_sha256(response_root / name) for name in source_names}
    source_bundle_id = sha256_bytes(canonical_json(source_material).encode("utf-8"))
    graph_root.mkdir(parents=True)
    writer = SupportGFGWriter(
        graph_root / "response_factor_analysis_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id=SCOPE_ID,
        source_bundle_id=source_bundle_id,
        contract_sha256=file_sha256(contract_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )

    writer.start_block("source_admission_and_contract_freeze", 0)
    source_refs: list[tuple[GraphRef, str]] = []
    for name in source_names:
        path = response_root / name
        source_object = {
            "object_id": "source_file_" + file_sha256(path),
            "content_sha256": file_sha256(path),
            "optimizer_step": 0,
            "role": f"validated_source_{name}",
            "semantic_key": f"adjacent-response-evidence:{name}",
        }
        origin = writer.origin(
            source_object,
            source_bundle_id=source_bundle_id,
            source_graph_schema="nanogpt-adjacent-response-transport-evidence-v1",
        )
        source_refs.append((origin, f"source_{name}"))
    task_ref = _object(writer, analysis_root / "ORIGINAL_TASK.txt", "frozen_user_task")
    freeze_occurrence = _occurrence(writer, "contract_freeze", 0)
    contract_ref = _object(writer, contract_path, "frozen_factor_analysis_contract")
    writer.bind(freeze_occurrence, [(task_ref, "task_instruction")], contract_ref, payload={"status": "FROZEN_BEFORE_FACTOR_RESULTS"})
    freeze_ref = _object(writer, analysis_root / "ANALYSIS_FREEZE.json", "factor_analysis_freeze_receipt")
    writer.bind(
        freeze_occurrence,
        [(contract_ref, "frozen_contract"), *source_refs],
        freeze_ref,
        payload={"global_unseen_accessed_at_freeze_time": False, "later_diagnostic_access_disclosed": unseen_diagnostic_accessed},
    )
    writer.flush_block()

    prior_refs: list[tuple[GraphRef, str]] = [(contract_ref, "frozen_contract"), (freeze_ref, "freeze_receipt"), *source_refs]
    output_refs: dict[str, GraphRef] = {}
    ordinal = 1
    for phase, names in PHASE_OUTPUTS.items():
        writer.start_block(phase, ordinal)
        occurrence = _occurrence(writer, phase, ordinal)
        phase_sources = list(prior_refs)
        for name in names:
            path = analysis_root / name
            require(path.is_file(), f"GFG_OUTPUT_MISSING:{name}")
            outcome = _object(writer, path, f"{phase}_result")
            writer.bind(
                occurrence,
                phase_sources,
                outcome,
                payload={
                    "outcome_file": name,
                    "outcome_kind": "content_addressed_analysis_artifact",
                    "row_level_material_retained_in_content_addressed_ledgers": True,
                },
            )
            output_refs[name] = outcome
        writer.flush_block()
        prior_refs = [(output_refs[name], f"prior_{name}") for name in names]
        ordinal += 1

    graph_manifest = writer.close()
    graph_manifest.update(
        {
            "status": "PASS",
            "graph_schema": GRAPH_SCHEMA,
            "block_schema": BLOCK_SCHEMA,
            "contract_sha256": file_sha256(contract_path),
            "source_files": source_material,
            "represented_output_files": sorted(output_refs),
            "current_step_alpha_positive_used_as_condition": False,
            "global_unseen_entry_accessed": unseen_diagnostic_accessed,
            "global_unseen_entry_used_in_factor_records": False,
            "prediction_model_trained": False,
        }
    )
    write_json(graph_root / "GFG_MANIFEST.json", graph_manifest)
    return graph_manifest


def validate_factor_analysis_gfg(*, analysis_root: Path, response_root: Path, graph_root: Path) -> dict[str, Any]:
    manifest_path = graph_root / "GFG_MANIFEST.json"
    manifest = read_json(manifest_path)
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "GFG_DATABASE_HASH_MISMATCH")
    require(manifest["graph_schema"] == GRAPH_SCHEMA, "GFG_SCHEMA_MISMATCH")
    require(manifest["global_unseen_entry_accessed"] is True, "GFG_DISCLOSED_UNSEEN_ACCESS_FLAG_MISSING")
    require(manifest["global_unseen_entry_used_in_factor_records"] is False, "GFG_UNSEEN_RECORD_USE_FLAG_INVALID")
    required_outputs = sorted(name for names in PHASE_OUTPUTS.values() for name in names)
    require(sorted(manifest["represented_output_files"]) == required_outputs, "GFG_OUTPUT_COVERAGE_MISMATCH")

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    prior_sha: str | None = None
    fact_count = object_count = occurrence_count = relation_count = 0
    represented_files: dict[str, str] = {}
    checks = 0
    try:
        metadata = {row["key"]: json.loads(row["value_json"]) for row in connection.execute("SELECT key,value_json FROM metadata")}
        require(metadata["schema"] == GRAPH_SCHEMA, "GFG_METADATA_SCHEMA_MISMATCH")
        require(metadata["contract_sha256"] == file_sha256(analysis_root / "FACTOR_ANALYSIS_CONTRACT.md"), "GFG_CONTRACT_HASH_MISMATCH")
        checks += 2
        for row in connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal"):
            payload = json.loads(zlib.decompress(row["payload_zlib"]))
            raw = canonical_json(payload).encode("utf-8")
            require(sha256_bytes(raw) == row["payload_sha256"], f"GFG_BLOCK_PAYLOAD_HASH_MISMATCH:{row['block_ordinal']}")
            require(row["prior_block_sha256"] == prior_sha, f"GFG_BLOCK_CHAIN_MISMATCH:{row['block_ordinal']}")
            prior_sha = row["block_sha256"]
            require(
                sum(len(value["sources"]) for value in payload["fact_blocks"]) == row["fact_count"],
                "GFG_FACT_COUNT_MISMATCH",
            )
            require(len(payload["objects"]) == row["object_count"], "GFG_OBJECT_COUNT_MISMATCH")
            require(len(payload["occurrences"]) == row["occurrence_count"], "GFG_OCCURRENCE_COUNT_MISMATCH")
            require(len(payload["relations"]) == row["relation_count"], "GFG_RELATION_COUNT_MISMATCH")
            fact_count += sum(len(value["sources"]) for value in payload["fact_blocks"])
            object_count += len(payload["objects"])
            occurrence_count += len(payload["occurrences"])
            relation_count += len(payload["relations"])
            object_ids = {value["object_id"] for value in payload["objects"]}
            occurrence_ids = {value["occurrence_id"] for value in payload["occurrences"]}
            for fact in payload["fact_blocks"]:
                require(bool(fact["sources"]), "GFG_FACT_WITHOUT_SOURCE")
                require(fact["occurrence_id"] in occurrence_ids, "GFG_FACT_OCCURRENCE_MISSING")
                require(fact["outcome"]["object_id"] in object_ids, "GFG_FACT_OUTCOME_MISSING")
                checks += 3
            for value in payload["objects"]:
                file_name = value.get("payload", {}).get("file_name")
                file_hash = value.get("payload", {}).get("file_sha256")
                if file_name and file_hash:
                    represented_files[str(file_name)] = str(file_hash)
            checks += 5
    finally:
        connection.close()

    for name in required_outputs:
        path = analysis_root / name
        require(path.is_file(), f"GFG_REPRESENTED_FILE_MISSING:{name}")
        require(represented_files.get(name) == file_sha256(path), f"GFG_REPRESENTED_FILE_HASH_MISMATCH:{name}")
        checks += 2
    records_path = analysis_root / "PRETARGET_FACTOR_RECORDS.jsonl.gz"
    with gzip.open(records_path, "rt", encoding="utf-8") as handle:
        record_count = 0
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            require(value["entry_id"] != "entry-362ded584a953f360aec", "GFG_GLOBAL_UNSEEN_RECORD_PRESENT")
            record_count += 1
    require(record_count == 15264, "GFG_RECORD_COUNT_INVALID")
    checks += record_count
    result = {
        "schema": "nanogpt-response-factor-analysis-gfg-validation-v1",
        "status": "PASS_WITH_DISCLOSED_BOUNDARY_VIOLATION",
        "check_count": checks,
        "record_count": record_count,
        "fact_count": fact_count,
        "object_count": object_count,
        "occurrence_count": occurrence_count,
        "relation_count": relation_count,
        "represented_output_count": len(required_outputs),
        "database_sha256": file_sha256(database),
        "manifest_sha256": file_sha256(manifest_path),
        "global_unseen_entry_accessed": True,
        "global_unseen_entry_used_in_factor_records": False,
        "prediction_model_trained": False,
    }
    write_json(graph_root / "GFG_VALIDATION.json", result)
    return result


__all__ = ["build_factor_analysis_gfg", "validate_factor_analysis_gfg"]
