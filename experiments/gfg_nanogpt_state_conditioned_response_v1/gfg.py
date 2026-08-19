from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import GraphRef, SupportGFGWriter

from .model import canonical_json, file_sha256, require
from .runner import SOURCE_ROOT, write_json


GRAPH_SCHEMA = "nanogpt-state-conditioned-response-gfg-v1"
BLOCK_SCHEMA = "nanogpt-state-conditioned-response-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-state-conditioned-response-gfg-manifest-v1"
SCOPE_ID = "nanogpt-state-conditioned-response-v1"

PHASE_OUTPUTS = {
    "contract_and_inputs": (
        "MODEL_CONTRACT.md",
        "EXPERIMENT_FREEZE.json",
        "BOUNDARY_VIOLATION.json",
        "FEATURE_MANIFEST.json",
    ),
    "run_isolated_modeling": (
        "TRAINING_SPLIT_MANIFEST.json",
        "MODEL_SPEC.json",
        "FINAL_M4_MODEL.npz",
        "FINAL_M4_MODEL_METADATA.json",
    ),
    "held_out_predictions": (
        "RESPONSE_CURVE_PREDICTIONS.json",
        "RESPONSE_CURVE_PREDICTIONS.jsonl.gz",
        "BASELINE_RESULTS.json",
        "NONLINEAR_RESPONSE_MODEL_RESULTS.json",
        "ABLATION_RESULTS.json",
    ),
    "failure_and_challenge_audit": (
        "FAILURE_CASE_ANALYSIS.json",
        "FAILURE_CASE_LEDGER.jsonl.gz",
        "FROZEN_CHALLENGE_AUDIT.jsonl.gz",
    ),
    "assessment_and_replay": (
        "SCIENTIFIC_ASSESSMENT.md",
        "INDEPENDENT_CHECKER.py",
        "REPRODUCE_RUN.py",
        "INDEPENDENT_CHECK.json",
        "DEVELOPMENT_MODEL_READY_WITH_DISCLOSED_BOUNDARY_VIOLATION.json",
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"file_name": path.name, "file_sha256": file_sha256(path), "bytes": path.stat().st_size}
    if path.suffix == ".json":
        value = read_json(path)
        payload["document_schema"] = value.get("schema")
        payload["document_status"] = value.get("status")
    return payload


def artifact(writer: SupportGFGWriter, path: Path, role: str) -> GraphRef:
    return writer.object(
        semantic_key=f"state-conditioned-response:file:{path.name}",
        role=role,
        optimizer_step=0,
        payload=file_payload(path),
        object_kind="content_addressed_analysis_artifact",
    )


def occurrence(writer: SupportGFGWriter, phase: str, ordinal: int) -> str:
    return writer.occurrence(
        occurrence_type=f"state_conditioned_response_{phase}_occurrence",
        optimizer_step=ordinal,
        transform_reference={
            "transform_id": f"state-conditioned-response:{phase}:v1",
            "implementation": "experiments.gfg_nanogpt_state_conditioned_response_v1",
        },
        payload={
            "phase": phase,
            "outer_split_unit": "entry_id",
            "positive_alpha_response_used_as_model_input": False,
            "current_step_functional_jk_used_as_model_input": False,
            "strict_global_unseen_claim_valid": False,
        },
    )


def build_gfg(*, analysis_root: Path, graph_root: Path) -> dict[str, Any]:
    require(not graph_root.exists(), f"GFG_ROOT_EXISTS:{graph_root}")
    independent = read_json(analysis_root / "INDEPENDENT_CHECK.json")
    require(independent["status"] == "PASS", "INDEPENDENT_CHECK_NOT_PASS")
    source_names = (
        "PRETARGET_FACTOR_RECORDS.jsonl.gz",
        "PRETARGET_FEATURE_AVAILABILITY.json",
        "SURVIVING_CONDITIONAL_COUNTEREXAMPLES.json",
        "BOUNDARY_VIOLATION.json",
    )
    source_material = {name: file_sha256(SOURCE_ROOT / name) for name in source_names}
    source_bundle_id = sha256_bytes(canonical_json(source_material).encode("utf-8"))
    graph_root.mkdir(parents=True)
    writer = SupportGFGWriter(
        graph_root / "state_conditioned_response_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id=SCOPE_ID,
        source_bundle_id=source_bundle_id,
        contract_sha256=file_sha256(analysis_root / "MODEL_CONTRACT.md"),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )

    writer.start_block("source_admission", 0)
    source_refs: list[tuple[GraphRef, str]] = []
    for name in source_names:
        path = SOURCE_ROOT / name
        source = {"object_id": "source_" + file_sha256(path), "content_sha256": file_sha256(path), "optimizer_step": 0, "role": f"validated_source_{name}", "semantic_key": f"response-factor-analysis:{name}"}
        source_refs.append((writer.origin(source, source_bundle_id=source_bundle_id, source_graph_schema="nanogpt-response-factor-analysis-gfg-v1"), f"source_{name}"))
    freeze_occurrence = occurrence(writer, "contract_freeze", 0)
    contract = artifact(writer, analysis_root / "MODEL_CONTRACT.md", "frozen_model_contract")
    freeze = artifact(writer, analysis_root / "EXPERIMENT_FREEZE.json", "experiment_freeze_receipt")
    writer.bind(freeze_occurrence, [(contract, "frozen_contract"), *source_refs], freeze, payload={"status": "FROZEN_BEFORE_MODEL_RESULTS"})
    writer.flush_block()

    prior: list[tuple[GraphRef, str]] = [(freeze, "freeze_receipt"), *source_refs]
    represented: dict[str, str] = {}
    for ordinal, (phase, names) in enumerate(PHASE_OUTPUTS.items(), 1):
        writer.start_block(phase, ordinal)
        phase_occurrence = occurrence(writer, phase, ordinal)
        phase_outputs: list[tuple[GraphRef, str]] = []
        for name in names:
            path = analysis_root / name
            require(path.is_file(), f"GFG_OUTPUT_MISSING:{name}")
            outcome = artifact(writer, path, f"{phase}_result")
            writer.bind(
                phase_occurrence,
                prior,
                outcome,
                payload={"outcome_file": name, "row_level_material_retained_in_content_addressed_ledger": name.endswith((".jsonl.gz", ".npz"))},
            )
            phase_outputs.append((outcome, f"prior_{name}"))
            represented[name] = file_sha256(path)
        writer.flush_block()
        prior = phase_outputs
    graph_manifest = writer.close()
    graph_manifest.update(
        {
            "status": "PASS_WITH_DISCLOSED_BOUNDARY_VIOLATION",
            "graph_schema": GRAPH_SCHEMA,
            "block_schema": BLOCK_SCHEMA,
            "represented_outputs": represented,
            "source_material": source_material,
            "strict_global_unseen_claim_valid": False,
        }
    )
    write_json(graph_root / "GFG_MANIFEST.json", graph_manifest)
    return graph_manifest


def validate_gfg(*, analysis_root: Path, graph_root: Path) -> dict[str, Any]:
    manifest_path = graph_root / "GFG_MANIFEST.json"
    manifest = read_json(manifest_path)
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "GFG_DATABASE_HASH_MISMATCH")
    require(manifest["graph_schema"] == GRAPH_SCHEMA, "GFG_SCHEMA_MISMATCH")
    required_outputs = sorted(name for names in PHASE_OUTPUTS.values() for name in names)
    require(sorted(manifest["represented_outputs"]) == required_outputs, "GFG_OUTPUT_COVERAGE_MISMATCH")
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    prior_sha: str | None = None
    fact_count = object_count = occurrence_count = relation_count = checks = 0
    represented: dict[str, str] = {}
    try:
        metadata = {row["key"]: json.loads(row["value_json"]) for row in connection.execute("SELECT key,value_json FROM metadata")}
        require(metadata["schema"] == GRAPH_SCHEMA, "GFG_METADATA_SCHEMA_MISMATCH")
        require(metadata["contract_sha256"] == file_sha256(analysis_root / "MODEL_CONTRACT.md"), "GFG_CONTRACT_HASH_MISMATCH")
        checks += 2
        for row in connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal"):
            payload = json.loads(zlib.decompress(row["payload_zlib"]))
            require(sha256_bytes(canonical_json(payload).encode("utf-8")) == row["payload_sha256"], "GFG_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_sha, "GFG_BLOCK_CHAIN_MISMATCH")
            prior_sha = row["block_sha256"]
            require(sum(len(value["sources"]) for value in payload["fact_blocks"]) == row["fact_count"], "GFG_FACT_COUNT_MISMATCH")
            require(len(payload["objects"]) == row["object_count"], "GFG_OBJECT_COUNT_MISMATCH")
            require(len(payload["occurrences"]) == row["occurrence_count"], "GFG_OCCURRENCE_COUNT_MISMATCH")
            require(len(payload["relations"]) == row["relation_count"], "GFG_RELATION_COUNT_MISMATCH")
            fact_count += int(row["fact_count"])
            object_count += int(row["object_count"])
            occurrence_count += int(row["occurrence_count"])
            relation_count += int(row["relation_count"])
            for value in payload["objects"]:
                file_name = value.get("payload", {}).get("file_name")
                file_hash = value.get("payload", {}).get("file_sha256")
                if file_name and file_hash:
                    represented[str(file_name)] = str(file_hash)
            checks += 6
    finally:
        connection.close()
    for name in required_outputs:
        require(represented.get(name) == file_sha256(analysis_root / name), f"GFG_REPRESENTED_HASH_MISMATCH:{name}")
        checks += 1
    result = {
        "schema": "nanogpt-state-conditioned-response-gfg-validation-v1",
        "status": "PASS_WITH_DISCLOSED_BOUNDARY_VIOLATION",
        "check_count": checks,
        "fact_count": fact_count,
        "object_count": object_count,
        "occurrence_count": occurrence_count,
        "relation_count": relation_count,
        "represented_output_count": len(required_outputs),
        "database_sha256": file_sha256(database),
        "strict_global_unseen_claim_valid": False,
    }
    write_json(graph_root / "GFG_VALIDATION.json", result)
    return result


__all__ = ["build_gfg", "validate_gfg"]
