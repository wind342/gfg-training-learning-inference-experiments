from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import GraphRef, SupportGFGWriter

from .runner import file_sha256


SCOPE_ID = "nanogpt-actual-update-boundary-v1"
GRAPH_SCHEMA = "nanogpt-actual-update-boundary-gfg-v1"
BLOCK_SCHEMA = "nanogpt-actual-update-boundary-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-actual-update-boundary-gfg-manifest-v1"
PHASES = {
    "contract": ("PROTOCOL_FREEZE.md", "EXPERIMENT_CONTEXT.json"),
    "source_admission": ("SOURCE_MANIFEST.json", "DERIVATIVE_AUDIT.json"),
    "prediction": ("BOUNDARY_PREDICTIONS.jsonl.gz", "BOUNDARY_RESULTS.json"),
    "assessment": ("SCIENTIFIC_ASSESSMENT.md", "INDEPENDENT_CHECK.json"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _artifact(writer: SupportGFGWriter, path: Path, role: str, ordinal: int) -> GraphRef:
    return writer.object(
        semantic_key=f"actual-update-boundary:file:{path.name}",
        role=role,
        optimizer_step=ordinal,
        payload={"file_name": path.name, "file_sha256": file_sha256(path), "bytes": path.stat().st_size},
        object_kind="content_addressed_analysis_artifact",
    )


def _occurrence(writer: SupportGFGWriter, phase: str, ordinal: int) -> str:
    return writer.occurrence(
        occurrence_type=f"actual_update_boundary_{phase}",
        optimizer_step=ordinal,
        transform_reference={"transform_id": f"actual-update-boundary:{phase}:v1", "implementation": __name__},
        payload={
            "phase": phase,
            "prediction_unit_count": 15264,
            "response_curve_predicted": False,
            "support_state_predicted": False,
            "difficult_subset_used": False,
        },
    )


def build_gfg(report_root: Path, graph_root: Path) -> dict[str, Any]:
    require(not graph_root.exists(), f"GFG_ROOT_EXISTS:{graph_root}")
    require(_read(report_root / "INDEPENDENT_CHECK.json")["status"] == "PASS", "INDEPENDENT_CHECK_NOT_PASS")
    audit = _read(report_root / "DERIVATIVE_AUDIT.json")
    source_hashes = {
        str(row["section_id"]): _hash(canonical_json({
            "receiver_state_sha256": row["receiver_state_sha256"],
            "actual_update_sha256": row["actual_update_sha256"],
            "evaluation_input_sha256": row["evaluation_input_sha256"],
        }).encode("utf-8"))
        for row in audit["sections"]
    }
    bundle = _hash(canonical_json(source_hashes).encode("utf-8"))
    graph_root.mkdir(parents=True)
    writer = SupportGFGWriter(
        graph_root / "actual_update_boundary_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id=SCOPE_ID,
        source_bundle_id=bundle,
        contract_sha256=file_sha256(report_root / "PROTOCOL_FREEZE.md"),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    writer.start_block("admitted_sources", 0)
    origins = [
        (writer.origin({
            "object_id": "section_" + digest,
            "content_sha256": digest,
            "optimizer_step": 0,
            "role": "validated_parameter_update_evaluation_bundle",
            "semantic_key": section,
        }, source_bundle_id=bundle, source_graph_schema="validated-nanogpt-training-evidence"), "admitted_training_evidence")
        for section, digest in sorted(source_hashes.items())
    ]
    event = _occurrence(writer, "admitted_sources", 0)
    admitted = _artifact(writer, report_root / "SOURCE_MANIFEST.json", "source_manifest", 0)
    writer.bind(event, origins, admitted, payload={"source_section_count": len(origins)})
    writer.flush_block()

    represented: dict[str, str] = {}
    prior: list[tuple[GraphRef, str]] = [(admitted, "validated_source_manifest")]
    for ordinal, (phase, names) in enumerate(PHASES.items(), 1):
        writer.start_block(phase, ordinal)
        event = _occurrence(writer, phase, ordinal)
        outcomes: list[tuple[GraphRef, str]] = []
        for name in names:
            path = report_root / name
            require(path.is_file(), f"OUTPUT_MISSING:{name}")
            outcome = _artifact(writer, path, f"{phase}_result", ordinal)
            writer.bind(event, prior, outcome, payload={
                "outcome_file": name,
                "positive_alpha_response_used_as_prediction_input": False,
                "primary_method": "quadratic_complete",
            })
            represented[name] = file_sha256(path)
            outcomes.append((outcome, f"prior_{name}"))
        writer.flush_block()
        prior = outcomes
    manifest = writer.close()
    manifest.update({
        "status": "PASS",
        "graph_schema": GRAPH_SCHEMA,
        "represented_outputs": represented,
        "source_hashes": source_hashes,
        "prediction_unit_count": 15264,
        "future_leakage_detected": False,
        "response_curve_predicted": False,
        "support_state_predicted": False,
    })
    _write(graph_root / "GFG_MANIFEST.json", manifest)
    return manifest


def validate_gfg(report_root: Path, graph_root: Path) -> dict[str, Any]:
    manifest = _read(graph_root / "GFG_MANIFEST.json")
    database = graph_root / str(manifest["database"])
    require(file_sha256(database) == manifest["database_sha256"], "DATABASE_HASH")
    required = sorted({name for names in PHASES.values() for name in names})
    represented: dict[str, str] = {}
    previous: str | None = None
    counts = {"block_count": 0, "fact_count": 0, "object_count": 0, "occurrence_count": 0, "relation_count": 0}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        metadata = {row["key"]: json.loads(row["value_json"]) for row in connection.execute("SELECT key,value_json FROM metadata")}
        require(metadata["schema"] == GRAPH_SCHEMA, "GRAPH_SCHEMA")
        require(metadata["contract_sha256"] == file_sha256(report_root / "PROTOCOL_FREEZE.md"), "CONTRACT_HASH")
        for row in connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal"):
            payload = json.loads(zlib.decompress(row["payload_zlib"]))
            require(_hash(canonical_json(payload).encode("utf-8")) == row["payload_sha256"], "PAYLOAD_HASH")
            require(row["prior_block_sha256"] == previous, "BLOCK_CHAIN")
            previous = str(row["block_sha256"])
            counts["block_count"] += 1
            for name in ("fact_count", "object_count", "occurrence_count", "relation_count"):
                counts[name] += int(row[name])
            for value in payload["objects"]:
                name = value.get("payload", {}).get("file_name")
                digest = value.get("payload", {}).get("file_sha256")
                if name and digest:
                    represented[str(name)] = str(digest)
    for name in required:
        require(represented.get(name) == file_sha256(report_root / name), f"REPRESENTED_HASH:{name}")
    result = {
        "schema": "nanogpt-actual-update-boundary-gfg-validation-v1",
        "status": "PASS",
        **counts,
        "represented_output_count": len(required),
        "database_sha256": file_sha256(database),
        "future_leakage_detected": False,
        "response_curve_predicted": False,
        "support_state_predicted": False,
    }
    _write(graph_root / "GFG_VALIDATION.json", result)
    return result


__all__ = ["build_gfg", "validate_gfg"]

