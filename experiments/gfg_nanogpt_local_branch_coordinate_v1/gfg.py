from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import GraphRef, SupportGFGWriter

from .analysis import canonical_json, file_sha256


SCOPE_ID = "nanogpt-local-branch-coordinate-v1"
GRAPH_SCHEMA = "nanogpt-local-branch-coordinate-gfg-v1"
BLOCK_SCHEMA = "nanogpt-local-branch-coordinate-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-local-branch-coordinate-gfg-manifest-v1"
PHASES = {
    "contract": ("PROTOCOL_FREEZE.md", "EXPERIMENT_FREEZE.json"),
    "coordinate_formation": ("AVAILABILITY_AUDIT.json", "FEATURE_MANIFEST.json", "SOURCE_OBJECT_LEDGER.jsonl.gz"),
    "development_selection": ("CANDIDATE_SELECTION.json", "DEVELOPMENT_RESULTS.json"),
    "sealed_confirmation": ("RESULTS.json", "CONFIRMATION_LEDGER.jsonl.gz", "MATCHED_PAIRS.json"),
    "adjudication": ("DECISION.json", "REPORT.md"),
    "independent_recomputation": ("INDEPENDENT_RECOMPUTATION.json",),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def artifact(writer: SupportGFGWriter, path: Path, role: str, ordinal: int) -> GraphRef:
    payload: dict[str, Any] = {"file_name": path.name, "file_sha256": file_sha256(path), "bytes": path.stat().st_size}
    if path.suffix == ".json":
        value = read_json(path)
        payload.update({"document_schema": value.get("schema"), "document_status": value.get("status")})
    return writer.object(
        semantic_key=f"local-branch-coordinate:file:{path.name}",
        role=role,
        optimizer_step=ordinal,
        payload=payload,
        object_kind="content_addressed_analysis_artifact",
    )


def occurrence(writer: SupportGFGWriter, phase: str, ordinal: int) -> str:
    return writer.occurrence(
        occurrence_type=f"local_branch_coordinate_{phase}",
        optimizer_step=ordinal,
        transform_reference={"transform_id": f"local-branch-coordinate:{phase}:v1", "implementation": __name__},
        payload={
            "phase": phase,
            "new_nanogpt_training": False,
            "gpu_used": False,
            "vm_ai_used": False,
            "new_response_probe": False,
            "confirmation_labels_are_adjudication_only": True,
        },
    )


def build_gfg(report_root: Path, graph_root: Path) -> dict[str, Any]:
    if graph_root.exists():
        raise RuntimeError(f"GFG_ROOT_EXISTS:{graph_root}")
    if read_json(report_root / "INDEPENDENT_RECOMPUTATION.json")["status"] != "PASS":
        raise RuntimeError("INDEPENDENT_RECOMPUTATION_NOT_PASS")
    freeze = read_json(report_root / "EXPERIMENT_FREEZE.json")
    source_material = freeze["source_hashes"]
    source_bundle = sha256_bytes(canonical_json(source_material).encode("utf-8"))
    graph_root.mkdir(parents=True)
    writer = SupportGFGWriter(
        graph_root / "local_branch_coordinate_gfg.sqlite3",
        graph_root / "tensor-objects",
        scope_id=SCOPE_ID,
        source_bundle_id=source_bundle,
        contract_sha256=file_sha256(report_root / "PROTOCOL_FREEZE.md"),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    writer.start_block("source_admission", 0)
    origins = []
    for source, digest in sorted(source_material.items()):
        origins.append((writer.origin({"object_id": "source_" + digest, "content_sha256": digest, "optimizer_step": 0, "role": "admitted_validated_source", "semantic_key": source}, source_bundle_id=source_bundle, source_graph_schema="validated-prior-nanogpt-evidence"), "admitted_source"))
    source_event = occurrence(writer, "source_admission", 0)
    source_ledger = artifact(writer, report_root / "SOURCE_OBJECT_LEDGER.jsonl.gz", "source_object_ledger", 0)
    writer.bind(source_event, origins, source_ledger, payload={"status": "SOURCE_IDENTITIES_AND_HASHES_ADMITTED"})
    writer.flush_block()
    prior: list[tuple[GraphRef, str]] = [(source_ledger, "validated_source_ledger")]
    represented: dict[str, str] = {}
    for ordinal, (phase, names) in enumerate(PHASES.items(), 1):
        writer.start_block(phase, ordinal)
        event = occurrence(writer, phase, ordinal)
        outputs = []
        for name in names:
            path = report_root / name
            value = artifact(writer, path, f"{phase}_result", ordinal)
            writer.bind(event, prior, value, payload={"outcome_file": name, "confirmation_used_for_selection": False})
            outputs.append((value, f"prior_{name}"))
            represented[name] = file_sha256(path)
        writer.flush_block()
        prior = outputs
    manifest = writer.close()
    manifest.update({"status": "PASS", "represented_outputs": represented, "source_material": source_material, "future_leakage_detected": False})
    write_json(graph_root / "GFG_MANIFEST.json", manifest)
    return manifest


def validate_gfg(report_root: Path, graph_root: Path) -> dict[str, Any]:
    manifest = read_json(graph_root / "GFG_MANIFEST.json")
    database = graph_root / manifest["database"]
    if file_sha256(database) != manifest["database_sha256"]:
        raise RuntimeError("GFG_DATABASE_HASH_MISMATCH")
    required = sorted(name for names in PHASES.values() for name in names)
    represented: dict[str, str] = {}
    prior = None
    counts = CounterLike()
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal"):
            payload = json.loads(zlib.decompress(row["payload_zlib"]))
            if sha256_bytes(canonical_json(payload).encode("utf-8")) != row["payload_sha256"]:
                raise RuntimeError("GFG_PAYLOAD_HASH_MISMATCH")
            if row["prior_block_sha256"] != prior:
                raise RuntimeError("GFG_BLOCK_CHAIN_MISMATCH")
            prior = row["block_sha256"]
            counts.add(row)
            for value in payload["objects"]:
                name = value.get("payload", {}).get("file_name")
                digest = value.get("payload", {}).get("file_sha256")
                if name and digest:
                    represented[str(name)] = str(digest)
    for name in required:
        if represented.get(name) != file_sha256(report_root / name):
            raise RuntimeError(f"GFG_REPRESENTED_HASH_MISMATCH:{name}")
    result = {"schema": "nanogpt-local-branch-coordinate-gfg-validation-v1", "status": "PASS", **counts.values, "represented_output_count": len(required), "database_sha256": file_sha256(database), "future_leakage_detected": False}
    write_json(graph_root / "GFG_VALIDATION.json", result)
    return result


class CounterLike:
    def __init__(self) -> None:
        self.values = {"block_count": 0, "fact_count": 0, "object_count": 0, "occurrence_count": 0, "relation_count": 0}

    def add(self, row: sqlite3.Row) -> None:
        self.values["block_count"] += 1
        for target, source in (("fact_count", "fact_count"), ("object_count", "object_count"), ("occurrence_count", "occurrence_count"), ("relation_count", "relation_count")):
            self.values[target] += int(row[source])


__all__ = ["build_gfg", "validate_gfg"]
