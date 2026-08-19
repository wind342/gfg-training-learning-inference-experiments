from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator
import zlib

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    canonical_bytes,
    file_sha256,
    payload_sha256,
    require,
    write_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_capture import (
    decode_block,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import (
    TrainingGFG,
)


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS origin_catalog (
    origin_id TEXT PRIMARY KEY,
    source_bundle_id TEXT NOT NULL,
    source_object_id TEXT NOT NULL,
    source_content_sha256 TEXT NOT NULL,
    source_optimizer_step INTEGER NOT NULL,
    source_role TEXT NOT NULL,
    source_semantic_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(source_bundle_id, source_object_id)
);
CREATE TABLE IF NOT EXISTS graph_blocks (
    block_ordinal INTEGER PRIMARY KEY,
    block_id TEXT NOT NULL UNIQUE,
    optimizer_step INTEGER NOT NULL,
    stage TEXT NOT NULL,
    prior_block_sha256 TEXT,
    payload_sha256 TEXT NOT NULL,
    block_sha256 TEXT NOT NULL UNIQUE,
    object_count INTEGER NOT NULL,
    occurrence_count INTEGER NOT NULL,
    fact_count INTEGER NOT NULL,
    relation_count INTEGER NOT NULL,
    payload_zlib BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_support_block_step
    ON graph_blocks(optimizer_step, stage);
CREATE TABLE IF NOT EXISTS checkpoint_index (
    optimizer_step INTEGER PRIMARY KEY,
    current_baseline_logits_sha256 TEXT NOT NULL,
    historical_baseline_logits_sha256 TEXT NOT NULL,
    historical_raw_logits_exact INTEGER NOT NULL,
    historical_predictions_exact INTEGER NOT NULL,
    historical_capability_exact INTEGER NOT NULL,
    historical_max_abs_logit_error REAL NOT NULL,
    single_gate_hashes_json TEXT NOT NULL,
    pair_gate_hashes_json TEXT NOT NULL,
    derived_object_ids_json TEXT NOT NULL,
    actual_forward_count INTEGER NOT NULL,
    status TEXT NOT NULL
);
"""


def _json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def _raw_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous().cpu().numpy()
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class GraphRef:
    object_id: str
    content_sha256: str
    role: str
    source_kind: str


class SupportGFGWriter:
    def __init__(
        self,
        database_path: Path,
        tensor_directory: Path,
        *,
        scope_id: str,
        source_bundle_id: str,
        contract_sha256: str,
        graph_schema: str = "nanogpt-support-redundancy-gfg-v1",
        block_schema: str = "nanogpt-support-redundancy-gfg-block-v1",
        manifest_schema: str = "nanogpt-support-redundancy-gfg-manifest-v1",
    ) -> None:
        self.database_path = database_path.resolve()
        self.tensor_directory = tensor_directory.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.tensor_directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.executescript(SCHEMA)
        self.scope_id = scope_id
        self.source_bundle_id = source_bundle_id
        self.graph_schema = graph_schema
        self.block_schema = block_schema
        self.manifest_schema = manifest_schema
        self._block_ordinal = 0
        self._occurrence_ordinal = 0
        self._last_block_sha256: str | None = None
        self._last_occurrence_id: str | None = None
        self._group: dict[str, Any] | None = None
        self._origins: dict[str, GraphRef] = {}
        self._put_metadata("schema", graph_schema)
        self._put_metadata("scope_id", scope_id)
        self._put_metadata("source_bundle_id", source_bundle_id)
        self._put_metadata("contract_sha256", contract_sha256)
        self.connection.commit()

    def _put_metadata(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value_json) VALUES (?,?)",
            (key, _json(value)),
        )

    def origin(
        self,
        source_object: dict[str, Any],
        *,
        source_bundle_id: str | None = None,
        source_graph_schema: str = "participant-safe-training-gfg-bundle-v1",
    ) -> GraphRef:
        source_id = str(source_object["object_id"])
        admitted_bundle_id = self.source_bundle_id if source_bundle_id is None else source_bundle_id
        cache_key = source_graph_schema + "\0" + admitted_bundle_id + "\0" + source_id
        cached = self._origins.get(cache_key)
        if cached is not None:
            return cached
        material = {
            "source_bundle_id": admitted_bundle_id,
            "source_content_sha256": source_object["content_sha256"],
            "source_graph_schema": source_graph_schema,
            "source_object_id": source_id,
            "source_optimizer_step": int(source_object["optimizer_step"]),
            "source_role": source_object["role"],
            "source_semantic_key": source_object["semantic_key"],
        }
        payload = {
            "origin_kind": "GeneratedOrigin",
            "source_graph_schema": source_graph_schema,
            **material,
        }
        origin_id = "origin_" + payload_sha256(material)
        self.connection.execute(
            """
            INSERT OR IGNORE INTO origin_catalog(
              origin_id,source_bundle_id,source_object_id,
              source_content_sha256,source_optimizer_step,source_role,
              source_semantic_key,payload_json
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                origin_id,
                admitted_bundle_id,
                source_id,
                source_object["content_sha256"],
                int(source_object["optimizer_step"]),
                source_object["role"],
                source_object["semantic_key"],
                _json(payload),
            ),
        )
        result = GraphRef(
            origin_id,
            payload_sha256(payload),
            str(source_object["role"]),
            "generated_origin",
        )
        self._origins[cache_key] = result
        return result

    def start_block(self, stage: str, optimizer_step: int) -> None:
        require(self._group is None, "CSRG_BLOCK_ALREADY_OPEN")
        self._group = {
            "fact_blocks": [],
            "objects": [],
            "occurrences": [],
            "optimizer_step": int(optimizer_step),
            "relations": [],
            "schema": self.block_schema,
            "stage": stage,
        }

    def object(
        self,
        *,
        semantic_key: str,
        role: str,
        optimizer_step: int,
        payload: dict[str, Any],
        object_kind: str = "analysis_result",
    ) -> GraphRef:
        require(self._group is not None, "CSRG_OBJECT_OUTSIDE_BLOCK")
        content_sha = payload_sha256(payload)
        material = {
            "content_sha256": content_sha,
            "domain_scope_id": self.scope_id,
            "semantic_key": semantic_key,
        }
        object_id = "obj_" + payload_sha256(material)
        self._group["objects"].append(
            {
                "content_sha256": content_sha,
                "object_id": object_id,
                "object_kind": object_kind,
                "optimizer_step": int(optimizer_step),
                "payload": payload,
                "role": role,
                "semantic_key": semantic_key,
            }
        )
        return GraphRef(object_id, content_sha, role, "derived_object")

    def tensor_object(
        self,
        *,
        semantic_key: str,
        role: str,
        optimizer_step: int,
        value: torch.Tensor | np.ndarray,
        representation: str,
        extra_payload: dict[str, Any] | None = None,
    ) -> GraphRef:
        array = _raw_array(value)
        raw_sha = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        path = self.tensor_directory / f"{raw_sha}.npy"
        if not path.exists():
            np.save(path, array, allow_pickle=False)
        payload = {
            "dtype": str(array.dtype),
            "file_sha256": file_sha256(path),
            "locator": f"tensor-objects/{path.name}",
            "raw_tensor_sha256": raw_sha,
            "representation": representation,
            "shape": list(array.shape),
            **(extra_payload or {}),
        }
        return self.object(
            semantic_key=semantic_key,
            role=role,
            optimizer_step=optimizer_step,
            payload=payload,
            object_kind="content_addressed_tensor",
        )

    def occurrence(
        self,
        *,
        occurrence_type: str,
        optimizer_step: int,
        transform_reference: dict[str, Any],
        payload: dict[str, Any],
    ) -> str:
        require(self._group is not None, "CSRG_OCCURRENCE_OUTSIDE_BLOCK")
        ordinal = self._occurrence_ordinal
        self._occurrence_ordinal += 1
        material = {
            "occurrence_type": occurrence_type,
            "optimizer_step": int(optimizer_step),
            "ordinal": ordinal,
            "scope_id": self.scope_id,
        }
        occurrence_id = "occ_" + payload_sha256(material)
        self._group["occurrences"].append(
            {
                "occurrence_id": occurrence_id,
                "occurrence_type": occurrence_type,
                "optimizer_step": int(optimizer_step),
                "ordinal": ordinal,
                "payload": payload,
                "transform_reference": transform_reference,
            }
        )
        if self._last_occurrence_id is not None:
            self.relation(
                "program_order",
                self._last_occurrence_id,
                occurrence_id,
                {"basis": "synchronous_csrg_analysis_control_flow"},
            )
        self._last_occurrence_id = occurrence_id
        return occurrence_id

    def bind(
        self,
        occurrence_id: str,
        sources: Iterable[tuple[GraphRef, str]],
        outcome: GraphRef,
        *,
        payload: dict[str, Any] | None = None,
    ) -> str:
        require(self._group is not None, "CSRG_BINDING_OUTSIDE_BLOCK")
        source_rows = [
            {
                "content_sha256": ref.content_sha256,
                "relation_role": relation_role,
                "source_id": ref.object_id,
                "source_kind": ref.source_kind,
            }
            for ref, relation_role in sources
        ]
        require(bool(source_rows), "CSRG_BINDING_WITHOUT_SOURCE")
        material = {
            "domain_scope_id": self.scope_id,
            "occurrence_id": occurrence_id,
            "outcome": {
                "content_sha256": outcome.content_sha256,
                "object_id": outcome.object_id,
                "outcome_role": outcome.role,
            },
            "payload": payload or {},
            "sources": source_rows,
        }
        block_id = "factblock_" + payload_sha256(material)
        self._group["fact_blocks"].append({"fact_block_id": block_id, **material})
        for source in source_rows:
            self.relation(
                "reads_from",
                source["source_id"],
                occurrence_id,
                {"fact_block_id": block_id, "relation_role": source["relation_role"]},
            )
            if source["source_kind"] == "generated_origin":
                self.relation(
                    "generated_origin_dependency",
                    source["source_id"],
                    outcome.object_id,
                    {"fact_block_id": block_id},
                )
        self.relation(
            "realizes_fact",
            occurrence_id,
            block_id,
            {"outcome_object_id": outcome.object_id},
        )
        return block_id

    def relation(
        self,
        relation_type: str,
        source_id: str,
        target_id: str,
        payload: dict[str, Any],
    ) -> None:
        require(self._group is not None, "CSRG_RELATION_OUTSIDE_BLOCK")
        material = {
            "payload": payload,
            "relation_type": relation_type,
            "source_id": source_id,
            "target_id": target_id,
        }
        self._group["relations"].append(
            {"relation_id": "rel_" + payload_sha256(material), **material}
        )

    def flush_block(self) -> None:
        require(self._group is not None, "CSRG_BLOCK_NOT_OPEN")
        payload = self._group
        self._group = None
        raw = canonical_bytes(payload)
        payload_sha = hashlib.sha256(raw).hexdigest()
        material = {
            "block_ordinal": self._block_ordinal,
            "optimizer_step": payload["optimizer_step"],
            "payload_sha256": payload_sha,
            "prior_block_sha256": self._last_block_sha256,
            "scope_id": self.scope_id,
            "stage": payload["stage"],
        }
        block_sha = payload_sha256(material)
        fact_count = sum(len(row["sources"]) for row in payload["fact_blocks"])
        self.connection.execute(
            """
            INSERT INTO graph_blocks(
              block_ordinal,block_id,optimizer_step,stage,
              prior_block_sha256,payload_sha256,block_sha256,
              object_count,occurrence_count,fact_count,relation_count,payload_zlib
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                self._block_ordinal,
                "block_" + block_sha,
                payload["optimizer_step"],
                payload["stage"],
                self._last_block_sha256,
                payload_sha,
                block_sha,
                len(payload["objects"]),
                len(payload["occurrences"]),
                fact_count,
                len(payload["relations"]),
                zlib.compress(raw, level=9),
            ),
        )
        self._last_block_sha256 = block_sha
        self._block_ordinal += 1

    def add_checkpoint(self, row: dict[str, Any]) -> None:
        columns = [
            "optimizer_step",
            "current_baseline_logits_sha256",
            "historical_baseline_logits_sha256",
            "historical_raw_logits_exact",
            "historical_predictions_exact",
            "historical_capability_exact",
            "historical_max_abs_logit_error",
            "single_gate_hashes_json",
            "pair_gate_hashes_json",
            "derived_object_ids_json",
            "actual_forward_count",
            "status",
        ]
        values = dict(row)
        for key in ("single_gate_hashes_json", "pair_gate_hashes_json", "derived_object_ids_json"):
            values[key] = _json(values[key])
        self.connection.execute(
            "INSERT INTO checkpoint_index(" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
            tuple(values[column] for column in columns),
        )

    def close(self) -> dict[str, Any]:
        require(self._group is None, "CSRG_BLOCK_LEFT_OPEN")
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        counts = {
            "blocks": self.connection.execute("SELECT COUNT(*) FROM graph_blocks").fetchone()[0],
            "checkpoints": self.connection.execute("SELECT COUNT(*) FROM checkpoint_index").fetchone()[0],
            "facts": self.connection.execute("SELECT COALESCE(SUM(fact_count),0) FROM graph_blocks").fetchone()[0],
            "objects": self.connection.execute("SELECT COALESCE(SUM(object_count),0) FROM graph_blocks").fetchone()[0],
            "occurrences": self.connection.execute("SELECT COALESCE(SUM(occurrence_count),0) FROM graph_blocks").fetchone()[0],
            "origins": self.connection.execute("SELECT COUNT(*) FROM origin_catalog").fetchone()[0],
            "relations": self.connection.execute("SELECT COALESCE(SUM(relation_count),0) FROM graph_blocks").fetchone()[0],
            "tensor_payloads": len(list(self.tensor_directory.glob("*.npy"))),
        }
        self.connection.close()
        return {
            "counts": counts,
            "database": self.database_path.name,
            "database_sha256": file_sha256(self.database_path),
            "final_block_sha256": self._last_block_sha256,
            "schema": self.manifest_schema,
            "scope_id": self.scope_id,
            "source_bundle_id": self.source_bundle_id,
            "status": "CAPTURE_CLOSED",
        }


class SupportGFG:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.connection = sqlite3.connect(
            f"file:{self.database_path.as_posix()}?mode=ro", uri=True
        )
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def metadata(self, key: str) -> Any:
        row = self.connection.execute(
            "SELECT value_json FROM metadata WHERE key=?", (key,)
        ).fetchone()
        require(row is not None, f"CSRG_METADATA_MISSING:{key}")
        return json.loads(row[0])

    def blocks(self) -> Iterator[tuple[sqlite3.Row, dict[str, Any]]]:
        for row in self.connection.execute(
            "SELECT * FROM graph_blocks ORDER BY block_ordinal"
        ):
            yield row, decode_block(row["payload_zlib"])

    def checkpoints(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM checkpoint_index ORDER BY optimizer_step"
            )
        ]


def _source_objects_for_step(graph: TrainingGFG, step: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _row, block in graph.blocks(min_step=step, max_step=step):
        for value in block.get("objects", []):
            result[value["object_id"]] = value
    return result


def validate_support_gfg(
    database_path: Path,
    *,
    source_database_path: Path,
    tensor_directory: Path,
    report_path: Path | None = None,
    expected_schema: str = "nanogpt-support-redundancy-gfg-v1",
    require_checkpoint_grid: bool = True,
) -> dict[str, Any]:
    graph = SupportGFG(database_path)
    source_graph = TrainingGFG(source_database_path)
    checks = 0
    try:
        require(graph.metadata("schema") == expected_schema, "CSRG_SCHEMA_INVALID")
        prior_block: str | None = None
        known_objects: dict[str, str] = {}
        known_occurrences: set[str] = set()
        known_facts: set[str] = set()
        origins = {
            row["origin_id"]: dict(row)
            for row in graph.connection.execute("SELECT * FROM origin_catalog")
        }
        source_cache: dict[int, dict[str, dict[str, Any]]] = {}
        for origin in origins.values():
            step = int(origin["source_optimizer_step"])
            if step not in source_cache:
                source_cache[step] = _source_objects_for_step(source_graph, step)
            source = source_cache[step].get(origin["source_object_id"])
            require(source is not None, "CSRG_GENERATED_ORIGIN_SOURCE_MISSING")
            require(source["content_sha256"] == origin["source_content_sha256"], "CSRG_GENERATED_ORIGIN_HASH_MISMATCH")
            require(source["role"] == origin["source_role"], "CSRG_GENERATED_ORIGIN_ROLE_MISMATCH")
            checks += 3

        for row, block in graph.blocks():
            raw = canonical_bytes(block)
            require(hashlib.sha256(raw).hexdigest() == row["payload_sha256"], "CSRG_BLOCK_PAYLOAD_HASH_MISMATCH")
            require(row["prior_block_sha256"] == prior_block, "CSRG_BLOCK_CHAIN_MISMATCH")
            require(len(block["objects"]) == row["object_count"], "CSRG_OBJECT_COUNT_MISMATCH")
            require(len(block["occurrences"]) == row["occurrence_count"], "CSRG_OCCURRENCE_COUNT_MISMATCH")
            require(len(block["relations"]) == row["relation_count"], "CSRG_RELATION_COUNT_MISMATCH")
            require(sum(len(value["sources"]) for value in block["fact_blocks"]) == row["fact_count"], "CSRG_FACT_COUNT_MISMATCH")
            for value in block["objects"]:
                require(value["object_id"] not in known_objects, "CSRG_DUPLICATE_OBJECT_ID")
                require(payload_sha256(value["payload"]) == value["content_sha256"], "CSRG_OBJECT_CONTENT_HASH_MISMATCH")
                known_objects[value["object_id"]] = value["content_sha256"]
                if value["object_kind"] == "content_addressed_tensor":
                    payload = value["payload"]
                    path = tensor_directory.parent / payload["locator"]
                    require(path.is_file(), "CSRG_TENSOR_PAYLOAD_MISSING")
                    require(file_sha256(path) == payload["file_sha256"], "CSRG_TENSOR_FILE_HASH_MISMATCH")
                    array = np.load(path, allow_pickle=False)
                    require(list(array.shape) == payload["shape"], "CSRG_TENSOR_SHAPE_MISMATCH")
                    require(str(array.dtype) == payload["dtype"], "CSRG_TENSOR_DTYPE_MISMATCH")
                    require(hashlib.sha256(array.tobytes(order="C")).hexdigest() == payload["raw_tensor_sha256"], "CSRG_TENSOR_RAW_HASH_MISMATCH")
                    checks += 5
            for occurrence in block["occurrences"]:
                occurrence_id = occurrence["occurrence_id"]
                require(occurrence_id not in known_occurrences, "CSRG_DUPLICATE_OCCURRENCE_ID")
                require(bool(occurrence["transform_reference"]), "CSRG_TRANSFORM_REFERENCE_MISSING")
                known_occurrences.add(occurrence_id)
            for fact in block["fact_blocks"]:
                fact_id = fact["fact_block_id"]
                require(fact_id not in known_facts, "CSRG_DUPLICATE_FACT_BLOCK_ID")
                require(fact["occurrence_id"] in known_occurrences, "CSRG_FACT_OCCURRENCE_MISSING")
                outcome_id = fact["outcome"]["object_id"]
                require(outcome_id in known_objects, "CSRG_FACT_OUTCOME_MISSING")
                require(known_objects[outcome_id] == fact["outcome"]["content_sha256"], "CSRG_FACT_OUTCOME_HASH_MISMATCH")
                require(bool(fact["sources"]), "CSRG_FACT_SOURCE_EMPTY")
                for source in fact["sources"]:
                    source_id = source["source_id"]
                    source_hash = known_objects.get(source_id)
                    if source_hash is None and source_id in origins:
                        source_hash = payload_sha256(json.loads(origins[source_id]["payload_json"]))
                    require(source_hash is not None, "CSRG_FACT_SOURCE_MISSING")
                    require(source_hash == source["content_sha256"], "CSRG_FACT_SOURCE_HASH_MISMATCH")
                    require(bool(source["relation_role"]), "CSRG_FACT_RELATION_ROLE_MISSING")
                known_facts.add(fact_id)
            valid_relation_types = {"generated_origin_dependency", "program_order", "reads_from", "realizes_fact"}
            for relation in block["relations"]:
                require(relation["relation_type"] in valid_relation_types, "CSRG_RELATION_TYPE_INVALID")
                if relation["relation_type"] == "realizes_fact":
                    require(relation["source_id"] in known_occurrences, "CSRG_INCIDENCE_SOURCE_MISSING")
                    require(relation["target_id"] in known_facts, "CSRG_INCIDENCE_TARGET_MISSING")
                elif relation["relation_type"] == "program_order":
                    require(relation["source_id"] in known_occurrences, "CSRG_ORDER_SOURCE_MISSING")
                    require(relation["target_id"] in known_occurrences, "CSRG_ORDER_TARGET_MISSING")
            prior_block = row["block_sha256"]
            checks += 7

        checkpoints = graph.checkpoints()
        if require_checkpoint_grid:
            require(bool(checkpoints), "CSRG_CHECKPOINT_GRID_EMPTY")
            expected_grid = list(range(100, int(checkpoints[-1]["optimizer_step"]) + 1, 100))
            require([row["optimizer_step"] for row in checkpoints] == expected_grid, "CSRG_CHECKPOINT_GRID_INCOMPLETE")
            require(all(row["actual_forward_count"] == 12 for row in checkpoints), "CSRG_FORWARD_COVERAGE_INCOMPLETE")
            require(all(row["status"] == "PASS" for row in checkpoints), "CSRG_CHECKPOINT_STATUS_NOT_PASS")
        result = {
            "checks": checks,
            "counts": {
                "blocks": graph.connection.execute("SELECT COUNT(*) FROM graph_blocks").fetchone()[0],
                "checkpoints": len(checkpoints),
                "facts": graph.connection.execute("SELECT COALESCE(SUM(fact_count),0) FROM graph_blocks").fetchone()[0],
                "objects": len(known_objects),
                "occurrences": len(known_occurrences),
                "origins": len(origins),
                "relations": graph.connection.execute("SELECT COALESCE(SUM(relation_count),0) FROM graph_blocks").fetchone()[0],
            },
            "database_sha256": file_sha256(database_path),
            "schema": "nanogpt-support-redundancy-gfg-validation-v1",
            "status": "PASS",
        }
        result["validation_sha256"] = payload_sha256(result)
        if report_path is not None:
            write_json(report_path, result)
        return result
    finally:
        source_graph.close()
        graph.close()
