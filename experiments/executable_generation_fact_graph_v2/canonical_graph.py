from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from generation_relation_core.canonical import canonical_bytes


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONTRACT_ROOT = EXPERIMENT_ROOT / "contracts"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def content_id(prefix: str, value: Any) -> str:
    return prefix + canonical_hash(value)


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def load_contract(name: str) -> dict[str, Any]:
    return json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hash(name: str) -> str:
    return file_sha256(EXPERIMENT_ROOT / name)


def graph_material(document: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(document["metadata"])
    metadata.pop("graph_id", None)
    return {
        "metadata": metadata,
        "fact_nodes": sorted(
            document["fact_nodes"], key=lambda row: row["graph_node_id"]
        ),
        "occurrence_nodes": sorted(
            document["occurrence_nodes"],
            key=lambda row: row["graph_node_id"],
        ),
        "incidence_edges": sorted(
            document["incidence_edges"],
            key=lambda row: row["graph_edge_id"],
        ),
        "relation_edges": sorted(
            document["relation_edges"],
            key=lambda row: row["graph_edge_id"],
        ),
    }


def graph_id(document: dict[str, Any]) -> str:
    return content_id("gfg2_", graph_material(document))


def canonical_graph_document(
    document: dict[str, Any],
) -> dict[str, Any]:
    result = graph_material(document)
    result["metadata"]["graph_id"] = graph_id(result)
    return result
