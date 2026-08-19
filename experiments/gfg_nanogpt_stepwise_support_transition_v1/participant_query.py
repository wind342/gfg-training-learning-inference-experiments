from __future__ import annotations

import argparse
import ast
from collections import deque
import json
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any, Iterator
import zlib


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


_NPY_FORMATS = {
    "<f4": "<f",
    "<f8": "<d",
    "<i8": "<q",
    "=f4": "=f",
    "=f8": "=d",
    "=i8": "=q",
}


def _npy_values(path: Path) -> tuple[tuple[int, ...], str, list[float | int]]:
    payload = path.read_bytes()
    require(payload[:6] == b"\x93NUMPY", "SST_QUERY_TENSOR_NOT_NPY")
    major = payload[6]
    if major == 1:
        require(len(payload) >= 10, "SST_QUERY_TENSOR_HEADER_TRUNCATED")
        header_length = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        require(len(payload) >= 12, "SST_QUERY_TENSOR_HEADER_TRUNCATED")
        header_length = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
    else:
        raise RuntimeError(f"SST_QUERY_TENSOR_NPY_VERSION_UNSUPPORTED:{major}")
    header_end = header_start + header_length
    require(header_end <= len(payload), "SST_QUERY_TENSOR_HEADER_TRUNCATED")
    header = ast.literal_eval(payload[header_start:header_end].decode("latin1").strip())
    require(isinstance(header, dict), "SST_QUERY_TENSOR_HEADER_INVALID")
    require(header.get("fortran_order") is False, "SST_QUERY_TENSOR_FORTRAN_ORDER_UNSUPPORTED")
    descriptor = str(header.get("descr"))
    require(descriptor in _NPY_FORMATS, f"SST_QUERY_TENSOR_DTYPE_UNSUPPORTED:{descriptor}")
    shape = tuple(int(value) for value in header.get("shape", ()))
    require(all(value >= 0 for value in shape), "SST_QUERY_TENSOR_SHAPE_INVALID")
    element_count = math.prod(shape) if shape else 1
    fmt = _NPY_FORMATS[descriptor]
    raw = payload[header_end:]
    require(len(raw) == element_count * struct.calcsize(fmt), "SST_QUERY_TENSOR_PAYLOAD_LENGTH_INVALID")
    return shape, descriptor, [item[0] for item in struct.iter_unpack(fmt, raw)]


def _reshape(values: list[float | int], shape: tuple[int, ...]) -> Any:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    stride = math.prod(shape[1:])
    return [_reshape(values[index * stride : (index + 1) * stride], shape[1:]) for index in range(shape[0])]


class StepwiseGraph:
    def __init__(self, graph_directory: Path) -> None:
        self.graph_directory = graph_directory.resolve()
        candidates = [
            path
            for path in self.graph_directory.glob("*.sqlite3")
            if not path.name.startswith("failed-")
        ]
        require(len(candidates) == 1, f"SST_QUERY_DATABASE_NOT_UNIQUE:{self.graph_directory}")
        self.connection = sqlite3.connect(f"file:{candidates[0].as_posix()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def blocks(self, optimizer_step: int | None = None) -> Iterator[dict[str, Any]]:
        query = "SELECT payload_zlib FROM graph_blocks"
        parameters: tuple[Any, ...] = ()
        if optimizer_step is not None:
            query += " WHERE optimizer_step=?"
            parameters = (optimizer_step,)
        query += " ORDER BY block_ordinal"
        for row in self.connection.execute(query, parameters):
            yield json.loads(zlib.decompress(row["payload_zlib"]))

    def summary(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT COUNT(*) blocks,COALESCE(SUM(object_count),0) objects,"
            "COALESCE(SUM(occurrence_count),0) occurrences,COALESCE(SUM(fact_count),0) facts,"
            "COALESCE(SUM(relation_count),0) relations FROM graph_blocks"
        ).fetchone()
        occurrence_types: dict[str, int] = {}
        stages: dict[str, int] = {}
        for block in self.blocks():
            stages[block["stage"]] = stages.get(block["stage"], 0) + 1
            for occurrence in block["occurrences"]:
                key = occurrence["occurrence_type"]
                occurrence_types[key] = occurrence_types.get(key, 0) + 1
        return {
            "graph_directory": self.graph_directory.name,
            "blocks": int(row["blocks"]),
            "objects": int(row["objects"]),
            "occurrences": int(row["occurrences"]),
            "facts": int(row["facts"]),
            "relations": int(row["relations"]),
            "origins": int(self.connection.execute("SELECT COUNT(*) FROM origin_catalog").fetchone()[0]),
            "occurrence_types": dict(sorted(occurrence_types.items())),
            "stages": dict(sorted(stages.items())),
        }

    def _origin(self, node_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM origin_catalog WHERE origin_id=?", (node_id,)
        ).fetchone()
        return None if row is None else {"node_kind": "GeneratedOrigin", "origin_id": node_id, **json.loads(row[0])}

    def node(self, node_id: str) -> dict[str, Any]:
        origin = self._origin(node_id)
        if origin is not None:
            return origin
        for block in self.blocks():
            for key, kind, identity in (
                ("objects", "object", "object_id"),
                ("occurrences", "occurrence", "occurrence_id"),
                ("fact_blocks", "atomic_fact_block", "fact_block_id"),
            ):
                for row in block[key]:
                    if row[identity] == node_id:
                        return {"node_kind": kind, "graph_stage": block["stage"], **row}
        raise KeyError(node_id)

    def find_occurrences(
        self,
        *,
        occurrence_type: str | None = None,
        optimizer_step: int | None = None,
        branch: str | None = None,
        horizon: int | None = None,
        seed_id: str | None = None,
        window_id: str | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for block in self.blocks(optimizer_step):
            if window_id is not None and f"window:{window_id}:" not in str(block["stage"]):
                continue
            for row in block["occurrences"]:
                payload = row.get("payload", {})
                if occurrence_type is not None and row["occurrence_type"] != occurrence_type:
                    continue
                if branch is not None and payload.get("branch") != branch:
                    continue
                if horizon is not None and int(payload.get("horizon", -1)) != horizon:
                    continue
                if seed_id is not None and payload.get("seed_id") != seed_id:
                    continue
                result.append({**row, "graph_stage": block["stage"]})
        return sorted(result, key=lambda row: (int(row["optimizer_step"]), int(row["ordinal"])))

    def occurrence_bundle(self, occurrence_id: str) -> dict[str, Any]:
        occurrence = None
        facts = []
        relations = []
        object_ids: set[str] = set()
        for block in self.blocks():
            for row in block["occurrences"]:
                if row["occurrence_id"] == occurrence_id:
                    occurrence = {**row, "graph_stage": block["stage"]}
            for row in block["fact_blocks"]:
                if row["occurrence_id"] == occurrence_id:
                    facts.append(row)
                    object_ids.add(row["outcome"]["object_id"])
                    object_ids.update(source["source_id"] for source in row["sources"] if source["source_kind"] != "generated_origin")
            for row in block["relations"]:
                if row["source_id"] == occurrence_id or row["target_id"] == occurrence_id:
                    relations.append(row)
        require(occurrence is not None, f"SST_QUERY_OCCURRENCE_MISSING:{occurrence_id}")
        objects = [
            row
            for block in self.blocks()
            for row in block["objects"]
            if row["object_id"] in object_ids
        ]
        return {"occurrence": occurrence, "facts": facts, "objects": objects, "relations": relations}

    def traverse(self, node_id: str, *, depth: int, direction: str) -> dict[str, Any]:
        require(direction in {"in", "out", "both"}, "SST_QUERY_DIRECTION_INVALID")
        require(0 <= depth <= 8, "SST_QUERY_DEPTH_INVALID")
        relations = [relation for block in self.blocks() for relation in block["relations"]]
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for relation in relations:
            if direction in {"out", "both"}:
                adjacency.setdefault(relation["source_id"], []).append(relation)
            if direction in {"in", "both"}:
                reverse = {**relation, "source_id": relation["target_id"], "target_id": relation["source_id"], "traversed_reverse": True}
                adjacency.setdefault(reverse["source_id"], []).append(reverse)
        queue = deque([(node_id, 0)])
        visited = {node_id}
        edges = []
        while queue:
            current, level = queue.popleft()
            if level >= depth:
                continue
            for relation in adjacency.get(current, []):
                edges.append(relation)
                target = relation["target_id"]
                if target not in visited:
                    visited.add(target)
                    queue.append((target, level + 1))
        return {"start_node_id": node_id, "nodes": [self.node(value) for value in sorted(visited)], "edges": edges}

    def tensor(self, object_id: str, *, mode: str, max_elements: int) -> dict[str, Any]:
        row = self.node(object_id)
        require(row["node_kind"] == "object" and row["object_kind"] == "content_addressed_tensor", "SST_QUERY_OBJECT_NOT_TENSOR")
        path = self.graph_directory / row["payload"]["locator"]
        shape, descriptor, values = _npy_values(path)
        result: dict[str, Any] = {"object": row, "size": len(values)}
        if mode == "stats":
            is_float = descriptor.endswith(("f4", "f8"))
            finite = [float(value) for value in values if not is_float or math.isfinite(value)]
            result["statistics"] = {
                "finite_count": len(finite),
                "maximum": max(finite) if finite else None,
                "mean": sum(finite) / len(finite) if finite else None,
                "minimum": min(finite) if finite else None,
                "nan_count": sum(1 for value in values if is_float and math.isnan(value)),
            }
        else:
            require(len(values) <= max_elements, "SST_QUERY_TENSOR_EXCEEDS_MAX_ELEMENTS")
            result["values"] = _reshape(values, shape)
        return result


def _graph(root: Path, entry_id: str, graph_kind: str) -> StepwiseGraph:
    require(graph_kind in {"stepwise", "causal-branch"}, "SST_QUERY_GRAPH_KIND_INVALID")
    return StepwiseGraph(root / "entries" / entry_id / graph_kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-entries")
    sub.add_parser("list-receipts")
    receipt = sub.add_parser("receipt")
    receipt.add_argument("--name", required=True)
    for command in ("summary", "find-occurrences", "occurrence", "node", "traverse", "tensor"):
        item = sub.add_parser(command)
        item.add_argument("--entry-id", required=True)
        item.add_argument("--graph-kind", choices=("stepwise", "causal-branch"), default="stepwise")
        if command == "find-occurrences":
            item.add_argument("--occurrence-type")
            item.add_argument("--optimizer-step", type=int)
            item.add_argument("--branch")
            item.add_argument("--horizon", type=int)
            item.add_argument("--seed-id")
            item.add_argument("--window-id")
        elif command == "occurrence":
            item.add_argument("--occurrence-id", required=True)
        elif command == "node":
            item.add_argument("--node-id", required=True)
        elif command == "traverse":
            item.add_argument("--node-id", required=True)
            item.add_argument("--depth", type=int, default=2)
            item.add_argument("--direction", choices=("in", "out", "both"), default="both")
        elif command == "tensor":
            item.add_argument("--object-id", required=True)
            item.add_argument("--mode", choices=("stats", "full"), default="stats")
            item.add_argument("--max-elements", type=int, default=100000)
    args = parser.parse_args()
    if args.command == "list-entries":
        result = {
            path.name: sorted(child.name for child in path.iterdir() if child.is_dir())
            for path in sorted((args.root / "entries").glob("entry-*"))
        }
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "list-receipts":
        print(json.dumps(sorted(path.name for path in (args.root / "machine-receipts").glob("*.json")), sort_keys=True))
        return
    if args.command == "receipt":
        path = (args.root / "machine-receipts" / args.name).resolve()
        require(path.parent == (args.root / "machine-receipts").resolve(), "SST_QUERY_RECEIPT_PATH_INVALID")
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True))
        return
    graph = _graph(args.root, args.entry_id, args.graph_kind)
    try:
        if args.command == "summary":
            result = graph.summary()
        elif args.command == "find-occurrences":
            result = graph.find_occurrences(occurrence_type=args.occurrence_type, optimizer_step=args.optimizer_step, branch=args.branch, horizon=args.horizon, seed_id=args.seed_id, window_id=args.window_id)
        elif args.command == "occurrence":
            result = graph.occurrence_bundle(args.occurrence_id)
        elif args.command == "node":
            result = graph.node(args.node_id)
        elif args.command == "traverse":
            result = graph.traverse(args.node_id, depth=args.depth, direction=args.direction)
        elif args.command == "tensor":
            result = graph.tensor(args.object_id, mode=args.mode, max_elements=args.max_elements)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    finally:
        graph.close()


if __name__ == "__main__":
    main()
