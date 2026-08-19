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
    """Read the numeric C-order NPY payloads admitted by this experiment.

    The participant image intentionally has no third-party Python packages.
    This reader therefore implements only the frozen payload contract:
    float32, float64, or int64; no pickle; and C-order layout. Unsupported
    descriptors fail closed instead of being guessed or coerced.
    """

    payload = path.read_bytes()
    require(payload[:6] == b"\x93NUMPY", "CST_QUERY_TENSOR_NOT_NPY")
    major = payload[6]
    if major == 1:
        require(len(payload) >= 10, "CST_QUERY_TENSOR_HEADER_TRUNCATED")
        header_length = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        require(len(payload) >= 12, "CST_QUERY_TENSOR_HEADER_TRUNCATED")
        header_length = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
    else:
        raise RuntimeError(f"CST_QUERY_TENSOR_NPY_VERSION_UNSUPPORTED:{major}")
    header_end = header_start + header_length
    require(header_end <= len(payload), "CST_QUERY_TENSOR_HEADER_TRUNCATED")
    header = ast.literal_eval(payload[header_start:header_end].decode("latin1").strip())
    require(isinstance(header, dict), "CST_QUERY_TENSOR_HEADER_INVALID")
    require(header.get("fortran_order") is False, "CST_QUERY_TENSOR_FORTRAN_ORDER_UNSUPPORTED")
    descriptor = str(header.get("descr"))
    require(descriptor in _NPY_FORMATS, f"CST_QUERY_TENSOR_DTYPE_UNSUPPORTED:{descriptor}")
    shape = tuple(int(value) for value in header.get("shape", ()))
    require(all(value >= 0 for value in shape), "CST_QUERY_TENSOR_SHAPE_INVALID")
    element_count = math.prod(shape) if shape else 1
    fmt = _NPY_FORMATS[descriptor]
    item_size = struct.calcsize(fmt)
    raw = payload[header_end:]
    require(
        len(raw) == element_count * item_size,
        "CST_QUERY_TENSOR_PAYLOAD_LENGTH_INVALID",
    )
    values = [item[0] for item in struct.iter_unpack(fmt, raw)]
    return shape, descriptor, values


def _reshape(values: list[float | int], shape: tuple[int, ...]) -> Any:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return list(values)
    stride = math.prod(shape[1:])
    return [
        _reshape(values[index * stride : (index + 1) * stride], shape[1:])
        for index in range(shape[0])
    ]


class TransitionGraph:
    def __init__(self, entry_directory: Path) -> None:
        self.entry_directory = entry_directory.resolve()
        database = self.entry_directory / "support_transition_gfg.sqlite3"
        require(database.is_file(), f"CST_QUERY_DATABASE_MISSING:{database}")
        self.connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.objects: dict[str, dict[str, Any]] = {}
        self.occurrences: dict[str, dict[str, Any]] = {}
        self.facts: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []
        self.origins = {
            row["origin_id"]: json.loads(row["payload_json"])
            for row in self.connection.execute("SELECT origin_id,payload_json FROM origin_catalog")
        }
        for payload in self.blocks():
            self.objects.update((row["object_id"], row) for row in payload["objects"])
            self.occurrences.update((row["occurrence_id"], row) for row in payload["occurrences"])
            self.facts.update((row["fact_block_id"], row) for row in payload["fact_blocks"])
            self.relations.extend(payload["relations"])

    def close(self) -> None:
        self.connection.close()

    def blocks(self) -> Iterator[dict[str, Any]]:
        rows = self.connection.execute("SELECT payload_zlib FROM graph_blocks ORDER BY block_ordinal")
        for row in rows:
            yield json.loads(zlib.decompress(row["payload_zlib"]))

    def node(self, node_id: str) -> dict[str, Any]:
        if node_id in self.objects:
            return {"node_kind": "object", **self.objects[node_id]}
        if node_id in self.occurrences:
            return {"node_kind": "occurrence", **self.occurrences[node_id]}
        if node_id in self.facts:
            return {"node_kind": "atomic_fact_block", **self.facts[node_id]}
        if node_id in self.origins:
            return {"node_kind": "GeneratedOrigin", "origin_id": node_id, **self.origins[node_id]}
        raise KeyError(node_id)

    def occurrence_bundle(self, occurrence_id: str) -> dict[str, Any]:
        occurrence = self.occurrences[occurrence_id]
        facts = [row for row in self.facts.values() if row["occurrence_id"] == occurrence_id]
        return {
            "occurrence": occurrence,
            "facts": facts,
            "outcomes": [self.objects[row["outcome"]["object_id"]] for row in facts],
            "relations": [
                row
                for row in self.relations
                if row["source_id"] == occurrence_id or row["target_id"] == occurrence_id
            ],
        }

    def find_occurrences(
        self,
        *,
        occurrence_type: str | None = None,
        optimizer_step: int | None = None,
        branch: str | None = None,
        horizon: int | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for row in self.occurrences.values():
            if occurrence_type is not None and row["occurrence_type"] != occurrence_type:
                continue
            if optimizer_step is not None and int(row["optimizer_step"]) != optimizer_step:
                continue
            if branch is not None and row["payload"].get("branch") != branch:
                continue
            if horizon is not None and int(row["payload"].get("horizon", -1)) != horizon:
                continue
            result.append(row)
        return sorted(result, key=lambda row: (int(row["optimizer_step"]), int(row["ordinal"])))

    def traverse(self, node_id: str, *, depth: int, direction: str) -> dict[str, Any]:
        require(direction in {"in", "out", "both"}, "CST_QUERY_DIRECTION_INVALID")
        require(0 <= depth <= 8, "CST_QUERY_DEPTH_INVALID")
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for relation in self.relations:
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
        return {
            "edges": edges,
            "nodes": [self.node(value) for value in sorted(visited)],
            "start_node_id": node_id,
        }

    def tensor(self, object_id: str, *, mode: str, max_elements: int) -> dict[str, Any]:
        row = self.objects[object_id]
        require(row["object_kind"] == "content_addressed_tensor", "CST_QUERY_OBJECT_NOT_TENSOR")
        path = self.entry_directory / row["payload"]["locator"]
        shape, descriptor, values = _npy_values(path)
        result: dict[str, Any] = {"object": row, "size": len(values)}
        if mode == "stats":
            is_float = descriptor.endswith(("f4", "f8"))
            finite = [float(value) for value in values if not is_float or math.isfinite(value)]
            result["statistics"] = {
                "finite_count": len(finite),
                "maximum": max(finite) if finite else None,
                "mean": (sum(finite) / len(finite)) if finite else None,
                "minimum": min(finite) if finite else None,
                "nan_count": sum(1 for value in values if is_float and math.isnan(value)),
            }
        else:
            require(len(values) <= max_elements, "CST_QUERY_TENSOR_EXCEEDS_MAX_ELEMENTS")
            result["values"] = _reshape(values, shape)
        return result

    def summary(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for row in self.occurrences.values():
            types[row["occurrence_type"]] = types.get(row["occurrence_type"], 0) + 1
        return {
            "entry_id": self.entry_directory.name,
            "fact_blocks": len(self.facts),
            "objects": len(self.objects),
            "occurrence_types": dict(sorted(types.items())),
            "occurrences": len(self.occurrences),
            "origins": len(self.origins),
            "relations": len(self.relations),
        }


def _entry(root: Path, entry_id: str) -> TransitionGraph:
    return TransitionGraph(root / entry_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-entries")
    summary = sub.add_parser("summary")
    summary.add_argument("--entry-id", required=True)
    find = sub.add_parser("find-occurrences")
    find.add_argument("--entry-id", required=True)
    find.add_argument("--occurrence-type")
    find.add_argument("--optimizer-step", type=int)
    find.add_argument("--branch")
    find.add_argument("--horizon", type=int)
    occurrence = sub.add_parser("occurrence")
    occurrence.add_argument("--entry-id", required=True)
    occurrence.add_argument("--occurrence-id", required=True)
    node = sub.add_parser("node")
    node.add_argument("--entry-id", required=True)
    node.add_argument("--node-id", required=True)
    traverse = sub.add_parser("traverse")
    traverse.add_argument("--entry-id", required=True)
    traverse.add_argument("--node-id", required=True)
    traverse.add_argument("--depth", type=int, default=2)
    traverse.add_argument("--direction", choices=("in", "out", "both"), default="both")
    tensor = sub.add_parser("tensor")
    tensor.add_argument("--entry-id", required=True)
    tensor.add_argument("--object-id", required=True)
    tensor.add_argument("--mode", choices=("stats", "full"), default="stats")
    tensor.add_argument("--max-elements", type=int, default=100000)
    args = parser.parse_args()
    if args.command == "list-entries":
        result = sorted(path.name for path in args.root.glob("entry-*") if (path / "support_transition_gfg.sqlite3").is_file())
        print(json.dumps(result, sort_keys=True))
        return
    graph = _entry(args.root, args.entry_id)
    try:
        if args.command == "summary":
            result = graph.summary()
        elif args.command == "find-occurrences":
            result = graph.find_occurrences(
                occurrence_type=args.occurrence_type,
                optimizer_step=args.optimizer_step,
                branch=args.branch,
                horizon=args.horizon,
            )
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
