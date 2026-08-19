from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _node_by_key(raw_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {row["key"]: row for row in raw_graph["nodes"]}
    if len(result) != len(raw_graph["nodes"]):
        raise ValueError("DUPLICATE_RAW_GRAPH_NODE_KEY")
    if raw_graph["root_key"] not in result:
        raise ValueError("RAW_GRAPH_ROOT_MISSING")
    return result


def _edges_by_source(raw_graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    occupied_slots: set[tuple[str, int]] = set()
    node_keys = {row["key"] for row in raw_graph["nodes"]}
    for edge in raw_graph["edges"]:
        source = edge["source_key"]
        target = edge["target_key"]
        slot = edge["slot"]
        if source not in node_keys or (target is not None and target not in node_keys):
            raise ValueError("RAW_GRAPH_EDGE_ENDPOINT_MISSING")
        if not isinstance(slot, int) or slot < 0:
            raise ValueError("RAW_GRAPH_SLOT_INVALID")
        slot_key = (source, slot)
        if slot_key in occupied_slots:
            raise ValueError("RAW_GRAPH_SLOT_DUPLICATED")
        occupied_slots.add(slot_key)
        result.setdefault(source, []).append(edge)
    for source in result:
        result[source].sort(key=lambda row: row["slot"])
        slots = [row["slot"] for row in result[source]]
        if slots != list(range(len(slots))):
            raise ValueError("RAW_GRAPH_SLOT_GAP")
    return result


def _root_paths(raw_graph: dict[str, Any]) -> dict[str, list[tuple[int, ...]]]:
    nodes = _node_by_key(raw_graph)
    outgoing = _edges_by_source(raw_graph)
    paths: dict[str, list[tuple[int, ...]]] = {key: [] for key in nodes}
    active: set[str] = set()

    def visit(node_key: str, path: tuple[int, ...]) -> None:
        if node_key in active:
            raise ValueError("RAW_GRAPH_CYCLE")
        paths[node_key].append(path)
        active.add(node_key)
        for edge in outgoing.get(node_key, []):
            target = edge["target_key"]
            if target is not None:
                visit(target, (*path, edge["slot"]))
        active.remove(node_key)

    visit(raw_graph["root_key"], ())
    if any(not node_paths for node_paths in paths.values()):
        raise ValueError("RAW_GRAPH_UNREACHABLE_NODE")
    return {key: sorted(set(node_paths)) for key, node_paths in paths.items()}


def _path_text(path: tuple[int, ...]) -> str:
    return "root" if not path else "root/" + "/".join(str(item) for item in path)


def canonicalize_graph(raw_graph: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a rooted ordered multigraph without retaining raw IDs."""
    nodes = _node_by_key(raw_graph)
    outgoing = _edges_by_source(raw_graph)
    paths = _root_paths(raw_graph)
    canonical_ids = {
        key: "agn_" + hashlib.sha256(canonical_bytes({
            "node_type": nodes[key]["node_type"],
            "root_paths": [_path_text(path) for path in paths[key]],
        })).hexdigest()
        for key in nodes
    }
    if len(set(canonical_ids.values())) != len(canonical_ids):
        raise ValueError("CANONICAL_NODE_ID_COLLISION")

    incoming: dict[str, list[dict[str, Any]]] = {key: [] for key in nodes}
    canonical_edges = []
    for source_key, source_edges in outgoing.items():
        for edge in source_edges:
            target_key = edge["target_key"]
            canonical_edge = {
                "output_nr": edge["output_nr"],
                "slot": edge["slot"],
                "source_node_id": canonical_ids[source_key],
                "target_node_id": None if target_key is None else canonical_ids[target_key],
            }
            canonical_edges.append(canonical_edge)
            if target_key is not None:
                incoming[target_key].append({
                    "output_nr": edge["output_nr"],
                    "slot": edge["slot"],
                    "source_node_id": canonical_ids[source_key],
                    "source_node_type": nodes[source_key]["node_type"],
                })

    canonical_nodes = []
    for key in nodes:
        node_outgoing = []
        for edge in outgoing.get(key, []):
            target = edge["target_key"]
            node_outgoing.append({
                "output_nr": edge["output_nr"],
                "slot": edge["slot"],
                "target_node_id": None if target is None else canonical_ids[target],
                "target_node_type": None if target is None else nodes[target]["node_type"],
            })
        path_texts = [_path_text(path) for path in paths[key]]
        canonical_nodes.append({
            "dependency_in_degree": len(incoming[key]),
            "dependency_out_degree": len([row for row in node_outgoing if row["target_node_id"] is not None]),
            "first_canonical_path": path_texts[0],
            "incoming_slot_signatures": sorted(
                incoming[key],
                key=lambda row: (
                    row["source_node_id"], row["slot"], row["output_nr"],
                ),
            ),
            "is_leaf_accumulator": nodes[key]["node_type"] == "struct torch::autograd::AccumulateGrad",
            "is_shared": len(path_texts) > 1,
            "node_id": canonical_ids[key],
            "node_type": nodes[key]["node_type"],
            "outgoing_slot_signatures": node_outgoing,
            "root_paths": path_texts,
            "shared_node_alias_closure": path_texts if len(path_texts) > 1 else [],
        })
    canonical_nodes.sort(key=lambda row: row["node_id"])
    canonical_edges.sort(key=lambda row: (
        row["source_node_id"],
        row["slot"],
        "" if row["target_node_id"] is None else row["target_node_id"],
        row["output_nr"],
    ))
    type_multiplicity: dict[str, int] = {}
    for row in canonical_nodes:
        type_multiplicity[row["node_type"]] = type_multiplicity.get(row["node_type"], 0) + 1
    payload = {
        "canonicalization": "root_ordered_paths_and_shared_alias_closure_v1",
        "edge_count": len(canonical_edges),
        "edges": canonical_edges,
        "node_count": len(canonical_nodes),
        "node_type_multiplicity": dict(sorted(type_multiplicity.items())),
        "nodes": canonical_nodes,
        "none_edge_count": sum(row["target_node_id"] is None for row in canonical_edges),
        "non_null_edge_count": sum(row["target_node_id"] is not None for row in canonical_edges),
        "root_node_id": canonical_ids[raw_graph["root_key"]],
        "shared_node_count": sum(row["is_shared"] for row in canonical_nodes),
    }
    return {
        **payload,
        "canonical_graph_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
    }
