from __future__ import annotations

import json
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def compare_canonical_graphs(native: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    native_nodes = {row["node_id"]: row for row in native["nodes"]}
    candidate_nodes = {row["node_id"]: row for row in candidate["nodes"]}
    native_edges = {_canonical(row) for row in native["edges"]}
    candidate_edges = {_canonical(row) for row in candidate["edges"]}
    common_nodes = set(native_nodes) & set(candidate_nodes)
    node_type_mismatches = sum(
        native_nodes[key]["node_type"] != candidate_nodes[key]["node_type"]
        for key in common_nodes
    )
    shared_mismatches = sum(
        (
            native_nodes[key]["is_shared"],
            native_nodes[key]["shared_node_alias_closure"],
        )
        != (
            candidate_nodes[key]["is_shared"],
            candidate_nodes[key]["shared_node_alias_closure"],
        )
        for key in common_nodes
    )
    edge_slot_mismatches = 0
    native_by_source_slot = {
        (row["source_node_id"], row["slot"]): row for row in native["edges"]
    }
    candidate_by_source_slot = {
        (row["source_node_id"], row["slot"]): row for row in candidate["edges"]
    }
    for key in set(native_by_source_slot) & set(candidate_by_source_slot):
        edge_slot_mismatches += native_by_source_slot[key] != candidate_by_source_slot[key]
    payload = {
        "canonical_bytes_exact": _canonical(native) == _canonical(candidate),
        "candidate_edge_count": candidate["edge_count"],
        "candidate_node_count": candidate["node_count"],
        "edge_mismatch": len(native_edges ^ candidate_edges),
        "edge_slot_mismatch": edge_slot_mismatches,
        "fabricated_edge": len(candidate_edges - native_edges),
        "fabricated_node": len(set(candidate_nodes) - set(native_nodes)),
        "missing_edge": len(native_edges - candidate_edges),
        "missing_leaf": len({
            key for key, row in native_nodes.items()
            if row["is_leaf_accumulator"] and key not in candidate_nodes
        }),
        "missing_node": len(set(native_nodes) - set(candidate_nodes)),
        "multiplicity_mismatch": native["node_type_multiplicity"] != candidate["node_type_multiplicity"],
        "native_edge_count": native["edge_count"],
        "native_node_count": native["node_count"],
        "node_type_mismatch": node_type_mismatches,
        "root_mismatch": native["root_node_id"] != candidate["root_node_id"],
        "shared_node_mismatch": shared_mismatches,
    }
    payload["exact"] = all([
        payload["canonical_bytes_exact"],
        payload["candidate_edge_count"] == payload["native_edge_count"],
        payload["candidate_node_count"] == payload["native_node_count"],
        payload["edge_mismatch"] == 0,
        payload["edge_slot_mismatch"] == 0,
        payload["fabricated_edge"] == 0,
        payload["fabricated_node"] == 0,
        payload["missing_edge"] == 0,
        payload["missing_leaf"] == 0,
        payload["missing_node"] == 0,
        not payload["multiplicity_mismatch"],
        payload["node_type_mismatch"] == 0,
        not payload["root_mismatch"],
        payload["shared_node_mismatch"] == 0,
    ])
    return payload
