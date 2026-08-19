from __future__ import annotations

from typing import Any

import torch

from .graph_canonicalization import canonicalize_graph


def _find_node(nodes: list[torch.autograd.graph.Node], target: torch.autograd.graph.Node) -> int | None:
    for index, node in enumerate(nodes):
        if node is target:
            return index
    return None


def observe_native_autograd_graph(loss: torch.Tensor) -> dict[str, Any]:
    """Traverse only the installed public Autograd graph surface."""
    root = loss.grad_fn
    if root is None:
        raise ValueError("NATIVE_AUTOGRAD_ROOT_MISSING")
    nodes: list[torch.autograd.graph.Node] = [root]
    queue: list[torch.autograd.graph.Node] = [root]
    visited: list[torch.autograd.graph.Node] = []
    raw_nodes: list[dict[str, Any]] = []
    raw_edges: list[dict[str, Any]] = []

    while queue:
        node = queue.pop(0)
        if any(node is item for item in visited):
            continue
        visited.append(node)
        source_index = _find_node(nodes, node)
        if source_index is None:
            raise RuntimeError("NATIVE_TRAVERSAL_IDENTITY_LOST")
        raw_nodes.append({"key": f"runtime_node_{source_index}", "node_type": node.name()})
        for slot, (target, output_nr) in enumerate(node.next_functions):
            target_key = None
            if target is not None:
                target_index = _find_node(nodes, target)
                if target_index is None:
                    nodes.append(target)
                    target_index = len(nodes) - 1
                target_key = f"runtime_node_{target_index}"
                queue.append(target)
            raw_edges.append({
                "output_nr": int(output_nr),
                "slot": slot,
                "source_key": f"runtime_node_{source_index}",
                "target_key": target_key,
            })
    if len(raw_nodes) != len(nodes):
        raise RuntimeError("NATIVE_TRAVERSAL_NODE_COVERAGE_FAILED")
    return canonicalize_graph({
        "edges": raw_edges,
        "nodes": raw_nodes,
        "root_key": "runtime_node_0",
    })
