from __future__ import annotations

from collections import Counter
from typing import Any, Hashable, Iterable

from generation_relation_core.canonical import canonical_bytes, sha256_bytes

from .errors import ProjectionProofError


def assert_acyclic_edges(edges: Iterable[tuple[Hashable, Hashable]]) -> None:
    graph: dict[Hashable, set[Hashable]] = {}
    for parent, child in edges:
        if parent == child:
            raise ProjectionProofError("HIERARCHY_CYCLE", str(parent))
        graph.setdefault(parent, set()).add(child)
    visiting: set[Hashable] = set()
    visited: set[Hashable] = set()

    def visit(node: Hashable) -> None:
        if node in visiting:
            raise ProjectionProofError("HIERARCHY_CYCLE", str(node))
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, set()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in tuple(graph):
        visit(node)


def compare_hierarchical(
    *,
    profile_id: str,
    direct: list[dict[str, Any]],
    hierarchical: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_counter = Counter(canonical_bytes(row) for row in direct)
    hierarchical_counter = Counter(canonical_bytes(row) for row in hierarchical)
    false_positive = sum((hierarchical_counter - direct_counter).values())
    false_negative = sum((direct_counter - hierarchical_counter).values())
    exact = false_positive == 0 and false_negative == 0
    return {
        "profile_id": profile_id,
        "direct_projection_hash": sha256_bytes(
            canonical_bytes(sorted(direct, key=canonical_bytes))
        ),
        "hierarchical_projection_hash": sha256_bytes(
            canonical_bytes(sorted(hierarchical, key=canonical_bytes))
        ),
        "record_false_positive": false_positive,
        "record_false_negative": false_negative,
        "edge_mismatch": 0 if exact else false_positive + false_negative,
        "attribute_mismatch": 0,
        "exact_equal": exact,
        "status": "SUPPORTED" if exact else "NOT_SUPPORTED",
    }


def require_hierarchical_equality(report: dict[str, Any]) -> None:
    if not report.get("exact_equal"):
        raise ProjectionProofError(
            "HIERARCHICAL_MISMATCH", report.get("profile_id", "")
        )


def not_evaluated_hierarchical(profile_id: str, reason: str) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "direct_projection_hash": None,
        "hierarchical_projection_hash": None,
        "record_false_positive": None,
        "record_false_negative": None,
        "edge_mismatch": None,
        "attribute_mismatch": None,
        "exact_equal": None,
        "reason": reason,
        "status": "NOT_EVALUATED",
    }
