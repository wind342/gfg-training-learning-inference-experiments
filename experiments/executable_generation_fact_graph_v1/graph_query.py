from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Iterable

from .canonical_graph import canonical_hash
from .graph_model import (
    GraphFactNode,
    GraphRelationEdge,
    ValidatedGenerationFactGraph,
)
from .graph_projections import (
    project_atomic_generation_state,
    project_relation_store,
    project_signed_generation_algebra,
)


def _intersects(left: dict[str, float], right: dict[str, float]) -> bool:
    return (
        max(left["x"], right["x"])
        < min(left["x"] + left["width"], right["x"] + right["width"])
        and max(left["y"], right["y"])
        < min(left["y"] + left["height"], right["y"] + right["height"])
    )


class GenerationFactGraphQueryEngine:
    def __init__(self, graph: ValidatedGenerationFactGraph) -> None:
        if graph.validation.status != "PASS":
            raise ValueError("VALIDATED_GENERATION_FACT_GRAPH_REQUIRED")
        self.validated_graph = graph
        self.graph = graph.graph
        self.nodes = {row.graph_node_id: row for row in self.graph.nodes}
        self.edges = {row.graph_edge_id: row for row in self.graph.edges}
        self._outgoing: dict[str, list[GraphRelationEdge]] = defaultdict(list)
        self._incoming: dict[str, list[GraphRelationEdge]] = defaultdict(list)
        for edge in self.graph.edges:
            self._outgoing[edge.source_graph_node_id].append(edge)
            self._incoming[edge.target_graph_node_id].append(edge)
            if edge.relation_semantics == "symmetric":
                self._outgoing[edge.target_graph_node_id].append(edge)
                self._incoming[edge.source_graph_node_id].append(edge)
        for rows in (*self._outgoing.values(), *self._incoming.values()):
            rows.sort(key=lambda row: row.graph_edge_id)
        self.global_transitive_closure_materialized = False

    def nodes_for_support(self, query_window: dict[str, Any]) -> list[str]:
        selected: list[str] = []
        for node in self.graph.nodes:
            reference = node.outcome_reference["reference"]
            if reference["kind"] != "support":
                continue
            support = node.outcome_reference["entity"]
            if (
                query_window.get("support_space_id")
                and support["support_space_id"] != query_window["support_space_id"]
            ):
                continue
            payload = support["support_payload"]
            predicate = query_window["predicate"]
            if predicate == "rectangle_intersection":
                matched = _intersects(
                    payload["rectangle"], query_window["rectangle"]
                )
            elif predicate == "native_support_key_membership":
                matched = (
                    payload["native_support_key"]
                    in set(query_window["native_support_keys"])
                )
            elif predicate == "outcome_kind":
                matched = payload.get("kind") in set(query_window["kinds"])
            elif predicate == "all":
                matched = True
            else:
                raise ValueError("GRAPH_SUPPORT_PREDICATE_UNKNOWN")
            if matched:
                selected.append(node.graph_node_id)
        return sorted(selected)

    @staticmethod
    def _allowed(
        edge: GraphRelationEdge, relation_filter: Iterable[str] | None
    ) -> bool:
        return relation_filter is None or edge.relation_type in set(relation_filter)

    def relation_edges(
        self,
        node_id: str,
        direction: str,
        relation_filter: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        if direction == "out":
            rows = self._outgoing.get(node_id, [])
        elif direction == "in":
            rows = self._incoming.get(node_id, [])
        elif direction == "both":
            rows = [
                *self._outgoing.get(node_id, []),
                *self._incoming.get(node_id, []),
            ]
        else:
            raise ValueError("GRAPH_DIRECTION_INVALID")
        unique = {
            row.graph_edge_id: row
            for row in rows
            if self._allowed(row, relation_filter)
        }
        return [unique[key].to_dict() for key in sorted(unique)]

    def predecessors(
        self, node_id: str, relation_filter: Iterable[str] | None = None
    ) -> list[str]:
        result = []
        for edge in self._incoming.get(node_id, []):
            if not self._allowed(edge, relation_filter):
                continue
            other = (
                edge.source_graph_node_id
                if edge.target_graph_node_id == node_id
                else edge.target_graph_node_id
            )
            result.append(other)
        return sorted(set(result))

    def successors(
        self, node_id: str, relation_filter: Iterable[str] | None = None
    ) -> list[str]:
        result = []
        for edge in self._outgoing.get(node_id, []):
            if not self._allowed(edge, relation_filter):
                continue
            other = (
                edge.target_graph_node_id
                if edge.source_graph_node_id == node_id
                else edge.source_graph_node_id
            )
            result.append(other)
        return sorted(set(result))

    def paths(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_filter: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            raise KeyError("GRAPH_PATH_ENDPOINT_UNKNOWN")
        result: list[dict[str, Any]] = []

        def walk(
            node_id: str,
            node_path: list[str],
            edge_path: list[str],
            active: set[str],
        ) -> None:
            if node_id == target_node_id:
                result.append(
                    {"node_ids": node_path, "edge_ids": edge_path}
                )
                return
            for edge in self._outgoing.get(node_id, []):
                if not self._allowed(edge, relation_filter):
                    continue
                next_id = (
                    edge.target_graph_node_id
                    if edge.source_graph_node_id == node_id
                    else edge.source_graph_node_id
                )
                if next_id in active:
                    continue
                walk(
                    next_id,
                    [*node_path, next_id],
                    [*edge_path, edge.graph_edge_id],
                    {*active, next_id},
                )

        walk(source_node_id, [source_node_id], [], {source_node_id})
        return sorted(
            result,
            key=lambda row: (tuple(row["node_ids"]), tuple(row["edge_ids"])),
        )

    def formation_subgraph(
        self,
        query_window: dict[str, Any],
        relation_policy: dict[str, Any],
    ) -> dict[str, Any]:
        selected = self.nodes_for_support(query_window)
        allowed = set(
            relation_policy.get(
                "relation_types", ["generated_origin_dependency"]
            )
        )
        stop_at_registered = relation_policy.get(
            "stop_at_registered_source", True
        )
        path_instances: list[dict[str, Any]] = []

        def descend(
            node_id: str,
            reverse_nodes: list[str],
            reverse_edges: list[str],
            active: set[str],
        ) -> None:
            node = self.nodes[node_id]
            is_registered = (
                node.source_reference["reference"]["kind"]
                == "registered_source"
            )
            incoming = [
                edge
                for edge in self._incoming.get(node_id, [])
                if edge.relation_type in allowed
                and edge.relation_semantics == "directed"
            ]
            if (stop_at_registered and is_registered) or not incoming:
                path_instances.append(
                    {
                        "node_ids": list(reversed(reverse_nodes)),
                        "edge_ids": list(reversed(reverse_edges)),
                    }
                )
                return
            for edge in incoming:
                source = edge.source_graph_node_id
                if source in active:
                    raise ValueError("FORMATION_SUBGRAPH_CYCLE")
                descend(
                    source,
                    [*reverse_nodes, source],
                    [*reverse_edges, edge.graph_edge_id],
                    {*active, source},
                )

        for node_id in selected:
            descend(node_id, [node_id], [], {node_id})

        path_instances.sort(
            key=lambda row: (tuple(row["node_ids"]), tuple(row["edge_ids"]))
        )
        included_node_ids = sorted(
            {node for row in path_instances for node in row["node_ids"]}
        )
        included_edge_ids = sorted(
            {edge for row in path_instances for edge in row["edge_ids"]}
        )
        source_nodes = sorted(
            node_id
            for node_id in included_node_ids
            if self.nodes[node_id].source_reference["reference"]["kind"]
            == "registered_source"
        )
        explicit = sorted(
            node_id
            for node_id in included_node_ids
            if self.nodes[node_id].outcome_reference["reference"]["kind"]
            == "disposition"
        )
        relation_counts = Counter(
            self.edges[edge_id].relation_type
            for edge_id in included_edge_ids
        )
        occurrence_counts = Counter(
            self.nodes[node_id].occurrence_identity
            for row in path_instances
            for node_id in row["node_ids"]
        )
        material = {
            "selected_result_nodes": selected,
            "included_nodes": included_node_ids,
            "included_edges": included_edge_ids,
            "path_instances": path_instances,
            "source_nodes": source_nodes,
            "occurrence_references": sorted(
                {
                    self.nodes[node_id].occurrence_identity
                    for node_id in included_node_ids
                }
            ),
            "explicit_disposition_nodes": explicit,
            "multiplicity_summary": {
                "path_count": len(path_instances),
                "node_path_occurrence_counts": dict(
                    sorted(occurrence_counts.items())
                ),
            },
            "relation_type_summary": dict(sorted(relation_counts.items())),
        }
        return {**material, "graph_slice_hash": canonical_hash(material)}

    def conflicts(self, node_id: str) -> list[str]:
        return sorted(
            {
                (
                    edge.target_graph_node_id
                    if edge.source_graph_node_id == node_id
                    else edge.source_graph_node_id
                )
                for edge in (
                    *self._outgoing.get(node_id, []),
                    *self._incoming.get(node_id, []),
                )
                if edge.relation_type == "conflicts_with"
            }
        )

    def downstream_impact(
        self, node_id: str, relation_filter: Iterable[str] | None = None
    ) -> list[str]:
        seen = {node_id}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for successor in self.successors(current, relation_filter):
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        return sorted(seen - {node_id})

    def compensation_target(self, node_id: str) -> list[str]:
        return self.downstream_impact(
            node_id,
            {
                "generated_origin_dependency",
                "message_send_receive",
                "reads_from",
            },
        )

    def project_atomic_generation_state(self) -> dict[str, Any]:
        return project_atomic_generation_state(self.validated_graph)

    def project_relation_store(self) -> dict[str, Any]:
        return project_relation_store(self.validated_graph)

    def project_signed_generation_algebra(
        self, contract: dict[str, Any]
    ) -> dict[str, Any]:
        return project_signed_generation_algebra(
            self.validated_graph, contract
        )

