from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Iterable

from .canonical_graph import canonical_hash
from .graph_model import ValidatedGenerationFactGraphV2
from .graph_projections import (
    project_fact_only_graph,
    project_gamma,
    project_occurrence_view,
    project_primitive_relation_sidecar,
    project_signed_algebra,
)


def _rectangle_intersects(
    left: dict[str, float], right: dict[str, float]
) -> bool:
    return (
        max(left["x"], right["x"])
        < min(left["x"] + left["width"], right["x"] + right["width"])
        and max(left["y"], right["y"])
        < min(left["y"] + left["height"], right["y"] + right["height"])
    )


class ExecutableGenerationFactGraphQueryEngineV2:
    def __init__(
        self, validated_graph: ValidatedGenerationFactGraphV2
    ) -> None:
        if validated_graph.validation.status != "PASS":
            raise ValueError("VALIDATED_GENERATION_FACT_GRAPH_V2_REQUIRED")
        self.validated_graph = validated_graph
        self.graph = validated_graph.graph
        self.fact_nodes = {
            row.graph_node_id: row for row in self.graph.fact_nodes
        }
        self.occurrence_nodes = {
            row.graph_node_id: row
            for row in self.graph.occurrence_nodes
        }
        self.nodes = {**self.fact_nodes, **self.occurrence_nodes}
        self.relation_edges_by_id = {
            row.graph_edge_id: row
            for row in self.graph.relation_edges
        }
        self.incidence_edges_by_id = {
            row.graph_edge_id: row
            for row in self.graph.incidence_edges
        }
        self._outgoing: dict[str, list[Any]] = defaultdict(list)
        self._incoming: dict[str, list[Any]] = defaultdict(list)
        for edge in self.graph.relation_edges:
            self._outgoing[edge.source_node_id].append(edge)
            self._incoming[edge.target_node_id].append(edge)
            if edge.relation_semantics == "symmetric":
                self._outgoing[edge.target_node_id].append(edge)
                self._incoming[edge.source_node_id].append(edge)
        self._fact_to_occurrence: dict[str, str] = {}
        self._occurrence_to_facts: dict[str, list[str]] = defaultdict(
            list
        )
        for edge in self.graph.incidence_edges:
            self._fact_to_occurrence[
                edge.target_fact_node_id
            ] = edge.source_occurrence_node_id
            self._occurrence_to_facts[
                edge.source_occurrence_node_id
            ].append(edge.target_fact_node_id)
        for rows in (
            *self._outgoing.values(),
            *self._incoming.values(),
            *self._occurrence_to_facts.values(),
        ):
            rows.sort(
                key=lambda row: (
                    row.graph_edge_id
                    if hasattr(row, "graph_edge_id")
                    else row
                )
            )
        self.global_transitive_closure_materialized = False

    def fact_nodes_for_support(
        self, query_window: dict[str, Any]
    ) -> list[str]:
        selected = []
        for node in self.graph.fact_nodes:
            reference = node.z["reference"]
            if reference["kind"] != "support":
                continue
            support = node.z["entity"]
            if (
                query_window.get("support_space_id")
                and support["support_space_id"]
                != query_window["support_space_id"]
            ):
                continue
            payload = support["support_payload"]
            predicate = query_window["predicate"]
            if predicate == "rectangle_intersection":
                matched = _rectangle_intersects(
                    payload["rectangle"], query_window["rectangle"]
                )
            elif predicate == "native_support_key_membership":
                matched = (
                    payload["native_support_key"]
                    in set(query_window["native_support_keys"])
                )
            elif predicate == "outcome_kind":
                matched = payload.get("kind") in set(
                    query_window["kinds"]
                )
            elif predicate == "all":
                matched = True
            else:
                raise ValueError("GRAPH_SUPPORT_PREDICATE_UNKNOWN")
            if matched:
                selected.append(node.graph_node_id)
        return sorted(selected)

    def occurrence_for_fact(self, fact_node_id: str) -> str:
        if fact_node_id not in self.fact_nodes:
            raise KeyError(fact_node_id)
        return self._fact_to_occurrence[fact_node_id]

    def facts_realized_by_occurrence(
        self, occurrence_node_id: str
    ) -> list[str]:
        if occurrence_node_id not in self.occurrence_nodes:
            raise KeyError(occurrence_node_id)
        return list(self._occurrence_to_facts.get(occurrence_node_id, []))

    @staticmethod
    def _allowed(
        edge: Any, relation_filter: Iterable[str] | None
    ) -> bool:
        return (
            relation_filter is None
            or edge.relation_type in set(relation_filter)
        )

    def relation_edges(
        self,
        node_id: str,
        direction: str,
        relation_filter: Iterable[str] | None = None,
        endpoint_kind_filter: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        rows = []
        if direction in {"out", "both"}:
            rows.extend(self._outgoing.get(node_id, []))
        if direction in {"in", "both"}:
            rows.extend(self._incoming.get(node_id, []))
        if direction not in {"out", "in", "both"}:
            raise ValueError("GRAPH_DIRECTION_INVALID")
        kinds = set(endpoint_kind_filter or [])
        unique = {
            row.graph_edge_id: row
            for row in rows
            if self._allowed(row, relation_filter)
            and (
                not kinds
                or row.source_node_kind in kinds
                or row.target_node_kind in kinds
            )
        }
        return [unique[key].to_dict() for key in sorted(unique)]

    def all_relation_edges(
        self,
        relation_filter: Iterable[str] | None = None,
        endpoint_signature_filter: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        relation_types = set(relation_filter or [])
        signatures = set(endpoint_signature_filter or [])
        rows = [
            edge
            for edge in self.graph.relation_edges
            if (
                not relation_types
                or edge.relation_type in relation_types
            )
            and (
                not signatures
                or (
                    edge.source_node_kind
                    + "->"
                    + edge.target_node_kind
                )
                in signatures
            )
        ]
        return [
            edge.to_dict()
            for edge in sorted(rows, key=lambda row: row.graph_edge_id)
        ]

    def predecessors(
        self, node_id: str, relation_filter: Iterable[str] | None = None
    ) -> list[str]:
        rows = []
        for edge in self._incoming.get(node_id, []):
            if self._allowed(edge, relation_filter):
                rows.append(
                    edge.source_node_id
                    if edge.target_node_id == node_id
                    else edge.target_node_id
                )
        return sorted(set(rows))

    def successors(
        self, node_id: str, relation_filter: Iterable[str] | None = None
    ) -> list[str]:
        rows = []
        for edge in self._outgoing.get(node_id, []):
            if self._allowed(edge, relation_filter):
                rows.append(
                    edge.target_node_id
                    if edge.source_node_id == node_id
                    else edge.source_node_id
                )
        return sorted(set(rows))

    def paths(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_policy: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            raise KeyError("GRAPH_PATH_ENDPOINT_UNKNOWN")
        allowed = set(relation_policy.get("relation_types", []))
        maximum = relation_policy.get("maximum_edges")
        result = []

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
            if maximum is not None and len(edge_path) >= maximum:
                return
            for edge in self._outgoing.get(node_id, []):
                if allowed and edge.relation_type not in allowed:
                    continue
                next_id = (
                    edge.target_node_id
                    if edge.source_node_id == node_id
                    else edge.source_node_id
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
            key=lambda row: (
                tuple(row["node_ids"]),
                tuple(row["edge_ids"]),
            ),
        )

    def shortest_path(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_policy: dict[str, Any],
    ) -> dict[str, Any] | None:
        allowed = set(relation_policy.get("relation_types", []))
        queue = deque([(source_node_id, [source_node_id], [])])
        visited = {source_node_id}
        while queue:
            current, node_path, edge_path = queue.popleft()
            if current == target_node_id:
                return {
                    "node_ids": node_path,
                    "edge_ids": edge_path,
                    "edge_count": len(edge_path),
                    "semantics": (
                        "minimum admitted edge count under the supplied "
                        "relation policy"
                    ),
                }
            for edge in self._outgoing.get(current, []):
                if allowed and edge.relation_type not in allowed:
                    continue
                next_id = (
                    edge.target_node_id
                    if edge.source_node_id == current
                    else edge.source_node_id
                )
                if next_id in visited:
                    continue
                visited.add(next_id)
                queue.append(
                    (
                        next_id,
                        [*node_path, next_id],
                        [*edge_path, edge.graph_edge_id],
                    )
                )
        return None

    def formation_subgraph(
        self,
        query_window: dict[str, Any],
        traversal_policy: dict[str, Any],
    ) -> dict[str, Any]:
        selected = self.fact_nodes_for_support(query_window)
        allowed = set(
            traversal_policy.get(
                "relation_types", ["generated_origin_dependency"]
            )
        )
        stop_at_registered = traversal_policy.get(
            "stop_at_registered_source", True
        )
        paths = []

        def descend(
            node_id: str,
            reverse_nodes: list[str],
            reverse_edges: list[str],
            active: set[str],
        ) -> None:
            node = self.fact_nodes[node_id]
            is_registered = (
                node.u["reference"]["kind"] == "registered_source"
            )
            incoming = [
                edge
                for edge in self._incoming.get(node_id, [])
                if edge.relation_type in allowed
                and edge.source_node_kind == "generation_fact"
                and edge.target_node_kind == "generation_fact"
                and edge.relation_semantics == "directed"
            ]
            if (stop_at_registered and is_registered) or not incoming:
                paths.append(
                    {
                        "node_ids": list(reversed(reverse_nodes)),
                        "edge_ids": list(reversed(reverse_edges)),
                    }
                )
                return
            for edge in incoming:
                source = edge.source_node_id
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
        paths.sort(
            key=lambda row: (
                tuple(row["node_ids"]),
                tuple(row["edge_ids"]),
            )
        )
        node_ids = sorted(
            {node for path in paths for node in path["node_ids"]}
        )
        edge_ids = sorted(
            {edge for path in paths for edge in path["edge_ids"]}
        )
        occurrence_ids = sorted(
            {
                self._fact_to_occurrence[node_id]
                for node_id in node_ids
            }
        )
        source_nodes = sorted(
            node_id
            for node_id in node_ids
            if self.fact_nodes[node_id].u["reference"]["kind"]
            == "registered_source"
        )
        material = {
            "selected_result_nodes": selected,
            "included_fact_nodes": node_ids,
            "included_occurrence_nodes": occurrence_ids,
            "included_relation_edges": edge_ids,
            "included_incidence_edges": sorted(
                edge.graph_edge_id
                for edge in self.graph.incidence_edges
                if edge.target_fact_node_id in node_ids
            ),
            "path_instances": paths,
            "source_fact_nodes": source_nodes,
            "multiplicity_summary": {"path_count": len(paths)},
            "relation_type_summary": dict(
                sorted(
                    Counter(
                        self.relation_edges_by_id[edge_id].relation_type
                        for edge_id in edge_ids
                    ).items()
                )
            ),
        }
        return {**material, "graph_slice_hash": canonical_hash(material)}

    def execution_subgraph(
        self,
        occurrence_anchor: str,
        traversal_policy: dict[str, Any],
    ) -> dict[str, Any]:
        if occurrence_anchor not in self.occurrence_nodes:
            raise KeyError(occurrence_anchor)
        allowed = set(traversal_policy.get("relation_types", []))
        maximum = traversal_policy.get("maximum_edges", 4)
        queue = deque([(occurrence_anchor, 0)])
        included = {occurrence_anchor}
        relation_edges = set()
        while queue:
            current, depth = queue.popleft()
            if depth >= maximum:
                continue
            for edge in (
                *self._outgoing.get(current, []),
                *self._incoming.get(current, []),
            ):
                if allowed and edge.relation_type not in allowed:
                    continue
                other = (
                    edge.target_node_id
                    if edge.source_node_id == current
                    else edge.source_node_id
                )
                relation_edges.add(edge.graph_edge_id)
                if other not in included:
                    included.add(other)
                    queue.append((other, depth + 1))
        facts = sorted(
            {
                fact
                for occurrence in included & set(self.occurrence_nodes)
                for fact in self._occurrence_to_facts.get(
                    occurrence, []
                )
            }
        )
        incidence = sorted(
            edge.graph_edge_id
            for edge in self.graph.incidence_edges
            if edge.target_fact_node_id in facts
        )
        material = {
            "anchor_occurrence_node": occurrence_anchor,
            "occurrence_nodes": sorted(
                included & set(self.occurrence_nodes)
            ),
            "fact_nodes": facts,
            "incidence_edges": incidence,
            "relation_edges": sorted(relation_edges),
        }
        return {**material, "graph_slice_hash": canonical_hash(material)}

    def conflicts(self, node_id: str) -> list[str]:
        return sorted(
            {
                edge.target_node_id
                if edge.source_node_id == node_id
                else edge.source_node_id
                for edge in (
                    *self._outgoing.get(node_id, []),
                    *self._incoming.get(node_id, []),
                )
                if edge.relation_type == "conflicts_with"
            }
        )

    def downstream_impact(
        self,
        node_id: str,
        relation_filter: Iterable[str] | None = None,
    ) -> list[str]:
        seen = {node_id}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for successor in self.successors(
                current, relation_filter
            ):
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        return sorted(seen - {node_id})

    def project_gamma(self) -> dict[str, Any]:
        return project_gamma(self.validated_graph)

    def project_occurrence_view(self) -> dict[str, Any]:
        return project_occurrence_view(self.validated_graph)

    def project_primitive_relation_sidecar(self) -> dict[str, Any]:
        return project_primitive_relation_sidecar(
            self.validated_graph
        )

    def project_fact_only_graph(self) -> dict[str, Any]:
        return project_fact_only_graph(self.validated_graph)

    def project_signed_algebra(
        self, contract: dict[str, Any]
    ) -> dict[str, Any]:
        return project_signed_algebra(
            self.validated_graph, contract
        )
