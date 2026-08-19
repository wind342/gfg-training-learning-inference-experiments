from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .graph_model import ValidatedGenerationFactGraphV2


class ScaleGraphQueryResolver:
    """Resolve frozen scale queries from the validated v2 graph only."""

    _EVENT_RELATIONS = {
        "program_order",
        "message_send_receive",
        "synchronizes_with",
        "generated_origin_dependency",
        "reads_from",
    }

    def __init__(
        self,
        validated_graph: ValidatedGenerationFactGraphV2,
        capture_audit: dict[str, Any],
    ) -> None:
        self.graph = validated_graph.graph
        self.run_id = self.graph.metadata.execution_run_id
        self.fact_by_native = {
            row.native_fact_id: row
            for row in self.graph.fact_nodes
            if row.native_fact_id is not None
        }
        self.occurrence_by_native = {
            row.concrete_occurrence_instance_id: row
            for row in self.graph.occurrence_nodes
        }
        self.fact_to_occurrence = {
            row.native_fact_id: row.concrete_occurrence_instance_id
            for row in self.graph.fact_nodes
            if row.native_fact_id is not None
        }
        self.occurrence_to_facts: dict[str, list[str]] = defaultdict(list)
        for fact_id, occurrence_id in self.fact_to_occurrence.items():
            self.occurrence_to_facts[occurrence_id].append(fact_id)
        for rows in self.occurrence_to_facts.values():
            rows.sort()
        self.adjacency: dict[str, set[str]] = defaultdict(set)
        self.reverse: dict[str, set[str]] = defaultdict(set)
        self.conflict_index: dict[str, set[str]] = defaultdict(set)
        self._build_indexes()
        self.scope_complete = {
            row["scope_id"]: row["status"] == "CAPTURE_COMPLETE"
            for row in capture_audit.get("scopes", [])
        }
        self.query_count = 0

    def _event_endpoint(self, node_id: str, node_kind: str) -> str:
        if node_kind == "generation_occurrence":
            return next(
                row.concrete_occurrence_instance_id
                for row in self.graph.occurrence_nodes
                if row.graph_node_id == node_id
            )
        fact = next(
            row
            for row in self.graph.fact_nodes
            if row.graph_node_id == node_id
        )
        return fact.concrete_occurrence_instance_id

    def _build_indexes(self) -> None:
        for edge in self.graph.relation_edges:
            source_native = edge.native_source_id
            target_native = edge.native_target_id
            if edge.relation_type == "conflicts_with":
                self.conflict_index[source_native].add(target_native)
                self.conflict_index[target_native].add(source_native)
                continue
            if edge.relation_type not in self._EVENT_RELATIONS:
                continue
            if edge.source_node_kind == "generation_fact":
                source = self.fact_to_occurrence[source_native]
            else:
                source = source_native
            if edge.target_node_kind == "generation_fact":
                target = self.fact_to_occurrence[target_native]
            else:
                target = target_native
            self.adjacency[source].add(target)
            self.reverse[target].add(source)

    def _path(self, source: str, target: str) -> list[str] | None:
        if source == target:
            return [source]
        queue = deque([source])
        predecessor: dict[str, str] = {}
        seen = {source}
        while queue:
            current = queue.popleft()
            for next_id in sorted(self.adjacency[current]):
                if next_id in seen:
                    continue
                seen.add(next_id)
                predecessor[next_id] = current
                if next_id == target:
                    path = [target]
                    while path[-1] != source:
                        path.append(predecessor[path[-1]])
                    return list(reversed(path))
                queue.append(next_id)
        return None

    def _reachable(
        self, source: str, adjacency: dict[str, set[str]]
    ) -> list[str]:
        seen = {source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for next_id in sorted(adjacency[current]):
                if next_id not in seen:
                    seen.add(next_id)
                    queue.append(next_id)
        return sorted(seen - {source})

    def _scope(self, occurrence_id: str) -> str | None:
        payload = self.occurrence_by_native[
            occurrence_id
        ].occurrence_payload
        receipt = payload.get("native_occurrence_receipt", payload)
        return receipt.get("scope_id")

    def answer(self, query: dict[str, Any]) -> dict[str, Any]:
        self.query_count += 1
        query_type = query["query_type"]
        if query_type == "happens_before":
            result: Any = (
                query["source_id"] != query["target_id"]
                and self._path(
                    query["source_id"], query["target_id"]
                )
                is not None
            )
        elif query_type == "concurrent_with":
            left = query["source_id"]
            right = query["target_id"]
            left_scope = self._scope(left)
            right_scope = self._scope(right)
            if (
                left_scope != right_scope
                or not self.scope_complete.get(left_scope, False)
            ):
                result = {
                    "status": "CONCURRENCY_NOT_ESTABLISHED",
                    "value": None,
                }
            else:
                result = {
                    "status": "ESTABLISHED",
                    "value": self._path(left, right) is None
                    and self._path(right, left) is None,
                }
        elif query_type == "relation_path":
            result = self._path(
                query["source_id"], query["target_id"]
            )
        elif query_type == "predecessors":
            result = self._reachable(query["target_id"], self.reverse)
        elif query_type == "successors":
            result = self._reachable(query["source_id"], self.adjacency)
        elif query_type == "conflicts":
            result = sorted(self.conflict_index[query["fact_id"]])
        elif query_type == "fact_to_occurrence":
            result = self.fact_to_occurrence[query["fact_id"]]
        elif query_type == "occurrence_to_selected_facts":
            available = self.occurrence_to_facts[
                query["occurrence_id"]
            ]
            requested = query.get("selected_fact_ids")
            result = (
                list(available)
                if requested is None
                else sorted(set(available) & set(requested))
            )
        else:
            raise ValueError("SCALE_GRAPH_QUERY_TYPE_UNSUPPORTED")
        return {
            "query_id": query["query_id"],
            "query_type": query_type,
            "result": result,
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "occurrence_count": len(self.occurrence_by_native),
            "fact_count": len(self.fact_by_native),
            "query_count": self.query_count,
            "candidate_kind": "validated_generation_fact_graph_v2",
            "full_transitive_closure_materialized": False,
            "global_closure_pair_count": 0,
            "event_adjacency_pair_count": sum(
                len(rows) for rows in self.adjacency.values()
            ),
        }
