from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from ..common import ExperimentError
from .capture_auditor import CAPTURE_COMPLETE
from .relation_model import CAUSAL_PRIMITIVE_TYPES, make_relation


ALLOWED_CANDIDATE_INPUT_KEYS = {
    "execution_run_id",
    "primitive_store",
    "capture_audit",
    "lifting_rules",
    "queries",
    "schema_version",
}
FORBIDDEN_CANDIDATE_INPUT_KEYS = {
    "runtime_receipts",
    "reference_receipts",
    "reference_output",
    "scenario_fixture",
    "oracle",
}


class IndexedCandidateResolver:
    """Disposable query index rebuilt only from validated candidate inputs."""

    def __init__(
        self,
        *,
        execution_run_id: str,
        primitive_store: dict[str, Any],
        capture_audit: dict[str, Any],
        lifting_rules: dict[str, Any],
        reusable_cache: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = execution_run_id
        if primitive_store["execution_run_id"] != execution_run_id:
            raise ExperimentError("CANDIDATE_STORE_RUN_ID_MISMATCH")
        if capture_audit["execution_run_id"] != execution_run_id:
            raise ExperimentError("CANDIDATE_CAPTURE_AUDIT_RUN_ID_MISMATCH")
        if reusable_cache is not None and reusable_cache.get("execution_run_id") != (
            execution_run_id
        ):
            raise ExperimentError("CROSS_RUN_QUERY_CACHE_REUSE")
        self.lifting_rules = lifting_rules
        self.occurrence_scope = {
            row["occurrence_id"]: row["scope_id"]
            for row in primitive_store["occurrence_catalog"]
        }
        self.occurrences = set(self.occurrence_scope)
        self.fact_to_occurrence_map = {
            row["fact_id"]: row["occurrence_id"]
            for row in primitive_store["fact_catalog"]
        }
        self.occurrence_to_facts_map: defaultdict[str, list[str]] = defaultdict(list)
        for fact_id, occurrence_id in self.fact_to_occurrence_map.items():
            self.occurrence_to_facts_map[occurrence_id].append(fact_id)
        for values in self.occurrence_to_facts_map.values():
            values.sort()
        self.capture_by_scope = {
            row["scope_id"]: row for row in capture_audit["scopes"]
        }
        self.relations = primitive_store["primitive_relations"]
        self.relation_by_id: dict[str, dict[str, Any]] = {}
        self.adjacency: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        self.reverse_adjacency: defaultdict[str, list[tuple[str, str]]] = (
            defaultdict(list)
        )
        self.pair_relation_ids: defaultdict[tuple[str, str], list[str]] = (
            defaultdict(list)
        )
        self.conflict_index: defaultdict[str, list[tuple[str, str]]] = defaultdict(
            list
        )
        self._build_indexes()
        self.topological_order = self._topological_sort_or_fail()
        self.topological_position = {
            occurrence_id: index
            for index, occurrence_id in enumerate(self.topological_order)
        }
        self.component_id = self._weak_components()
        self._reachability_cache: dict[tuple[str, str], bool] = {}
        self._path_cache: dict[
            tuple[str, str], tuple[list[str], list[str]] | None
        ] = {}
        self.query_count = 0

    def _build_indexes(self) -> None:
        relation_ids: set[str] = set()
        for relation in self.relations:
            relation_id = relation["relation_id"]
            if relation_id in relation_ids:
                raise ExperimentError("DUPLICATE_PRIMITIVE_RELATION")
            relation_ids.add(relation_id)
            self.relation_by_id[relation_id] = relation
            relation_type = relation["relation_type"]
            if relation_type == "conflicts_with":
                self.conflict_index[relation["source_id"]].append(
                    (relation["target_id"], relation_id)
                )
                self.conflict_index[relation["target_id"]].append(
                    (relation["source_id"], relation_id)
                )
                continue
            if relation_type not in CAUSAL_PRIMITIVE_TYPES:
                continue
            if relation["endpoint_level"] == "occurrence":
                source = relation["source_id"]
                target = relation["target_id"]
            else:
                source = self.fact_to_occurrence_map[relation["source_id"]]
                target = self.fact_to_occurrence_map[relation["target_id"]]
            if source == target:
                raise ExperimentError("HAPPENS_BEFORE_SELF_CYCLE")
            self.pair_relation_ids[(source, target)].append(relation_id)
        for (source, target), relation_ids_for_pair in sorted(
            self.pair_relation_ids.items()
        ):
            for relation_id in sorted(relation_ids_for_pair):
                self.adjacency[source].append((target, relation_id))
                self.reverse_adjacency[target].append((source, relation_id))
        for node in self.occurrences:
            self.adjacency[node].sort()
            self.reverse_adjacency[node].sort()
        for values in self.conflict_index.values():
            values.sort()

    def _topological_sort_or_fail(self) -> list[str]:
        distinct_targets = {
            source: {target for target, _ in self.adjacency[source]}
            for source in self.occurrences
        }
        indegree = {node: 0 for node in self.occurrences}
        for targets in distinct_targets.values():
            for target in targets:
                indegree[target] += 1
        queue = deque(sorted(node for node, value in indegree.items() if value == 0))
        ordered: list[str] = []
        while queue:
            node = queue.popleft()
            ordered.append(node)
            for target in sorted(distinct_targets[node]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(ordered) != len(self.occurrences):
            raise ExperimentError("HAPPENS_BEFORE_CYCLE")
        return ordered

    def _weak_components(self) -> dict[str, int]:
        neighbors: defaultdict[str, set[str]] = defaultdict(set)
        for source, target in self.pair_relation_ids:
            neighbors[source].add(target)
            neighbors[target].add(source)
        component: dict[str, int] = {}
        component_number = 0
        for start in sorted(self.occurrences):
            if start in component:
                continue
            queue = deque([start])
            component[start] = component_number
            while queue:
                node = queue.popleft()
                for neighbor in sorted(neighbors[node]):
                    if neighbor not in component:
                        component[neighbor] = component_number
                        queue.append(neighbor)
            component_number += 1
        return component

    def _require_occurrences(self, source: str, target: str) -> None:
        if source not in self.occurrences or target not in self.occurrences:
            raise ExperimentError("QUERY_OCCURRENCE_UNKNOWN")

    def relation_path(
        self, source: str, target: str
    ) -> tuple[list[str], list[str]] | None:
        self._require_occurrences(source, target)
        key = (source, target)
        if key in self._path_cache:
            return self._path_cache[key]
        if source == target:
            result: tuple[list[str], list[str]] = ([source], [])
            self._path_cache[key] = result
            return result
        if self.component_id[source] != self.component_id[target]:
            self._path_cache[key] = None
            return None
        if self.topological_position[source] >= self.topological_position[target]:
            self._path_cache[key] = None
            return None
        queue = deque([source])
        predecessor: dict[str, tuple[str, str]] = {}
        visited = {source}
        found = False
        while queue and not found:
            current = queue.popleft()
            by_target: defaultdict[str, list[str]] = defaultdict(list)
            for next_node, relation_id in self.adjacency[current]:
                by_target[next_node].append(relation_id)
            for next_node in sorted(by_target):
                if next_node in visited:
                    continue
                visited.add(next_node)
                predecessor[next_node] = (
                    current,
                    min(by_target[next_node]),
                )
                if next_node == target:
                    found = True
                    break
                queue.append(next_node)
        if not found:
            self._path_cache[key] = None
            return None
        nodes = [target]
        relation_ids: list[str] = []
        current = target
        while current != source:
            previous, relation_id = predecessor[current]
            nodes.append(previous)
            relation_ids.append(relation_id)
            current = previous
        nodes.reverse()
        relation_ids.reverse()
        result = (nodes, relation_ids)
        self._path_cache[key] = result
        return result

    def happens_before(self, source: str, target: str) -> bool:
        key = (source, target)
        if key not in self._reachability_cache:
            path = self.relation_path(source, target)
            self._reachability_cache[key] = path is not None and source != target
        return self._reachability_cache[key]

    def concurrent_with(self, left: str, right: str) -> dict[str, Any]:
        self._require_occurrences(left, right)
        left_scope = self.occurrence_scope[left]
        right_scope = self.occurrence_scope[right]
        if left_scope != right_scope:
            return {
                "status": "CONCURRENCY_NOT_ESTABLISHED",
                "value": None,
            }
        audit = self.capture_by_scope[left_scope]
        if audit["status"] != CAPTURE_COMPLETE:
            return {
                "status": "CONCURRENCY_NOT_ESTABLISHED",
                "value": None,
            }
        value = not self.happens_before(left, right) and not self.happens_before(
            right, left
        )
        return {"status": "ESTABLISHED", "value": value}

    def _reachable_set(
        self, source: str, adjacency: dict[str, list[tuple[str, str]]]
    ) -> list[str]:
        if source not in self.occurrences:
            raise ExperimentError("QUERY_OCCURRENCE_UNKNOWN")
        visited = {source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for target, _ in adjacency[current]:
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
        return sorted(visited - {source})

    def successors(self, source: str) -> list[str]:
        return self._reachable_set(source, self.adjacency)

    def predecessors(self, target: str) -> list[str]:
        return self._reachable_set(target, self.reverse_adjacency)

    def conflicts(self, fact_id: str) -> list[str]:
        if fact_id not in self.fact_to_occurrence_map:
            raise ExperimentError("QUERY_FACT_UNKNOWN")
        return sorted({other for other, _ in self.conflict_index[fact_id]})

    def fact_to_occurrence(self, fact_id: str) -> str:
        if fact_id not in self.fact_to_occurrence_map:
            raise ExperimentError("QUERY_FACT_UNKNOWN")
        return self.fact_to_occurrence_map[fact_id]

    def occurrence_to_selected_facts(
        self, occurrence_id: str, selected_fact_ids: list[str] | None = None
    ) -> list[str]:
        if occurrence_id not in self.occurrences:
            raise ExperimentError("QUERY_OCCURRENCE_UNKNOWN")
        available = self.occurrence_to_facts_map[occurrence_id]
        if selected_fact_ids is None:
            return list(available)
        if not set(selected_fact_ids) <= set(available):
            raise ExperimentError("LIFTING_SELECTED_FACT_ENDPOINT_INVALID")
        return sorted(selected_fact_ids)

    def derived_happens_before(
        self, source: str, target: str
    ) -> dict[str, Any] | None:
        path = self.relation_path(source, target)
        if path is None or source == target:
            return None
        _, relation_ids = path
        row = make_relation(
            endpoint_level="occurrence",
            relation_type="happens_before",
            source_id=source,
            target_id=target,
            establishment_source="inferred",
            authority_id="indexed-candidate-resolver-v1",
            execution_run_id=self.run_id,
            rule_id="HB-SHORTEST-PATH-INDEXED-V1",
            input_relation_refs=relation_ids,
        )
        self.validate_derived_proof(row)
        return row

    def validate_derived_proof(self, row: dict[str, Any]) -> None:
        if not row.get("input_relation_refs"):
            raise ExperimentError("DERIVED_INPUT_RELATION_IDS_MISSING")
        path = self.relation_path(row["source_id"], row["target_id"])
        if path is None:
            raise ExperimentError("DERIVED_RELATION_PATH_MISSING")
        _, expected_ids = path
        if row.get("rule_id") != "HB-SHORTEST-PATH-INDEXED-V1":
            raise ExperimentError("DERIVED_RULE_ID_MISMATCH")
        if row["input_relation_refs"] != expected_ids:
            raise ExperimentError(
                "SHORTEST_PATH_INPUT_RELATION_IDS_MISMATCH"
            )
        if any(relation_id not in self.relation_by_id for relation_id in expected_ids):
            raise ExperimentError("DERIVED_INPUT_RELATION_UNKNOWN")

    def answer_query(self, query: dict[str, Any]) -> dict[str, Any]:
        self.query_count += 1
        query_type = query["query_type"]
        if query_type == "happens_before":
            result: Any = self.happens_before(
                query["source_id"], query["target_id"]
            )
        elif query_type == "concurrent_with":
            result = self.concurrent_with(query["source_id"], query["target_id"])
        elif query_type == "relation_path":
            path = self.relation_path(query["source_id"], query["target_id"])
            result = None if path is None else path[0]
        elif query_type == "predecessors":
            result = self.predecessors(query["target_id"])
        elif query_type == "successors":
            result = self.successors(query["source_id"])
        elif query_type == "conflicts":
            result = self.conflicts(query["fact_id"])
        elif query_type == "fact_to_occurrence":
            result = self.fact_to_occurrence(query["fact_id"])
        elif query_type == "occurrence_to_selected_facts":
            result = self.occurrence_to_selected_facts(
                query["occurrence_id"], query.get("selected_fact_ids")
            )
        else:
            raise ExperimentError("QUERY_TYPE_UNSUPPORTED")
        return {
            "query_id": query["query_id"],
            "query_type": query_type,
            "result": result,
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "occurrence_count": len(self.occurrences),
            "fact_count": len(self.fact_to_occurrence_map),
            "primitive_relation_count": len(self.relations),
            "causal_pair_count": len(self.pair_relation_ids),
            "retained_primitive_relation_count": sum(
                len(values) for values in self.pair_relation_ids.values()
            ),
            "query_count": self.query_count,
            "reachability_cache_entry_count": len(self._reachability_cache),
            "path_cache_entry_count": len(self._path_cache),
            "full_transitive_closure_materialized": False,
            "global_closure_pair_count": 0,
        }


def validate_candidate_input(payload: dict[str, Any]) -> None:
    if "hidden_primitive_relation_store" in payload:
        raise ExperimentError("CANDIDATE_HIDDEN_PRIMITIVE_STORE")
    forbidden = FORBIDDEN_CANDIDATE_INPUT_KEYS & set(payload)
    if forbidden:
        raise ExperimentError("CANDIDATE_FORBIDDEN_REFERENCE_INPUT")
    unknown = set(payload) - ALLOWED_CANDIDATE_INPUT_KEYS
    if unknown:
        raise ExperimentError("CANDIDATE_INPUT_KEY_UNREGISTERED")


def resolve_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validate_candidate_input(payload)
    resolver = IndexedCandidateResolver(
        execution_run_id=payload["execution_run_id"],
        primitive_store=payload["primitive_store"],
        capture_audit=payload["capture_audit"],
        lifting_rules=payload["lifting_rules"],
    )
    answers = [resolver.answer_query(query) for query in payload["queries"]]
    return {
        "process_role": "candidate",
        "execution_run_id": payload["execution_run_id"],
        "answers": answers,
        "metrics": resolver.metrics(),
        "schema_version": "candidate-query-output-v1",
    }
