from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from ..common import ExperimentError


class EagerReferenceResolver:
    """Small-only receipt oracle that materializes complete occurrence closure."""

    def __init__(self, runtime_receipts: dict[str, Any]) -> None:
        self.receipts = runtime_receipts
        self.run_id = runtime_receipts["execution_run_id"]
        self.occurrences = {
            row["concrete_occurrence_instance_id"]
            for row in runtime_receipts["occurrences"]
        }
        self.fact_to_occurrence = {
            row["fact_id"]: row["occurrence_id"]
            for row in runtime_receipts["facts"]
        }
        self.adjacency: defaultdict[str, set[str]] = defaultdict(set)
        self._construct_edges_from_receipts()
        self._reject_cycle()
        self.paths = self._materialize_all_shortest_paths()

    def _construct_edges_from_receipts(self) -> None:
        for row in self.receipts.get("program_order_receipts", []):
            self.adjacency[row["source_occurrence_id"]].add(
                row["target_occurrence_id"]
            )
        for row in self.receipts.get("message_receipts", []):
            self.adjacency[row["send_occurrence_id"]].add(
                row["receive_occurrence_id"]
            )
        for row in self.receipts.get("synchronization_receipts", []):
            for source in row["pre_occurrence_ids"]:
                self.adjacency[source].add(row["release_occurrence_id"])
        for row in self.receipts.get("generated_origin_receipts", []):
            self.adjacency[self.fact_to_occurrence[row["producer_fact_id"]]].add(
                self.fact_to_occurrence[row["consumer_fact_id"]]
            )
        for row in self.receipts.get("reads_from_receipts", []):
            self.adjacency[self.fact_to_occurrence[row["source_fact_id"]]].add(
                self.fact_to_occurrence[row["target_fact_id"]]
            )

    def _reject_cycle(self) -> None:
        indegree = {node: 0 for node in self.occurrences}
        for targets in self.adjacency.values():
            for target in targets:
                indegree[target] += 1
        queue = deque(node for node in sorted(indegree) if indegree[node] == 0)
        visited = 0
        while queue:
            source = queue.popleft()
            visited += 1
            for target in sorted(self.adjacency[source]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(self.occurrences):
            raise ExperimentError("REFERENCE_EVENT_DAG_CYCLE")

    def _materialize_all_shortest_paths(
        self,
    ) -> dict[tuple[str, str], list[str]]:
        paths: dict[tuple[str, str], list[str]] = {}
        for source in sorted(self.occurrences):
            queue = deque([source])
            predecessor: dict[str, str] = {}
            visited = {source}
            while queue:
                current = queue.popleft()
                for target in sorted(self.adjacency[current]):
                    if target in visited:
                        continue
                    visited.add(target)
                    predecessor[target] = current
                    queue.append(target)
                    path = [target]
                    cursor = target
                    while cursor != source:
                        cursor = predecessor[cursor]
                        path.append(cursor)
                    paths[(source, target)] = list(reversed(path))
        return paths

    def happens_before(self, source: str, target: str) -> bool:
        return (source, target) in self.paths

    def relation_path(self, source: str, target: str) -> list[str] | None:
        return self.paths.get((source, target))

    def successors(self, source: str) -> list[str]:
        return sorted(target for left, target in self.paths if left == source)

    def predecessors(self, target: str) -> list[str]:
        return sorted(source for source, right in self.paths if right == target)

    def metrics(self) -> dict[str, Any]:
        return {
            "reference_kind": "small_eager_receipt_oracle",
            "full_transitive_closure_materialized": True,
            "global_closure_pair_count": len(self.paths),
        }
