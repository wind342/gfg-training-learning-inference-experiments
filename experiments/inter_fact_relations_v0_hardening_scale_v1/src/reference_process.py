from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from ..common import ExperimentError, load_json, write_json
from .eager_reference_resolver import EagerReferenceResolver


ALLOWED_REFERENCE_INPUT_KEYS = {
    "execution_run_id",
    "runtime_receipts",
    "capture_contract",
    "queries",
    "reference_mode",
    "schema_version",
}
FORBIDDEN_REFERENCE_INPUT_KEYS = {
    "primitive_store",
    "candidate_graph",
    "candidate_output",
    "candidate_index",
}


class ReceiptOracleResolver:
    """Independent receipt-DAG oracle; it never reads candidate relations."""

    def __init__(
        self,
        runtime_receipts: dict[str, Any],
        capture_contract: dict[str, Any],
        *,
        eager: bool,
    ) -> None:
        self.receipts = runtime_receipts
        self.run_id = runtime_receipts["execution_run_id"]
        self.occurrence_rows = {
            row["concrete_occurrence_instance_id"]: row
            for row in runtime_receipts["occurrences"]
        }
        self.occurrences = set(self.occurrence_rows)
        self.fact_to_occurrence = {
            row["fact_id"]: row["occurrence_id"]
            for row in runtime_receipts["facts"]
        }
        self.occurrence_to_facts: defaultdict[str, list[str]] = defaultdict(list)
        for fact_id, occurrence_id in self.fact_to_occurrence.items():
            self.occurrence_to_facts[occurrence_id].append(fact_id)
        for values in self.occurrence_to_facts.values():
            values.sort()
        self.adjacency: defaultdict[str, set[str]] = defaultdict(set)
        self.reverse: defaultdict[str, set[str]] = defaultdict(set)
        self._construct_event_dag_from_receipts()
        self.topological_order = self._topological_order_or_fail()
        self.topological_position = {
            node: index for index, node in enumerate(self.topological_order)
        }
        self.components = self._weak_components()
        self.eager = (
            EagerReferenceResolver(runtime_receipts) if eager else None
        )
        self.scope_complete = self._independent_scope_completeness(
            capture_contract
        )
        self.conflict_index: defaultdict[str, set[str]] = defaultdict(set)
        access_index = {
            row["access_id"]: row
            for row in runtime_receipts.get("resource_access_receipts", [])
        }
        for row in runtime_receipts.get("conflict_receipts", []):
            left = access_index[row["left_access_id"]]["fact_id"]
            right = access_index[row["right_access_id"]]["fact_id"]
            self.conflict_index[left].add(right)
            self.conflict_index[right].add(left)
        self.query_count = 0

    def _construct_event_dag_from_receipts(self) -> None:
        edges: list[tuple[str, str]] = []
        edges.extend(
            (row["source_occurrence_id"], row["target_occurrence_id"])
            for row in self.receipts.get("program_order_receipts", [])
        )
        edges.extend(
            (row["send_occurrence_id"], row["receive_occurrence_id"])
            for row in self.receipts.get("message_receipts", [])
        )
        for row in self.receipts.get("synchronization_receipts", []):
            edges.extend(
                (source, row["release_occurrence_id"])
                for source in row["pre_occurrence_ids"]
            )
        edges.extend(
            (
                self.fact_to_occurrence[row["producer_fact_id"]],
                self.fact_to_occurrence[row["consumer_fact_id"]],
            )
            for row in self.receipts.get("generated_origin_receipts", [])
        )
        edges.extend(
            (
                self.fact_to_occurrence[row["source_fact_id"]],
                self.fact_to_occurrence[row["target_fact_id"]],
            )
            for row in self.receipts.get("reads_from_receipts", [])
        )
        for source, target in edges:
            self.adjacency[source].add(target)
            self.reverse[target].add(source)

    def _topological_order_or_fail(self) -> list[str]:
        indegree = {node: len(self.reverse[node]) for node in self.occurrences}
        queue = deque(sorted(node for node, count in indegree.items() if count == 0))
        ordered: list[str] = []
        while queue:
            node = queue.popleft()
            ordered.append(node)
            for target in sorted(self.adjacency[node]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(ordered) != len(self.occurrences):
            raise ExperimentError("REFERENCE_EVENT_DAG_CYCLE")
        return ordered

    def _weak_components(self) -> dict[str, int]:
        result: dict[str, int] = {}
        component_id = 0
        for start in sorted(self.occurrences):
            if start in result:
                continue
            queue = deque([start])
            result[start] = component_id
            while queue:
                node = queue.popleft()
                for neighbor in sorted(
                    self.adjacency[node] | self.reverse[node]
                ):
                    if neighbor not in result:
                        result[neighbor] = component_id
                        queue.append(neighbor)
            component_id += 1
        return result

    def _independent_scope_completeness(
        self, contract: dict[str, Any]
    ) -> dict[str, bool]:
        if contract["execution_run_id"] != self.run_id:
            raise ExperimentError("REFERENCE_CONTRACT_RUN_ID_MISMATCH")
        required = {
            "program_order",
            "messages",
            "synchronization",
            "generated_origin",
            "reads_from",
            "resource_access",
        }
        contract_by_scope = {
            row["scope_id"]: row for row in contract["scopes"]
        }
        occurrence_by_scope: defaultdict[str, set[str]] = defaultdict(set)
        for occurrence_id, row in self.occurrence_rows.items():
            occurrence_by_scope[row["scope_id"]].add(occurrence_id)
        executor_covered = {
            row["occurrence_id"]
            for row in self.receipts.get("executor_coverage_receipts", [])
        }
        blocked_scopes = {
            row["scope_id"]
            for key in (
                "unknown_edges",
                "unclassified_messages",
                "unclassified_operations",
                "external_communications",
                "unclassified_synchronization_operations",
                "unclassified_resource_accesses",
            )
            for row in self.receipts.get(key, [])
        }
        result: dict[str, bool] = {}
        for scope_id, occurrence_ids in occurrence_by_scope.items():
            row = contract_by_scope.get(scope_id)
            result[scope_id] = bool(
                row
                and set(row["covered_occurrence_ids"]) == occurrence_ids
                and required
                <= {
                    key
                    for key, value in row["planned_capture"].items()
                    if value is True
                }
                and row.get("external_communication_absent") is True
                and row.get("unobserved_scheduler_relation_ruled_out") is True
                and occurrence_ids <= executor_covered
                and scope_id not in blocked_scopes
            )
        return result

    def _path(self, source: str, target: str) -> list[str] | None:
        if self.eager is not None:
            return self.eager.relation_path(source, target)
        if source == target:
            return [source]
        if self.components[source] != self.components[target]:
            return None
        if self.topological_position[source] >= self.topological_position[target]:
            return None
        queue = deque([source])
        predecessor: dict[str, str] = {}
        seen = {source}
        while queue:
            current = queue.popleft()
            for next_node in sorted(self.adjacency[current]):
                if next_node in seen:
                    continue
                seen.add(next_node)
                predecessor[next_node] = current
                if next_node == target:
                    path = [target]
                    while path[-1] != source:
                        path.append(predecessor[path[-1]])
                    return list(reversed(path))
                queue.append(next_node)
        return None

    def _happens_before(self, source: str, target: str) -> bool:
        if self.eager is not None:
            return self.eager.happens_before(source, target)
        return source != target and self._path(source, target) is not None

    def _reachable(
        self, source: str, adjacency: dict[str, set[str]]
    ) -> list[str]:
        seen = {source}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for target in sorted(adjacency[node]):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return sorted(seen - {source})

    def answer(self, query: dict[str, Any]) -> dict[str, Any]:
        self.query_count += 1
        query_type = query["query_type"]
        if query_type == "happens_before":
            result: Any = self._happens_before(
                query["source_id"], query["target_id"]
            )
        elif query_type == "concurrent_with":
            left = query["source_id"]
            right = query["target_id"]
            left_scope = self.occurrence_rows[left]["scope_id"]
            right_scope = self.occurrence_rows[right]["scope_id"]
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
                    "value": not self._happens_before(left, right)
                    and not self._happens_before(right, left),
                }
        elif query_type == "relation_path":
            result = self._path(query["source_id"], query["target_id"])
        elif query_type == "predecessors":
            if self.eager is not None:
                result = self.eager.predecessors(query["target_id"])
            else:
                result = self._reachable(query["target_id"], self.reverse)
        elif query_type == "successors":
            if self.eager is not None:
                result = self.eager.successors(query["source_id"])
            else:
                result = self._reachable(query["source_id"], self.adjacency)
        elif query_type == "conflicts":
            result = sorted(self.conflict_index[query["fact_id"]])
        elif query_type == "fact_to_occurrence":
            result = self.fact_to_occurrence[query["fact_id"]]
        elif query_type == "occurrence_to_selected_facts":
            available = self.occurrence_to_facts[query["occurrence_id"]]
            requested = query.get("selected_fact_ids")
            result = (
                list(available)
                if requested is None
                else sorted(set(requested) & set(available))
            )
        else:
            raise ExperimentError("REFERENCE_QUERY_TYPE_UNSUPPORTED")
        return {
            "query_id": query["query_id"],
            "query_type": query_type,
            "result": result,
        }

    def metrics(self) -> dict[str, Any]:
        eager_metrics = self.eager.metrics() if self.eager is not None else {}
        return {
            "occurrence_count": len(self.occurrences),
            "fact_count": len(self.fact_to_occurrence),
            "query_count": self.query_count,
            "reference_kind": (
                "small_eager_receipt_oracle"
                if self.eager is not None
                else "lazy_runtime_receipt_event_dag_oracle"
            ),
            "full_transitive_closure_materialized": self.eager is not None,
            "global_closure_pair_count": eager_metrics.get(
                "global_closure_pair_count", 0
            ),
        }


def validate_reference_input(payload: dict[str, Any]) -> None:
    if FORBIDDEN_REFERENCE_INPUT_KEYS & set(payload):
        raise ExperimentError("REFERENCE_FORBIDDEN_CANDIDATE_INPUT")
    if set(payload) - ALLOWED_REFERENCE_INPUT_KEYS:
        raise ExperimentError("REFERENCE_INPUT_KEY_UNREGISTERED")


def resolve_reference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validate_reference_input(payload)
    if payload["runtime_receipts"]["execution_run_id"] != payload[
        "execution_run_id"
    ]:
        raise ExperimentError("REFERENCE_RECEIPT_RUN_ID_MISMATCH")
    mode = payload["reference_mode"]
    if mode not in {"eager", "lazy_oracle"}:
        raise ExperimentError("REFERENCE_MODE_INVALID")
    resolver = ReceiptOracleResolver(
        payload["runtime_receipts"],
        payload["capture_contract"],
        eager=mode == "eager",
    )
    answers = [resolver.answer(query) for query in payload["queries"]]
    return {
        "process_role": "reference",
        "execution_run_id": payload["execution_run_id"],
        "answers": answers,
        "metrics": resolver.metrics(),
        "schema_version": "reference-query-output-v1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output)
    try:
        result = resolve_reference_payload(load_json(Path(args.input)))
    except ExperimentError as error:
        write_json(
            output_path,
            {
                "status": "FAIL",
                "reason_code": str(error),
                "partial_success": False,
                "process_role": "reference",
            },
        )
        return 2
    write_json(output_path, {"status": "PASS", **result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
