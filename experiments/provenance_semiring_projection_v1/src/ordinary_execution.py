from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .structural import canonical_json_bytes, logical_output_key
from .workloads import select_query


Collector = Callable[[dict[str, Any]], object]


@dataclass(frozen=True)
class ExecutionRow:
    identity: str
    values: dict[str, Any]


def _emit(collector: Collector | None, event: dict[str, Any]) -> None:
    if collector is not None:
        collector(event)


def _predicate(values: dict[str, Any], spec: dict[str, Any]) -> bool:
    actual = values[spec["field"]]
    expected = spec["value"]
    operator = spec["operator"]
    if operator == "eq":
        return actual == expected
    if operator == "ge":
        return actual >= expected
    if operator == "gt":
        return actual > expected
    raise ValueError(f"unsupported predicate operator: {operator!r}")


class OrdinaryPositiveRAExecutor:
    """Deterministic positive-RA bag executor with an optional write-only observer."""

    def __init__(self, workload: dict[str, Any], collector: Collector | None = None) -> None:
        self.workload = workload
        self.workload_id = workload["id"]
        self.collector = collector
        self._occurrence_count = 0

    def _base(self, relation: str) -> list[ExecutionRow]:
        rows: list[ExecutionRow] = []
        for row in self.workload["relations"][relation]:
            item = ExecutionRow(row["source_identity"], dict(row["values"]))
            rows.append(item)
            _emit(
                self.collector,
                {
                    "event": "source",
                    "workload_id": self.workload_id,
                    "relation": relation,
                    "source_identity": item.identity,
                    "values": item.values,
                },
            )
        return rows

    def _output(
        self,
        *,
        stage: str,
        operator: str,
        values: dict[str, Any],
        inputs: list[ExecutionRow],
        roles: list[str],
        terminal: bool,
        details: dict[str, Any],
    ) -> ExecutionRow:
        ordinal = self._occurrence_count
        self._occurrence_count += 1
        identity = f"{self.workload_id}:{stage}:o{ordinal:08d}"
        row = ExecutionRow(identity, dict(values))
        _emit(
            self.collector,
            {
                "event": "occurrence",
                "workload_id": self.workload_id,
                "occurrence_identity": f"{self.workload_id}:{stage}:g{ordinal:08d}",
                "stage": stage,
                "operator": operator,
                "output_identity": identity,
                "logical_output_key": logical_output_key(self.workload_id, values),
                "terminal": terminal,
                "values": row.values,
                "inputs": [item.identity for item in inputs],
                "roles": roles,
                "details": details,
            },
        )
        return row

    def _dispose(self, *, stage: str, operator: str, item: ExecutionRow, reason: str, details: dict[str, Any]) -> None:
        _emit(
            self.collector,
            {
                "event": "disposition",
                "workload_id": self.workload_id,
                "stage": stage,
                "operator": operator,
                "input_identity": item.identity,
                "reason": reason,
                "details": details,
            },
        )

    def evaluate(self, node: dict[str, Any], *, terminal: bool = False) -> list[ExecutionRow]:
        operator = node["op"]
        if operator == "base":
            return self._base(node["relation"])
        stage = node["stage"]
        if operator == "select":
            rows = self.evaluate(node["input"])
            result: list[ExecutionRow] = []
            for item in rows:
                passed = _predicate(item.values, node["predicate"])
                details = {"predicate": node["predicate"], "predicate_result": passed}
                if passed:
                    result.append(self._output(stage=stage, operator=operator, values=item.values, inputs=[item], roles=["selection_input:0"], terminal=terminal, details=details))
                else:
                    self._dispose(stage=stage, operator=operator, item=item, reason="selection_excluded", details=details)
            return result
        if operator in {"project", "rename"}:
            rows = self.evaluate(node["input"])
            result = []
            for item in rows:
                values = {field["output"]: item.values[field["input"]] for field in node["fields"]}
                result.append(self._output(stage=stage, operator=operator, values=values, inputs=[item], roles=[f"{operator}_input:0"], terminal=terminal, details={"fields": node["fields"]}))
            return result
        if operator == "union":
            result = []
            for branch_index, child in enumerate(node["inputs"]):
                for item in self.evaluate(child):
                    result.append(self._output(stage=stage, operator=operator, values=item.values, inputs=[item], roles=[f"union_branch:{branch_index}"], terminal=terminal, details={"branch_index": branch_index}))
            return result
        if operator == "join":
            left = self.evaluate(node["left"])
            right = self.evaluate(node["right"])
            right_index: dict[tuple[Any, ...], list[ExecutionRow]] = {}
            for item in right:
                key = tuple(item.values[name] for name in node["right_keys"])
                right_index.setdefault(key, []).append(item)
            result = []
            matched_right: set[str] = set()
            for left_item in left:
                key = tuple(left_item.values[name] for name in node["left_keys"])
                matches = right_index.get(key, [])
                if not matches:
                    self._dispose(stage=stage, operator=operator, item=left_item, reason="join_unmatched_left", details={"join_key": list(key)})
                for right_item in matches:
                    matched_right.add(right_item.identity)
                    values = dict(left_item.values)
                    for name, value in right_item.values.items():
                        target = name if name not in values else f"{node.get('right_prefix', 'right_')}{name}"
                        if target in values:
                            raise ValueError(f"join column collision: {target}")
                        values[target] = value
                    result.append(self._output(stage=stage, operator=operator, values=values, inputs=[left_item, right_item], roles=["join_left_slot:0", "join_right_slot:1"], terminal=terminal, details={"left_keys": node["left_keys"], "right_keys": node["right_keys"], "join_key": list(key)}))
            for right_item in right:
                if right_item.identity not in matched_right:
                    key = [right_item.values[name] for name in node["right_keys"]]
                    self._dispose(stage=stage, operator=operator, item=right_item, reason="join_unmatched_right", details={"join_key": key})
            return result
        raise ValueError(f"unsupported positive-RA operator: {operator!r}")


def execute_ordinary(workload: dict[str, Any], *, variant: str | None = None, collector: Collector | None = None) -> tuple[bytes, dict[str, Any]]:
    selected_variant, query = select_query(workload, variant)
    executor = OrdinaryPositiveRAExecutor(workload, collector)
    rows = executor.evaluate(query, terminal=True)
    ordinary_document = {"workload_id": workload["id"], "variant": selected_variant, "rows": [row.values for row in rows]}
    return canonical_json_bytes(ordinary_document), {
        "source_count": sum(len(rows) for rows in workload["relations"].values()),
        "occurrence_count": executor._occurrence_count,
        "ordinary_row_count": len(rows),
    }
