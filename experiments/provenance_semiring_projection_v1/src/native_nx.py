from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .native_polynomial_oracle import NativePolynomialOracle, native_variable_for_source
from .workloads import select_query


@dataclass(frozen=True)
class KRow:
    values: dict[str, Any]
    annotation: NativePolynomialOracle


KRelation = dict[str, KRow]


def _canonical_values(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _logical_output_key(workload_id: str, values: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_values(values).encode("utf-8")).hexdigest()
    return f"{workload_id}:value:{digest}"


def _insert(
    relation: KRelation,
    values: dict[str, Any],
    annotation: NativePolynomialOracle,
) -> None:
    key = _canonical_values(values)
    prior = relation.get(key)
    relation[key] = KRow(
        dict(values), annotation if prior is None else prior.annotation.add(annotation)
    )


def _predicate(values: dict[str, Any], spec: dict[str, Any]) -> bool:
    actual = values[spec["field"]]
    expected = spec["value"]
    if spec["operator"] == "eq":
        return actual == expected
    if spec["operator"] == "ge":
        return actual >= expected
    if spec["operator"] == "gt":
        return actual > expected
    raise ValueError(f"unsupported predicate operator: {spec['operator']!r}")


class NativeNXEvaluator:
    """Direct K-relation evaluator whose carrier is N[X].

    This implementation consumes only the relational fixture and its positive-RA
    AST. It neither executes nor imports the ordinary executor or Core capture.
    """

    def __init__(self, workload: dict[str, Any]) -> None:
        self.workload = workload
        self.source_variables: dict[str, str] = {}

    def _base(self, relation_name: str) -> KRelation:
        result: KRelation = {}
        for source in self.workload["relations"][relation_name]:
            identity = source["source_identity"]
            variable = native_variable_for_source(identity)
            prior_identity = self.source_variables.get(variable)
            if prior_identity is not None and prior_identity != identity:
                raise ValueError("source-variable collision")
            self.source_variables[variable] = identity
            _insert(result, source["values"], NativePolynomialOracle.variable(variable))
        return result

    def evaluate(self, node: dict[str, Any]) -> KRelation:
        operator = node["op"]
        if operator == "base":
            return self._base(node["relation"])
        if operator == "select":
            child = self.evaluate(node["input"])
            return {key: row for key, row in child.items() if _predicate(row.values, node["predicate"])}
        if operator in {"project", "rename"}:
            result: KRelation = {}
            for row in self.evaluate(node["input"]).values():
                values = {field["output"]: row.values[field["input"]] for field in node["fields"]}
                _insert(result, values, row.annotation)
            return result
        if operator == "union":
            result = {}
            for child_node in node["inputs"]:
                for row in self.evaluate(child_node).values():
                    _insert(result, row.values, row.annotation)
            return result
        if operator == "join":
            left = self.evaluate(node["left"])
            right = self.evaluate(node["right"])
            result = {}
            for left_row in left.values():
                left_key = tuple(left_row.values[name] for name in node["left_keys"])
                for right_row in right.values():
                    right_key = tuple(right_row.values[name] for name in node["right_keys"])
                    if left_key != right_key:
                        continue
                    values = dict(left_row.values)
                    for name, value in right_row.values.items():
                        target = name if name not in values else f"{node.get('right_prefix', 'right_')}{name}"
                        if target in values:
                            raise ValueError(f"join column collision: {target}")
                        values[target] = value
                    _insert(result, values, left_row.annotation.multiply(right_row.annotation))
            return result
        raise ValueError(f"unsupported positive-RA operator: {operator!r}")


def evaluate_native_nx(workload: dict[str, Any], *, variant: str | None = None) -> dict[str, object]:
    selected_variant, query = select_query(workload, variant)
    evaluator = NativeNXEvaluator(workload)
    relation = evaluator.evaluate(query)
    outputs = []
    for row in relation.values():
        outputs.append(
            {
                "logical_output_key": _logical_output_key(workload["id"], row.values),
                "values": row.values,
                "polynomial": row.annotation.to_document(),
            }
        )
    outputs.sort(key=lambda item: item["logical_output_key"])
    return {
        "schema_version": "native-nx-result-v1",
        "workload_id": workload["id"],
        "variant": selected_variant,
        "source_variables": [
            {"variable": variable, "source_identity": identity}
            for variable, identity in sorted(evaluator.source_variables.items())
        ],
        "outputs": outputs,
    }
