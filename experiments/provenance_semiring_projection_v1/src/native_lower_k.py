from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from .workloads import select_query


class Carrier(Protocol):
    domain_id: str

    def zero(self) -> object: ...
    def one(self) -> object: ...
    def variable(self, name: str) -> object: ...
    def plus(self, left: object, right: object) -> object: ...
    def times(self, left: object, right: object) -> object: ...
    def document(self, value: object) -> object: ...


class BagCarrier:
    domain_id = "bag_naturals"
    def zero(self) -> int: return 0
    def one(self) -> int: return 1
    def variable(self, name: str) -> int: return 1
    def plus(self, left: object, right: object) -> int: return int(left) + int(right)
    def times(self, left: object, right: object) -> int: return int(left) * int(right)
    def document(self, value: object) -> int: return int(value)


class BooleanCarrier:
    domain_id = "boolean"
    def zero(self) -> bool: return False
    def one(self) -> bool: return True
    def variable(self, name: str) -> bool: return True
    def plus(self, left: object, right: object) -> bool: return bool(left) or bool(right)
    def times(self, left: object, right: object) -> bool: return bool(left) and bool(right)
    def document(self, value: object) -> bool: return bool(value)


class FlatSourceSupportCarrier:
    """Flat variable-union task view, not a complete semiring carrier."""

    domain_id = "flat_source_support_view"
    def zero(self) -> frozenset[str]: return frozenset()
    def one(self) -> frozenset[str]: return frozenset()
    def variable(self, name: str) -> frozenset[str]: return frozenset({name})
    def plus(self, left: object, right: object) -> frozenset[str]: return frozenset(left) | frozenset(right)  # type: ignore[arg-type]
    def times(self, left: object, right: object) -> frozenset[str]: return frozenset(left) | frozenset(right)  # type: ignore[arg-type]
    def document(self, value: object) -> dict[str, list[str]]: return {"variables": sorted(frozenset(value))}  # type: ignore[arg-type]


def _absorb(terms: frozenset[frozenset[str]]) -> frozenset[frozenset[str]]:
    return frozenset(term for term in terms if not any(other < term for other in terms))


class PositiveBooleanCarrier:
    domain_id = "positive_boolean_lineage"
    def zero(self) -> frozenset[frozenset[str]]: return frozenset()
    def one(self) -> frozenset[frozenset[str]]: return frozenset({frozenset()})
    def variable(self, name: str) -> frozenset[frozenset[str]]: return frozenset({frozenset({name})})
    def plus(self, left: object, right: object) -> frozenset[frozenset[str]]:
        return _absorb(frozenset(left) | frozenset(right))  # type: ignore[arg-type]
    def times(self, left: object, right: object) -> frozenset[frozenset[str]]:
        left_terms = frozenset(left)  # type: ignore[arg-type]
        right_terms = frozenset(right)  # type: ignore[arg-type]
        if not left_terms or not right_terms:
            return frozenset()
        return _absorb(frozenset(a | b for a in left_terms for b in right_terms))
    def document(self, value: object) -> dict[str, list[list[str]]]:
        terms = sorted((sorted(term) for term in frozenset(value)), key=lambda term: (len(term), term))  # type: ignore[arg-type]
        return {"terms": terms}


CARRIERS: tuple[Carrier, ...] = (
    BagCarrier(),
    BooleanCarrier(),
    PositiveBooleanCarrier(),
    FlatSourceSupportCarrier(),
)


def _direct_canonical_values(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _direct_logical_output_key(workload_id: str, values: dict[str, Any]) -> str:
    digest = hashlib.sha256(_direct_canonical_values(values).encode("utf-8")).hexdigest()
    return f"{workload_id}:value:{digest}"


def _direct_variable_for_source(source_identity: str) -> str:
    if not isinstance(source_identity, str) or not source_identity:
        raise ValueError("source identity must be a non-empty string")
    return "x_" + hashlib.sha256(source_identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LowerKRow:
    values: dict[str, Any]
    annotation: object


def _predicate(values: dict[str, Any], spec: dict[str, Any]) -> bool:
    actual = values[spec["field"]]
    expected = spec["value"]
    if spec["operator"] == "eq": return actual == expected
    if spec["operator"] == "ge": return actual >= expected
    if spec["operator"] == "gt": return actual > expected
    raise ValueError("unsupported predicate")


class DirectLowerKEvaluator:
    """Direct target-semiring evaluator; it has no N[X] dependency."""

    def __init__(self, workload: dict[str, Any], carrier: Carrier) -> None:
        self.workload = workload
        self.carrier = carrier

    def _insert(self, relation: dict[str, LowerKRow], values: dict[str, Any], annotation: object) -> None:
        key = _direct_canonical_values(values)
        prior = relation.get(key)
        relation[key] = LowerKRow(dict(values), annotation if prior is None else self.carrier.plus(prior.annotation, annotation))

    def evaluate(self, node: dict[str, Any]) -> dict[str, LowerKRow]:
        op = node["op"]
        if op == "base":
            result: dict[str, LowerKRow] = {}
            for source in self.workload["relations"][node["relation"]]:
                self._insert(
                    result,
                    source["values"],
                    self.carrier.variable(_direct_variable_for_source(source["source_identity"])),
                )
            return result
        if op == "select":
            return {key: row for key, row in self.evaluate(node["input"]).items() if _predicate(row.values, node["predicate"])}
        if op in {"project", "rename"}:
            result = {}
            for row in self.evaluate(node["input"]).values():
                values = {field["output"]: row.values[field["input"]] for field in node["fields"]}
                self._insert(result, values, row.annotation)
            return result
        if op == "union":
            result = {}
            for child in node["inputs"]:
                for row in self.evaluate(child).values():
                    self._insert(result, row.values, row.annotation)
            return result
        if op == "join":
            result = {}
            left = self.evaluate(node["left"])
            right = self.evaluate(node["right"])
            for left_row in left.values():
                left_key = tuple(left_row.values[name] for name in node["left_keys"])
                for right_row in right.values():
                    if left_key != tuple(right_row.values[name] for name in node["right_keys"]):
                        continue
                    values = dict(left_row.values)
                    for name, value in right_row.values.items():
                        target = name if name not in values else f"{node.get('right_prefix', 'right_')}{name}"
                        if target in values: raise ValueError("join collision")
                        values[target] = value
                    self._insert(result, values, self.carrier.times(left_row.annotation, right_row.annotation))
            return result
        raise ValueError(f"unsupported operator: {op!r}")


def evaluate_direct_lower_domains(workload: dict[str, Any], *, variant: str | None = None) -> dict[str, Any]:
    selected_variant, query = select_query(workload, variant)
    domains = []
    for carrier in CARRIERS:
        relation = DirectLowerKEvaluator(workload, carrier).evaluate(query)
        outputs = [
            {"logical_output_key": _direct_logical_output_key(workload["id"], row.values), "values": row.values, "annotation": carrier.document(row.annotation)}
            for row in relation.values()
        ]
        outputs.sort(key=lambda item: item["logical_output_key"])
        domains.append({"domain_id": carrier.domain_id, "outputs": outputs})
    return {"workload_id": workload["id"], "variant": selected_variant, "domains": domains}
