from __future__ import annotations

import json
import re
from typing import Any


_ORDINAL = re.compile(r"(?:^|\|)ordinal=(\d+)(?:\||$)")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


class NativeOracleLineageReference:
    """Receipt boundary reference with gradient edges supplied by native evidence."""

    def __init__(
        self,
        receipts: list[dict[str, Any]],
        native_relations: list[dict[str, Any]],
        workload_key: str,
    ) -> None:
        self._sources = {
            row["payload"]["source_ref"]
            for row in receipts
            if row["kind"] in {"source", "source_metadata"}
        }
        self._incoming: dict[str, list[dict[str, Any]]] = {}
        self._outgoing: dict[str, list[dict[str, Any]]] = {}
        loss_ref = next(
            row["payload"]["loss_ref"]
            for row in receipts
            if row["kind"] == "backward_start"
        )

        def occurrence_key(receipt: dict[str, Any]) -> str:
            return (
                f"{receipt['step_key']}:{receipt['stage']}:"
                f"{receipt['kind']}:{receipt['ordinal']}"
            )

        def edge(origin: str, outcome: str, occurrence: str, role: str) -> None:
            match = _ORDINAL.search(role)
            row = {
                "occurrence_key": occurrence,
                "ordinal": None if match is None else int(match.group(1)),
                "origin_key": origin,
                "outcome_key": outcome,
                "role": role,
            }
            self._incoming.setdefault(outcome, []).append(row)
            self._outgoing.setdefault(origin, []).append(row)

        for receipt in [row for row in receipts if row["kind"] == "operation"]:
            output = receipt["payload"]["output_ref"]
            for slot, origin in enumerate(receipt["payload"]["input_refs"]):
                edge(
                    origin,
                    output,
                    occurrence_key(receipt),
                    f"operation_input|slot={slot}|ordinal={slot}",
                )

        backward_receipt = next(
            row for row in receipts if row["kind"] == "backward_start"
        )
        backward_key = f"{backward_receipt['step_key']}:backward:completion"
        edge(
            loss_ref,
            backward_key,
            occurrence_key(backward_receipt),
            "backward_from_loss|ordinal=0",
        )
        dependencies_by_target: dict[str, list[str]] = {}
        for relation in native_relations:
            if relation["workload_key"] != workload_key:
                continue
            dependencies_by_target.setdefault(
                relation["target_gradient_key"], []
            ).append(relation["dependency_key"])

        gradient_by_parameter: dict[str, str] = {}
        for receipt in [row for row in receipts if row["kind"] == "gradient"]:
            leaf_ref = receipt["payload"]["leaf_ref"]
            gradient_key = f"{receipt['step_key']}:gradient:{leaf_ref}"
            source_ref = self._leaf_source_ref(leaf_ref)
            edge(
                source_ref,
                gradient_key,
                occurrence_key(receipt),
                "gradient_for_leaf|ordinal=1",
            )
            for ordinal, dependency_ref in enumerate(
                sorted(dependencies_by_target.get(gradient_key, [])),
                start=2,
            ):
                edge(
                    dependency_ref,
                    gradient_key,
                    occurrence_key(receipt),
                    f"gradient_value_dependency|ordinal={ordinal}",
                )
            if leaf_ref.startswith("parameter:"):
                gradient_by_parameter[leaf_ref.split(":", 1)[1]] = gradient_key

        after = next(row for row in receipts if row["kind"] == "optimizer_after")
        optimizer_occurrence = occurrence_key(after)
        optimizer_source = "source:optimizer:state:before"
        optimizer_after = f"{after['step_key']}:optimizer_state:after"
        edge(
            optimizer_source,
            optimizer_after,
            optimizer_occurrence,
            "optimizer_state_transition|ordinal=0",
        )
        for parameter_name in sorted(after["payload"]["parameter_values"]):
            parameter_after = f"{after['step_key']}:parameter:{parameter_name}:after"
            gradient = gradient_by_parameter.get(parameter_name)
            if gradient is None:
                continue
            edge(
                f"source:parameter:{parameter_name}:before",
                parameter_after,
                optimizer_occurrence,
                "parameter_previous_version|ordinal=0",
            )
            edge(
                gradient,
                parameter_after,
                optimizer_occurrence,
                "optimizer_gradient_input|ordinal=1",
            )
            edge(
                gradient,
                optimizer_after,
                optimizer_occurrence,
                f"optimizer_gradient_state_input|parameter={parameter_name}",
            )
            edge(
                optimizer_source,
                parameter_after,
                optimizer_occurrence,
                "optimizer_state_input|ordinal=2",
            )
        for mapping in (self._incoming, self._outgoing):
            for key in mapping:
                mapping[key].sort(key=_canonical)

    @staticmethod
    def _leaf_source_ref(leaf_ref: str) -> str:
        kind, name = leaf_ref.split(":", 1)
        return (
            f"source:sample:{name}"
            if kind == "input"
            else f"source:parameter:{name}:before"
        )

    def _reverse(
        self,
        target: str,
        active: frozenset[str],
    ) -> list[list[dict[str, Any]]]:
        if target in active:
            raise ValueError("REFERENCE_V2_LINEAGE_CYCLE")
        if target in self._sources:
            return [[]]
        result = []
        for relation in self._incoming.get(target, []):
            for prefix in self._reverse(relation["origin_key"], active | {target}):
                result.append([*prefix, relation])
        if not result:
            raise ValueError(f"REFERENCE_V2_ORIGIN_MISSING:{target}")
        return result

    def reverse_paths(self, target_key: str) -> list[dict[str, Any]]:
        return self._normalize(self._reverse(target_key, frozenset()))

    def _forward(
        self,
        origin: str,
        prefix: list[dict[str, Any]],
        active: frozenset[str],
    ) -> list[list[dict[str, Any]]]:
        if origin in active:
            raise ValueError("REFERENCE_V2_LINEAGE_CYCLE")
        result = []
        for relation in self._outgoing.get(origin, []):
            path = [*prefix, relation]
            result.append(path)
            result.extend(
                self._forward(
                    relation["outcome_key"],
                    path,
                    active | {origin},
                )
            )
        return result

    def forward_paths(self, source_ref: str) -> list[dict[str, Any]]:
        if source_ref not in self._sources:
            raise ValueError("REFERENCE_V2_SOURCE_UNKNOWN")
        return self._normalize(self._forward(source_ref, [], frozenset()))

    @staticmethod
    def _normalize(paths: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        counts: dict[bytes, int] = {}
        values: dict[bytes, dict[str, Any]] = {}
        for path in paths:
            row = {
                "relations": path,
                "source_key": path[0]["origin_key"],
                "target_key": path[-1]["outcome_key"],
            }
            key = _canonical(row)
            counts[key] = counts.get(key, 0) + 1
            values[key] = row
        return [
            {**values[key], "multiplicity": counts[key]}
            for key in sorted(values)
        ]
