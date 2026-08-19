from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cmp_to_key
from typing import Any, Iterable, Sequence, TYPE_CHECKING

from .operators import Aggregate, PredicateFunction, Projection, SortKey

if TYPE_CHECKING:
    from .core_adapter import CoreAdapter


@dataclass(frozen=True)
class RelationTuple:
    tuple_id: str
    table_identity: str
    values: dict[str, Any]
    order_key: tuple[Any, ...]
    support_id: str | None = None

    def with_support(self, support_id: str) -> "RelationTuple":
        return replace(self, support_id=support_id)


def base_tuple(
    tuple_id: str,
    table_identity: str,
    values: dict[str, Any],
    order_index: int,
) -> RelationTuple:
    if not tuple_id or not table_identity:
        raise ValueError("tuple_id and table_identity are required")
    return RelationTuple(
        tuple_id, table_identity, dict(values), (order_index, tuple_id)
    )


class RelationalExecutor:
    """A small deterministic bag-semantics executor.

    Tuple identities are control-plane state. They never enter ``values`` and
    therefore never enter ordinary CSV/JSON output.
    """

    def __init__(self, contract: "CoreAdapter | None" = None) -> None:
        self.contract = contract
        self._stage_names: set[str] = set()

    @property
    def contract_enabled(self) -> bool:
        return self.contract is not None

    def _claim_stage(self, stage: str) -> None:
        if not stage or stage in self._stage_names:
            raise ValueError(f"stage name must be unique and non-empty: {stage!r}")
        self._stage_names.add(stage)

    @staticmethod
    def _output(stage: str, ordinal: int, values: dict[str, Any]) -> RelationTuple:
        tuple_id = f"{stage}:{ordinal:08d}"
        return RelationTuple(tuple_id, stage, dict(values), (ordinal, tuple_id))

    def _capture_output(
        self,
        stage: str,
        operator_type: str,
        output: RelationTuple,
        inputs: Sequence[RelationTuple],
        roles: Sequence[str],
        payload: dict[str, Any],
    ) -> RelationTuple:
        if self.contract is None:
            return output
        support_id = self.contract.capture_output(
            stage=stage,
            operator_type=operator_type,
            output=output,
            inputs=list(inputs),
            roles=list(roles),
            occurrence_payload=payload,
        )
        return output.with_support(support_id)

    def _capture_disposition(
        self,
        stage: str,
        operator_type: str,
        item: RelationTuple,
        reason: str,
        payload: dict[str, Any],
    ) -> None:
        if self.contract is not None:
            self.contract.capture_disposition(
                stage=stage,
                operator_type=operator_type,
                input_tuple=item,
                reason=reason,
                occurrence_payload=payload,
            )

    def selection(
        self,
        rows: Sequence[RelationTuple],
        *,
        stage: str,
        predicate: PredicateFunction,
        predicate_description: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[RelationTuple]:
        self._claim_stage(stage)
        result: list[RelationTuple] = []
        params = parameters or {}
        for item in rows:
            passed = bool(predicate(item.values))
            payload = {
                "predicate": predicate_description,
                "parameters": params,
                "input_tuple_identity": item.tuple_id,
                "predicate_result": passed,
            }
            if passed:
                output = self._output(stage, len(result), item.values)
                result.append(
                    self._capture_output(
                        stage,
                        "selection",
                        output,
                        [item],
                        ["selection_input"],
                        payload,
                    )
                )
            else:
                self._capture_disposition(
                    stage,
                    "selection",
                    item,
                    "selection_excluded",
                    payload,
                )
        return result

    def projection(
        self,
        rows: Sequence[RelationTuple],
        *,
        stage: str,
        projections: Sequence[Projection],
    ) -> list[RelationTuple]:
        self._claim_stage(stage)
        if not projections:
            raise ValueError("at least one projection is required")
        result = []
        descriptions = [
            {"output_name": item.output_name, "expression": item.expression}
            for item in projections
        ]
        for ordinal, item in enumerate(rows):
            values = {
                column.output_name: column.evaluate(item.values)
                for column in projections
            }
            output = self._output(stage, ordinal, values)
            payload = {
                "projections": descriptions,
                "dropped_columns": sorted(
                    set(item.values) - {p.output_name for p in projections}
                ),
                "input_tuple_identity": item.tuple_id,
            }
            result.append(
                self._capture_output(
                    stage,
                    "projection",
                    output,
                    [item],
                    ["projection_input"],
                    payload,
                )
            )
        return result

    def equi_join(
        self,
        left: Sequence[RelationTuple],
        right: Sequence[RelationTuple],
        *,
        stage: str,
        left_keys: Sequence[str],
        right_keys: Sequence[str],
        right_prefix: str = "right_",
    ) -> list[RelationTuple]:
        self._claim_stage(stage)
        if not left_keys or len(left_keys) != len(right_keys):
            raise ValueError("join keys must be non-empty and have equal arity")
        right_index: dict[tuple[Any, ...], list[RelationTuple]] = {}
        for item in right:
            key = tuple(item.values[name] for name in right_keys)
            right_index.setdefault(key, []).append(item)
        result: list[RelationTuple] = []
        matched_right: set[str] = set()
        for left_item in left:
            key = tuple(left_item.values[name] for name in left_keys)
            matches = right_index.get(key, [])
            if not matches:
                self._capture_disposition(
                    stage,
                    "equi_join",
                    left_item,
                    "join_unmatched_left",
                    {
                        "left_keys": list(left_keys),
                        "right_keys": list(right_keys),
                        "join_key": list(key),
                        "side": "left",
                    },
                )
                continue
            for right_item in matches:
                matched_right.add(right_item.tuple_id)
                values = dict(left_item.values)
                for name, value in right_item.values.items():
                    target = name if name not in values else f"{right_prefix}{name}"
                    if target in values:
                        raise ValueError(f"join column collision: {target}")
                    values[target] = value
                output = self._output(stage, len(result), values)
                result.append(
                    self._capture_output(
                        stage,
                        "equi_join",
                        output,
                        [left_item, right_item],
                        ["join_left_input", "join_right_input"],
                        {
                            "left_keys": list(left_keys),
                            "right_keys": list(right_keys),
                            "join_key": list(key),
                            "condition": " AND ".join(
                                f"left.{left_name} = right.{right_name}"
                                for left_name, right_name in zip(
                                    left_keys, right_keys, strict=True
                                )
                            ),
                            "left_tuple_identity": left_item.tuple_id,
                            "right_tuple_identity": right_item.tuple_id,
                        },
                    )
                )
        for right_item in right:
            if right_item.tuple_id in matched_right:
                continue
            key = tuple(right_item.values[name] for name in right_keys)
            self._capture_disposition(
                stage,
                "equi_join",
                right_item,
                "join_unmatched_right",
                {
                    "left_keys": list(left_keys),
                    "right_keys": list(right_keys),
                    "join_key": list(key),
                    "side": "right",
                },
            )
        return result

    def group_by(
        self,
        rows: Sequence[RelationTuple],
        *,
        stage: str,
        group_keys: Sequence[str],
        aggregates: Sequence[Aggregate],
    ) -> list[RelationTuple]:
        self._claim_stage(stage)
        if not aggregates:
            raise ValueError("at least one aggregate is required")
        groups: dict[tuple[Any, ...], list[RelationTuple]] = {}
        if not group_keys and not rows:
            groups[()] = []
        for item in rows:
            key = tuple(item.values[name] for name in group_keys)
            groups.setdefault(key, []).append(item)
        result = []
        aggregate_descriptions = [
            {
                "output_name": agg.output_name,
                "function": agg.function,
                "expression": agg.expression,
            }
            for agg in aggregates
        ]
        for ordinal, (key, contributors) in enumerate(groups.items()):
            values = {name: value for name, value in zip(group_keys, key, strict=True)}
            for aggregate in aggregates:
                if aggregate.function == "COUNT":
                    value: Any = len(contributors)
                else:
                    terms = [aggregate.evaluate(item.values) for item in contributors]  # type: ignore[misc]
                    total = sum(terms, Decimal("0"))
                    if aggregate.function == "SUM":
                        value = total if contributors else None
                    else:
                        value = (
                            total / Decimal(len(contributors)) if contributors else None
                        )
                values[aggregate.output_name] = value
            output = self._output(stage, ordinal, values)
            roles = ["aggregation_contributor" for _ in contributors]
            payload = {
                "group_keys": list(group_keys),
                "group_key_values": list(key),
                "aggregates": aggregate_descriptions,
                "participant_count": len(contributors),
                "avg_semantics": "exact Decimal SUM divided by exact COUNT",
            }
            if contributors:
                result.append(
                    self._capture_output(
                        stage,
                        "group_by",
                        output,
                        contributors,
                        roles,
                        payload,
                    )
                )
            else:
                # SQL scalar aggregation over an empty input has no source
                # relation. It remains outside the mandatory tested fixtures.
                result.append(output)
        return result

    def sort(
        self,
        rows: Sequence[RelationTuple],
        *,
        stage: str,
        sort_keys: Sequence[SortKey],
    ) -> list[RelationTuple]:
        self._claim_stage(stage)
        if not sort_keys:
            raise ValueError("at least one sort key is required")

        def compare(left: RelationTuple, right: RelationTuple) -> int:
            for spec in sort_keys:
                lv = left.values[spec.field]
                rv = right.values[spec.field]
                if lv == rv:
                    continue
                value = -1 if lv < rv else 1
                return -value if spec.descending else value
            return (
                -1
                if left.tuple_id < right.tuple_id
                else (1 if left.tuple_id > right.tuple_id else 0)
            )

        ordered = sorted(rows, key=cmp_to_key(compare))
        result = []
        description = [
            {"field": spec.field, "direction": "DESC" if spec.descending else "ASC"}
            for spec in sort_keys
        ]
        for ordinal, item in enumerate(ordered):
            output = self._output(stage, ordinal, item.values)
            result.append(
                self._capture_output(
                    stage,
                    "sort",
                    output,
                    [item],
                    ["sort_input"],
                    {
                        "sort_keys": description,
                        "tie_breaker": "input tuple identity ascending",
                    },
                )
            )
        return result

    def limit(
        self,
        rows: Sequence[RelationTuple],
        *,
        stage: str,
        count: int,
    ) -> list[RelationTuple]:
        self._claim_stage(stage)
        if count < 0:
            raise ValueError("limit must be non-negative")
        result = []
        for ordinal, item in enumerate(rows):
            payload = {
                "limit": count,
                "input_position": ordinal,
                "input_tuple_identity": item.tuple_id,
            }
            if ordinal < count:
                output = self._output(stage, ordinal, item.values)
                result.append(
                    self._capture_output(
                        stage,
                        "limit",
                        output,
                        [item],
                        ["limit_retained"],
                        payload,
                    )
                )
            else:
                self._capture_disposition(
                    stage, "limit", item, "limit_excluded", payload
                )
        return result


def ensure_unique_tuple_identities(rows: Iterable[RelationTuple]) -> None:
    ids = [row.tuple_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("tuple identities must be unique under bag semantics")
