from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


RowValues = dict[str, Any]
ExpressionFunction = Callable[[RowValues], Any]
PredicateFunction = Callable[[RowValues], bool]


@dataclass(frozen=True)
class Projection:
    output_name: str
    expression: str
    evaluate: ExpressionFunction


@dataclass(frozen=True)
class Aggregate:
    output_name: str
    function: str
    expression: str | None = None
    evaluate: ExpressionFunction | None = None

    def __post_init__(self) -> None:
        normalized = self.function.upper()
        if normalized not in {"SUM", "COUNT", "AVG"}:
            raise ValueError(f"unsupported aggregate: {self.function}")
        if normalized in {"SUM", "AVG"} and self.evaluate is None:
            raise ValueError(f"{normalized} requires an evaluator")
        object.__setattr__(self, "function", normalized)


@dataclass(frozen=True)
class SortKey:
    field: str
    descending: bool = False
