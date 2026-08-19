from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from generation_relation_core.snapshots import (
    SnapshotValidation,
    ValidatedSnapshot,
    validate_snapshot,
)

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.operators import Aggregate, Projection
from experiments.database_lineage.src.relational_executor import (
    RelationTuple,
    RelationalExecutor,
    base_tuple,
)
from experiments.database_lineage.src.synthetic_cases import execute_business_query

from .execution_capture import ProjectionCaptureContract
from .native_otel_capture import (
    NativeOtelCapture,
    canonicalize_native_trace,
)


Workload = Callable[[object | None], list[RelationTuple]]


@dataclass
class CapturedRun:
    rows: list[RelationTuple]
    core: CoreAdapter | None
    native: NativeOtelCapture | None
    snapshot: ValidatedSnapshot | None
    validation: SnapshotValidation | None
    native_trace: dict | None


def selection_fixture(contract: object | None) -> list[RelationTuple]:
    rows = [
        base_tuple("oracle:a", "Oracle", {"value": 1, "keep": True}, 0),
        base_tuple("oracle:b", "Oracle", {"value": 2, "keep": False}, 1),
        base_tuple("oracle:c", "Oracle", {"value": 3, "keep": True}, 2),
    ]
    return RelationalExecutor(contract).selection(  # type: ignore[arg-type]
        rows,
        stage="oracle_selection",
        predicate=lambda row: row["keep"],
        predicate_description="keep = true",
    )


def business_fixture(contract: object | None) -> list[RelationTuple]:
    rows, _executor = execute_business_query(contract)  # type: ignore[arg-type]
    return rows


def run_captured(
    workload: Workload,
    *,
    run_id: str,
    core_enabled: bool,
    otel_enabled: bool,
) -> CapturedRun:
    core = CoreAdapter(run_id=run_id) if core_enabled else None
    native = NativeOtelCapture(run_id=run_id) if otel_enabled else None
    contract = (
        ProjectionCaptureContract(core=core, native=native)
        if core is not None or native is not None
        else None
    )
    rows = workload(contract)
    snapshot = validation = None
    if core is not None:
        snapshot = core.validated_snapshot()
        validation = validate_snapshot(snapshot, core.registry)
    native_trace = None
    if native is not None:
        native_trace = canonicalize_native_trace(
            native.finish(), expected_run_id=run_id
        )
    return CapturedRun(
        rows=rows,
        core=core,
        native=native,
        snapshot=snapshot,
        validation=validation,
        native_trace=native_trace,
    )


def _selection_counterexample(
    prefix: str, contract: object | None
) -> list[RelationTuple]:
    rows = [
        base_tuple(f"{prefix}:source:1", "Strict", {"value": 11}, 0),
        base_tuple(f"{prefix}:source:2", "Strict", {"value": 22}, 1),
    ]
    return RelationalExecutor(contract).selection(  # type: ignore[arg-type]
        rows,
        stage="strict_selection",
        predicate=lambda _row: True,
        predicate_description="true",
    )


def strict_selection_workload(prefix: str) -> Workload:
    return lambda contract: _selection_counterexample(prefix, contract)


def _many_to_many_counterexample(
    prefix: str, contract: object | None
) -> list[RelationTuple]:
    left = [
        base_tuple(f"{prefix}:left:1", "Left", {"key": "k", "left": "same"}, 0),
        base_tuple(f"{prefix}:left:2", "Left", {"key": "k", "left": "same"}, 1),
    ]
    right = [
        base_tuple(f"{prefix}:right:1", "Right", {"key": "k", "right": "same"}, 0),
        base_tuple(f"{prefix}:right:2", "Right", {"key": "k", "right": "same"}, 1),
    ]
    return RelationalExecutor(contract).equi_join(  # type: ignore[arg-type]
        left,
        right,
        stage="strict_many_to_many",
        left_keys=["key"],
        right_keys=["key"],
        right_prefix="r_",
    )


def strict_many_to_many_workload(prefix: str) -> Workload:
    return lambda contract: _many_to_many_counterexample(prefix, contract)


def q6_like_small_fixture(contract: object | None) -> list[RelationTuple]:
    rows = [
        base_tuple("q6s:1", "Lineitem", {"price": 100, "discount": 2}, 0),
        base_tuple("q6s:2", "Lineitem", {"price": 200, "discount": 0}, 1),
        base_tuple("q6s:3", "Lineitem", {"price": 50, "discount": 3}, 2),
    ]
    executor = RelationalExecutor(contract)  # type: ignore[arg-type]
    selected = executor.selection(
        rows,
        stage="q6s_selection",
        predicate=lambda row: row["discount"] > 0,
        predicate_description="discount > 0",
    )
    projected = executor.projection(
        selected,
        stage="q6s_revenue",
        projections=[
            Projection(
                "revenue_term",
                "price * discount",
                lambda row: row["price"] * row["discount"],
            )
        ],
    )
    return executor.group_by(
        projected,
        stage="q6s_aggregation",
        group_keys=[],
        aggregates=[
            Aggregate("revenue", "SUM", "revenue_term", lambda row: row["revenue_term"])
        ],
    )


def ordinary_value_rows(rows: Sequence[RelationTuple]) -> list[dict]:
    return [dict(row.values) for row in rows]
