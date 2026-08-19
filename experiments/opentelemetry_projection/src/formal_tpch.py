from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import duckdb
import psutil

from generation_relation_core.snapshots import validate_snapshot

from experiments.database_lineage.src.core_adapter import CoreAdapter
from experiments.database_lineage.src.duckdb_reference import (
    compare_official_typed,
    compare_rows,
    execute_reference,
    parse_official_answer,
)
from experiments.database_lineage.src.result_serializer import (
    csv_bytes,
    json_bytes,
    ordinary_rows,
)
from experiments.database_lineage.src.tpch_loader import (
    load_tables,
    official_sql_and_answers,
)
from experiments.database_lineage.src.tpch_plans import PLANS

from .canonical_otel import canonical_trace_sha256
from .core_to_otel_projection import project_core_to_otel
from .database_projection import project_core_to_database
from .database_to_otel_projection import project_database_to_otel
from .execution_capture import ProjectionCaptureContract
from .native_otel_capture import NativeOtelCapture, canonicalize_native_trace
from .projection_validator import assert_trace_equal, trace_diff


def _rss_bytes() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _type_signature(rows) -> list[dict[str, str]]:
    return [
        {key: type(value).__qualname__ for key, value in row.values.items()}
        for row in rows
    ]


def run_tpch_q6(database_path: Path, *, scale_factor: float = 0.01) -> dict:
    """Run SF0.01 Q6 with official answers, Core, and official OTel SDK."""

    connection = duckdb.connect(str(database_path), read_only=True)
    connection.execute("LOAD tpch")
    official = official_sql_and_answers(connection)
    sql = official["queries"][6]
    expected_answer = parse_official_answer(official["answers"][f"{scale_factor}:6"])
    tables = load_tables(connection, ("lineitem",))

    ordinary_start = time.perf_counter()
    ordinary = PLANS[6](tables, None)
    ordinary_seconds = time.perf_counter() - ordinary_start

    reference_start = time.perf_counter()
    duckdb_reference = execute_reference(connection, sql)
    reference_seconds = time.perf_counter() - reference_start

    run_id = f"otel-tpch-sf-{scale_factor}-q6"
    core = CoreAdapter(
        run_id=run_id,
        dependencies={
            "duckdb": duckdb.__version__,
            "opentelemetry": "api-1.44.0_sdk-1.44.0_semconv-0.65b0",
        },
    )
    native = NativeOtelCapture(run_id=run_id)
    contract = ProjectionCaptureContract(core=core, native=native)
    captured_start = time.perf_counter()
    captured = PLANS[6](tables, contract)
    captured_seconds = time.perf_counter() - captured_start

    snapshot_start = time.perf_counter()
    snapshot = core.validated_snapshot()
    snapshot_seconds = time.perf_counter() - snapshot_start
    validation_start = time.perf_counter()
    validation = validate_snapshot(snapshot, core.registry)
    validation_seconds = time.perf_counter() - validation_start

    native_start = time.perf_counter()
    native_trace = canonicalize_native_trace(native.finish(), expected_run_id=run_id)
    native_normalization_seconds = time.perf_counter() - native_start

    direct_start = time.perf_counter()
    direct_trace = project_core_to_otel(snapshot, validation)
    direct_seconds = time.perf_counter() - direct_start
    native_direct_diff = trace_diff(direct_trace, native_trace)
    assert_trace_equal(direct_trace, native_trace)
    native_span_count = len(native_trace["spans"])
    native_sha256 = canonical_trace_sha256(native_trace)
    native.clear_native_records()
    del native_trace
    gc.collect()

    database_start = time.perf_counter()
    database_projection = project_core_to_database(snapshot, validation)
    database_projection_seconds = time.perf_counter() - database_start
    hierarchical_start = time.perf_counter()
    hierarchical_trace = project_database_to_otel(database_projection)
    hierarchical_seconds = time.perf_counter() - hierarchical_start
    hierarchical_diff = trace_diff(direct_trace, hierarchical_trace)
    assert_trace_equal(
        direct_trace,
        hierarchical_trace,
        mismatch_reason="HIERARCHICAL_PROJECTION_MISMATCH",
    )

    ordinary_csv = csv_bytes(ordinary)
    captured_csv = csv_bytes(captured)
    ordinary_json = json_bytes(ordinary)
    captured_json = json_bytes(captured)
    forbidden = {
        "trace_id",
        "span_id",
        "core_id",
        "tuple_id",
        "lineage",
        "provenance",
        "token",
    }
    output_fields = {field for row in captured for field in row.values}
    reference_comparison = compare_rows(
        ordinary_rows(captured), duckdb_reference["text_rows"]
    )
    answer_comparison = compare_official_typed(
        [row.values for row in captured], expected_answer["text_rows"]
    )
    occurrence_count = len(snapshot.tables.generation_occurrences)
    span_count = len(direct_trace["spans"])
    link_count = sum(
        len(span["linked_semantic_keys"]) for span in direct_trace["spans"]
    )
    direct_sha256 = canonical_trace_sha256(direct_trace)
    hierarchical_sha256 = canonical_trace_sha256(hierarchical_trace)
    connection.close()
    return {
        "workload": "TPC-H-derived SF0.01 Q6",
        "scale_factor": scale_factor,
        "query_number": 6,
        "sql": sql,
        "ordinary_output_row_count": len(ordinary),
        "duckdb_exact": reference_comparison["exact"],
        "official_answer_exact": answer_comparison["exact_after_typed_parse"],
        "core_snapshot_id": snapshot.snapshot_id,
        "core_snapshot_validated": True,
        "core_occurrence_count": occurrence_count,
        "core_binding_count": len(snapshot.tables.generation_bindings),
        "validated_relation_evidence_count": len(validation.relation_evidence),
        "native_span_count": native_span_count,
        "direct_projected_span_count": span_count,
        "hierarchical_projected_span_count": len(hierarchical_trace["spans"]),
        "causal_link_count": link_count,
        "native_vs_direct": asdict(native_direct_diff),
        "direct_vs_hierarchical": asdict(hierarchical_diff),
        "trace_sha256": {
            "native": native_sha256,
            "direct": direct_sha256,
            "hierarchical": hierarchical_sha256,
        },
        "output_orthogonality": {
            "ordinary_vs_core_and_otel_csv_byte_identical": ordinary_csv
            == captured_csv,
            "ordinary_vs_core_and_otel_json_byte_identical": ordinary_json
            == captured_json,
            "values_and_order_equal": [row.values for row in ordinary]
            == [row.values for row in captured],
            "types_equal": _type_signature(ordinary) == _type_signature(captured),
            "forbidden_output_fields": sorted(output_fields & forbidden),
            "csv_sha256": hashlib.sha256(captured_csv).hexdigest(),
            "json_sha256": hashlib.sha256(captured_json).hexdigest(),
        },
        "performance_seconds": {
            "ordinary_execution": ordinary_seconds,
            "core_and_otel_execution": captured_seconds,
            "snapshot_build": snapshot_seconds,
            "snapshot_validation": validation_seconds,
            "native_normalization": native_normalization_seconds,
            "direct_projection": direct_seconds,
            "database_projection": database_projection_seconds,
            "hierarchical_projection": hierarchical_seconds,
            "duckdb_reference": reference_seconds,
        },
        "peak_process_rss_bytes": _rss_bytes(),
    }
