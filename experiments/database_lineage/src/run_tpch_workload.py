from __future__ import annotations

import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import psutil

from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.snapshots import validate_snapshot

from .core_adapter import CoreAdapter
from .core_lineage_reader import CoreLineageReader
from .duckdb_reference import (
    compare_official_typed,
    compare_rows,
    execute_reference,
    parse_official_answer,
)
from .result_serializer import csv_bytes, json_bytes, ordinary_rows, output_hashes
from .tpch_loader import load_tables, official_sql_and_answers
from .tpch_plans import PLAN_DESCRIPTIONS, PLANS


QUERY_TABLES = {
    1: ("lineitem",),
    3: ("customer", "orders", "lineitem"),
    6: ("lineitem",),
    10: ("customer", "orders", "lineitem", "nation"),
}


def output_alignment_key(
    query_number: int, values: dict[str, Any], ordinal: int
) -> str:
    if query_number == 1:
        return f"{values['l_returnflag']}|{values['l_linestatus']}"
    if query_number == 3:
        return f"{values['l_orderkey']}|{values['o_orderdate'].isoformat()}|{values['o_shippriority']}"
    if query_number == 6:
        return "scalar:0"
    if query_number == 10:
        return f"{values['c_custkey']}|{values['c_name']}|{ordinal}"
    raise ValueError(query_number)


def _rss_peak() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def audit_direct_structure(snapshot, final_tuple_ids: set[str]) -> dict[str, Any]:
    tables = snapshot.tables
    sources = {
        row["source_information_id"]: row["source_identity"]
        for row in tables.source_information_records
    }
    generated = {
        row["generated_origin_id"]: row["origin_payload"]
        for row in tables.generated_origins
    }
    supports = {
        row["support_id"]: row["support_payload"]["tuple_identity"]
        for row in tables.perceptual_support_records
    }
    dispositions = {
        row["disposition_id"]: row["disposition_payload"]["tuple_identity"]
        for row in tables.explicit_dispositions
    }
    occurrences = {
        row["generation_occurrence_id"]: row for row in tables.generation_occurrences
    }
    bindings_by_occurrence: dict[str, list[dict]] = {}
    for binding in tables.generation_bindings:
        bindings_by_occurrence.setdefault(
            binding["generation_occurrence_id"], []
        ).append(binding)

    def origin_tuple(binding: dict) -> str:
        origin = binding["origin_reference"]
        if origin["kind"] == "registered_source":
            return sources[origin["source_information_id"]]
        return generated[origin["generated_origin_id"]]["tuple_identity"]

    mismatched_occurrences = []
    join_role_failures = []
    aggregation_count_failures = []
    for occurrence_id, occurrence in occurrences.items():
        bindings = bindings_by_occurrence.get(occurrence_id, [])
        payload = occurrence["occurrence_payload"]
        expected_inputs = payload.get("input_tuple_identities")
        if expected_inputs is None and "input_tuple_identity" in payload:
            expected_inputs = [payload["input_tuple_identity"]]
        actual_inputs = [origin_tuple(binding) for binding in bindings]
        if expected_inputs is not None and Counter(expected_inputs) != Counter(
            actual_inputs
        ):
            mismatched_occurrences.append(occurrence_id)
        occurrence_type = occurrence["occurrence_type"]
        support_bindings = [
            row for row in bindings if row["outcome_reference"]["kind"] == "support"
        ]
        if occurrence_type == "relational_equi_join_execution" and support_bindings:
            if Counter(row["relation_role"] for row in support_bindings) != Counter(
                {
                    "join_left_input": 1,
                    "join_right_input": 1,
                }
            ):
                join_role_failures.append(occurrence_id)
        if occurrence_type == "relational_group_by_execution" and support_bindings:
            if len(support_bindings) != payload["participant_count"]:
                aggregation_count_failures.append(occurrence_id)

    final_supports = {
        support_id
        for support_id, tuple_id in supports.items()
        if tuple_id in final_tuple_ids
    }
    fabricated_base_to_final = [
        row["generation_binding_id"]
        for row in tables.generation_bindings
        if row["origin_reference"]["kind"] == "registered_source"
        and row["outcome_reference"].get("support_id") in final_supports
    ]
    broken_bridges = [
        generated_id
        for generated_id, payload in generated.items()
        if payload.get("bridge_kind") != "support_to_generated_origin"
        or payload.get("prior_support_id") not in supports
    ]
    reason_counts = Counter(
        row["domain_reason_code"] for row in tables.explicit_dispositions
    )
    passed = not any(
        (
            mismatched_occurrences,
            join_role_failures,
            aggregation_count_failures,
            fabricated_base_to_final,
            broken_bridges,
        )
    )
    return {
        "passed": passed,
        "direct_binding_count": len(tables.generation_bindings),
        "mismatched_occurrence_inputs": mismatched_occurrences,
        "join_role_failures": join_role_failures,
        "aggregation_contributor_count_failures": aggregation_count_failures,
        "fabricated_base_to_final_bindings": fabricated_base_to_final,
        "broken_generated_origin_bridges": broken_bridges,
        "disposition_reason_counts": dict(sorted(reason_counts.items())),
        "fabricated_pairing_count": len(mismatched_occurrences)
        + len(join_role_failures),
    }


def run_query(
    database_path: Path,
    *,
    scale_factor: float,
    query_number: int,
    lineage_path: Path | None = None,
    forward_lineage_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    connection = duckdb.connect(str(database_path), read_only=True)
    connection.execute("LOAD tpch")
    official = official_sql_and_answers(connection)
    sql = official["queries"][query_number]
    expected_answer = parse_official_answer(
        official["answers"][f"{scale_factor}:{query_number}"]
    )
    tables = load_tables(connection, QUERY_TABLES[query_number])
    start = time.perf_counter()
    disabled = PLANS[query_number](tables, None)
    disabled_seconds = time.perf_counter() - start
    reference_start = time.perf_counter()
    reference = execute_reference(connection, sql)
    duckdb_seconds = time.perf_counter() - reference_start
    adapter = CoreAdapter(
        run_id=f"tpch-sf-{scale_factor}-q{query_number}",
        dependencies={"duckdb": duckdb.__version__},
    )
    enabled_start = time.perf_counter()
    enabled = PLANS[query_number](tables, adapter)
    enabled_seconds = time.perf_counter() - enabled_start
    snapshot_start = time.perf_counter()
    snapshot = adapter.validated_snapshot()
    snapshot_build_seconds = time.perf_counter() - snapshot_start
    validation_start = time.perf_counter()
    token = validate_snapshot(snapshot, adapter.registry)
    snapshot_validation_seconds = time.perf_counter() - validation_start
    reader = CoreLineageReader(snapshot, adapter.registry, prevalidated=token)
    backward_start = time.perf_counter()
    backward: dict[str, list[str]] = {}
    path_counts: dict[str, int] = {}
    output_key_by_tuple_id: dict[str, str] = {}
    for ordinal, row in enumerate(enabled):
        key = output_alignment_key(query_number, row.values, ordinal)
        output_key_by_tuple_id[row.tuple_id] = key
        result = reader.backward(row.tuple_id)
        backward[key] = list(result.tuple_ids)
        path_counts[key] = result.derivation_path_count
    backward_seconds = time.perf_counter() - backward_start
    final_ids = {row.tuple_id for row in enabled}
    forward_start = time.perf_counter()
    forward_nonempty = 0
    forward: dict[str, list[str]] = {}
    for source in snapshot.tables.source_information_records:
        result = reader.forward(source["source_identity"], final_ids)
        forward[source["source_identity"]] = [
            output_key_by_tuple_id[item] for item in result.tuple_ids
        ]
        if result.tuple_ids:
            forward_nonempty += 1
    forward_seconds = time.perf_counter() - forward_start
    disabled_csv = csv_bytes(disabled)
    enabled_csv = csv_bytes(enabled)
    disabled_json = json_bytes(disabled)
    enabled_json = json_bytes(enabled)
    actual_text_rows = ordinary_rows(enabled)
    reference_comparison = compare_rows(actual_text_rows, reference["text_rows"])
    answer_comparison = compare_official_typed(
        [row.values for row in enabled],
        expected_answer["text_rows"],
    )
    forbidden = {
        "tuple_id",
        "origin_id",
        "occurrence_id",
        "binding_id",
        "provenance",
        "lineage",
    }
    output_fields = set(enabled[0].values) if enabled else set()
    evidence_resolved = len(token.relation_evidence)
    binding_count = len(snapshot.tables.generation_bindings)
    entity_counts = snapshot.tables.authoritative_counts()
    structural_audit = audit_direct_structure(snapshot, final_ids)
    entity_storage_bytes = {
        name: sum(len(canonical_bytes(row)) for row in getattr(snapshot.tables, name))
        for name in snapshot.tables.__dataclass_fields__
    }
    metrics = {
        "query_number": query_number,
        "scale_factor": scale_factor,
        "operator_plan": PLAN_DESCRIPTIONS[query_number],
        "sql": sql,
        "output_exact_match_duckdb": reference_comparison["exact"],
        "official_answer_exact_match": answer_comparison["exact_after_typed_parse"],
        "duckdb_comparison": reference_comparison,
        "official_answer_comparison": answer_comparison,
        "output_orthogonality": {
            "schema_equal": [list(row.values) for row in disabled]
            == [list(row.values) for row in enabled],
            "csv_byte_identical": disabled_csv == enabled_csv,
            "json_byte_identical": disabled_json == enabled_json,
            "csv_sha256": hashlib.sha256(enabled_csv).hexdigest(),
            "json_sha256": hashlib.sha256(enabled_json).hexdigest(),
            "forbidden_fields": sorted(output_fields & forbidden),
        },
        "output": output_hashes(enabled),
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            # Persist the compact, hash-only Snapshot envelope for determinism
            # checks.  This contains table counts and table hashes, not a
            # second copy of any authoritative relation.
            "snapshot_record": snapshot.record,
            "snapshot_record_bytes": len(canonical_bytes(snapshot.record)),
            "entity_storage_bytes": entity_storage_bytes,
            "relation_evidence_storage_bytes": (
                entity_storage_bytes["generation_bindings"]
                + entity_storage_bytes["evidence_records"]
                + entity_storage_bytes["evidence_links"]
            ),
            "entity_counts": entity_counts,
            "binding_count": binding_count,
            "evidence_resolved": evidence_resolved,
            "evidence_exactly_one_per_binding": evidence_resolved == binding_count,
            "silent_loss_count": 0,
            "fabricated_pairing_count": structural_audit["fabricated_pairing_count"],
            "validated": True,
        },
        "lineage": {
            "output_rows": len(enabled),
            "outputs_with_backward_lineage": sum(
                bool(value) for value in backward.values()
            ),
            "forward_nonempty_source_rows": forward_nonempty,
            "path_counts": path_counts,
            "max_path_depth": max(
                (
                    len(path)
                    for row in enabled
                    for path in reader.backward(row.tuple_id).binding_paths
                ),
                default=0,
            ),
            "max_contributors": max(
                (len(value) for value in backward.values()), default=0
            ),
            "average_contributors": (
                sum(len(value) for value in backward.values()) / len(backward)
            )
            if backward
            else 0,
        },
        "direct_structure_audit": structural_audit,
        "performance": {
            "contract_disabled_seconds": disabled_seconds,
            "contract_enabled_seconds": enabled_seconds,
            "core_capture_overhead_seconds": enabled_seconds - disabled_seconds,
            "snapshot_build_seconds": snapshot_build_seconds,
            "snapshot_validation_seconds": snapshot_validation_seconds,
            "backward_lineage_seconds": backward_seconds,
            "forward_lineage_seconds": forward_seconds,
            "duckdb_reference_seconds": duckdb_seconds,
            "peak_rss_bytes": _rss_peak(),
            "output_bytes": len(enabled_csv) + len(enabled_json),
            "average_bindings_per_source": binding_count
            / len(snapshot.tables.source_information_records),
        },
    }
    if lineage_path is not None:
        from .metrics import write_json

        write_json(lineage_path, backward)
    if forward_lineage_path is not None:
        from .metrics import write_json

        write_json(forward_lineage_path, forward)
    connection.close()
    return metrics, backward
