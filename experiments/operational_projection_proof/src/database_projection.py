"""Read-only database which-lineage projection from a validated Core snapshot.

This module deliberately has no Oracle, native database-result, or filesystem
dependency. All indexes are function-local and rebuildable from the snapshot.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.snapshots import SnapshotValidation, ValidatedSnapshot

from .errors import ProjectionProofError
from .projection_profile import ProjectionProfile
from .projection_result import empty_result, normalize_result


def _require_validation(
    snapshot: ValidatedSnapshot, validation: SnapshotValidation
) -> None:
    binding_ids = {
        row["generation_binding_id"] for row in snapshot.tables.generation_bindings
    }
    if (
        validation.snapshot_id != snapshot.snapshot_id
        or set(validation.relation_evidence) != binding_ids
    ):
        raise ProjectionProofError("SNAPSHOT_VALIDATION_REQUIRED", snapshot.snapshot_id)


def _semantic_rows_with_ordinals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[bytes, int] = defaultdict(int)
    result = []
    for row in sorted(rows, key=canonical_bytes):
        base = canonical_bytes(row)
        result.append({**row, "relation_ordinal": counts[base]})
        counts[base] += 1
    return result


def _paths_to_output(
    *,
    output_tuple_id: str,
    reverse_edges: dict[str, list[dict[str, Any]]],
    base_sources: set[str],
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    def walk(
        current: str, visited: frozenset[str]
    ) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
        if current in visited:
            raise ProjectionProofError("HIERARCHY_CYCLE", current)
        if current in base_sources:
            return [(current, (current,), ())]
        result: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for edge in sorted(reverse_edges.get(current, []), key=canonical_bytes):
            for source, nodes, roles in walk(
                edge["input_tuple_id"], visited | {current}
            ):
                result.append(
                    (source, (*nodes, current), (*roles, edge["relation_role"]))
                )
        return result

    return sorted(
        walk(output_tuple_id, frozenset()),
        key=lambda item: canonical_bytes([item[0], list(item[1]), list(item[2])]),
    )


def project_database_snapshot(
    *,
    snapshot: ValidatedSnapshot,
    validation: SnapshotValidation,
    profile: ProjectionProfile,
    workload_id: str,
    final_stages: Iterable[str],
    include_dispositions: bool,
    duplicate_cases: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    _require_validation(snapshot, validation)
    tables = snapshot.tables
    sources = {
        row["source_information_id"]: row for row in tables.source_information_records
    }
    generated = {row["generated_origin_id"]: row for row in tables.generated_origins}
    supports = {row["support_id"]: row for row in tables.perceptual_support_records}
    dispositions = {row["disposition_id"]: row for row in tables.explicit_dispositions}

    direct_without_ordinals: list[dict[str, Any]] = []
    disposition_rows: list[dict[str, Any]] = []
    for binding in sorted(
        tables.generation_bindings, key=lambda row: row["generation_binding_id"]
    ):
        origin = binding["origin_reference"]
        if origin["kind"] == "registered_source":
            input_tuple_id = sources[origin["source_information_id"]]["source_identity"]
        else:
            payload = generated[origin["generated_origin_id"]]["origin_payload"]
            input_tuple_id = payload["tuple_identity"]
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            output_tuple_id = supports[outcome["support_id"]]["support_payload"][
                "tuple_identity"
            ]
        else:
            if not include_dispositions:
                continue
            disposition = dispositions[outcome["disposition_id"]]
            output_tuple_id = disposition["disposition_payload"]["tuple_identity"]
            disposition_rows.append(
                {
                    "workload_id": workload_id,
                    "input_tuple_id": input_tuple_id,
                    "output_tuple_id": output_tuple_id,
                    "core_disposition_category": disposition[
                        "core_disposition_category"
                    ],
                    "reason_code": disposition["domain_reason_code"],
                }
            )
        direct_without_ordinals.append(
            {
                "workload_id": workload_id,
                "input_tuple_id": input_tuple_id,
                "output_tuple_id": output_tuple_id,
                "outcome_kind": outcome["kind"],
                "relation_role": binding["relation_role"],
            }
        )
    direct_rows = _semantic_rows_with_ordinals(direct_without_ordinals)
    support_edges = [row for row in direct_rows if row["outcome_kind"] == "support"]
    reverse_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in support_edges:
        reverse_edges[row["output_tuple_id"]].append(row)
    base_source_ids = {row["source_identity"] for row in sources.values()}
    target_stages = set(final_stages)
    final_output_ids = sorted(
        row["support_payload"]["tuple_identity"]
        for row in supports.values()
        if row["support_payload"].get("operator_stage") in target_stages
    )

    path_rows: list[dict[str, Any]] = []
    backward_rows: list[dict[str, Any]] = []
    paths_by_source: dict[str, list[str]] = defaultdict(list)
    for output_tuple_id in final_output_ids:
        paths = _paths_to_output(
            output_tuple_id=output_tuple_id,
            reverse_edges=reverse_edges,
            base_sources=base_source_ids,
        )
        ordinals: dict[str, int] = defaultdict(int)
        for source_id, tuple_path, role_path in paths:
            path_rows.append(
                {
                    "workload_id": workload_id,
                    "output_tuple_id": output_tuple_id,
                    "source_tuple_id": source_id,
                    "path_ordinal": ordinals[source_id],
                    "tuple_path": list(tuple_path),
                    "relation_roles": list(role_path),
                    "path_length": len(role_path),
                }
            )
            ordinals[source_id] += 1
            paths_by_source[source_id].append(output_tuple_id)
        backward_rows.append(
            {
                "workload_id": workload_id,
                "output_tuple_id": output_tuple_id,
                "source_tuple_ids": sorted(
                    {source_id for source_id, _nodes, _roles in paths}
                ),
                "derivation_path_count": len(paths),
            }
        )
    forward_rows = (
        [
            {
                "workload_id": workload_id,
                "source_tuple_id": source_id,
                "output_tuple_ids": sorted(set(paths_by_source.get(source_id, []))),
                "derivation_path_count": len(paths_by_source.get(source_id, [])),
            }
            for source_id in sorted(base_source_ids)
        ]
        if final_output_ids
        else []
    )

    source_by_identity = {row["source_identity"]: row for row in sources.values()}
    duplicate_rows = []
    for case in duplicate_cases:
        identities = list(case["source_tuple_ids"])
        payloads = [
            source_by_identity[item]["source_payload"].get("field_values")
            for item in identities
        ]
        matching = [row for row in support_edges if row["input_tuple_id"] in identities]
        duplicate_rows.append(
            {
                "workload_id": workload_id,
                "case_id": case["case_id"],
                "source_tuple_ids": identities,
                "distinct_identity_count": len(set(identities)),
                "source_payload_equal": len(
                    {canonical_bytes(payload) for payload in payloads}
                )
                == 1,
                "output_tuple_ids": sorted(
                    {row["output_tuple_id"] for row in matching}
                ),
                "direct_relation_count": len(matching),
            }
        )
    result = empty_result(profile)
    result["records"].update(
        {
            "direct_relations": direct_rows,
            "backward_lineage": backward_rows,
            "forward_lineage": forward_rows,
            "derivation_paths": path_rows,
            "explicit_dispositions": sorted(disposition_rows, key=canonical_bytes),
            "multiplicity": [
                {
                    "workload_id": workload_id,
                    "total_relation_count": len(direct_rows),
                    "support_relation_count": len(support_edges),
                    "disposition_relation_count": len(disposition_rows),
                    "derivation_path_count": len(path_rows),
                    "final_output_count": len(final_output_ids),
                }
            ],
            "duplicate_identities": duplicate_rows,
        }
    )
    return normalize_result(result, profile)
