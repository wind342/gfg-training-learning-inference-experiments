from __future__ import annotations

from collections import defaultdict
from typing import Any

from generation_relation_core.snapshots import ValidatedSnapshot

from .canonical_graph import canonical_hash, content_id


def _core_occurrence_rows(
    snapshot_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in snapshot_inputs:
        snapshot = item["snapshot"]
        if not isinstance(snapshot, ValidatedSnapshot):
            raise TypeError("VALIDATED_SNAPSHOT_OBJECT_REQUIRED")
        run_id = item["execution_run_id"]
        aliases_by_core: dict[str, set[str]] = defaultdict(set)
        binding_occurrences = {
            row["generation_binding_id"]: row[
                "generation_occurrence_id"
            ]
            for row in snapshot.tables.generation_bindings
        }
        for binding_id, identity in item.get(
            "native_binding_identities", {}
        ).items():
            native = identity.get("native_occurrence_id")
            core = binding_occurrences.get(binding_id)
            if core and native:
                aliases_by_core[core].add(native)
        for occurrence in snapshot.tables.generation_occurrences:
            core_id = occurrence["generation_occurrence_id"]
            aliases = aliases_by_core.get(core_id, set())
            if len(aliases) > 1:
                raise ValueError("CORE_OCCURRENCE_NATIVE_ALIAS_AMBIGUOUS")
            concrete_id = next(iter(aliases), core_id)
            rows.append(
                {
                    "execution_run_id": run_id,
                    "concrete_occurrence_instance_id": concrete_id,
                    "generation_occurrence_id": core_id,
                    "occurrence_type": occurrence["occurrence_type"],
                    "occurrence_stage": occurrence["occurrence_stage"],
                    "stable_instance_key": occurrence[
                        "stable_instance_key"
                    ],
                    "occurrence_index": occurrence["occurrence_index"],
                    "transform_reference": occurrence[
                        "transform_reference"
                    ],
                    "occurrence_payload": occurrence[
                        "occurrence_payload"
                    ],
                    "generator_manifest_id": occurrence[
                        "generator_manifest_id"
                    ],
                    "evidence_refs": [],
                    "catalog_authority": "validated-core-snapshot",
                }
            )
    return rows


def build_core_occurrence_catalog(
    snapshot_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    run_ids = {row["execution_run_id"] for row in snapshot_inputs}
    if len(run_ids) != 1:
        raise ValueError("CATALOG_RUN_SCOPE_MISMATCH")
    rows = _core_occurrence_rows(snapshot_inputs)
    material = {
        "schema_version": "occurrence-endpoint-catalog-v2",
        "execution_run_id": next(iter(run_ids)),
        "occurrences": sorted(
            rows,
            key=lambda row: row["concrete_occurrence_instance_id"],
        ),
        "establishment_source": "validated_core_snapshot",
    }
    return {
        **material,
        "occurrence_catalog_id": content_id("gfoc2_", material),
    }


def merge_occurrence_catalogs(
    *catalogs: dict[str, Any],
) -> dict[str, Any]:
    if not catalogs:
        raise ValueError("OCCURRENCE_CATALOG_REQUIRED")
    run_ids = {row["execution_run_id"] for row in catalogs}
    if len(run_ids) != 1:
        raise ValueError("CATALOG_RUN_SCOPE_MISMATCH")
    by_id: dict[str, dict[str, Any]] = {}
    for catalog in catalogs:
        for occurrence in catalog["occurrences"]:
            occurrence_id = occurrence[
                "concrete_occurrence_instance_id"
            ]
            existing = by_id.get(occurrence_id)
            if existing is None:
                by_id[occurrence_id] = occurrence
            elif existing != occurrence:
                raise ValueError(
                    "OCCURRENCE_CATALOG_ENTRY_CONFLICT:" + occurrence_id
                )
    material = {
        "schema_version": "occurrence-endpoint-catalog-v2",
        "execution_run_id": next(iter(run_ids)),
        "occurrences": [by_id[key] for key in sorted(by_id)],
        "establishment_source": "merged_validated_catalogs",
        "input_catalog_ids": sorted(
            row["occurrence_catalog_id"] for row in catalogs
        ),
    }
    return {
        **material,
        "occurrence_catalog_id": content_id("gfoc2_", material),
    }


def validate_occurrence_catalog(
    catalog: dict[str, Any],
    *,
    required_occurrence_ids: set[str],
) -> dict[str, Any]:
    rows = catalog["occurrences"]
    ids = [
        row["concrete_occurrence_instance_id"] for row in rows
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_OCCURRENCE_CATALOG_ID")
    for row in rows:
        if row["execution_run_id"] != catalog["execution_run_id"]:
            raise ValueError("OCCURRENCE_CATALOG_RUN_SCOPE_MISMATCH")
        for field in (
            "occurrence_type",
            "occurrence_stage",
            "stable_instance_key",
            "catalog_authority",
        ):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(
                    "OCCURRENCE_CATALOG_FIELD_INVALID:" + field
                )
    missing = sorted(required_occurrence_ids - set(ids))
    if missing:
        raise ValueError(
            "REFERENCED_OCCURRENCE_MISSING_FROM_CATALOG:" + missing[0]
        )
    material = {
        "occurrence_catalog_id": catalog["occurrence_catalog_id"],
        "occurrence_count": len(rows),
        "required_occurrence_count": len(required_occurrence_ids),
        "missing_occurrence_count": 0,
        "status": "PASS",
    }
    return {**material, "validation_sha256": canonical_hash(material)}
