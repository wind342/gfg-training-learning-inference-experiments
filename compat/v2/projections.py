from __future__ import annotations

"""Legacy compatibility projections derived only from Core v3 bindings.

These views are outputs, never authoritative inputs. In particular, they must
not be joined to reconstruct ``GenerationBinding`` rows because that would
invent Cartesian-product relations that the generator never asserted.
"""

from collections import defaultdict


def derive_legacy_projections(
    sources: list[dict], occurrences: list[dict], bindings: list[dict],
    *, validate_schema: bool = True,
) -> tuple[list[dict], list[dict]]:
    # Deferred import keeps this compatibility module independently importable
    # while the Core package initializes snapshots.
    from generation_relation_core.canonical import finalize_entity

    source_by_id = {row["source_information_id"]: row for row in sources}
    occurrence_by_id = {row["generation_occurrence_id"]: row for row in occurrences}
    source_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    occurrence_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for binding in bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] != "support":
            continue
        support_id = outcome["support_id"]
        origin = binding["origin_reference"]
        if origin["kind"] == "registered_source":
            source_groups[(support_id, origin["source_information_id"])].append(binding["generation_binding_id"])
        occurrence_groups[(support_id, binding["generation_occurrence_id"])].append(binding["generation_binding_id"])

    source_rows = []
    for (support_id, source_id), binding_ids in source_groups.items():
        source = source_by_id[source_id]
        provenance_status = "verified"
        if isinstance(source["source_payload"], dict):
            provenance_status = source["source_payload"].get("legacy_provenance_status", "verified")
        source_rows.append(finalize_entity("LegacySourceBindingProjection", {
            "support_id": support_id,
            "source_information_id": source_id,
            "source_element_id": source["source_identity"],
            "source_parent_id": source["source_parent_id"],
            "source_granularity": source["source_granularity"],
            "provenance_status": provenance_status,
            "derived_from_generation_binding_ids": binding_ids,
            "schema_version": "3.0.0",
        }, validate_schema=validate_schema))

    occurrence_rows = []
    binding_by_id = {row["generation_binding_id"]: row for row in bindings}
    for (support_id, occurrence_id), binding_ids in occurrence_groups.items():
        occurrence = occurrence_by_id[occurrence_id]
        roles = sorted({binding_by_id[item]["relation_role"] for item in binding_ids})
        role = roles[0] if len(roles) == 1 else "multiple_relation_roles"
        occurrence_rows.append(finalize_entity("LegacyOccurrenceBindingProjection", {
            "support_id": support_id,
            "generation_occurrence_id": occurrence_id,
            "render_occurrence_id": occurrence["stable_instance_key"],
            "binding_role": role,
            "occurrence_index": occurrence["occurrence_index"],
            "derived_from_generation_binding_ids": binding_ids,
            "schema_version": "3.0.0",
        }, validate_schema=validate_schema))
    source_rows.sort(key=lambda row: row["source_binding_id"])
    occurrence_rows.sort(key=lambda row: row["occurrence_binding_id"])
    return source_rows, occurrence_rows


def projections_equal(left: list[dict], right: list[dict], id_field: str) -> bool:
    from generation_relation_core.canonical import canonical_bytes

    return canonical_bytes(sorted(left, key=lambda row: row[id_field])) == canonical_bytes(
        sorted(right, key=lambda row: row[id_field])
    )
