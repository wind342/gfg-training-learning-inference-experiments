"""Snapshot-only final-support selection and recursive GeneratedOrigin traversal."""

from __future__ import annotations

import hashlib

from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.entities import query_request
from generation_relation_core.predicate_registry import PredicateRegistry
from generation_relation_core.query_engine import QueryEngine
from generation_relation_core.snapshots import ValidatedSnapshot

from .contract import DOMAIN_SCOPE_ID, QUERY_RECTANGLE


def build_query(snapshot: ValidatedSnapshot) -> dict:
    visual = next(
        row
        for row in snapshot.tables.support_space_records
        if row["support_space_name"]
        == "signal_svg_spectrogram_css_rectangle"
    )
    profile = next(
        row
        for row in snapshot.tables.predicate_profiles
        if row["support_space_id"] == visual["support_space_id"]
    )
    return query_request(
        domain_scope_id=DOMAIN_SCOPE_ID,
        support_space_id=visual["support_space_id"],
        predicate_profile_id=profile["predicate_profile_id"],
        predicate="intersection",
        query_payload={"rectangle": QUERY_RECTANGLE},
    )


def _signature(row: dict) -> str:
    return "|".join(
        [
            *row["support_keys"],
            *row["roles"],
            *row["occurrence_keys"],
            row["raw_source_identity"],
        ]
    )


def answer_from_snapshot(
    snapshot: ValidatedSnapshot, registry: PredicateRegistry
) -> dict:
    query = build_query(snapshot)
    query_result = QueryEngine(snapshot, registry).execute(query).result
    tables = snapshot.tables
    supports = {
        row["support_id"]: row for row in tables.perceptual_support_records
    }
    generated = {
        row["generated_origin_id"]: row for row in tables.generated_origins
    }
    sources = {
        row["source_information_id"]: row
        for row in tables.source_information_records
    }
    occurrences = {
        row["generation_occurrence_id"]: row
        for row in tables.generation_occurrences
    }
    bindings_by_support: dict[str, list[dict]] = {}
    for binding in tables.generation_bindings:
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            bindings_by_support.setdefault(outcome["support_id"], []).append(
                binding
            )
    for rows in bindings_by_support.values():
        rows.sort(key=lambda row: row["generation_binding_id"])

    paths: list[dict] = []

    def descend(
        binding: dict,
        support_keys: list[str],
        roles: list[str],
        occurrence_keys: list[str],
        binding_ids: list[str],
        generated_ids: list[str],
        active_supports: set[str],
    ) -> None:
        role_path = [*roles, binding["relation_role"]]
        occurrence_path = [
            *occurrence_keys,
            occurrences[binding["generation_occurrence_id"]][
                "stable_instance_key"
            ],
        ]
        binding_path = [*binding_ids, binding["generation_binding_id"]]
        origin = binding["origin_reference"]
        if origin["kind"] == "registered_source":
            source = sources[origin["source_information_id"]]
            paths.append(
                {
                    "support_keys": support_keys,
                    "roles": role_path,
                    "occurrence_keys": occurrence_path,
                    "binding_ids": binding_path,
                    "generated_origin_ids": generated_ids,
                    "raw_source_identity": source["source_identity"],
                }
            )
            return
        generated_row = generated[origin["generated_origin_id"]]
        prior_support_id = generated_row["origin_payload"][
            "prior_support_id"
        ]
        if prior_support_id in active_supports:
            raise ValueError("GENERATED_ORIGIN_CYCLE")
        prior = supports.get(prior_support_id)
        if prior is None:
            raise ValueError("GENERATED_ORIGIN_PRIOR_SUPPORT_MISSING")
        producer_bindings = bindings_by_support.get(prior_support_id, [])
        if not producer_bindings:
            raise ValueError("GENERATED_ORIGIN_PRODUCER_BINDING_MISSING")
        prior_key = prior["support_payload"]["native_support_key"]
        for producer in producer_bindings:
            descend(
                producer,
                [*support_keys, prior_key],
                role_path,
                occurrence_path,
                binding_path,
                [*generated_ids, generated_row["generated_origin_id"]],
                {*active_supports, prior_support_id},
            )

    selected_support_ids = query_result["support_ids"]
    for support_id in selected_support_ids:
        support = supports[support_id]
        final_key = support["support_payload"]["native_support_key"]
        for binding in bindings_by_support.get(support_id, []):
            descend(
                binding,
                [final_key],
                [],
                [],
                [],
                [],
                {support_id},
            )
    signatures = sorted(_signature(row) for row in paths)
    return {
        "query_id": query["query_id"],
        "query_status": query_result["query_status"],
        "query_rectangle": QUERY_RECTANGLE,
        "selected_final_support_keys": sorted(
            supports[item]["support_payload"]["native_support_key"]
            for item in selected_support_ids
        ),
        "raw_source_identities": sorted(
            {row["raw_source_identity"] for row in paths}
        ),
        "path_count": len(paths),
        "path_signature_multiset_sha256": hashlib.sha256(
            canonical_bytes(signatures)
        ).hexdigest(),
        "traversed_binding_count": len(
            {
                binding_id
                for row in paths
                for binding_id in row["binding_ids"]
            }
        ),
        "traversed_generated_origin_count": len(
            {
                origin_id
                for row in paths
                for origin_id in row["generated_origin_ids"]
            }
        ),
        "path_signatures": signatures,
    }
