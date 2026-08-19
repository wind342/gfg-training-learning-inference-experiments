from __future__ import annotations

from dataclasses import dataclass

from .canonical import canonical_bytes, finalize_entity, verify_entity
from .errors import CoreV3Error
from .predicate_registry import PredicateRegistry
from .snapshots import SnapshotValidation, ValidatedSnapshot, implementation_hashes, validate_snapshot


def precise_relation(binding: dict) -> dict:
    return {
        "generation_binding_id": binding["generation_binding_id"],
        "origin": binding["origin_reference"],
        "generation_occurrence_id": binding["generation_occurrence_id"],
        "outcome": binding["outcome_reference"],
        "relation_role": binding["relation_role"],
        "evidence_ids": binding["evidence_ids"],
    }


def _canonical_distinct_union(values: list[object]) -> list[object]:
    """Construct a deterministic union without weakening duplicate validation."""
    by_canonical_bytes = {canonical_bytes(value): value for value in values}
    return [by_canonical_bytes[key] for key in sorted(by_canonical_bytes)]


@dataclass(frozen=True)
class QueryExecution:
    result: dict
    candidate_count: int


class QueryEngine:
    """Audit-per-query dispatch from authoritative GenerationBinding rows."""

    def __init__(self, snapshot: ValidatedSnapshot, predicate_registry: PredicateRegistry) -> None:
        self.snapshot = snapshot
        self.predicate_registry = predicate_registry
        self._implementation_hashes = implementation_hashes()
        validate_snapshot(snapshot, predicate_registry, expected_implementation_hashes=self._implementation_hashes)

    def execute(self, request: dict) -> QueryExecution:
        token = validate_snapshot(
            self.snapshot,
            self.predicate_registry,
            expected_implementation_hashes=self._implementation_hashes,
        )
        verify_entity("QueryRequest", request)
        return self._execute_prevalidated(request, token)

    def execute_controlled(self, request: dict) -> QueryExecution:
        try:
            return self.execute(request)
        except CoreV3Error as exc:
            query_id = request.get("query_id")
            if not isinstance(query_id, str) or not query_id.startswith("qr3_"):
                raise
            result = finalize_entity("QueryResult", {
                "snapshot_id": self.snapshot.snapshot_id,
                "query_id": query_id,
                "query_status": "controlled_error",
                "reason_code": exc.reason_code,
                "hits": [],
                "support_ids": [],
                "generation_binding_ids": [],
                "source_information_ids": [],
                "parent_ids": [],
                "occurrence_ids": [],
                "status_values": [],
                "reason_codes": [exc.reason_code],
                "evidence_ids": [],
                "schema_version": "3.0.0",
            })
            return QueryExecution(result, 0)

    def _execute_prevalidated(self, request: dict, token: SnapshotValidation) -> QueryExecution:
        """Private batch path; callers cannot obtain a token without full snapshot validation."""
        if token.snapshot_id != self.snapshot.snapshot_id:
            raise CoreV3Error("SNAPSHOT_HASH_MISMATCH", "PREVALIDATION_TOKEN")
        tables = self.snapshot.tables
        spaces = {row["support_space_id"]: row for row in tables.support_space_records}
        profiles = {row["predicate_profile_id"]: row for row in tables.predicate_profiles}
        space = spaces.get(request["support_space_id"])
        if space is None:
            raise CoreV3Error("PREDICATE_PROFILE_UNKNOWN", request["support_space_id"])
        profile = profiles.get(request["predicate_profile_id"])
        if profile is None:
            raise CoreV3Error("PREDICATE_PROFILE_UNKNOWN", request["predicate_profile_id"])
        if profile["support_space_id"] != space["support_space_id"]:
            raise CoreV3Error("SUPPORT_PAYLOAD_PROFILE_MISMATCH", request["predicate_profile_id"])
        self.predicate_registry.validate_query(
            request["predicate_profile_id"], request["predicate"], request["query_payload"],
        )
        sources = {row["source_information_id"]: row for row in tables.source_information_records}
        supports = [
            row for row in tables.perceptual_support_records
            if row["domain_scope_id"] == request["domain_scope_id"]
            and row["support_space_id"] == request["support_space_id"]
            and row["predicate_profile_id"] == request["predicate_profile_id"]
        ]
        bindings_by_support: dict[str, list[dict]] = {}
        for binding in tables.generation_bindings:
            outcome = binding["outcome_reference"]
            if outcome["kind"] == "support":
                bindings_by_support.setdefault(outcome["support_id"], []).append(binding)
        hits = []
        for support in sorted(supports, key=lambda row: row["support_id"]):
            if not self.predicate_registry.evaluate(
                request["predicate_profile_id"], support, request["query_payload"], request["predicate"],
            ):
                continue
            bindings = sorted(
                bindings_by_support.get(support["support_id"], []),
                key=lambda row: row["generation_binding_id"],
            )
            source_ids = _canonical_distinct_union([
                row["origin_reference"]["source_information_id"] for row in bindings
                if row["origin_reference"]["kind"] == "registered_source"
            ])
            parent_ids = _canonical_distinct_union([
                sources[item]["source_parent_id"] for item in source_ids
                if sources[item]["source_parent_id"] is not None
            ])
            hits.append(finalize_entity("QueryHit", {
                "support_id": support["support_id"],
                "generation_relations": [precise_relation(row) for row in bindings],
                "generation_binding_ids": _canonical_distinct_union([
                    row["generation_binding_id"] for row in bindings
                ]),
                "source_information_ids": source_ids,
                "parent_ids": parent_ids,
                "occurrence_ids": _canonical_distinct_union([
                    row["generation_occurrence_id"] for row in bindings
                ]),
                "status_values": _canonical_distinct_union([support["support_status"]]),
                "reason_codes": [],
                "evidence_ids": _canonical_distinct_union([
                    item for row in bindings for item in row["evidence_ids"]
                ]),
                "schema_version": "3.0.0",
            }))
        support_ids = _canonical_distinct_union([row["support_id"] for row in hits])
        binding_ids = _canonical_distinct_union([
            item for row in hits for item in row["generation_binding_ids"]
        ])
        source_ids = _canonical_distinct_union([
            item for row in hits for item in row["source_information_ids"]
        ])
        parent_ids = _canonical_distinct_union([item for row in hits for item in row["parent_ids"]])
        occurrence_ids = _canonical_distinct_union([
            item for row in hits for item in row["occurrence_ids"]
        ])
        statuses = _canonical_distinct_union([item for row in hits for item in row["status_values"]])
        reasons = _canonical_distinct_union([item for row in hits for item in row["reason_codes"]])
        evidence_ids = _canonical_distinct_union([item for row in hits for item in row["evidence_ids"]])
        status = "valid_nonempty" if hits else "valid_empty"
        reason = "MATCHES_FOUND" if hits else "NO_PREDICATE_MATCH"
        result = finalize_entity("QueryResult", {
            "snapshot_id": self.snapshot.snapshot_id,
            "query_id": request["query_id"],
            "query_status": status,
            "reason_code": reason,
            "hits": hits,
            "support_ids": support_ids,
            "generation_binding_ids": binding_ids,
            "source_information_ids": source_ids,
            "parent_ids": parent_ids,
            "occurrence_ids": occurrence_ids,
            "status_values": statuses,
            "reason_codes": reasons,
            "evidence_ids": evidence_ids,
            "schema_version": "3.0.0",
        })
        return QueryExecution(result, len(supports))

    def disposition_relations(self, domain_scope_id: str, category: str | None = None) -> list[dict]:
        validate_snapshot(
            self.snapshot,
            self.predicate_registry,
            expected_implementation_hashes=self._implementation_hashes,
        )
        dispositions = {
            row["disposition_id"]: row for row in self.snapshot.tables.explicit_dispositions
            if row["domain_scope_id"] == domain_scope_id
            and (category is None or row["core_disposition_category"] == category)
        }
        return [
            {
                **precise_relation(binding),
                "status": dispositions[binding["outcome_reference"]["disposition_id"]]["core_disposition_category"],
                "reason_code": dispositions[binding["outcome_reference"]["disposition_id"]]["domain_reason_code"],
            }
            for binding in sorted(self.snapshot.tables.generation_bindings, key=lambda row: row["generation_binding_id"])
            if binding["outcome_reference"]["kind"] == "disposition"
            and binding["outcome_reference"]["disposition_id"] in dispositions
        ]


def legacy_projection_for_support(snapshot: ValidatedSnapshot, support_id: str) -> dict:
    """Compatibility view only. It never reconstructs GenerationBinding rows."""
    source_rows = [row for row in snapshot.tables.legacy_source_binding_projections if row["support_id"] == support_id]
    occurrence_rows = [row for row in snapshot.tables.legacy_occurrence_binding_projections if row["support_id"] == support_id]
    support = next((row for row in snapshot.tables.perceptual_support_records if row["support_id"] == support_id), None)
    if support is None:
        raise CoreV3Error("EXTERNAL_KEY_MISSING", support_id)
    binding_ids = {
        item for row in [*source_rows, *occurrence_rows]
        for item in row["derived_from_generation_binding_ids"]
    }
    migration = next((
        row for row in snapshot.tables.migration_records
        if binding_ids and binding_ids <= set(row["generated_binding_ids"])
    ), None)
    return {
        "support_id": support_id,
        "sources": sorted({row["source_element_id"] for row in source_rows}),
        "parents": sorted({row["source_parent_id"] for row in source_rows if row["source_parent_id"] is not None}),
        "occurrences": sorted({row["render_occurrence_id"] for row in occurrence_rows}),
        "statuses": [support["support_status"]],
        "reasons": [],
        "evidence": (
            migration["legacy_evidence_ids"] if migration is not None and migration["legacy_evidence_ids"] else
            sorted({
                evidence_id
                for binding_id in binding_ids
                for binding in snapshot.tables.generation_bindings
                if binding["generation_binding_id"] == binding_id
                for evidence_id in binding["evidence_ids"]
            })
        ),
    }
