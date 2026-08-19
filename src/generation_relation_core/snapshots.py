from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .canonical import canonical_bytes, finalize_entity, payload_sha256, table_hash, verify_entity
from .errors import CoreV3Error
from .implementation_identity import build_implementation_content_identity
from .predicate_registry import PredicateRegistry
from compat.v2.projections import derive_legacy_projections, projections_equal
from .relation_evidence import RelationEvidenceResolver, ResolvedRelationEvidence
from .schema_registry import protocol


AUTHORITATIVE_TABLE_SPECS: dict[str, tuple[str, str]] = {
    "source_information_records": ("SourceInformationRecord", "source_information_id"),
    "generation_occurrences": ("GenerationOccurrence", "generation_occurrence_id"),
    "generated_origins": ("GeneratedOrigin", "generated_origin_id"),
    "perceptual_support_records": ("PerceptualSupportRecord", "support_id"),
    "explicit_dispositions": ("ExplicitDisposition", "disposition_id"),
    "generation_bindings": ("GenerationBinding", "generation_binding_id"),
    "hierarchy_records": ("HierarchyRecord", "hierarchy_record_id"),
    "support_space_records": ("SupportSpaceRecord", "support_space_id"),
    "predicate_profiles": ("PredicateProfile", "predicate_profile_id"),
    "evidence_records": ("EvidenceRecord", "evidence_id"),
    "evidence_links": ("EvidenceLink", "evidence_link_id"),
    "generator_manifests": ("GeneratorManifest", "generator_manifest_id"),
    "generator_operation_results": ("GeneratorOperationResult", "operation_result_id"),
    "environment_records": ("EnvironmentRecord", "environment_record_id"),
    "migration_records": ("MigrationRecord", "migration_record_id"),
}

DERIVED_TABLE_SPECS: dict[str, tuple[str, str]] = {
    "legacy_source_binding_projections": ("LegacySourceBindingProjection", "source_binding_id"),
    "legacy_occurrence_binding_projections": ("LegacyOccurrenceBindingProjection", "occurrence_binding_id"),
}

ALL_TABLE_SPECS = {**AUTHORITATIVE_TABLE_SPECS, **DERIVED_TABLE_SPECS}


@dataclass
class CoreV3Tables:
    source_information_records: list[dict] = field(default_factory=list)
    generation_occurrences: list[dict] = field(default_factory=list)
    generated_origins: list[dict] = field(default_factory=list)
    perceptual_support_records: list[dict] = field(default_factory=list)
    explicit_dispositions: list[dict] = field(default_factory=list)
    generation_bindings: list[dict] = field(default_factory=list)
    hierarchy_records: list[dict] = field(default_factory=list)
    support_space_records: list[dict] = field(default_factory=list)
    predicate_profiles: list[dict] = field(default_factory=list)
    evidence_records: list[dict] = field(default_factory=list)
    evidence_links: list[dict] = field(default_factory=list)
    generator_manifests: list[dict] = field(default_factory=list)
    generator_operation_results: list[dict] = field(default_factory=list)
    environment_records: list[dict] = field(default_factory=list)
    migration_records: list[dict] = field(default_factory=list)
    legacy_source_binding_projections: list[dict] = field(default_factory=list)
    legacy_occurrence_binding_projections: list[dict] = field(default_factory=list)

    def authoritative_counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in AUTHORITATIVE_TABLE_SPECS}

    def derived_counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in DERIVED_TABLE_SPECS}

    def all_entity_ids(self) -> set[str]:
        result: set[str] = set()
        for name, (_entity_type, id_field) in ALL_TABLE_SPECS.items():
            result.update(row[id_field] for row in getattr(self, name))
        return result


@dataclass
class ValidatedSnapshot:
    record: dict[str, Any]
    tables: CoreV3Tables

    @property
    def snapshot_id(self) -> str:
        return self.record["snapshot_id"]


@dataclass(frozen=True)
class SnapshotValidation:
    snapshot_id: str
    relation_evidence: dict[str, ResolvedRelationEvidence]


def implementation_hashes() -> dict[str, str]:
    return build_implementation_content_identity()["implementation_hashes"]


def protocol_sha256() -> str:
    return payload_sha256(protocol())


def _table_hashes(tables: CoreV3Tables, specs: dict[str, tuple[str, str]]) -> dict[str, str]:
    return {
        name: table_hash(getattr(tables, name), entity_type, verify_rows=False)
        for name, (entity_type, _id_field) in specs.items()
    }


def _verify_payload_digests(tables: CoreV3Tables) -> None:
    checks = [
        (tables.source_information_records, "source_payload", "source_payload_sha256"),
        (tables.generation_occurrences, "occurrence_payload", "occurrence_payload_sha256"),
        (tables.generated_origins, "origin_payload", "origin_payload_sha256"),
        (tables.perceptual_support_records, "support_payload", "support_payload_sha256"),
        (tables.explicit_dispositions, "disposition_payload", "disposition_payload_sha256"),
        (tables.support_space_records, "support_payload_schema", "support_payload_schema_sha256"),
        (tables.support_space_records, "query_payload_schema", "query_payload_schema_sha256"),
    ]
    for rows, payload_field, digest_field in checks:
        for row in rows:
            if payload_sha256(row[payload_field]) != row[digest_field]:
                raise CoreV3Error("HASH_OR_ID_MISMATCH", digest_field)


def _validate_unique_ids(tables: CoreV3Tables) -> None:
    all_ids: list[str] = []
    for name, (entity_type, id_field) in ALL_TABLE_SPECS.items():
        rows = getattr(tables, name)
        ids = [row.get(id_field) for row in rows]
        if len(ids) != len(set(ids)):
            raise CoreV3Error("DUPLICATE_ENTITY_ID", name)
        for row in rows:
            verify_entity(entity_type, row)
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise CoreV3Error("DUPLICATE_ENTITY_ID", "GLOBAL")


def _reject_cycles(edges: dict[str, set[str]], detail: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", f"HIERARCHY_CYCLE:{detail}")
        if node in visited:
            return
        visiting.add(node)
        for child in edges.get(node, set()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(edges):
        visit(node)


def _validate_source_identity_hierarchy(rows: list[dict]) -> None:
    by_domain: dict[str, dict[str, str | None]] = {}
    for row in rows:
        domain = by_domain.setdefault(row["domain_scope_id"], {})
        identity = row["source_identity"]
        parent = row["source_parent_id"]
        if identity in domain and domain[identity] != parent:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", f"SOURCE_PARENT_CONFLICT:{identity}")
        if parent == identity:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", f"SOURCE_PARENT_SELF:{identity}")
        domain[identity] = parent
    for domain_scope_id, identities in by_domain.items():
        edges: dict[str, set[str]] = {}
        for child, parent in identities.items():
            if parent in identities:
                edges.setdefault(parent, set()).add(child)
        _reject_cycles(edges, f"SOURCE:{domain_scope_id}")


def validate_tables(tables: CoreV3Tables, predicate_registry: PredicateRegistry) -> dict[str, ResolvedRelationEvidence]:
    _validate_unique_ids(tables)
    _verify_payload_digests(tables)
    _validate_source_identity_hierarchy(tables.source_information_records)

    sources = {row["source_information_id"]: row for row in tables.source_information_records}
    occurrences = {row["generation_occurrence_id"]: row for row in tables.generation_occurrences}
    generated = {row["generated_origin_id"]: row for row in tables.generated_origins}
    supports = {row["support_id"]: row for row in tables.perceptual_support_records}
    dispositions = {row["disposition_id"]: row for row in tables.explicit_dispositions}
    bindings = {row["generation_binding_id"]: row for row in tables.generation_bindings}
    spaces = {row["support_space_id"]: row for row in tables.support_space_records}
    profiles = {row["predicate_profile_id"]: row for row in tables.predicate_profiles}
    evidence = {row["evidence_id"]: row for row in tables.evidence_records}
    manifests = {row["generator_manifest_id"]: row for row in tables.generator_manifests}
    environments = {row["environment_record_id"]: row for row in tables.environment_records}

    if predicate_registry.profile_ids != frozenset(profiles):
        raise CoreV3Error("PREDICATE_PROFILE_UNKNOWN", "SNAPSHOT_REGISTRY_MISMATCH")

    for profile in tables.predicate_profiles:
        if profile["support_space_id"] not in spaces:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", profile["support_space_id"])
        if not profile["authorized"]:
            raise CoreV3Error("PREDICATE_PROFILE_UNAUTHORIZED", profile["predicate_profile_id"])

    for support in tables.perceptual_support_records:
        profile = profiles.get(support["predicate_profile_id"])
        if profile is None or support["support_space_id"] not in spaces:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", support["support_id"])
        if profile["support_space_id"] != support["support_space_id"]:
            raise CoreV3Error("SUPPORT_PAYLOAD_PROFILE_MISMATCH", support["support_id"])
        predicate_registry.validate_support(profile["predicate_profile_id"], support)

    for manifest in tables.generator_manifests:
        if any(item not in spaces for item in manifest["supported_support_space_ids"]):
            raise CoreV3Error("EXTERNAL_KEY_MISSING", manifest["generator_manifest_id"])
        if any(item not in profiles for item in manifest["supported_predicate_profile_ids"]):
            raise CoreV3Error("EXTERNAL_KEY_MISSING", manifest["generator_manifest_id"])

    for occurrence in tables.generation_occurrences:
        if occurrence["generator_manifest_id"] not in manifests:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", occurrence["generation_occurrence_id"])
    for origin in tables.generated_origins:
        if origin["generator_manifest_id"] not in manifests:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", origin["generated_origin_id"])

    used_sources: set[str] = set()
    used_occurrences: set[str] = set()
    used_generated: set[str] = set()
    explained_supports: set[str] = set()
    explained_dispositions: set[str] = set()
    for binding in tables.generation_bindings:
        origin = binding["origin_reference"]
        if origin["kind"] == "registered_source":
            origin_id = origin["source_information_id"]
            record = sources.get(origin_id)
            if record is None:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", origin_id)
            used_sources.add(origin_id)
        elif origin["kind"] == "generated_origin":
            origin_id = origin["generated_origin_id"]
            record = generated.get(origin_id)
            if record is None:
                raise CoreV3Error("GENERATED_ORIGIN_MISCLASSIFIED", origin_id)
            used_generated.add(origin_id)
        else:
            raise CoreV3Error("GENERATED_ORIGIN_MISCLASSIFIED", binding["generation_binding_id"])
        occurrence = occurrences.get(binding["generation_occurrence_id"])
        if occurrence is None:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", binding["generation_occurrence_id"])
        used_occurrences.add(occurrence["generation_occurrence_id"])
        outcome = binding["outcome_reference"]
        if outcome["kind"] == "support":
            outcome_id = outcome["support_id"]
            outcome_record = supports.get(outcome_id)
            if outcome_record is None:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", outcome_id)
            explained_supports.add(outcome_id)
        elif outcome["kind"] == "disposition":
            outcome_id = outcome["disposition_id"]
            outcome_record = dispositions.get(outcome_id)
            if outcome_record is None:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", outcome_id)
            explained_dispositions.add(outcome_id)
        else:
            raise CoreV3Error("BINDING_OUTCOME_CARDINALITY_INVALID", binding["generation_binding_id"])
        if not (binding["domain_scope_id"] == record["domain_scope_id"] == occurrence["domain_scope_id"] == outcome_record["domain_scope_id"]):
            raise CoreV3Error("EXTERNAL_KEY_MISSING", f"DOMAIN_SCOPE:{binding['generation_binding_id']}")

    if used_sources != set(sources):
        raise CoreV3Error("SOURCE_COVERAGE_FAILED", "REGISTERED_SOURCE")
    if used_generated != set(generated):
        raise CoreV3Error("SOURCE_COVERAGE_FAILED", "GENERATED_ORIGIN")
    if used_occurrences != set(occurrences):
        raise CoreV3Error("SOURCE_COVERAGE_FAILED", "OCCURRENCE")
    if explained_supports != set(supports):
        raise CoreV3Error("SUPPORT_ORIGIN_UNEXPLAINED", "SUPPORT")
    if explained_dispositions != set(dispositions):
        raise CoreV3Error("SUPPORT_ORIGIN_UNEXPLAINED", "DISPOSITION")

    entity_ids = tables.all_entity_ids()
    entity_domains = {
        **{row["source_information_id"]: row["domain_scope_id"] for row in tables.source_information_records},
        **{row["support_id"]: row["domain_scope_id"] for row in tables.perceptual_support_records},
    }
    hierarchy_graphs: dict[tuple[str, str], dict[str, set[str]]] = {}
    for hierarchy in tables.hierarchy_records:
        for reference_name in ("parent_reference", "child_reference"):
            reference = hierarchy[reference_name]
            target = reference["entity_id"]
            if target not in entity_ids:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", target)
            if reference["entity_type"] == "source_information" and target not in sources:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", target)
            if reference["entity_type"] == "perceptual_support" and target not in supports:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", target)
            if entity_domains.get(target) != hierarchy["domain_scope_id"]:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", f"HIERARCHY_DOMAIN:{target}")
        parent_id = hierarchy["parent_reference"]["entity_id"]
        child_id = hierarchy["child_reference"]["entity_id"]
        if parent_id == child_id:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", f"HIERARCHY_SELF:{parent_id}")
        graph = hierarchy_graphs.setdefault(
            (hierarchy["domain_scope_id"], hierarchy["hierarchy_kind"]), {},
        )
        graph.setdefault(parent_id, set()).add(child_id)
    for (domain_scope_id, kind), graph in hierarchy_graphs.items():
        _reject_cycles(graph, f"{kind}:{domain_scope_id}")

    environment_hashes = {row["environment_payload_sha256"] for row in environments.values()}
    for record in tables.evidence_records:
        if record["environment_hash"] not in environment_hashes:
            raise CoreV3Error("SNAPSHOT_ENVIRONMENT_MISMATCH", record["evidence_id"])
    for link in tables.evidence_links:
        if link["evidence_id"] not in evidence:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", link["evidence_id"])
        if link["subject_type"] == "generation_binding" and link["subject_id"] not in bindings:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", link["subject_id"])
        if link["subject_type"] == "entity" and link["subject_id"] not in entity_ids:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", link["subject_id"])

    for operation in tables.generator_operation_results:
        if operation["generator_manifest_id"] not in manifests:
            raise CoreV3Error("EXTERNAL_KEY_MISSING", operation["generator_manifest_id"])
        if any(item not in evidence for item in operation["evidence_ids"]):
            raise CoreV3Error("EXTERNAL_KEY_MISSING", operation["operation_result_id"])
        if operation["status"] == "success":
            if operation["reason_code"] is not None:
                raise CoreV3Error("OPERATION_BINDING_CLOSURE_FAILED", operation["operation_result_id"])
            if any(item not in entity_ids for item in operation["produced_entity_ids"]):
                raise CoreV3Error("OPERATION_BINDING_CLOSURE_FAILED", operation["operation_result_id"])
        else:
            if operation["reason_code"] is None or operation["produced_entity_ids"]:
                raise CoreV3Error("OPERATION_BINDING_CLOSURE_FAILED", operation["operation_result_id"])

    for migration in tables.migration_records:
        generated_ids = set(migration["generated_binding_ids"])
        if not generated_ids <= set(bindings):
            raise CoreV3Error("EXTERNAL_KEY_MISSING", migration["migration_record_id"])
        if migration["migration_classification"] == "MIGRATION_PAIRING_UNRESOLVED" and generated_ids:
            raise CoreV3Error("AMBIGUOUS_RELATION_REJECTED", migration["legacy_record_id"])
        if migration["fabricated_cross_product_count"] != 0:
            raise CoreV3Error("AMBIGUOUS_RELATION_REJECTED", migration["legacy_record_id"])

    expected_source, expected_occurrence = derive_legacy_projections(
        tables.source_information_records, tables.generation_occurrences, tables.generation_bindings,
        validate_schema=False,
    )
    if not projections_equal(expected_source, tables.legacy_source_binding_projections, "source_binding_id"):
        raise CoreV3Error("PROJECTION_DRIFT", "SOURCE")
    if not projections_equal(expected_occurrence, tables.legacy_occurrence_binding_projections, "occurrence_binding_id"):
        raise CoreV3Error("PROJECTION_DRIFT", "OCCURRENCE")

    return RelationEvidenceResolver().resolve(tables, preverified=True)


def build_snapshot(
    tables: CoreV3Tables,
    predicate_registry: PredicateRegistry,
    *, expected_implementation_hashes: dict[str, str] | None = None,
) -> ValidatedSnapshot:
    validate_tables(tables, predicate_registry)
    impl = expected_implementation_hashes or implementation_hashes()
    record = finalize_entity("ValidatedSnapshot", {
        "protocol_version": "sidecar-core-v3-generation-relation-v1",
        "protocol_sha256": protocol_sha256(),
        "implementation_hashes": impl,
        "environment_record_ids": [row["environment_record_id"] for row in tables.environment_records],
        "authoritative_table_counts": tables.authoritative_counts(),
        "authoritative_table_hashes": _table_hashes(tables, AUTHORITATIVE_TABLE_SPECS),
        "derived_projection_table_counts": tables.derived_counts(),
        "derived_projection_table_hashes": _table_hashes(tables, DERIVED_TABLE_SPECS),
        "generator_manifest_ids": [row["generator_manifest_id"] for row in tables.generator_manifests],
        "generator_operation_result_ids": [row["operation_result_id"] for row in tables.generator_operation_results],
        "schema_version": "3.0.0",
    })
    snapshot = ValidatedSnapshot(record=record, tables=tables)
    validate_snapshot(snapshot, predicate_registry, expected_implementation_hashes=impl)
    return snapshot


def validate_snapshot(
    snapshot: ValidatedSnapshot,
    predicate_registry: PredicateRegistry,
    *, expected_implementation_hashes: dict[str, str] | None = None,
) -> SnapshotValidation:
    verify_entity("ValidatedSnapshot", snapshot.record)
    if snapshot.record["protocol_version"] != "sidecar-core-v3-generation-relation-v1" or snapshot.record["protocol_sha256"] != protocol_sha256():
        raise CoreV3Error("SNAPSHOT_PROTOCOL_MISMATCH")
    expected_impl = expected_implementation_hashes or implementation_hashes()
    if snapshot.record["implementation_hashes"] != expected_impl:
        raise CoreV3Error("IMPLEMENTATION_HASH_MISMATCH", "SNAPSHOT")
    relation_evidence = validate_tables(snapshot.tables, predicate_registry)
    if snapshot.record["environment_record_ids"] != sorted(
        [row["environment_record_id"] for row in snapshot.tables.environment_records]
    ):
        raise CoreV3Error("SNAPSHOT_ENVIRONMENT_MISMATCH")
    if snapshot.record["authoritative_table_counts"] != snapshot.tables.authoritative_counts():
        raise CoreV3Error("SNAPSHOT_TABLE_COUNT_MISMATCH", "AUTHORITATIVE")
    if snapshot.record["derived_projection_table_counts"] != snapshot.tables.derived_counts():
        raise CoreV3Error("SNAPSHOT_TABLE_COUNT_MISMATCH", "PROJECTION")
    if snapshot.record["authoritative_table_hashes"] != _table_hashes(snapshot.tables, AUTHORITATIVE_TABLE_SPECS):
        raise CoreV3Error("SNAPSHOT_TABLE_HASH_MISMATCH", "AUTHORITATIVE")
    if snapshot.record["derived_projection_table_hashes"] != _table_hashes(snapshot.tables, DERIVED_TABLE_SPECS):
        raise CoreV3Error("SNAPSHOT_TABLE_HASH_MISMATCH", "PROJECTION")
    if snapshot.record["generator_manifest_ids"] != sorted(
        [row["generator_manifest_id"] for row in snapshot.tables.generator_manifests]
    ):
        raise CoreV3Error("SNAPSHOT_HASH_MISMATCH", "MANIFEST_IDS")
    if snapshot.record["generator_operation_result_ids"] != sorted(
        [row["operation_result_id"] for row in snapshot.tables.generator_operation_results]
    ):
        raise CoreV3Error("SNAPSHOT_HASH_MISMATCH", "OPERATION_IDS")
    return SnapshotValidation(snapshot.snapshot_id, relation_evidence)
