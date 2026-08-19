from __future__ import annotations

from typing import Any

from .canonical import canonical_bytes, finalize_entity, payload_sha256, sha256_bytes


SCHEMA_VERSION = "3.0.0"
PROTOCOL_VERSION = "sidecar-core-v3-generation-relation-v1"


def source_information(
    *, domain_scope_id: str, source_identity: str, source_parent_id: str | None,
    source_granularity: str, source_payload: Any,
) -> dict:
    return finalize_entity("SourceInformationRecord", {
        "domain_scope_id": domain_scope_id,
        "source_identity": source_identity,
        "source_parent_id": source_parent_id,
        "source_granularity": source_granularity,
        "source_payload": source_payload,
        "source_payload_sha256": payload_sha256(source_payload),
        "schema_version": SCHEMA_VERSION,
    })


def generation_occurrence(
    *, domain_scope_id: str, generator_manifest_id: str, occurrence_stage: str,
    occurrence_type: str, stable_instance_key: str, occurrence_index: int,
    transform_reference: Any, occurrence_payload: Any,
) -> dict:
    return finalize_entity("GenerationOccurrence", {
        "domain_scope_id": domain_scope_id,
        "generator_manifest_id": generator_manifest_id,
        "occurrence_stage": occurrence_stage,
        "occurrence_type": occurrence_type,
        "stable_instance_key": stable_instance_key,
        "occurrence_index": occurrence_index,
        "transform_reference": transform_reference,
        "occurrence_payload": occurrence_payload,
        "occurrence_payload_sha256": payload_sha256(occurrence_payload),
        "schema_version": SCHEMA_VERSION,
    })


def generated_origin(
    *, domain_scope_id: str, generator_manifest_id: str, origin_type: str, origin_payload: Any,
) -> dict:
    return finalize_entity("GeneratedOrigin", {
        "domain_scope_id": domain_scope_id,
        "generator_manifest_id": generator_manifest_id,
        "origin_type": origin_type,
        "origin_payload": origin_payload,
        "origin_payload_sha256": payload_sha256(origin_payload),
        "schema_version": SCHEMA_VERSION,
    })


def perceptual_support(
    *, domain_scope_id: str, support_space_id: str, support_payload: Any,
    predicate_profile_id: str, support_status: str = "available",
) -> dict:
    return finalize_entity("PerceptualSupportRecord", {
        "domain_scope_id": domain_scope_id,
        "support_space_id": support_space_id,
        "support_payload": support_payload,
        "support_payload_sha256": payload_sha256(support_payload),
        "predicate_profile_id": predicate_profile_id,
        "support_status": support_status,
        "formal_evidence": True,
        "schema_version": SCHEMA_VERSION,
    })


def explicit_disposition(
    *, domain_scope_id: str, core_disposition_category: str, domain_reason_code: str,
    disposition_payload: Any,
) -> dict:
    return finalize_entity("ExplicitDisposition", {
        "domain_scope_id": domain_scope_id,
        "core_disposition_category": core_disposition_category,
        "domain_reason_code": domain_reason_code,
        "disposition_payload": disposition_payload,
        "disposition_payload_sha256": payload_sha256(disposition_payload),
        "formal_evidence": True,
        "schema_version": SCHEMA_VERSION,
    })


def relation_material(
    *, domain_scope_id: str, origin_reference: dict, generation_occurrence_id: str,
    outcome_reference: dict, relation_role: str,
) -> dict:
    return {
        "domain_scope_id": domain_scope_id,
        "origin_reference": origin_reference,
        "generation_occurrence_id": generation_occurrence_id,
        "outcome_reference": outcome_reference,
        "relation_role": relation_role,
        "schema_version": SCHEMA_VERSION,
    }


def generation_binding(
    *, domain_scope_id: str, origin_reference: dict, generation_occurrence_id: str,
    outcome_reference: dict, relation_role: str, evidence_ids: list[str],
) -> dict:
    material = relation_material(
        domain_scope_id=domain_scope_id,
        origin_reference=origin_reference,
        generation_occurrence_id=generation_occurrence_id,
        outcome_reference=outcome_reference,
        relation_role=relation_role,
    )
    return finalize_entity("GenerationBinding", {
        **material,
        "relation_material_sha256": payload_sha256(material),
        "evidence_ids": evidence_ids,
    })


def support_space(
    *, domain_scope_id: str, support_space_name: str, support_payload_schema: dict,
    query_payload_schema: dict, normalization_rule: str,
) -> dict:
    return finalize_entity("SupportSpaceRecord", {
        "domain_scope_id": domain_scope_id,
        "support_space_name": support_space_name,
        "support_payload_schema": support_payload_schema,
        "support_payload_schema_sha256": payload_sha256(support_payload_schema),
        "query_payload_schema": query_payload_schema,
        "query_payload_schema_sha256": payload_sha256(query_payload_schema),
        "normalization_rule": normalization_rule,
        "schema_version": SCHEMA_VERSION,
    })


def predicate_profile(
    *, domain_scope_id: str, support_space_id: str, predicate_kind: str,
    supported_predicates: list[str], predicate_authority: str, authorized: bool,
    implementation_module: str, implementation_symbol: str,
    predicate_implementation_sha256: str, normalization_rule: str,
    result_ordering_rule: str,
) -> dict:
    return finalize_entity("PredicateProfile", {
        "domain_scope_id": domain_scope_id,
        "support_space_id": support_space_id,
        "predicate_kind": predicate_kind,
        "supported_predicates": supported_predicates,
        "predicate_authority": predicate_authority,
        "authorized": authorized,
        "implementation_module": implementation_module,
        "implementation_symbol": implementation_symbol,
        "predicate_implementation_sha256": predicate_implementation_sha256,
        "normalization_rule": normalization_rule,
        "result_ordering_rule": result_ordering_rule,
        "schema_version": SCHEMA_VERSION,
    })


def evidence_record(
    *, artifact_locator: str, artifact_role: str, artifact_bytes: bytes,
    evidence_authority: str, extraction_method: str, extraction_code_hash: str,
    environment_hash: str, related_record_ids: list[str], formal_evidence: bool = True,
    diagnostic_only: bool = False, availability_status: str = "available",
) -> dict:
    return finalize_entity("EvidenceRecord", {
        "artifact_locator": artifact_locator,
        "artifact_role": artifact_role,
        "artifact_sha256": sha256_bytes(artifact_bytes),
        "artifact_size": len(artifact_bytes),
        "evidence_authority": evidence_authority,
        "extraction_method": extraction_method,
        "extraction_code_hash": extraction_code_hash,
        "environment_hash": environment_hash,
        "formal_evidence": formal_evidence,
        "diagnostic_only": diagnostic_only,
        "availability_status": availability_status,
        "related_record_ids": related_record_ids,
        "schema_version": SCHEMA_VERSION,
    })


def evidence_link(*, evidence_id: str, subject_type: str, subject_id: str, evidence_role: str) -> dict:
    return finalize_entity("EvidenceLink", {
        "evidence_id": evidence_id,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "evidence_role": evidence_role,
        "schema_version": SCHEMA_VERSION,
    })


def generator_manifest(
    *, generator_name: str, generator_version: str, generator_code_hash: str,
    supported_support_space_ids: list[str], supported_predicate_profile_ids: list[str],
    supported_operations: list[str], authorized_evidence_authorities: list[str],
    dependency_hashes: list[str],
) -> dict:
    return finalize_entity("GeneratorManifest", {
        "generator_name": generator_name,
        "generator_version": generator_version,
        "generator_code_hash": generator_code_hash,
        "protocol_version": PROTOCOL_VERSION,
        "supported_support_space_ids": supported_support_space_ids,
        "supported_predicate_profile_ids": supported_predicate_profile_ids,
        "supported_operations": supported_operations,
        "authorized_evidence_authorities": authorized_evidence_authorities,
        "dependency_hashes": dependency_hashes,
        "formal_evidence_capable": True,
        "schema_version": SCHEMA_VERSION,
    })


def generator_operation_result(
    *, generator_manifest_id: str, operation_name: str, produced_entity_ids: list[str],
    evidence_ids: list[str], status: str = "success", reason_code: str | None = None,
) -> dict:
    return finalize_entity("GeneratorOperationResult", {
        "generator_manifest_id": generator_manifest_id,
        "operation_name": operation_name,
        "status": status,
        "reason_code": reason_code,
        "produced_entity_ids": produced_entity_ids,
        "evidence_ids": evidence_ids,
        "schema_version": SCHEMA_VERSION,
    })


def hierarchy_record(
    *, domain_scope_id: str, hierarchy_kind: str, parent_reference: dict,
    child_reference: dict, relation_role: str,
) -> dict:
    return finalize_entity("HierarchyRecord", {
        "domain_scope_id": domain_scope_id,
        "hierarchy_kind": hierarchy_kind,
        "parent_reference": parent_reference,
        "child_reference": child_reference,
        "relation_role": relation_role,
        "schema_version": SCHEMA_VERSION,
    })


def migration_record(
    *, domain_scope_id: str, legacy_record_type: str, legacy_record_id: str, migration_classification: str,
    reason_code: str, uniqueness_proof: str, legacy_source_binding_ids: list[str],
    legacy_occurrence_binding_ids: list[str], legacy_evidence_ids: list[str], generated_binding_ids: list[str],
    carrier_artifact_sha256s: list[str],
) -> dict:
    return finalize_entity("MigrationRecord", {
        "domain_scope_id": domain_scope_id,
        "legacy_record_type": legacy_record_type,
        "legacy_record_id": legacy_record_id,
        "migration_classification": migration_classification,
        "reason_code": reason_code,
        "uniqueness_proof": uniqueness_proof,
        "legacy_source_binding_ids": legacy_source_binding_ids,
        "legacy_occurrence_binding_ids": legacy_occurrence_binding_ids,
        "legacy_evidence_ids": legacy_evidence_ids,
        "generated_binding_ids": generated_binding_ids,
        "carrier_artifact_sha256s": carrier_artifact_sha256s,
        "fabricated_cross_product_count": 0,
        "schema_version": SCHEMA_VERSION,
    })


def environment_record(
    *, runtime_name: str, runtime_version: str, operating_system: str,
    dependency_hashes: dict[str, str],
) -> dict:
    return finalize_entity("EnvironmentRecord", {
        "runtime_name": runtime_name,
        "runtime_version": runtime_version,
        "operating_system": operating_system,
        "dependency_hashes": dependency_hashes,
        "schema_version": SCHEMA_VERSION,
    })


def query_request(
    *, domain_scope_id: str, support_space_id: str, predicate_profile_id: str,
    predicate: str, query_payload: Any, requested_granularity: str = "generation_relation",
) -> dict:
    return finalize_entity("QueryRequest", {
        "domain_scope_id": domain_scope_id,
        "support_space_id": support_space_id,
        "predicate_profile_id": predicate_profile_id,
        "predicate": predicate,
        "query_payload": query_payload,
        "requested_granularity": requested_granularity,
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
    })


def relation_evidence_for_material(
    material: dict, *, artifact_locator: str, evidence_authority: str,
    extraction_method: str, extraction_code_hash: str, environment_hash: str,
    related_record_ids: list[str],
) -> dict:
    return evidence_record(
        artifact_locator=artifact_locator,
        artifact_role="generation_relation_material",
        artifact_bytes=canonical_bytes(material),
        evidence_authority=evidence_authority,
        extraction_method=extraction_method,
        extraction_code_hash=extraction_code_hash,
        environment_hash=environment_hash,
        related_record_ids=related_record_ids,
    )
