from __future__ import annotations

from dataclasses import dataclass

from .canonical import canonical_bytes, payload_sha256, verify_entity
from .entities import relation_material
from .errors import CoreV3Error


@dataclass(frozen=True)
class ResolvedRelationEvidence:
    generation_binding_id: str
    evidence_link_id: str
    evidence_id: str
    operation_result_id: str


class RelationEvidenceResolver:
    """Resolve one explicit primary relation path. There is deliberately no fallback."""

    def __init__(self) -> None:
        self.fallback_count = 0

    def resolve(self, tables: object, *, preverified: bool = False) -> dict[str, ResolvedRelationEvidence]:
        evidence = {row["evidence_id"]: row for row in tables.evidence_records}
        manifests = {row["generator_manifest_id"]: row for row in tables.generator_manifests}
        occurrences = {row["generation_occurrence_id"]: row for row in tables.generation_occurrences}
        entity_by_id = {}
        for table_name, id_field in (
            ("source_information_records", "source_information_id"),
            ("generation_occurrences", "generation_occurrence_id"),
            ("generated_origins", "generated_origin_id"),
            ("perceptual_support_records", "support_id"),
            ("explicit_dispositions", "disposition_id"),
            ("generation_bindings", "generation_binding_id"),
            ("hierarchy_records", "hierarchy_record_id"),
            ("support_space_records", "support_space_id"),
            ("predicate_profiles", "predicate_profile_id"),
            ("evidence_records", "evidence_id"),
            ("evidence_links", "evidence_link_id"),
            ("generator_manifests", "generator_manifest_id"),
            ("generator_operation_results", "operation_result_id"),
            ("environment_records", "environment_record_id"),
            ("migration_records", "migration_record_id"),
        ):
            for entity in getattr(tables, table_name):
                entity_by_id.setdefault(entity[id_field], entity)
        operation_by_key: dict[tuple[str, int], dict] = {}
        operations_by_produced_entity: dict[str, list[tuple[str, int]]] = {}
        operation_produced_entity_sets: dict[tuple[str, int], set[str]] = {}
        operation_evidence_sets: dict[tuple[str, int], set[str]] = {}
        for operation_index, operation in enumerate(tables.generator_operation_results):
            operation_key = (operation["operation_result_id"], operation_index)
            operation_by_key[operation_key] = operation
            produced_entities = set(operation["produced_entity_ids"])
            operation_produced_entity_sets[operation_key] = produced_entities
            operation_evidence_sets[operation_key] = set(operation["evidence_ids"])
            for entity_id in produced_entities:
                operations_by_produced_entity.setdefault(entity_id, []).append(operation_key)
        primary_evidence_candidates_by_binding: dict[str, list[dict]] = {}
        for link in tables.evidence_links:
            if not preverified:
                verify_entity("EvidenceLink", link)
            if link["subject_type"] == "generation_binding" and link["evidence_role"] == "primary_generation_relation":
                primary_evidence_candidates_by_binding.setdefault(link["subject_id"], []).append(link)
        result = {}
        for binding in tables.generation_bindings:
            if not preverified:
                verify_entity("GenerationBinding", binding)
            binding_id = binding["generation_binding_id"]
            material = relation_material(
                domain_scope_id=binding["domain_scope_id"],
                origin_reference=binding["origin_reference"],
                generation_occurrence_id=binding["generation_occurrence_id"],
                outcome_reference=binding["outcome_reference"],
                relation_role=binding["relation_role"],
            )
            material_bytes = canonical_bytes(material)
            if payload_sha256(material) != binding["relation_material_sha256"]:
                raise CoreV3Error("BINDING_EVIDENCE_MATERIAL_MISMATCH", binding_id)
            primary = primary_evidence_candidates_by_binding.get(binding_id, [])
            if not primary:
                raise CoreV3Error("BINDING_HAS_NO_PRIMARY_EVIDENCE", binding_id)
            if len(primary) != 1:
                raise CoreV3Error("BINDING_HAS_MULTIPLE_PRIMARY_EVIDENCE", binding_id)
            link = primary[0]
            record = evidence.get(link["evidence_id"])
            if record is None or record["evidence_id"] not in binding["evidence_ids"]:
                raise CoreV3Error("EVIDENCE_ENTITY_MISMATCH", binding_id)
            if not preverified:
                verify_entity("EvidenceRecord", record)
            if record["artifact_role"] == "query_output":
                raise CoreV3Error("QUERY_OUTPUT_EVIDENCE_PROHIBITED", binding_id)
            if record["artifact_role"] == "diagnostic_artifact" or record["diagnostic_only"]:
                raise CoreV3Error("DIAGNOSTIC_EVIDENCE_PROHIBITED", binding_id)
            if not record["formal_evidence"]:
                raise CoreV3Error("EVIDENCE_NOT_FORMAL", binding_id)
            if record["availability_status"] != "available":
                raise CoreV3Error("EVIDENCE_NOT_AVAILABLE", binding_id)
            if record["artifact_role"] != "generation_relation_material":
                raise CoreV3Error("BINDING_EVIDENCE_MATERIAL_MISMATCH", binding_id)
            if record["artifact_sha256"] != payload_sha256(material) or record["artifact_size"] != len(material_bytes):
                raise CoreV3Error("BINDING_EVIDENCE_MATERIAL_MISMATCH", binding_id)
            if record["artifact_locator"] != f"candidate://relation_materials.jsonl#sha256={record['artifact_sha256']}":
                raise CoreV3Error("ARTIFACT_HASH_MISMATCH", binding_id)
            occurrence = occurrences.get(binding["generation_occurrence_id"])
            if occurrence is None:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", binding["generation_occurrence_id"])
            manifest = manifests.get(occurrence["generator_manifest_id"])
            if manifest is None:
                raise CoreV3Error("EXTERNAL_KEY_MISSING", occurrence["generator_manifest_id"])
            if record["evidence_authority"] not in manifest["authorized_evidence_authorities"]:
                raise CoreV3Error("EVIDENCE_AUTHORITY_UNAUTHORIZED", binding_id)
            origin = binding["origin_reference"]
            origin_id = origin.get("source_information_id", origin.get("generated_origin_id"))
            outcome = binding["outcome_reference"]
            outcome_id = outcome.get("support_id", outcome.get("disposition_id"))
            required_related = {origin_id, binding["generation_occurrence_id"], outcome_id}
            if any(entity_id not in entity_by_id for entity_id in required_related):
                raise CoreV3Error("EVIDENCE_ENTITY_MISMATCH", binding_id)
            if not required_related <= set(record["related_record_ids"]):
                raise CoreV3Error("EVIDENCE_ENTITY_MISMATCH", binding_id)
            if any(item.startswith(("qh3_", "res3_")) for item in record["related_record_ids"]):
                raise CoreV3Error("QUERY_OUTPUT_EVIDENCE_PROHIBITED", binding_id)
            closure = [
                operation_by_key[operation_key]
                for operation_key in operations_by_produced_entity.get(binding_id, [])
                if binding_id in operation_produced_entity_sets[operation_key]
                and operation_by_key[operation_key]["status"] == "success"
                and operation_by_key[operation_key]["generator_manifest_id"] == manifest["generator_manifest_id"]
                and record["evidence_id"] in operation_evidence_sets[operation_key]
            ]
            if len(closure) != 1:
                raise CoreV3Error("OPERATION_BINDING_CLOSURE_FAILED", binding_id)
            result[binding_id] = ResolvedRelationEvidence(
                generation_binding_id=binding_id,
                evidence_link_id=link["evidence_link_id"],
                evidence_id=record["evidence_id"],
                operation_result_id=closure[0]["operation_result_id"],
            )
        if set(result) != {row["generation_binding_id"] for row in tables.generation_bindings}:
            raise CoreV3Error("OPERATION_BINDING_CLOSURE_FAILED", "COVERAGE")
        return result
