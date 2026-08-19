"""Frozen pre-index Core resolver used only by the independent evaluator."""

from __future__ import annotations

from generation_relation_core.canonical import (
    canonical_bytes,
    payload_sha256,
    verify_entity,
)
from generation_relation_core.entities import relation_material
from generation_relation_core.errors import CoreV3Error
from generation_relation_core.relation_evidence import ResolvedRelationEvidence


class LegacyScanRelationEvidenceResolver:
    """The pre-optimization full-scan algorithm, retained for equivalence tests."""

    def resolve(
        self, tables: object, *, preverified: bool = False
    ) -> dict[str, ResolvedRelationEvidence]:
        evidence = {row["evidence_id"]: row for row in tables.evidence_records}
        manifests = {
            row["generator_manifest_id"]: row for row in tables.generator_manifests
        }
        occurrences = {
            row["generation_occurrence_id"]: row
            for row in tables.generation_occurrences
        }
        links_by_binding: dict[str, list[dict]] = {}
        for link in tables.evidence_links:
            if not preverified:
                verify_entity("EvidenceLink", link)
            if (
                link["subject_type"] == "generation_binding"
                and link["evidence_role"] == "primary_generation_relation"
            ):
                links_by_binding.setdefault(link["subject_id"], []).append(link)
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
            primary = links_by_binding.get(binding_id, [])
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
            if (
                record["artifact_role"] == "diagnostic_artifact"
                or record["diagnostic_only"]
            ):
                raise CoreV3Error("DIAGNOSTIC_EVIDENCE_PROHIBITED", binding_id)
            if not record["formal_evidence"]:
                raise CoreV3Error("EVIDENCE_NOT_FORMAL", binding_id)
            if record["availability_status"] != "available":
                raise CoreV3Error("EVIDENCE_NOT_AVAILABLE", binding_id)
            if record["artifact_role"] != "generation_relation_material":
                raise CoreV3Error("BINDING_EVIDENCE_MATERIAL_MISMATCH", binding_id)
            if record["artifact_sha256"] != payload_sha256(material) or record[
                "artifact_size"
            ] != len(material_bytes):
                raise CoreV3Error("BINDING_EVIDENCE_MATERIAL_MISMATCH", binding_id)
            if (
                record["artifact_locator"]
                != f"candidate://relation_materials.jsonl#sha256={record['artifact_sha256']}"
            ):
                raise CoreV3Error("ARTIFACT_HASH_MISMATCH", binding_id)
            occurrence = occurrences.get(binding["generation_occurrence_id"])
            if occurrence is None:
                raise CoreV3Error(
                    "EXTERNAL_KEY_MISSING", binding["generation_occurrence_id"]
                )
            manifest = manifests.get(occurrence["generator_manifest_id"])
            if manifest is None:
                raise CoreV3Error(
                    "EXTERNAL_KEY_MISSING", occurrence["generator_manifest_id"]
                )
            if (
                record["evidence_authority"]
                not in manifest["authorized_evidence_authorities"]
            ):
                raise CoreV3Error("EVIDENCE_AUTHORITY_UNAUTHORIZED", binding_id)
            origin = binding["origin_reference"]
            origin_id = origin.get(
                "source_information_id", origin.get("generated_origin_id")
            )
            outcome = binding["outcome_reference"]
            outcome_id = outcome.get("support_id", outcome.get("disposition_id"))
            required_related = {
                origin_id,
                binding["generation_occurrence_id"],
                outcome_id,
            }
            if not required_related <= set(record["related_record_ids"]):
                raise CoreV3Error("EVIDENCE_ENTITY_MISMATCH", binding_id)
            if any(
                item.startswith(("qh3_", "res3_"))
                for item in record["related_record_ids"]
            ):
                raise CoreV3Error("QUERY_OUTPUT_EVIDENCE_PROHIBITED", binding_id)
            closure = [
                operation
                for operation in tables.generator_operation_results
                if operation["generator_manifest_id"]
                == manifest["generator_manifest_id"]
                and operation["status"] == "success"
                and binding_id in operation["produced_entity_ids"]
                and record["evidence_id"] in operation["evidence_ids"]
            ]
            if len(closure) != 1:
                raise CoreV3Error("OPERATION_BINDING_CLOSURE_FAILED", binding_id)
            result[binding_id] = ResolvedRelationEvidence(
                generation_binding_id=binding_id,
                evidence_link_id=link["evidence_link_id"],
                evidence_id=record["evidence_id"],
                operation_result_id=closure[0]["operation_result_id"],
            )
        if set(result) != {
            row["generation_binding_id"] for row in tables.generation_bindings
        }:
            raise CoreV3Error("OPERATION_BINDING_CLOSURE_FAILED", "COVERAGE")
        return result
