"""Build Core v3 facts directly from synchronous transformation receipts.

This module deliberately does not import a Source Map encoder, decoder,
consumer, native collector, expected fixture, or Oracle.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Any, Iterable

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.entities import (
    environment_record,
    evidence_link,
    evidence_record,
    explicit_disposition,
    generated_origin,
    generation_binding,
    generation_occurrence,
    generator_manifest,
    generator_operation_result,
    perceptual_support,
    predicate_profile,
    relation_material,
    source_information,
    support_space,
)
from generation_relation_core.predicate_registry import PredicateRegistry, implementation_sha256
from generation_relation_core.snapshots import CoreV3Tables, ValidatedSnapshot, build_snapshot


DOMAIN_SCOPE_ID = "ecma426_source_map_projection_v1"
EVIDENCE_AUTHORITY = "synchronous_source_to_source_generation_receipt_v1"


def exact_generated_anchor_membership(support: dict, query: dict, predicate: str) -> bool:
    return predicate == "membership" and all(
        support.get(field) == query.get(field)
        for field in ("generated_artifact", "generated_line", "generated_column")
    )


def experiment_code_hash(root: Path | None = None) -> str:
    base = root or Path(__file__).resolve().parents[1]
    paths = [
        path for folder in (base / "src", base / "contracts")
        for path in folder.rglob("*")
        if path.is_file() and path.name not in {"pnpm-lock.yaml"} and "__pycache__" not in path.parts
    ]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _position_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["line", "column"],
        "properties": {
            "line": {"type": "integer", "minimum": 0},
            "column": {"type": "integer", "minimum": 0},
        },
    }


class CoreProjectionCollector:
    def __init__(self, *, dependency_hashes: dict[str, str]) -> None:
        self.code_hash = experiment_code_hash()
        self.space = support_space(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_name="generated_javascript_utf16_segment_ranges",
            support_payload_schema={
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "native_support_key", "stage_id", "generated_artifact",
                    "generated_line", "generated_column", "generated_start",
                    "generated_end", "generated_bytes_sha256", "mapping_eligible",
                    "source_root",
                ],
                "properties": {
                    "native_support_key": {"type": "string", "minLength": 1},
                    "stage_id": {"type": "string", "minLength": 1},
                    "generated_artifact": {"type": "string", "minLength": 1},
                    "generated_line": {"type": "integer", "minimum": 0},
                    "generated_column": {"type": "integer", "minimum": 0},
                    "generated_start": _position_schema(),
                    "generated_end": _position_schema(),
                    "generated_bytes_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "mapping_eligible": {"type": "boolean"},
                    "source_root": {"type": ["string", "null"]},
                },
            },
            query_payload_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["generated_artifact", "generated_line", "generated_column"],
                "properties": {
                    "generated_artifact": {"type": "string", "minLength": 1},
                    "generated_line": {"type": "integer", "minimum": 0},
                    "generated_column": {"type": "integer", "minimum": 0},
                },
            },
            normalization_rule="zero-based line and UTF-16 column; half-open generated ranges; exact scalar equality",
        )
        self.profile = predicate_profile(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=self.space["support_space_id"],
            predicate_kind="generated_javascript_anchor_membership",
            supported_predicates=["membership"],
            predicate_authority=EVIDENCE_AUTHORITY,
            authorized=True,
            implementation_module=exact_generated_anchor_membership.__module__,
            implementation_symbol=exact_generated_anchor_membership.__name__,
            predicate_implementation_sha256=implementation_sha256(exact_generated_anchor_membership),
            normalization_rule=self.space["normalization_rule"],
            result_ordering_rule="ascending generated line, column, then support_id",
        )
        dependency_hashes = {key: value for key, value in sorted(dependency_hashes.items())}
        self.environment = environment_record(
            runtime_name="CPython and external Node Source Map comparator",
            runtime_version=platform.python_version(),
            operating_system=platform.platform(),
            dependency_hashes=dependency_hashes,
        )
        self.manifest = generator_manifest(
            generator_name="deterministic-ecma426-source-to-source-transformer",
            generator_version="source-map-projection-v1",
            generator_code_hash=self.code_hash,
            supported_support_space_ids=[self.space["support_space_id"]],
            supported_predicate_profile_ids=[self.profile["predicate_profile_id"]],
            supported_operations=[
                "COPY_RANGE", "RENAME_IDENTIFIER", "INSERT_LITERAL", "DELETE_RANGE",
                "REORDER_RANGE", "DUPLICATE_RANGE", "CONCATENATE_SOURCES",
                "COLLAPSE_WHITESPACE", "EMIT_SYNTHETIC_WRAPPER", "MINIFY_TO_SINGLE_LINE",
                "REWRITE_THEN_RESTORE", "register_generated_origin_bridge",
            ],
            authorized_evidence_authorities=[EVIDENCE_AUTHORITY],
            dependency_hashes=sorted(set(dependency_hashes.values())),
        )
        self.registry = PredicateRegistry(
            [self.space], [self.profile],
            {self.profile["predicate_profile_id"]: exact_generated_anchor_membership},
        )
        self.tables = CoreV3Tables(
            support_space_records=[self.space],
            predicate_profiles=[self.profile],
            generator_manifests=[self.manifest],
            environment_records=[self.environment],
        )
        self.sources: dict[tuple[Any, ...], dict] = {}
        self.support_by_key: dict[str, dict] = {}
        self.occurrence_by_key: dict[str, dict] = {}
        self.generated_by_prior_support: dict[str, dict] = {}
        self.stage_operations: dict[str, dict] = {}

    def _source(self, origin: dict) -> dict:
        key = (
            origin["source_file"], origin["source_artifact_sha256"],
            origin["source_start"]["line"], origin["source_start"]["column"],
            origin["source_end"]["line"], origin["source_end"]["column"],
            origin["source_bytes_sha256"], origin.get("original_name"),
        )
        row = self.sources.get(key)
        if row is None:
            row = source_information(
                domain_scope_id=DOMAIN_SCOPE_ID,
                source_identity=(
                    f"{origin['source_file']}#L{origin['source_start']['line']}C{origin['source_start']['column']}"
                    f"-L{origin['source_end']['line']}C{origin['source_end']['column']}@{origin['source_bytes_sha256']}"
                ),
                source_parent_id=origin["source_file"],
                source_granularity="javascript_utf16_source_range",
                source_payload={
                    "source_file": origin["source_file"],
                    "source_artifact_sha256": origin["source_artifact_sha256"],
                    "source_start": origin["source_start"],
                    "source_end": origin["source_end"],
                    "source_bytes_sha256": origin["source_bytes_sha256"],
                    "source_text": origin["source_text"],
                    "source_content": origin["source_content"],
                    "original_name": origin.get("original_name"),
                    "generated_name": origin.get("generated_name"),
                    "coordinate_unit": "UTF-16 code units",
                },
            )
            self.sources[key] = row
            self.tables.source_information_records.append(row)
        return row

    def _generated_origin(self, origin: dict, produced_ids: list[str]) -> dict:
        prior_key = origin.get("prior_support_key")
        prior = self.support_by_key.get(prior_key)
        if prior is None:
            raise ValueError(f"GENERATED_ORIGIN_PRODUCER_SUPPORT_MISSING:{prior_key}")
        row = self.generated_by_prior_support.get(prior_key)
        if row is not None:
            return row
        producer_stage = prior["support_payload"]["stage_id"]
        producer_operation = self.stage_operations.get(producer_stage)
        if producer_operation is None:
            raise ValueError(f"GENERATED_ORIGIN_OPERATION_RESULT_MISSING:{producer_stage}")
        row = generated_origin(
            domain_scope_id=DOMAIN_SCOPE_ID,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            origin_type="generated_javascript_segment_reintroduced_as_input",
            origin_payload={
                "bridge_kind": "support_to_generated_origin",
                "prior_support_id": prior["support_id"],
                "prior_support_key": prior_key,
                "producer_operation_result_id": producer_operation["operation_result_id"],
                "source_file": origin["source_file"],
                "source_artifact_sha256": origin["source_artifact_sha256"],
                "source_start": origin["source_start"],
                "source_end": origin["source_end"],
                "source_content": origin["source_content"],
                "original_name": origin.get("original_name"),
                "generated_name": origin.get("generated_name"),
            },
        )
        bridge_material = canonical_bytes({
            "generated_origin_id": row["generated_origin_id"],
            "prior_support_id": prior["support_id"],
            "producer_operation_result_id": producer_operation["operation_result_id"],
        })
        evidence = evidence_record(
            artifact_locator=f"candidate://generated_origin_bridges.jsonl#sha256={hashlib.sha256(bridge_material).hexdigest()}",
            artifact_role="validation_report",
            artifact_bytes=bridge_material,
            evidence_authority=EVIDENCE_AUTHORITY,
            extraction_method="explicit producer support key carried into the next-stage generation receipt",
            extraction_code_hash=self.code_hash,
            environment_hash=self.environment["environment_payload_sha256"],
            related_record_ids=sorted([
                row["generated_origin_id"], prior["support_id"], producer_operation["operation_result_id"],
            ]),
        )
        link = evidence_link(
            evidence_id=evidence["evidence_id"], subject_type="entity",
            subject_id=row["generated_origin_id"], evidence_role="entity_payload",
        )
        operation = generator_operation_result(
            generator_manifest_id=self.manifest["generator_manifest_id"],
            operation_name="register_generated_origin_bridge",
            produced_entity_ids=[row["generated_origin_id"], evidence["evidence_id"], link["evidence_link_id"]],
            evidence_ids=[evidence["evidence_id"]],
        )
        self.generated_by_prior_support[prior_key] = row
        self.tables.generated_origins.append(row)
        self.tables.evidence_records.append(evidence)
        self.tables.evidence_links.append(link)
        self.tables.generator_operation_results.append(operation)
        produced_ids.append(row["generated_origin_id"])
        return row

    def _occurrence(self, receipt: dict) -> dict:
        key = receipt["occurrence_key"]
        if key in self.occurrence_by_key:
            raise ValueError(f"DUPLICATE_OCCURRENCE_KEY:{key}")
        row = generation_occurrence(
            domain_scope_id=DOMAIN_SCOPE_ID,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            occurrence_stage=receipt["stage_id"],
            occurrence_type="javascript_source_to_source_emit" if receipt["receipt_type"] == "emit" else "javascript_source_range_disposition",
            stable_instance_key=f"{receipt['run_id']}:{key}",
            occurrence_index=receipt["occurrence_index"],
            transform_reference={
                "operation_type": receipt["operation_type"],
                "profile_id": "ecma426-source-map-projection-v1",
            },
            occurrence_payload={
                "run_id": receipt["run_id"],
                "stage_id": receipt["stage_id"],
                "occurrence_key": key,
                "operation_type": receipt["operation_type"],
                "transform_parameters": receipt["transform_parameters"],
                "generated_artifact": receipt["generated_artifact"],
                "capture_timing": "synchronous_at_emit_or_disposition",
            },
        )
        self.occurrence_by_key[key] = row
        self.tables.generation_occurrences.append(row)
        return row

    def _bind(
        self,
        *,
        origin_reference: dict,
        origin_id: str,
        occurrence: dict,
        outcome_reference: dict,
        outcome_id: str,
        role: str,
        evidence_ids: list[str],
        produced_ids: list[str],
    ) -> None:
        material = relation_material(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
        )
        material_bytes = canonical_bytes(material)
        digest = hashlib.sha256(material_bytes).hexdigest()
        evidence = evidence_record(
            artifact_locator=f"candidate://relation_materials.jsonl#sha256={digest}",
            artifact_role="generation_relation_material",
            artifact_bytes=material_bytes,
            evidence_authority=EVIDENCE_AUTHORITY,
            extraction_method="synchronous deterministic transformer receipt; no post-generation matching",
            extraction_code_hash=self.code_hash,
            environment_hash=self.environment["environment_payload_sha256"],
            related_record_ids=sorted([origin_id, occurrence["generation_occurrence_id"], outcome_id]),
        )
        binding = generation_binding(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
            evidence_ids=[evidence["evidence_id"]],
        )
        link = evidence_link(
            evidence_id=evidence["evidence_id"], subject_type="generation_binding",
            subject_id=binding["generation_binding_id"], evidence_role="primary_generation_relation",
        )
        self.tables.evidence_records.append(evidence)
        self.tables.generation_bindings.append(binding)
        self.tables.evidence_links.append(link)
        evidence_ids.append(evidence["evidence_id"])
        produced_ids.extend([binding["generation_binding_id"], evidence["evidence_id"], link["evidence_link_id"]])

    def collect_stage(self, receipts: Iterable[dict]) -> dict:
        rows = list(receipts)
        if not rows:
            raise ValueError("EMPTY_STAGE")
        stage_ids = {row["stage_id"] for row in rows}
        if len(stage_ids) != 1:
            raise ValueError("MIXED_STAGE_RECEIPTS")
        stage_id = next(iter(stage_ids))
        produced_ids: list[str] = []
        evidence_ids: list[str] = []
        for receipt in rows:
            occurrence = self._occurrence(receipt)
            produced_ids.append(occurrence["generation_occurrence_id"])
            if receipt["receipt_type"] == "emit":
                start = receipt["generated_start"]
                support = perceptual_support(
                    domain_scope_id=DOMAIN_SCOPE_ID,
                    support_space_id=self.space["support_space_id"],
                    support_payload={
                        "native_support_key": receipt["support_key"],
                        "stage_id": stage_id,
                        "generated_artifact": receipt["generated_artifact"],
                        "generated_line": start["line"],
                        "generated_column": start["column"],
                        "generated_start": start,
                        "generated_end": receipt["generated_end"],
                        "generated_bytes_sha256": receipt["generated_bytes_sha256"],
                        "mapping_eligible": receipt["mapping_eligible"],
                        "source_root": receipt["source_root"],
                    },
                    predicate_profile_id=self.profile["predicate_profile_id"],
                )
                if receipt["support_key"] in self.support_by_key:
                    raise ValueError(f"DUPLICATE_SUPPORT_KEY:{receipt['support_key']}")
                self.support_by_key[receipt["support_key"]] = support
                self.tables.perceptual_support_records.append(support)
                produced_ids.append(support["support_id"])
                outcome_reference = {"kind": "support", "support_id": support["support_id"]}
                outcome_id = support["support_id"]
            elif receipt["receipt_type"] == "disposition":
                disposition = explicit_disposition(
                    domain_scope_id=DOMAIN_SCOPE_ID,
                    core_disposition_category="suppressed",
                    domain_reason_code=receipt["disposition_reason_code"],
                    disposition_payload={
                        "stage_id": stage_id,
                        "generated_artifact": receipt["generated_artifact"],
                        "operation_type": receipt["operation_type"],
                        "source_file": receipt["origins"][0]["source_file"],
                        "source_start": receipt["origins"][0]["source_start"],
                        "source_end": receipt["origins"][0]["source_end"],
                    },
                )
                self.tables.explicit_dispositions.append(disposition)
                produced_ids.append(disposition["disposition_id"])
                outcome_reference = {"kind": "disposition", "disposition_id": disposition["disposition_id"]}
                outcome_id = disposition["disposition_id"]
            else:
                raise ValueError(f"RECEIPT_TYPE_UNKNOWN:{receipt['receipt_type']}")
            for origin in receipt["origins"]:
                if origin["origin_kind"] == "registered_source":
                    origin_row = self._source(origin)
                    origin_reference = {"kind": "registered_source", "source_information_id": origin_row["source_information_id"]}
                    origin_id = origin_row["source_information_id"]
                    produced_ids.append(origin_id)
                elif origin["origin_kind"] == "generated_origin":
                    origin_row = self._generated_origin(origin, produced_ids)
                    origin_reference = {"kind": "generated_origin", "generated_origin_id": origin_row["generated_origin_id"]}
                    origin_id = origin_row["generated_origin_id"]
                else:
                    raise ValueError(f"ORIGIN_KIND_UNKNOWN:{origin['origin_kind']}")
                self._bind(
                    origin_reference=origin_reference,
                    origin_id=origin_id,
                    occurrence=occurrence,
                    outcome_reference=outcome_reference,
                    outcome_id=outcome_id,
                    role=origin["relation_role"],
                    evidence_ids=evidence_ids,
                    produced_ids=produced_ids,
                )
        operation = generator_operation_result(
            generator_manifest_id=self.manifest["generator_manifest_id"],
            operation_name=f"collect_{stage_id}_generation_facts",
            produced_entity_ids=sorted(set(produced_ids)),
            evidence_ids=sorted(set(evidence_ids)),
        )
        self.tables.generator_operation_results.append(operation)
        self.stage_operations[stage_id] = operation
        return operation

    def finalize(self) -> ValidatedSnapshot:
        source_rows, occurrence_rows = derive_legacy_projections(
            self.tables.source_information_records,
            self.tables.generation_occurrences,
            self.tables.generation_bindings,
            validate_schema=False,
        )
        self.tables.legacy_source_binding_projections = source_rows
        self.tables.legacy_occurrence_binding_projections = occurrence_rows
        return build_snapshot(self.tables, self.registry)


def snapshot_document(snapshot: ValidatedSnapshot) -> dict[str, Any]:
    return {
        "snapshot": snapshot.record,
        "tables": {
            field: getattr(snapshot.tables, field)
            for field in snapshot.tables.__dataclass_fields__
        },
    }
