"""Synchronous Core v3 collection for signed-effect native operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import platform
from pathlib import Path
from typing import Any

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.entities import (
    environment_record,
    evidence_link,
    evidence_record,
    explicit_disposition,
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
from generation_relation_core.predicate_registry import (
    PredicateRegistry,
    implementation_sha256,
)
from generation_relation_core.snapshots import (
    CoreV3Tables,
    SnapshotValidation,
    ValidatedSnapshot,
    build_snapshot,
    validate_snapshot,
)

from .generator import NativeExecution, OperationReceipt
from .predicates import all_effect_supports


DOMAIN_SCOPE_ID = "signed-generation-algebra-v1"
EVIDENCE_AUTHORITY = "synchronous-signed-effect-native-execution-v1"
EXPERIMENT_ROOT = Path(__file__).resolve().parent


def experiment_code_hash() -> str:
    digest = hashlib.sha256()
    included = [
        path
        for path in EXPERIMENT_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (
            path.suffix == ".py"
            or (
                path.suffix == ".json"
                and "contracts" in path.relative_to(EXPERIMENT_ROOT).parts
            )
        )
    ]
    for path in sorted(
        included, key=lambda item: item.relative_to(EXPERIMENT_ROOT).as_posix()
    ):
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _effect_support_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "case_id",
            "effect_identity",
            "effect_multiplicity",
            "execution_id",
            "native_after",
            "native_before",
            "native_support_key",
            "outcome_kind",
        ],
        "properties": {
            "case_id": {"type": "string", "minLength": 1},
            "effect_identity": {
                "type": "string",
                "pattern": "^x_[a-z0-9_]+$",
            },
            "effect_multiplicity": {"type": "integer", "minimum": 1},
            "execution_id": {"type": "string", "minLength": 1},
            "native_after": {},
            "native_before": {},
            "native_support_key": {"type": "string", "minLength": 1},
            "outcome_kind": {"const": "realized_effect"},
        },
    }


def _query_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }


@dataclass(frozen=True)
class CollectedExecution:
    snapshot: ValidatedSnapshot
    validation: SnapshotValidation
    predicate_registry: PredicateRegistry


class SignedEffectCollector:
    """Build Core entities synchronously from completed native operations."""

    def __init__(self, execution_id: str, supported_executions: list[str]) -> None:
        self.execution_id = execution_id
        self.code_hash = experiment_code_hash()
        self.space = support_space(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_name="signed_effect_realized_outcome_v1",
            support_payload_schema=_effect_support_schema(),
            query_payload_schema=_query_schema(),
            normalization_rule=(
                "exact effect identity, multiplicity, execution and native "
                "before/after values"
            ),
        )
        self.profile = predicate_profile(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=self.space["support_space_id"],
            predicate_kind="all_signed_effect_supports_v1",
            supported_predicates=["membership"],
            predicate_authority=EVIDENCE_AUTHORITY,
            authorized=True,
            implementation_module=all_effect_supports.__module__,
            implementation_symbol=all_effect_supports.__name__,
            predicate_implementation_sha256=implementation_sha256(
                all_effect_supports
            ),
            normalization_rule=self.space["normalization_rule"],
            result_ordering_rule="ascending content-addressed support_id",
        )
        self.environment = environment_record(
            runtime_name="CPython sqlite3 deterministic signed-effect fixture",
            runtime_version=platform.python_version(),
            operating_system=platform.system(),
            dependency_hashes={
                "experiment_code": self.code_hash,
                "sqlite_runtime": hashlib.sha256(
                    f"sqlite:{__import__('sqlite3').sqlite_version}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            },
        )
        self.manifest = generator_manifest(
            generator_name="signed-generation-algebra-native-fixtures",
            generator_version="1",
            generator_code_hash=self.code_hash,
            supported_support_space_ids=[self.space["support_space_id"]],
            supported_predicate_profile_ids=[
                self.profile["predicate_profile_id"]
            ],
            supported_operations=sorted(supported_executions),
            authorized_evidence_authorities=[EVIDENCE_AUTHORITY],
            dependency_hashes=[self.code_hash],
        )
        self.sources: list[dict[str, Any]] = []
        self.occurrences: list[dict[str, Any]] = []
        self.supports: list[dict[str, Any]] = []
        self.dispositions: list[dict[str, Any]] = []
        self.bindings: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []

    def capture(self, receipt: OperationReceipt) -> None:
        if receipt.execution_id != self.execution_id:
            raise RuntimeError("receipt belongs to a different execution")
        if receipt.effect_multiplicity <= 0:
            raise RuntimeError("effect multiplicity must be positive")
        source = source_information(
            domain_scope_id=DOMAIN_SCOPE_ID,
            source_identity=receipt.source_identity,
            source_parent_id=None,
            source_granularity="native_effect_input",
            source_payload={
                "case_id": receipt.case_id,
                "effect_identity": receipt.effect_identity,
                "execution_id": receipt.execution_id,
                "sequence_index": receipt.sequence_index,
                "transform_operation": receipt.transform_operation,
            },
        )
        occurrence = generation_occurrence(
            domain_scope_id=DOMAIN_SCOPE_ID,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            occurrence_stage="native_effect_application",
            occurrence_type=receipt.occurrence_type,
            stable_instance_key=receipt.occurrence_key,
            occurrence_index=receipt.sequence_index,
            transform_reference={
                "contract_id": "signed-effect-interpretation-v1",
                "operation": receipt.transform_operation,
            },
            occurrence_payload={
                "capture_timing": (
                    "synchronous_after_native_operation_completion"
                ),
                "effect_identity": receipt.effect_identity,
                "effect_multiplicity": receipt.effect_multiplicity,
                "native_after": receipt.native_after,
                "native_before": receipt.native_before,
            },
        )
        if receipt.outcome_kind == "support":
            outcome = perceptual_support(
                domain_scope_id=DOMAIN_SCOPE_ID,
                support_space_id=self.space["support_space_id"],
                support_payload={
                    "case_id": receipt.case_id,
                    "effect_identity": receipt.effect_identity,
                    "effect_multiplicity": receipt.effect_multiplicity,
                    "execution_id": receipt.execution_id,
                    "native_after": receipt.native_after,
                    "native_before": receipt.native_before,
                    "native_support_key": receipt.occurrence_key,
                    "outcome_kind": "realized_effect",
                },
                predicate_profile_id=self.profile["predicate_profile_id"],
            )
            outcome_reference = {
                "kind": "support",
                "support_id": outcome["support_id"],
            }
            outcome_id = outcome["support_id"]
            self.supports.append(outcome)
        elif receipt.outcome_kind == "disposition":
            if receipt.disposition_reason is None:
                raise RuntimeError(
                    "disposition receipt requires an explicit reason"
                )
            outcome = explicit_disposition(
                domain_scope_id=DOMAIN_SCOPE_ID,
                core_disposition_category="suppressed",
                domain_reason_code=receipt.disposition_reason,
                disposition_payload={
                    "case_id": receipt.case_id,
                    "effect_identity": receipt.effect_identity,
                    "effect_multiplicity": receipt.effect_multiplicity,
                    "execution_id": receipt.execution_id,
                    "native_after": receipt.native_after,
                    "native_before": receipt.native_before,
                    "native_support_key": receipt.occurrence_key,
                    "outcome_kind": "explicit_filter_disposition",
                },
            )
            outcome_reference = {
                "kind": "disposition",
                "disposition_id": outcome["disposition_id"],
            }
            outcome_id = outcome["disposition_id"]
            self.dispositions.append(outcome)
        else:
            raise RuntimeError(f"unknown outcome kind: {receipt.outcome_kind}")
        origin_reference = {
            "kind": "registered_source",
            "source_information_id": source["source_information_id"],
        }
        material = relation_material(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence[
                "generation_occurrence_id"
            ],
            outcome_reference=outcome_reference,
            relation_role=receipt.relation_role,
        )
        material_bytes = canonical_bytes(material)
        material_sha256 = hashlib.sha256(material_bytes).hexdigest()
        evidence = evidence_record(
            artifact_locator=(
                "candidate://relation_materials.jsonl"
                f"#sha256={material_sha256}"
            ),
            artifact_role="generation_relation_material",
            artifact_bytes=material_bytes,
            evidence_authority=EVIDENCE_AUTHORITY,
            extraction_method=(
                "synchronous callback immediately after completed native "
                "mutation or filter decision"
            ),
            extraction_code_hash=self.code_hash,
            environment_hash=self.environment[
                "environment_payload_sha256"
            ],
            related_record_ids=sorted(
                [
                    source["source_information_id"],
                    occurrence["generation_occurrence_id"],
                    outcome_id,
                ]
            ),
        )
        binding = generation_binding(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence[
                "generation_occurrence_id"
            ],
            outcome_reference=outcome_reference,
            relation_role=receipt.relation_role,
            evidence_ids=[evidence["evidence_id"]],
        )
        link = evidence_link(
            evidence_id=evidence["evidence_id"],
            subject_type="generation_binding",
            subject_id=binding["generation_binding_id"],
            evidence_role="primary_generation_relation",
        )
        self.sources.append(source)
        self.occurrences.append(occurrence)
        self.evidence.append(evidence)
        self.bindings.append(binding)
        self.links.append(link)

    @staticmethod
    def _sorted(
        rows: list[dict[str, Any]], id_field: str
    ) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda row: row[id_field])

    def finalize(self, native: NativeExecution) -> CollectedExecution:
        if native.execution_id != self.execution_id:
            raise RuntimeError("native result belongs to another execution")
        if native.receipt_count != len(self.bindings):
            raise RuntimeError("native receipt and binding counts differ")
        execution_receipt_bytes = canonical_bytes(
            {
                "execution_id": native.execution_id,
                "final_output": native.final_output,
                "receipt_count": native.receipt_count,
            }
        )
        execution_receipt_sha256 = hashlib.sha256(
            execution_receipt_bytes
        ).hexdigest()
        execution_evidence = evidence_record(
            artifact_locator=(
                "candidate://native_execution_receipt.json"
                f"#sha256={execution_receipt_sha256}"
            ),
            artifact_role="validation_report",
            artifact_bytes=execution_receipt_bytes,
            evidence_authority=EVIDENCE_AUTHORITY,
            extraction_method=(
                "native execution finalization after all synchronous "
                "operation receipts"
            ),
            extraction_code_hash=self.code_hash,
            environment_hash=self.environment[
                "environment_payload_sha256"
            ],
            related_record_ids=sorted(
                [
                    self.space["support_space_id"],
                    self.profile["predicate_profile_id"],
                    self.manifest["generator_manifest_id"],
                ]
            ),
        )
        self.evidence.append(execution_evidence)
        infrastructure_ids = [
            self.space["support_space_id"],
            self.profile["predicate_profile_id"],
            self.environment["environment_record_id"],
            self.manifest["generator_manifest_id"],
        ]
        entity_ids = [
            *infrastructure_ids,
            *[row["source_information_id"] for row in self.sources],
            *[
                row["generation_occurrence_id"]
                for row in self.occurrences
            ],
            *[row["support_id"] for row in self.supports],
            *[row["disposition_id"] for row in self.dispositions],
            *[row["generation_binding_id"] for row in self.bindings],
            *[row["evidence_id"] for row in self.evidence],
            *[row["evidence_link_id"] for row in self.links],
        ]
        operation = generator_operation_result(
            generator_manifest_id=self.manifest["generator_manifest_id"],
            operation_name=self.execution_id,
            produced_entity_ids=sorted(entity_ids),
            evidence_ids=sorted(
                row["evidence_id"] for row in self.evidence
            ),
        )
        tables = CoreV3Tables(
            source_information_records=self._sorted(
                self.sources, "source_information_id"
            ),
            generation_occurrences=self._sorted(
                self.occurrences, "generation_occurrence_id"
            ),
            perceptual_support_records=self._sorted(
                self.supports, "support_id"
            ),
            explicit_dispositions=self._sorted(
                self.dispositions, "disposition_id"
            ),
            generation_bindings=self._sorted(
                self.bindings, "generation_binding_id"
            ),
            support_space_records=[self.space],
            predicate_profiles=[self.profile],
            evidence_records=self._sorted(
                self.evidence, "evidence_id"
            ),
            evidence_links=self._sorted(
                self.links, "evidence_link_id"
            ),
            generator_manifests=[self.manifest],
            generator_operation_results=[operation],
            environment_records=[self.environment],
        )
        source_projection, occurrence_projection = (
            derive_legacy_projections(
                tables.source_information_records,
                tables.generation_occurrences,
                tables.generation_bindings,
                validate_schema=False,
            )
        )
        tables.legacy_source_binding_projections = source_projection
        tables.legacy_occurrence_binding_projections = (
            occurrence_projection
        )
        registry = PredicateRegistry(
            [self.space],
            [self.profile],
            {
                self.profile[
                    "predicate_profile_id"
                ]: all_effect_supports
            },
        )
        snapshot = build_snapshot(tables, registry)
        validation = validate_snapshot(snapshot, registry)
        return CollectedExecution(snapshot, validation, registry)
