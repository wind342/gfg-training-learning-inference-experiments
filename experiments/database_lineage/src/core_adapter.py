from __future__ import annotations

import hashlib
import platform
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.canonical import canonical_bytes, payload_sha256
from generation_relation_core.entities import (
    environment_record,
    evidence_link,
    explicit_disposition,
    generated_origin,
    generation_binding,
    generation_occurrence,
    generator_manifest,
    generator_operation_result,
    perceptual_support,
    predicate_profile,
    relation_evidence_for_material,
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
    ValidatedSnapshot,
    build_snapshot,
)

from .relational_executor import RelationTuple


DOMAIN_SCOPE_ID = "database_lineage_deterministic_relational_execution"
EVIDENCE_AUTHORITY = "deterministic_operator_runtime_capture"


def tuple_identity_predicate(support: dict, query: dict, predicate: str) -> bool:
    return (
        predicate == "membership"
        and support["tuple_identity"] == query["tuple_identity"]
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": format(value, "f")}
    if isinstance(value, (date, datetime)):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    return value


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CoreAdapter:
    """Capture actual operator-time facts directly into unmodified Core v3."""

    def __init__(
        self, *, run_id: str, dependencies: dict[str, str] | None = None
    ) -> None:
        self.run_id = run_id
        self._occurrence_index = 0
        self._sources: dict[str, dict] = {}
        self._generated_by_support: dict[str, dict] = {}
        self._pending_operations: dict[str, dict[str, list[str]]] = {}
        self._operations_finalized = False
        path = Path(__file__)
        code_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        dependency_versions = {
            "python": platform.python_version(),
            "core": "3.0.0",
            **(dependencies or {}),
        }
        dependency_hashes = {
            name: _sha_text(value)
            for name, value in sorted(dependency_versions.items())
        }
        self.space = support_space(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_name="relational_tuple_occurrence",
            support_payload_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["native_support_key", "tuple_identity", "operator_stage"],
                "properties": {
                    "native_support_key": {"type": "string", "minLength": 1},
                    "tuple_identity": {"type": "string", "minLength": 1},
                    "operator_stage": {"type": "string", "minLength": 1},
                },
            },
            query_payload_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["tuple_identity"],
                "properties": {"tuple_identity": {"type": "string", "minLength": 1}},
            },
            normalization_rule="tuple identities and operator stages are exact UTF-8 strings",
        )
        self.profile = predicate_profile(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=self.space["support_space_id"],
            predicate_kind="relational_tuple_identity_membership",
            supported_predicates=["membership"],
            predicate_authority=EVIDENCE_AUTHORITY,
            authorized=True,
            implementation_module=tuple_identity_predicate.__module__,
            implementation_symbol=tuple_identity_predicate.__name__,
            predicate_implementation_sha256=implementation_sha256(
                tuple_identity_predicate
            ),
            normalization_rule=self.space["normalization_rule"],
            result_ordering_rule="ascending content-addressed support_id",
        )
        self.environment = environment_record(
            runtime_name="CPython",
            runtime_version=platform.python_version(),
            operating_system=platform.platform(),
            dependency_hashes=dependency_hashes,
        )
        self.manifest = generator_manifest(
            generator_name="deterministic-relational-executor",
            generator_version="1.0.0",
            generator_code_hash=code_hash,
            supported_support_space_ids=[self.space["support_space_id"]],
            supported_predicate_profile_ids=[self.profile["predicate_profile_id"]],
            supported_operations=[
                "capture_selection",
                "capture_projection",
                "capture_equi_join",
                "capture_group_by",
                "capture_sort",
                "capture_limit",
            ],
            authorized_evidence_authorities=[EVIDENCE_AUTHORITY],
            dependency_hashes=sorted(dependency_hashes.values()),
        )
        self.tables = CoreV3Tables(
            support_space_records=[self.space],
            predicate_profiles=[self.profile],
            generator_manifests=[self.manifest],
            environment_records=[self.environment],
        )
        self.registry = PredicateRegistry(
            [self.space],
            [self.profile],
            {self.profile["predicate_profile_id"]: tuple_identity_predicate},
        )

    def _origin(self, item: RelationTuple) -> tuple[dict[str, str], str]:
        if item.support_id is None:
            source = self._sources.get(item.tuple_id)
            if source is None:
                source = source_information(
                    domain_scope_id=DOMAIN_SCOPE_ID,
                    source_identity=item.tuple_id,
                    source_parent_id=item.table_identity,
                    source_granularity="database_tuple",
                    source_payload={
                        "tuple_identity": item.tuple_id,
                        "table_identity": item.table_identity,
                        "field_values": _json_value(item.values),
                        "deterministic_order": _json_value(list(item.order_key)),
                    },
                )
                self._sources[item.tuple_id] = source
                self.tables.source_information_records.append(source)
            source_id = source["source_information_id"]
            return {
                "kind": "registered_source",
                "source_information_id": source_id,
            }, source_id
        origin = self._generated_by_support.get(item.support_id)
        if origin is None:
            origin = generated_origin(
                domain_scope_id=DOMAIN_SCOPE_ID,
                generator_manifest_id=self.manifest["generator_manifest_id"],
                origin_type="generated_output_reintroduced_as_input",
                origin_payload={
                    "bridge_kind": "support_to_generated_origin",
                    "prior_support_id": item.support_id,
                    "tuple_identity": item.tuple_id,
                    "table_identity": item.table_identity,
                },
            )
            self._generated_by_support[item.support_id] = origin
            self.tables.generated_origins.append(origin)
        origin_id = origin["generated_origin_id"]
        return {"kind": "generated_origin", "generated_origin_id": origin_id}, origin_id

    def _occurrence(
        self,
        *,
        stage: str,
        operator_type: str,
        stable_instance_key: str,
        occurrence_payload: dict[str, Any],
    ) -> dict:
        occurrence = generation_occurrence(
            domain_scope_id=DOMAIN_SCOPE_ID,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            occurrence_stage=stage,
            occurrence_type=f"relational_{operator_type}_execution",
            stable_instance_key=stable_instance_key,
            occurrence_index=self._occurrence_index,
            transform_reference={"operator_type": operator_type, "stage": stage},
            occurrence_payload=_json_value(
                {"run_id": self.run_id, **occurrence_payload}
            ),
        )
        self._occurrence_index += 1
        self.tables.generation_occurrences.append(occurrence)
        return occurrence

    def _bind(
        self,
        *,
        origin_reference: dict[str, str],
        origin_id: str,
        occurrence: dict,
        outcome_reference: dict[str, str],
        outcome_id: str,
        role: str,
        operation_name: str,
    ) -> str:
        material = relation_material(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
        )
        evidence = relation_evidence_for_material(
            material,
            artifact_locator=f"candidate://relation_materials.jsonl#sha256={payload_sha256(material)}",
            evidence_authority=EVIDENCE_AUTHORITY,
            extraction_method="captured synchronously at deterministic operator output/disposition creation",
            extraction_code_hash=self.manifest["generator_code_hash"],
            environment_hash=self.environment["environment_payload_sha256"],
            related_record_ids=[
                origin_id,
                occurrence["generation_occurrence_id"],
                outcome_id,
            ],
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
            evidence_id=evidence["evidence_id"],
            subject_type="generation_binding",
            subject_id=binding["generation_binding_id"],
            evidence_role="primary_generation_relation",
        )
        self.tables.evidence_records.append(evidence)
        self.tables.generation_bindings.append(binding)
        self.tables.evidence_links.append(link)
        pending = self._pending_operations.setdefault(
            operation_name,
            {"produced_entity_ids": [], "evidence_ids": []},
        )
        pending["produced_entity_ids"].extend(
            [
                binding["generation_binding_id"],
                evidence["evidence_id"],
                link["evidence_link_id"],
            ]
        )
        pending["evidence_ids"].append(evidence["evidence_id"])
        return binding["generation_binding_id"]

    def capture_output(
        self,
        *,
        stage: str,
        operator_type: str,
        output: RelationTuple,
        inputs: list[RelationTuple],
        roles: list[str],
        occurrence_payload: dict[str, Any],
    ) -> str:
        if not inputs or len(inputs) != len(roles):
            raise ValueError("every captured output requires one role per real input")
        support = perceptual_support(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=self.space["support_space_id"],
            support_payload={
                "native_support_key": output.tuple_id,
                "tuple_identity": output.tuple_id,
                "operator_stage": stage,
            },
            predicate_profile_id=self.profile["predicate_profile_id"],
        )
        self.tables.perceptual_support_records.append(support)
        occurrence = self._occurrence(
            stage=stage,
            operator_type=operator_type,
            stable_instance_key=output.tuple_id,
            occurrence_payload={
                **occurrence_payload,
                "output_tuple_identity": output.tuple_id,
                "input_tuple_identities": [item.tuple_id for item in inputs],
            },
        )
        outcome_reference = {"kind": "support", "support_id": support["support_id"]}
        for item, role in zip(inputs, roles, strict=True):
            origin_reference, origin_id = self._origin(item)
            self._bind(
                origin_reference=origin_reference,
                origin_id=origin_id,
                occurrence=occurrence,
                outcome_reference=outcome_reference,
                outcome_id=support["support_id"],
                role=role,
                operation_name=f"capture_{operator_type}_{stage}",
            )
        return support["support_id"]

    def capture_disposition(
        self,
        *,
        stage: str,
        operator_type: str,
        input_tuple: RelationTuple,
        reason: str,
        occurrence_payload: dict[str, Any],
    ) -> str:
        occurrence = self._occurrence(
            stage=stage,
            operator_type=operator_type,
            stable_instance_key=f"{stage}:disposition:{input_tuple.tuple_id}",
            occurrence_payload={
                **occurrence_payload,
                "input_tuple_identity": input_tuple.tuple_id,
                "disposition_reason": reason,
            },
        )
        disposition = explicit_disposition(
            domain_scope_id=DOMAIN_SCOPE_ID,
            core_disposition_category="suppressed",
            domain_reason_code=reason,
            disposition_payload={
                "tuple_identity": input_tuple.tuple_id,
                "operator_stage": stage,
                "operator_type": operator_type,
            },
        )
        self.tables.explicit_dispositions.append(disposition)
        origin_reference, origin_id = self._origin(input_tuple)
        self._bind(
            origin_reference=origin_reference,
            origin_id=origin_id,
            occurrence=occurrence,
            outcome_reference={
                "kind": "disposition",
                "disposition_id": disposition["disposition_id"],
            },
            outcome_id=disposition["disposition_id"],
            role=reason,
            operation_name=f"capture_{operator_type}_{stage}",
        )
        return disposition["disposition_id"]

    def validated_snapshot(self) -> ValidatedSnapshot:
        if not self._operations_finalized:
            for operation_name, payload in sorted(self._pending_operations.items()):
                self.tables.generator_operation_results.append(
                    generator_operation_result(
                        generator_manifest_id=self.manifest["generator_manifest_id"],
                        operation_name=operation_name,
                        produced_entity_ids=payload["produced_entity_ids"],
                        evidence_ids=payload["evidence_ids"],
                    )
                )
            self._operations_finalized = True
        source_rows, occurrence_rows = derive_legacy_projections(
            self.tables.source_information_records,
            self.tables.generation_occurrences,
            self.tables.generation_bindings,
            validate_schema=False,
        )
        self.tables.legacy_source_binding_projections = source_rows
        self.tables.legacy_occurrence_binding_projections = occurrence_rows
        return build_snapshot(self.tables, self.registry)
