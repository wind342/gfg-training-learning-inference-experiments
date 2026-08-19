from __future__ import annotations

import hashlib
import platform
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
from generation_relation_core.predicate_registry import PredicateRegistry, implementation_sha256
from generation_relation_core.snapshots import CoreV3Tables, ValidatedSnapshot, build_snapshot, validate_snapshot

from .profile_runtime import support_membership_predicate


DOMAIN_SCOPE_ID = "provenance_semiring_positive_ra_v1"
EVIDENCE_AUTHORITY = "synchronous_positive_ra_runtime_capture_v1"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CoreEventCollector:
    """Write-only observer that records actual executor events in Core v3."""

    def __init__(self, *, run_id: str, variant_tag: str = "baseline", variation: dict[str, str] | None = None) -> None:
        self.run_id = run_id
        self.variant_tag = variant_tag
        self.variation = dict(variation or {})
        allowed_variations = {"evidence", "environment", "disposition", "operation_result"}
        if set(self.variation) - allowed_variations:
            raise ValueError("unknown Core capture variation")
        if any(not isinstance(value, str) or not value for value in self.variation.values()):
            raise ValueError("Core capture variation values must be non-empty strings")
        self._sources: dict[str, dict[str, Any]] = {}
        self._support_by_output: dict[str, dict[str, Any]] = {}
        self._generated_by_support: dict[str, dict[str, Any]] = {}
        self._operations: dict[str, dict[str, list[str]]] = {}
        self._finalized = False
        dependency_hashes = {
            "python": _sha_text(platform.python_version()),
            "core": _sha_text("3.0.0"),
        }
        if "environment" in self.variation:
            dependency_hashes["experiment_environment_variant"] = _sha_text(self.variation["environment"])
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        payload_properties = {
            "native_support_key": {"type": "string", "minLength": 1},
            "tuple_identity": {"type": "string", "minLength": 1},
            "operator_stage": {"type": "string", "minLength": 1},
            "logical_output_key": {"type": "string", "minLength": 1},
            "terminal": {"type": "boolean"},
            "values": {"type": "object"},
        }
        self.space = support_space(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_name="positive_ra_tuple_support_v1",
            support_payload_schema={
                "type": "object",
                "additionalProperties": False,
                "required": list(payload_properties),
                "properties": payload_properties,
            },
            query_payload_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["tuple_identity"],
                "properties": {"tuple_identity": {"type": "string", "minLength": 1}},
            },
            normalization_rule="all identity and stage fields are exact UTF-8 strings",
        )
        self.profile = predicate_profile(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=self.space["support_space_id"],
            predicate_kind="positive_ra_tuple_membership",
            supported_predicates=["membership"],
            predicate_authority=EVIDENCE_AUTHORITY,
            authorized=True,
            implementation_module=support_membership_predicate.__module__,
            implementation_symbol=support_membership_predicate.__name__,
            predicate_implementation_sha256=implementation_sha256(support_membership_predicate),
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
            generator_name="positive-ra-runtime-observer",
            generator_version="1.0.0",
            generator_code_hash=code_hash,
            supported_support_space_ids=[self.space["support_space_id"]],
            supported_predicate_profile_ids=[self.profile["predicate_profile_id"]],
            supported_operations=["capture_positive_ra_occurrence", "capture_explicit_disposition"],
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
            {self.profile["predicate_profile_id"]: support_membership_predicate},
        )

    def __call__(self, event: dict[str, Any]) -> None:
        if self._finalized:
            raise RuntimeError("collector is already finalized")
        event_type = event.get("event")
        if event_type == "source":
            self._capture_source(event)
        elif event_type == "occurrence":
            self._capture_occurrence(event)
        elif event_type == "disposition":
            self._capture_disposition(event)
        else:
            raise ValueError(f"unknown executor event: {event_type!r}")
        return None

    def _capture_source(self, event: dict[str, Any]) -> None:
        identity = event["source_identity"]
        existing = self._sources.get(identity)
        if existing is not None:
            if existing["source_payload"]["field_values"] != event["values"]:
                raise ValueError("source identity was reused with different values")
            return
        record = source_information(
            domain_scope_id=DOMAIN_SCOPE_ID,
            source_identity=identity,
            source_parent_id=event["relation"],
            source_granularity="database_tuple_identity",
            source_payload={
                "relation": event["relation"],
                "field_values": event["values"],
                "workload_id": event["workload_id"],
            },
        )
        self._sources[identity] = record
        self.tables.source_information_records.append(record)

    def _origin(self, input_identity: str) -> tuple[dict[str, str], str]:
        source = self._sources.get(input_identity)
        if source is not None:
            source_id = source["source_information_id"]
            return {"kind": "registered_source", "source_information_id": source_id}, source_id
        prior_support = self._support_by_output.get(input_identity)
        if prior_support is None:
            raise ValueError(f"input has neither source nor prior support: {input_identity}")
        support_id = prior_support["support_id"]
        origin = self._generated_by_support.get(support_id)
        if origin is None:
            origin = generated_origin(
                domain_scope_id=DOMAIN_SCOPE_ID,
                generator_manifest_id=self.manifest["generator_manifest_id"],
                origin_type="generated_output_reintroduced_as_positive_ra_input",
                origin_payload={
                    "bridge_kind": "support_to_generated_origin",
                    "prior_support_id": support_id,
                    "prior_output_identity": input_identity,
                },
            )
            self._generated_by_support[support_id] = origin
            self.tables.generated_origins.append(origin)
        origin_id = origin["generated_origin_id"]
        return {"kind": "generated_origin", "generated_origin_id": origin_id}, origin_id

    def _occurrence(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "variant_tag": self.variant_tag,
            "workload_id": event["workload_id"],
            "operator": event["operator"],
            "details": event["details"],
        }
        stable_instance_key = event.get("occurrence_identity")
        if stable_instance_key is None:
            stable_instance_key = f"disposition:{event['stage']}:{event['input_identity']}"
        return generation_occurrence(
            domain_scope_id=DOMAIN_SCOPE_ID,
            generator_manifest_id=self.manifest["generator_manifest_id"],
            occurrence_stage=event["stage"],
            occurrence_type=f"positive_ra_{event['operator']}",
            stable_instance_key=stable_instance_key,
            occurrence_index=len(self.tables.generation_occurrences),
            transform_reference={"operator": event["operator"], "stage": event["stage"]},
            occurrence_payload=payload,
        )

    def _bind(self, *, origin_reference: dict[str, str], origin_id: str, occurrence: dict[str, Any], outcome_reference: dict[str, str], outcome_id: str, role: str) -> None:
        material = relation_material(
            domain_scope_id=DOMAIN_SCOPE_ID,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=role,
        )
        extraction_method = "synchronous write-only observer at actual operator event"
        if "evidence" in self.variation:
            extraction_method += f"; frozen evidence variant={self.variation['evidence']}"
        evidence = relation_evidence_for_material(
            material,
            artifact_locator=f"candidate://relation_materials.jsonl#sha256={payload_sha256(material)}",
            evidence_authority=EVIDENCE_AUTHORITY,
            extraction_method=extraction_method,
            extraction_code_hash=self.manifest["generator_code_hash"],
            environment_hash=self.environment["environment_payload_sha256"],
            related_record_ids=[origin_id, occurrence["generation_occurrence_id"], outcome_id],
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
        operation_name = "capture_positive_ra_generation_relation"
        if "operation_result" in self.variation:
            operation_name += f"_{self.variation['operation_result']}"
        operation = self._operations.setdefault(operation_name, {"produced_entity_ids": [], "evidence_ids": []})
        operation["produced_entity_ids"].extend([binding["generation_binding_id"], evidence["evidence_id"], link["evidence_link_id"]])
        operation["evidence_ids"].append(evidence["evidence_id"])

    def _capture_occurrence(self, event: dict[str, Any]) -> None:
        if not event["inputs"] or len(event["inputs"]) != len(event["roles"]):
            raise ValueError("every output occurrence needs one explicit role per input")
        if event["output_identity"] in self._support_by_output:
            raise ValueError("duplicate generated output identity")
        support = perceptual_support(
            domain_scope_id=DOMAIN_SCOPE_ID,
            support_space_id=self.space["support_space_id"],
            support_payload={
                "native_support_key": event["output_identity"],
                "tuple_identity": event["output_identity"],
                "operator_stage": event["stage"],
                "logical_output_key": event["logical_output_key"],
                "terminal": bool(event["terminal"]),
                "values": event["values"],
            },
            predicate_profile_id=self.profile["predicate_profile_id"],
        )
        occurrence = self._occurrence(event)
        self.tables.perceptual_support_records.append(support)
        self.tables.generation_occurrences.append(occurrence)
        self._support_by_output[event["output_identity"]] = support
        outcome = {"kind": "support", "support_id": support["support_id"]}
        for input_identity, role in zip(event["inputs"], event["roles"], strict=True):
            origin_reference, origin_id = self._origin(input_identity)
            self._bind(origin_reference=origin_reference, origin_id=origin_id, occurrence=occurrence, outcome_reference=outcome, outcome_id=support["support_id"], role=role)

    def _capture_disposition(self, event: dict[str, Any]) -> None:
        occurrence = self._occurrence(event)
        disposition_payload = {
            "input_identity": event["input_identity"],
            "operator_stage": event["stage"],
            "variant_tag": self.variant_tag,
        }
        if "disposition" in self.variation:
            disposition_payload["scientific_disposition_variant"] = self.variation["disposition"]
        disposition = explicit_disposition(
            domain_scope_id=DOMAIN_SCOPE_ID,
            core_disposition_category="suppressed",
            domain_reason_code=event["reason"],
            disposition_payload=disposition_payload,
        )
        self.tables.generation_occurrences.append(occurrence)
        self.tables.explicit_dispositions.append(disposition)
        origin_reference, origin_id = self._origin(event["input_identity"])
        self._bind(
            origin_reference=origin_reference,
            origin_id=origin_id,
            occurrence=occurrence,
            outcome_reference={"kind": "disposition", "disposition_id": disposition["disposition_id"]},
            outcome_id=disposition["disposition_id"],
            role=event["reason"],
        )

    def validated_snapshot(self) -> tuple[ValidatedSnapshot, object]:
        if not self._finalized:
            for operation_name, payload in sorted(self._operations.items()):
                self.tables.generator_operation_results.append(
                    generator_operation_result(
                        generator_manifest_id=self.manifest["generator_manifest_id"],
                        operation_name=operation_name,
                        produced_entity_ids=payload["produced_entity_ids"],
                        evidence_ids=payload["evidence_ids"],
                    )
                )
            source_rows, occurrence_rows = derive_legacy_projections(
                self.tables.source_information_records,
                self.tables.generation_occurrences,
                self.tables.generation_bindings,
                validate_schema=False,
            )
            self.tables.legacy_source_binding_projections = source_rows
            self.tables.legacy_occurrence_binding_projections = occurrence_rows
            self._finalized = True
        snapshot = build_snapshot(self.tables, self.registry)
        validation = validate_snapshot(snapshot, self.registry)
        return snapshot, validation


def snapshot_document(snapshot: ValidatedSnapshot) -> dict[str, Any]:
    return {
        "record": snapshot.record,
        "tables": {
            name: getattr(snapshot.tables, name)
            for name in snapshot.tables.__dataclass_fields__
        },
    }


def core_snapshot_from_events(workload: dict[str, Any], *, variant: str | None = None, run_id: str = "scientific-execution", variant_tag: str = "baseline", variation: dict[str, str] | None = None) -> tuple[bytes, dict[str, Any], ValidatedSnapshot, object]:
    from .ordinary_execution import execute_ordinary

    collector = CoreEventCollector(run_id=run_id, variant_tag=variant_tag, variation=variation)
    ordinary_bytes, measurements = execute_ordinary(workload, variant=variant, collector=collector)
    snapshot, validation = collector.validated_snapshot()
    return ordinary_bytes, measurements, snapshot, validation
