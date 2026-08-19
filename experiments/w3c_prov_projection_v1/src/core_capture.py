from __future__ import annotations

import hashlib
import platform
from typing import Any

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

from .events import ActivityEvent, AgentEvent, BindingEvent, BridgeEvent, GeneratorVariant, OutcomeEvent, SourceEvent


DOMAIN_SCOPE = "w3c-prov-projection-fixture-v1"
EVIDENCE_AUTHORITY = "fixture-runtime-callbacks"


def support_membership(support: dict[str, Any], query: dict[str, Any], predicate: str) -> bool:
    return predicate == "membership" and support.get("native_support_key") == query.get("native_support_key")


def _unique_add(target: dict[str, Any], key: str, value: Any) -> None:
    if key in target:
        raise ValueError(f"duplicate callback key: {key}")
    target[key] = value


class CoreCaptureCollector:
    """Capture actual callbacks into Core v3 without any PROV knowledge."""

    def __init__(self, variant: GeneratorVariant = GeneratorVariant()) -> None:
        self.variant = variant
        self.agent: AgentEvent | None = None
        self.sources: dict[str, SourceEvent] = {}
        self.activities: dict[str, ActivityEvent] = {}
        self.outcomes: dict[str, OutcomeEvent] = {}
        self.bridges: dict[str, BridgeEvent] = {}
        self.bindings: list[BindingEvent] = []

    def on_agent(self, event: AgentEvent) -> None:
        if self.agent is not None:
            raise ValueError("duplicate agent callback")
        self.agent = event

    def on_source(self, event: SourceEvent) -> None:
        _unique_add(self.sources, event.key, event)

    def on_activity(self, event: ActivityEvent) -> None:
        _unique_add(self.activities, event.key, event)

    def on_outcome(self, event: OutcomeEvent) -> None:
        _unique_add(self.outcomes, event.key, event)

    def on_bridge(self, event: BridgeEvent) -> None:
        _unique_add(self.bridges, event.key, event)

    def on_binding(self, event: BindingEvent) -> None:
        self.bindings.append(event)

    def validated_snapshot(self) -> ValidatedSnapshot:
        if self.agent is None:
            raise ValueError("agent callback missing")
        dependency_hash = hashlib.sha256(self.variant.environment_detail.encode("utf-8")).hexdigest()
        space = support_space(
            domain_scope_id=DOMAIN_SCOPE,
            support_space_name="deterministic_tabular_outputs",
            support_payload_schema={
                "type": "object",
                "required": ["native_support_key", "result_category", "result_identity"],
                "properties": {
                    "native_support_key": {"type": "string", "minLength": 1},
                    "result_category": {"type": "string", "minLength": 1},
                    "result_identity": {"type": "string", "minLength": 1},
                },
            },
            query_payload_schema={
                "type": "object",
                "required": ["native_support_key"],
                "properties": {"native_support_key": {"type": "string", "minLength": 1}},
            },
            normalization_rule="Core v3 canonical JSON; support identity is explicit in native_support_key",
        )
        profile = predicate_profile(
            domain_scope_id=DOMAIN_SCOPE,
            support_space_id=space["support_space_id"],
            predicate_kind="native_support_key_membership",
            supported_predicates=["membership"],
            predicate_authority=EVIDENCE_AUTHORITY,
            authorized=True,
            implementation_module=support_membership.__module__,
            implementation_symbol=support_membership.__name__,
            predicate_implementation_sha256=implementation_sha256(support_membership),
            normalization_rule=space["normalization_rule"],
            result_ordering_rule="ascending content-addressed support_id",
        )
        environment = environment_record(
            runtime_name="CPython",
            runtime_version=platform.python_version(),
            operating_system="deterministic-fixture-runtime",
            dependency_hashes={"execution_context": dependency_hash},
        )
        manifest = generator_manifest(
            generator_name=self.agent.name,
            generator_version=self.agent.version,
            generator_code_hash=self.agent.code_identity,
            supported_support_space_ids=[space["support_space_id"]],
            supported_predicate_profile_ids=[profile["predicate_profile_id"]],
            supported_operations=["generate_fixture_base", "generate_fixture_alternate"],
            authorized_evidence_authorities=[EVIDENCE_AUTHORITY],
            dependency_hashes=[dependency_hash],
        )
        if self.variant.operation_name not in manifest["supported_operations"]:
            raise ValueError(f"unsupported operation variant: {self.variant.operation_name}")

        tables = CoreV3Tables(
            support_space_records=[space],
            predicate_profiles=[profile],
            generator_manifests=[manifest],
            environment_records=[environment],
        )
        source_ids: dict[str, str] = {}
        for key, event in sorted(self.sources.items()):
            row = source_information(
                domain_scope_id=DOMAIN_SCOPE,
                source_identity=event.source_identity,
                source_parent_id=None,
                source_granularity=event.source_granularity,
                source_payload={
                    "domain_type": event.domain_type,
                    "stable_domain_identity": event.stable_domain_identity,
                    "value": event.value,
                },
            )
            tables.source_information_records.append(row)
            source_ids[key] = row["source_information_id"]

        activity_ids: dict[str, str] = {}
        for key, event in sorted(self.activities.items(), key=lambda item: item[1].occurrence_index):
            row = generation_occurrence(
                domain_scope_id=DOMAIN_SCOPE,
                generator_manifest_id=manifest["generator_manifest_id"],
                occurrence_stage=event.stage,
                occurrence_type=event.occurrence_type,
                stable_instance_key=event.stable_instance_key,
                occurrence_index=event.occurrence_index,
                transform_reference=event.transform_reference,
                occurrence_payload={
                    "operation_type": event.operation_type,
                    "diagnostic_context": event.diagnostic_context,
                },
            )
            tables.generation_occurrences.append(row)
            activity_ids[key] = row["generation_occurrence_id"]

        outcome_refs: dict[str, dict[str, str]] = {}
        outcome_ids: dict[str, str] = {}
        for key, event in sorted(self.outcomes.items()):
            if event.kind == "support":
                if event.result_category is None or event.result_identity is None:
                    raise ValueError(f"support semantics missing: {key}")
                payload = {
                    "native_support_key": event.result_identity,
                    "result_category": event.result_category,
                    "result_identity": event.result_identity,
                    "value": event.payload,
                }
                row = perceptual_support(
                    domain_scope_id=DOMAIN_SCOPE,
                    support_space_id=space["support_space_id"],
                    support_payload=payload,
                    predicate_profile_id=profile["predicate_profile_id"],
                )
                tables.perceptual_support_records.append(row)
                outcome_refs[key] = {"kind": "support", "support_id": row["support_id"]}
                outcome_ids[key] = row["support_id"]
            elif event.kind == "disposition":
                if event.disposition_category is None or event.reason_code is None:
                    raise ValueError(f"disposition semantics missing: {key}")
                row = explicit_disposition(
                    domain_scope_id=DOMAIN_SCOPE,
                    core_disposition_category=event.disposition_category,
                    domain_reason_code=event.reason_code,
                    disposition_payload={"disposition_class": "unmatched-input", "value": event.payload},
                )
                tables.explicit_dispositions.append(row)
                outcome_refs[key] = {"kind": "disposition", "disposition_id": row["disposition_id"]}
                outcome_ids[key] = row["disposition_id"]
            else:
                raise ValueError(f"unknown outcome kind: {event.kind}")

        bridge_ids: dict[str, str] = {}
        for key, event in sorted(self.bridges.items()):
            prior_id = outcome_ids.get(event.prior_outcome_key)
            if prior_id is None or outcome_refs[event.prior_outcome_key]["kind"] != "support":
                raise ValueError(f"bridge prior support missing: {key}")
            row = generated_origin(
                domain_scope_id=DOMAIN_SCOPE,
                generator_manifest_id=manifest["generator_manifest_id"],
                origin_type="prior-generated-support",
                origin_payload={
                    "prior_support_id": prior_id,
                    "profile_external_detail": event.profile_external_detail,
                },
            )
            tables.generated_origins.append(row)
            bridge_ids[key] = row["generated_origin_id"]

        binding_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        link_rows: list[dict[str, Any]] = []
        seen_semantics: set[tuple[str, str, str, str, int]] = set()
        for event in sorted(
            self.bindings,
            key=lambda item: (item.activity_key, item.origin_kind, item.origin_key, item.outcome_key, item.role, item.ordinal),
        ):
            semantic = (event.origin_kind, event.origin_key, event.activity_key, event.role, event.ordinal)
            if semantic in seen_semantics:
                raise ValueError(f"duplicate binding ordinal: {semantic}")
            seen_semantics.add(semantic)
            if event.origin_kind == "source":
                origin_id = source_ids[event.origin_key]
                origin_ref = {"kind": "registered_source", "source_information_id": origin_id}
            elif event.origin_kind == "generated":
                origin_id = bridge_ids[event.origin_key]
                origin_ref = {"kind": "generated_origin", "generated_origin_id": origin_id}
            else:
                raise ValueError(f"unknown origin kind: {event.origin_kind}")
            occurrence_id = activity_ids[event.activity_key]
            outcome_ref = outcome_refs[event.outcome_key]
            outcome_id = outcome_ids[event.outcome_key]
            role = f"{event.role}|ordinal={event.ordinal:04d}"
            material = relation_material(
                domain_scope_id=DOMAIN_SCOPE,
                origin_reference=origin_ref,
                generation_occurrence_id=occurrence_id,
                outcome_reference=outcome_ref,
                relation_role=role,
            )
            material_bytes = canonical_bytes(material)
            evidence = evidence_record(
                artifact_locator=f"candidate://relation_materials.jsonl#sha256={hashlib.sha256(material_bytes).hexdigest()}",
                artifact_role="generation_relation_material",
                artifact_bytes=material_bytes,
                evidence_authority=EVIDENCE_AUTHORITY,
                extraction_method=f"synchronous fixture callback:{self.variant.evidence_detail}",
                extraction_code_hash=self.agent.code_identity,
                environment_hash=environment["environment_payload_sha256"],
                related_record_ids=sorted([origin_id, occurrence_id, outcome_id]),
            )
            binding = generation_binding(
                domain_scope_id=DOMAIN_SCOPE,
                origin_reference=origin_ref,
                generation_occurrence_id=occurrence_id,
                outcome_reference=outcome_ref,
                relation_role=role,
                evidence_ids=[evidence["evidence_id"]],
            )
            link = evidence_link(
                evidence_id=evidence["evidence_id"],
                subject_type="generation_binding",
                subject_id=binding["generation_binding_id"],
                evidence_role="primary_generation_relation",
            )
            binding_rows.append(binding)
            evidence_rows.append(evidence)
            link_rows.append(link)

        tables.generation_bindings = sorted(binding_rows, key=lambda row: row["generation_binding_id"])
        tables.evidence_records = sorted(evidence_rows, key=lambda row: row["evidence_id"])
        tables.evidence_links = sorted(link_rows, key=lambda row: row["evidence_link_id"])
        produced_ids = sorted(
            set(source_ids.values())
            | set(activity_ids.values())
            | set(outcome_ids.values())
            | set(bridge_ids.values())
            | {row["generation_binding_id"] for row in binding_rows}
            | {row["evidence_id"] for row in evidence_rows}
            | {row["evidence_link_id"] for row in link_rows}
        )
        operation = generator_operation_result(
            generator_manifest_id=manifest["generator_manifest_id"],
            operation_name=self.variant.operation_name,
            produced_entity_ids=produced_ids,
            evidence_ids=sorted(row["evidence_id"] for row in evidence_rows),
        )
        tables.generator_operation_results = [operation]
        source_projection, occurrence_projection = derive_legacy_projections(
            tables.source_information_records,
            tables.generation_occurrences,
            tables.generation_bindings,
            validate_schema=False,
        )
        tables.legacy_source_binding_projections = source_projection
        tables.legacy_occurrence_binding_projections = occurrence_projection
        registry = PredicateRegistry([space], [profile], {profile["predicate_profile_id"]: support_membership})
        return build_snapshot(tables, registry)
