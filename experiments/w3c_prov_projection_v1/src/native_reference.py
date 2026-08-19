from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .events import ActivityEvent, AgentEvent, BindingEvent, BridgeEvent, OutcomeEvent, SourceEvent
from .record_model import sorted_records


EX_IRI = "https://example.org/w3c-prov-projection-v1#"
PROV_IRI = "http://www.w3.org/ns/prov#"


def _semantic_id(prefix: str, key: dict[str, Any]) -> str:
    # Deliberately independent from candidate_projection._semantic_id.
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return "ex:" + prefix + "_" + digest


def _safe_role(role: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9-]*", role) is None:
        raise ValueError(f"invalid callback role: {role}")
    return "ex:" + role.replace("-", "_")


def _put(target: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    prior = target.get(record["id"])
    if prior is not None and prior != record:
        raise ValueError(f"native semantic collision: {record['id']}")
    target[record["id"]] = record


class NativeProvCollector:
    """Independent synchronous PROV collector; imports neither Core nor candidate."""

    def __init__(self) -> None:
        self.agent: AgentEvent | None = None
        self.sources: dict[str, SourceEvent] = {}
        self.activities: dict[str, ActivityEvent] = {}
        self.outcomes: dict[str, OutcomeEvent] = {}
        self.bridges: dict[str, BridgeEvent] = {}
        self.bindings: list[BindingEvent] = []

    def on_agent(self, event: AgentEvent) -> None:
        if self.agent is not None:
            raise ValueError("duplicate native agent")
        self.agent = event

    def on_source(self, event: SourceEvent) -> None:
        if event.key in self.sources:
            raise ValueError(f"duplicate native source: {event.key}")
        self.sources[event.key] = event

    def on_activity(self, event: ActivityEvent) -> None:
        if event.key in self.activities:
            raise ValueError(f"duplicate native activity: {event.key}")
        self.activities[event.key] = event

    def on_outcome(self, event: OutcomeEvent) -> None:
        if event.key in self.outcomes:
            raise ValueError(f"duplicate native outcome: {event.key}")
        self.outcomes[event.key] = event

    def on_bridge(self, event: BridgeEvent) -> None:
        if event.key in self.bridges:
            raise ValueError(f"duplicate native bridge: {event.key}")
        self.bridges[event.key] = event

    def on_binding(self, event: BindingEvent) -> None:
        self.bindings.append(event)

    def normalized_records(self) -> list[dict[str, Any]]:
        if self.agent is None:
            raise ValueError("native agent missing")
        records: dict[str, dict[str, Any]] = {}
        source_entities: dict[str, str] = {}
        outcome_entities: dict[str, str] = {}
        activities: dict[str, str] = {}

        agent_key = {
            "kind": "software-agent", "generator_name": self.agent.name,
            "generator_version": self.agent.version, "code_identity": self.agent.code_identity,
        }
        agent_id = _semantic_id("ag", agent_key)
        _put(records, {
            "kind": "agent", "id": agent_id, "types": ["prov:SoftwareAgent"],
            "attributes": {
                "ex:codeIdentity": self.agent.code_identity,
                "ex:generatorName": self.agent.name,
                "ex:generatorVersion": self.agent.version,
            },
        })
        for key, event in sorted(self.sources.items()):
            semantic = {
                "kind": "source", "source_identity": event.source_identity,
                "source_granularity": event.source_granularity, "domain_type": event.domain_type,
                "stable_domain_identity": event.stable_domain_identity,
            }
            record_id = _semantic_id("e", semantic)
            _put(records, {
                "kind": "entity", "id": record_id, "types": ["ex:SourceInformation"],
                "attributes": {
                    "ex:domainType": event.domain_type,
                    "ex:sourceGranularity": event.source_granularity,
                    "ex:sourceIdentity": event.source_identity,
                    "ex:stableDomainIdentity": event.stable_domain_identity,
                },
            })
            source_entities[key] = record_id
        for key, event in sorted(self.outcomes.items()):
            if event.kind == "support":
                if event.result_category is None or event.result_identity is None:
                    raise ValueError(f"native support semantics missing: {key}")
                semantic = {"kind": "support", "result_category": event.result_category, "result_identity": event.result_identity}
                record_id = _semantic_id("e", semantic)
                record = {
                    "kind": "entity", "id": record_id, "types": ["ex:GeneratedSupport"],
                    "attributes": {"ex:resultCategory": event.result_category, "ex:resultIdentity": event.result_identity},
                }
            elif event.kind == "disposition":
                if event.disposition_category is None or event.reason_code is None:
                    raise ValueError(f"native disposition semantics missing: {key}")
                semantic = {"kind": "disposition", "disposition_category": event.disposition_category, "reason_code": event.reason_code}
                record_id = _semantic_id("e", semantic)
                record = {
                    "kind": "entity", "id": record_id, "types": ["ex:DispositionRecord"],
                    "attributes": {"ex:dispositionCategory": event.disposition_category, "ex:reasonCode": event.reason_code},
                }
            else:
                raise ValueError(f"unknown native outcome kind: {event.kind}")
            _put(records, record)
            outcome_entities[key] = record_id
        for key, event in sorted(self.activities.items(), key=lambda item: item[1].occurrence_index):
            semantic = {
                "kind": "occurrence", "stage": event.stage, "occurrence_type": event.occurrence_type,
                "stable_instance_key": event.stable_instance_key, "occurrence_index": event.occurrence_index,
                "operation_type": event.operation_type,
            }
            record_id = _semantic_id("a", semantic)
            _put(records, {
                "kind": "activity", "id": record_id, "types": ["ex:GenerationOccurrence"],
                "attributes": {
                    "ex:occurrenceIndex": event.occurrence_index,
                    "ex:occurrenceStage": event.stage,
                    "ex:occurrenceType": event.occurrence_type,
                    "ex:operationType": event.operation_type,
                    "ex:stableInstanceKey": event.stable_instance_key,
                },
            })
            activities[key] = record_id
            association_key = {"activity": record_id, "agent": agent_id, "role": "ex:generator", "ordinal": 0}
            association_id = _semantic_id("as", association_key)
            _put(records, {"kind": "association", "id": association_id, **association_key})

        bridge_to_entity = {
            key: outcome_entities[event.prior_outcome_key]
            for key, event in self.bridges.items()
        }
        seen_bindings: set[tuple[str, str, str, str, int]] = set()
        for event in sorted(
            self.bindings,
            key=lambda item: (item.activity_key, item.origin_kind, item.origin_key, item.outcome_key, item.role, item.ordinal),
        ):
            binding_key = (event.origin_kind, event.origin_key, event.activity_key, event.role, event.ordinal)
            if binding_key in seen_bindings:
                raise ValueError(f"duplicate native binding ordinal: {binding_key}")
            seen_bindings.add(binding_key)
            origin_entity = source_entities[event.origin_key] if event.origin_kind == "source" else bridge_to_entity[event.origin_key]
            outcome_entity = outcome_entities[event.outcome_key]
            activity = activities[event.activity_key]
            role = _safe_role(event.role)
            usage_key = {"activity": activity, "entity": origin_entity, "role": role, "ordinal": event.ordinal}
            usage_id = _semantic_id("u", usage_key)
            _put(records, {"kind": "usage", "id": usage_id, **usage_key})
            generation_key = {"entity": outcome_entity, "activity": activity}
            generation_id = _semantic_id("g", generation_key)
            _put(records, {"kind": "generation", "id": generation_id, **generation_key})
            derivation_key = {
                "generated_entity": outcome_entity, "used_entity": origin_entity,
                "activity": activity, "generation": generation_id, "usage": usage_id,
                "role": role, "ordinal": event.ordinal,
            }
            derivation_id = _semantic_id("d", derivation_key)
            _put(records, {"kind": "derivation", "id": derivation_id, **derivation_key})
        return sorted_records(list(records.values()))

    def qualified_provo(self) -> bytes:
        return serialize_qualified_provo(self.normalized_records())


def _literal(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def serialize_qualified_provo(records: list[dict[str, Any]]) -> bytes:
    lines = [
        "@prefix ex: <https://example.org/w3c-prov-projection-v1#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]
    for record in sorted_records(records):
        kind = record["kind"]
        if kind in {"entity", "activity", "agent"}:
            base_type = {"entity": "prov:Entity", "activity": "prov:Activity", "agent": "prov:Agent"}[kind]
            types = [base_type, *record["types"]]
            predicates = ["a " + ", ".join(types)]
            predicates.extend(f"{key} {_literal(value)}" for key, value in sorted(record["attributes"].items()))
            lines.append(record["id"] + " " + " ;\n  ".join(predicates) + " .")
        elif kind == "usage":
            lines.append(f"{record['activity']} prov:qualifiedUsage {record['id']} .")
            lines.append(
                f"{record['id']} a prov:Usage ; prov:entity {record['entity']} ; "
                f"prov:hadRole {record['role']} ; ex:relationOrdinal {record['ordinal']} ."
            )
        elif kind == "generation":
            lines.append(f"{record['entity']} prov:qualifiedGeneration {record['id']} .")
            lines.append(f"{record['id']} a prov:Generation ; prov:activity {record['activity']} .")
        elif kind == "derivation":
            lines.append(f"{record['generated_entity']} prov:qualifiedDerivation {record['id']} .")
            lines.append(
                f"{record['id']} a prov:Derivation ; prov:entity {record['used_entity']} ; "
                f"prov:hadActivity {record['activity']} ; prov:hadGeneration {record['generation']} ; "
                f"prov:hadUsage {record['usage']} ; prov:hadRole {record['role']} ; "
                f"ex:relationOrdinal {record['ordinal']} ."
            )
        elif kind == "association":
            lines.append(f"{record['activity']} prov:qualifiedAssociation {record['id']} .")
            lines.append(
                f"{record['id']} a prov:Association ; prov:agent {record['agent']} ; "
                f"prov:hadRole {record['role']} ; ex:relationOrdinal {record['ordinal']} ."
            )
        else:
            raise ValueError(f"unknown native record kind: {kind}")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")

