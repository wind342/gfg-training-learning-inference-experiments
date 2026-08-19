from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from generation_relation_core.snapshots import ValidatedSnapshot

from .record_model import sorted_records, validate_normalized_records


_ROLE_RE = re.compile(r"^([a-z][a-z0-9-]*)\|ordinal=([0-9]{4})$")


def _semantic_id(prefix: str, key: dict[str, Any]) -> str:
    encoded = json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"ex:{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def _role(value: str) -> tuple[str, int]:
    match = _ROLE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"relation role does not satisfy frozen profile: {value}")
    return "ex:" + match.group(1).replace("-", "_"), int(match.group(2))


def _add(target: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    prior = target.get(record["id"])
    if prior is not None and prior != record:
        raise ValueError(f"duplicate semantic key with different payload: {record['id']}")
    target[record["id"]] = record


def project_snapshot(snapshot: ValidatedSnapshot) -> list[dict[str, Any]]:
    """Project one and only one input: a validated Core v3 Snapshot."""

    if not isinstance(snapshot, ValidatedSnapshot):
        raise TypeError("candidate input must be ValidatedSnapshot")
    tables = snapshot.tables
    records: dict[str, dict[str, Any]] = {}
    entity_by_core: dict[str, str] = {}
    activity_by_core: dict[str, str] = {}
    agent_by_manifest: dict[str, str] = {}

    for row in tables.source_information_records:
        payload = row["source_payload"]
        key = {
            "kind": "source",
            "source_identity": row["source_identity"],
            "source_granularity": row["source_granularity"],
            "domain_type": payload["domain_type"],
            "stable_domain_identity": payload["stable_domain_identity"],
        }
        record_id = _semantic_id("e", key)
        record = {
            "kind": "entity",
            "id": record_id,
            "types": ["ex:SourceInformation"],
            "attributes": {
                "ex:domainType": key["domain_type"],
                "ex:sourceGranularity": key["source_granularity"],
                "ex:sourceIdentity": key["source_identity"],
                "ex:stableDomainIdentity": key["stable_domain_identity"],
            },
        }
        _add(records, record)
        entity_by_core[row["source_information_id"]] = record_id

    for row in tables.perceptual_support_records:
        payload = row["support_payload"]
        key = {
            "kind": "support",
            "result_category": payload["result_category"],
            "result_identity": payload["result_identity"],
        }
        record_id = _semantic_id("e", key)
        record = {
            "kind": "entity",
            "id": record_id,
            "types": ["ex:GeneratedSupport"],
            "attributes": {
                "ex:resultCategory": key["result_category"],
                "ex:resultIdentity": key["result_identity"],
            },
        }
        _add(records, record)
        entity_by_core[row["support_id"]] = record_id

    for row in tables.explicit_dispositions:
        key = {
            "kind": "disposition",
            "disposition_category": row["core_disposition_category"],
            "reason_code": row["domain_reason_code"],
        }
        record_id = _semantic_id("e", key)
        record = {
            "kind": "entity",
            "id": record_id,
            "types": ["ex:DispositionRecord"],
            "attributes": {
                "ex:dispositionCategory": key["disposition_category"],
                "ex:reasonCode": key["reason_code"],
            },
        }
        _add(records, record)
        entity_by_core[row["disposition_id"]] = record_id

    for row in tables.generator_manifests:
        key = {
            "kind": "software-agent",
            "generator_name": row["generator_name"],
            "generator_version": row["generator_version"],
            "code_identity": row["generator_code_hash"],
        }
        record_id = _semantic_id("ag", key)
        record = {
            "kind": "agent",
            "id": record_id,
            "types": ["prov:SoftwareAgent"],
            "attributes": {
                "ex:codeIdentity": key["code_identity"],
                "ex:generatorName": key["generator_name"],
                "ex:generatorVersion": key["generator_version"],
            },
        }
        _add(records, record)
        agent_by_manifest[row["generator_manifest_id"]] = record_id

    for row in tables.generation_occurrences:
        payload = row["occurrence_payload"]
        key = {
            "kind": "occurrence",
            "stage": row["occurrence_stage"],
            "occurrence_type": row["occurrence_type"],
            "stable_instance_key": row["stable_instance_key"],
            "occurrence_index": row["occurrence_index"],
            "operation_type": payload["operation_type"],
        }
        record_id = _semantic_id("a", key)
        record = {
            "kind": "activity",
            "id": record_id,
            "types": ["ex:GenerationOccurrence"],
            "attributes": {
                "ex:occurrenceIndex": key["occurrence_index"],
                "ex:occurrenceStage": key["stage"],
                "ex:occurrenceType": key["occurrence_type"],
                "ex:operationType": key["operation_type"],
                "ex:stableInstanceKey": key["stable_instance_key"],
            },
        }
        _add(records, record)
        activity_by_core[row["generation_occurrence_id"]] = record_id
        agent_id = agent_by_manifest[row["generator_manifest_id"]]
        association_key = {"activity": record_id, "agent": agent_id, "role": "ex:generator", "ordinal": 0}
        association_id = _semantic_id("as", association_key)
        _add(records, {
            "kind": "association", "id": association_id, "activity": record_id,
            "agent": agent_id, "role": "ex:generator", "ordinal": 0,
        })

    generated_origin_to_entity: dict[str, str] = {}
    for row in tables.generated_origins:
        if row["origin_type"] != "prior-generated-support":
            raise ValueError(f"unsupported GeneratedOrigin type: {row['origin_type']}")
        prior_support_id = row["origin_payload"].get("prior_support_id")
        if prior_support_id not in entity_by_core:
            raise ValueError(f"GeneratedOrigin prior support missing: {row['generated_origin_id']}")
        generated_origin_to_entity[row["generated_origin_id"]] = entity_by_core[prior_support_id]

    generation_by_pair: dict[tuple[str, str], str] = {}
    projected_binding_count = 0
    for row in tables.generation_bindings:
        origin = row["origin_reference"]
        if origin["kind"] == "registered_source":
            origin_entity = entity_by_core[origin["source_information_id"]]
        elif origin["kind"] == "generated_origin":
            origin_entity = generated_origin_to_entity[origin["generated_origin_id"]]
        else:
            raise ValueError(f"unknown origin reference: {origin}")
        outcome = row["outcome_reference"]
        outcome_core_id = outcome.get("support_id", outcome.get("disposition_id"))
        outcome_entity = entity_by_core[outcome_core_id]
        activity = activity_by_core[row["generation_occurrence_id"]]
        role, ordinal = _role(row["relation_role"])

        usage_key = {"activity": activity, "entity": origin_entity, "role": role, "ordinal": ordinal}
        usage_id = _semantic_id("u", usage_key)
        _add(records, {
            "kind": "usage", "id": usage_id, "activity": activity, "entity": origin_entity,
            "role": role, "ordinal": ordinal,
        })
        generation_key = {"entity": outcome_entity, "activity": activity}
        generation_id = _semantic_id("g", generation_key)
        prior_generation = generation_by_pair.get((outcome_entity, activity))
        if prior_generation not in (None, generation_id):
            raise ValueError("generation identity conflict")
        generation_by_pair[(outcome_entity, activity)] = generation_id
        _add(records, {
            "kind": "generation", "id": generation_id, "entity": outcome_entity, "activity": activity,
        })
        derivation_key = {
            "generated_entity": outcome_entity,
            "used_entity": origin_entity,
            "activity": activity,
            "generation": generation_id,
            "usage": usage_id,
            "role": role,
            "ordinal": ordinal,
        }
        derivation_id = _semantic_id("d", derivation_key)
        _add(records, {"kind": "derivation", "id": derivation_id, **derivation_key})
        projected_binding_count += 1

    result = sorted_records(list(records.values()))
    if projected_binding_count != len(tables.generation_bindings):
        raise ValueError("missing binding projection")
    violations = validate_normalized_records(result)
    if violations:
        raise ValueError(f"candidate constraint violation: {violations}")
    return result

