from __future__ import annotations

import platform
from collections import defaultdict
from typing import Any

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.canonical import canonical_bytes
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
from generation_relation_core.snapshots import CoreV3Tables, build_snapshot

from ..canonical_graph import canonical_hash, content_id


ADAPTER_AUTHORITY = "executable-generation-fact-core-adapter-v1"


def native_key_membership(
    support: dict[str, Any], query: dict[str, Any], predicate: str
) -> bool:
    return (
        predicate == "membership"
        and support["native_support_key"] == query["native_support_key"]
    )


def _coordinates(fact: dict[str, Any]) -> dict[str, Any]:
    if "coordinates" in fact:
        return fact["coordinates"]
    return fact["semantic_projection"]["coordinates"]


def _native_fact_id(fact: dict[str, Any]) -> str:
    return fact["fact_id"]


def _native_result_id(fact: dict[str, Any]) -> str:
    return fact.get("result_id", fact.get("support_id", fact["fact_id"]))


def _native_occurrence_id(fact: dict[str, Any]) -> str:
    coordinates = _coordinates(fact)
    omega = coordinates["omega_bar"]
    return (
        omega.get("concrete_occurrence_id")
        or fact.get("occurrence_id")
        or omega.get("concrete_occurrence_instance_id")
        or omega.get("core_content_occurrence_id")
    )


def _tau_value(fact: dict[str, Any]) -> Any:
    return _coordinates(fact)["tau"]


def _rho_value(fact: dict[str, Any]) -> Any:
    return _coordinates(fact)["rho"]


def _role(fact: dict[str, Any]) -> str:
    value = _rho_value(fact)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("role"), str):
        return value["role"]
    return "native-role:" + canonical_hash(value)


def _is_disposition(fact: dict[str, Any]) -> bool:
    z = _coordinates(fact)["z"]
    return z.get("kind") == "ExplicitDisposition"


def _native_prior_support_id(fact: dict[str, Any]) -> str | None:
    u = _coordinates(fact)["u"]
    if u.get("kind") != "generated_origin":
        return None
    return u.get("prior_support_id")


def _sort_table(rows: list[dict], id_field: str) -> list[dict]:
    return sorted(rows, key=lambda row: row[id_field])


def build_core_snapshot_from_atomic_facts(
    *,
    atomic_fact_bundle: dict[str, Any],
    execution_run_id: str,
    domain_scope_id: str,
    generator_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    facts = list(atomic_fact_bundle["facts"])
    if atomic_fact_bundle["execution_run_id"] != execution_run_id:
        raise ValueError("ATOMIC_FACT_RUN_SCOPE_MISMATCH")
    fact_ids = [_native_fact_id(row) for row in facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("ATOMIC_FACT_ID_DUPLICATED")

    space = support_space(
        domain_scope_id=domain_scope_id,
        support_space_name=f"{generator_name}-native-outcome",
        support_payload_schema={
            "type": "object",
            "additionalProperties": False,
            "required": [
                "native_support_key",
                "native_fact_id",
                "native_result_id",
                "native_z",
            ],
            "properties": {
                "native_support_key": {"type": "string"},
                "native_fact_id": {"type": "string"},
                "native_result_id": {"type": "string"},
                "native_z": {},
            },
        },
        query_payload_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["native_support_key"],
            "properties": {"native_support_key": {"type": "string"}},
        },
        normalization_rule="exact native fact identity and canonical JSON payload",
    )
    profile = predicate_profile(
        domain_scope_id=domain_scope_id,
        support_space_id=space["support_space_id"],
        predicate_kind="native_key_membership",
        supported_predicates=["membership"],
        predicate_authority=ADAPTER_AUTHORITY,
        authorized=True,
        implementation_module=__name__,
        implementation_symbol=native_key_membership.__name__,
        predicate_implementation_sha256=implementation_sha256(
            native_key_membership
        ),
        normalization_rule="exact string equality",
        result_ordering_rule="generation_binding_id ascending",
    )
    registry = PredicateRegistry(
        [space], [profile], {profile["predicate_profile_id"]: native_key_membership}
    )
    environment = environment_record(
        runtime_name="CPython",
        runtime_version=platform.python_version(),
        operating_system=platform.platform(),
        dependency_hashes={
            "native_atomic_fact_bundle": canonical_hash(atomic_fact_bundle)
        },
    )
    manifest = generator_manifest(
        generator_name=generator_name,
        generator_version="1",
        generator_code_hash=canonical_hash(
            {
                "adapter": ADAPTER_AUTHORITY,
                "generator": generator_name,
            }
        ),
        supported_support_space_ids=[space["support_space_id"]],
        supported_predicate_profile_ids=[profile["predicate_profile_id"]],
        supported_operations=["adapt_native_atomic_fact_to_core_v3"],
        authorized_evidence_authorities=[ADAPTER_AUTHORITY],
        dependency_hashes=[canonical_hash(atomic_fact_bundle)],
    )

    occurrences_by_native: dict[str, dict] = {}
    fact_native_occurrence: dict[str, str] = {}
    for index, fact in enumerate(
        sorted(facts, key=lambda row: (_native_occurrence_id(row), _native_fact_id(row)))
    ):
        native_occurrence_id = _native_occurrence_id(fact)
        if not native_occurrence_id:
            raise ValueError("NATIVE_OCCURRENCE_ID_MISSING")
        fact_native_occurrence[_native_fact_id(fact)] = native_occurrence_id
        if native_occurrence_id in occurrences_by_native:
            existing = occurrences_by_native[native_occurrence_id]
            if existing["transform_reference"] != _tau_value(fact):
                raise ValueError("NATIVE_OCCURRENCE_TRANSFORM_CONFLICT")
            continue
        occurrences_by_native[native_occurrence_id] = generation_occurrence(
            domain_scope_id=domain_scope_id,
            generator_manifest_id=manifest["generator_manifest_id"],
            occurrence_stage=generator_name,
            occurrence_type="native_atomic_fact_occurrence",
            stable_instance_key=(
                f"{execution_run_id}:{native_occurrence_id}"
            ),
            occurrence_index=len(occurrences_by_native),
            transform_reference=_tau_value(fact),
            occurrence_payload={
                "execution_run_id": execution_run_id,
                "native_occurrence_id": native_occurrence_id,
                "native_omega_bar": _coordinates(fact)["omega_bar"],
            },
        )

    support_by_native: dict[str, dict] = {}
    disposition_by_fact: dict[str, dict] = {}
    for fact in facts:
        fact_id = _native_fact_id(fact)
        result_id = _native_result_id(fact)
        z = _coordinates(fact)["z"]
        if _is_disposition(fact):
            disposition_by_fact[fact_id] = explicit_disposition(
                domain_scope_id=domain_scope_id,
                core_disposition_category="suppressed",
                domain_reason_code=str(z.get("value", z.get("reason", "NATIVE_DISPOSITION"))),
                disposition_payload={
                    "native_fact_id": fact_id,
                    "native_result_id": result_id,
                    "native_z": z,
                },
            )
        else:
            support = perceptual_support(
                domain_scope_id=domain_scope_id,
                support_space_id=space["support_space_id"],
                support_payload={
                    "native_support_key": result_id,
                    "native_fact_id": fact_id,
                    "native_result_id": result_id,
                    "native_z": z,
                },
                predicate_profile_id=profile["predicate_profile_id"],
            )
            support_by_native[fact.get("support_id", result_id)] = support
            support_by_native[result_id] = support

    sources: list[dict] = []
    generated: list[dict] = []
    bindings: list[dict] = []
    evidence_records: list[dict] = []
    evidence_links: list[dict] = []
    operations: list[dict] = []
    native_identities: dict[str, dict[str, Any]] = {}
    fact_to_binding: dict[str, str] = {}
    native_occurrence_to_core: dict[str, str] = {}
    core_outcome_by_fact: dict[str, dict] = {}

    for native_id, occurrence in occurrences_by_native.items():
        native_occurrence_to_core[native_id] = occurrence[
            "generation_occurrence_id"
        ]

    for fact in sorted(facts, key=_native_fact_id):
        fact_id = _native_fact_id(fact)
        result_id = _native_result_id(fact)
        coordinates = _coordinates(fact)
        prior_support = _native_prior_support_id(fact)
        if prior_support is None:
            source = source_information(
                domain_scope_id=domain_scope_id,
                source_identity=f"{execution_run_id}:{fact_id}:u",
                source_parent_id=None,
                source_granularity="native_atomic_fact_origin",
                source_payload={
                    "native_fact_id": fact_id,
                    "native_u": coordinates["u"],
                },
            )
            sources.append(source)
            origin_reference = {
                "kind": "registered_source",
                "source_information_id": source["source_information_id"],
            }
        else:
            support = support_by_native.get(prior_support)
            if support is None:
                raise ValueError(
                    f"NATIVE_GENERATED_ORIGIN_SUPPORT_MISSING:{prior_support}"
                )
            origin = generated_origin(
                domain_scope_id=domain_scope_id,
                generator_manifest_id=manifest["generator_manifest_id"],
                origin_type="native_generated_origin",
                origin_payload={
                    "prior_support_id": support["support_id"],
                    "native_prior_support_id": prior_support,
                    "native_u": coordinates["u"],
                    "native_consumer_fact_id": fact_id,
                },
            )
            generated.append(origin)
            origin_reference = {
                "kind": "generated_origin",
                "generated_origin_id": origin["generated_origin_id"],
            }
        if _is_disposition(fact):
            outcome = disposition_by_fact[fact_id]
            outcome_reference = {
                "kind": "disposition",
                "disposition_id": outcome["disposition_id"],
            }
        else:
            support = support_by_native[result_id]
            outcome = support
            outcome_reference = {
                "kind": "support",
                "support_id": support["support_id"],
            }
        core_outcome_by_fact[fact_id] = outcome_reference
        occurrence = occurrences_by_native[fact_native_occurrence[fact_id]]
        material = relation_material(
            domain_scope_id=domain_scope_id,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=_role(fact),
        )
        evidence = relation_evidence_for_material(
            material,
            artifact_locator=(
                "candidate://relation_materials.jsonl#sha256="
                + canonical_hash(material)
            ),
            evidence_authority=ADAPTER_AUTHORITY,
            extraction_method="validated_native_atomic_fact_adapter",
            extraction_code_hash=canonical_hash(
                {"adapter": ADAPTER_AUTHORITY, "version": 1}
            ),
            environment_hash=environment["environment_payload_sha256"],
            related_record_ids=[
                origin_reference.get(
                    "source_information_id",
                    origin_reference.get("generated_origin_id"),
                ),
                occurrence["generation_occurrence_id"],
                outcome_reference.get(
                    "support_id", outcome_reference.get("disposition_id")
                ),
            ],
        )
        binding = generation_binding(
            domain_scope_id=domain_scope_id,
            origin_reference=origin_reference,
            generation_occurrence_id=occurrence["generation_occurrence_id"],
            outcome_reference=outcome_reference,
            relation_role=_role(fact),
            evidence_ids=[evidence["evidence_id"]],
        )
        link = evidence_link(
            evidence_id=evidence["evidence_id"],
            subject_type="generation_binding",
            subject_id=binding["generation_binding_id"],
            evidence_role="primary_generation_relation",
        )
        operation = generator_operation_result(
            generator_manifest_id=manifest["generator_manifest_id"],
            operation_name="adapt_native_atomic_fact_to_core_v3",
            produced_entity_ids=[binding["generation_binding_id"]],
            evidence_ids=[evidence["evidence_id"]],
        )
        bindings.append(binding)
        evidence_records.append(evidence)
        evidence_links.append(link)
        operations.append(operation)
        fact_to_binding[fact_id] = binding["generation_binding_id"]
        native_identities[binding["generation_binding_id"]] = {
            "native_fact_id": fact_id,
            "native_result_id": result_id,
            "native_occurrence_id": fact_native_occurrence[fact_id],
            "native_fact": fact,
            "coordinates_exact": {
                "u": coordinates["u"],
                "tau": coordinates["tau"],
                "omega_bar": coordinates["omega_bar"],
                "z": coordinates["z"],
                "rho": coordinates["rho"],
            },
        }

    occurrence_rows = list(occurrences_by_native.values())
    support_rows = {
        row["support_id"]: row for row in support_by_native.values()
    }
    disposition_rows = list(disposition_by_fact.values())
    legacy_source, legacy_occurrence = derive_legacy_projections(
        sources,
        occurrence_rows,
        bindings,
        validate_schema=False,
    )
    tables = CoreV3Tables(
        source_information_records=_sort_table(
            sources, "source_information_id"
        ),
        generation_occurrences=_sort_table(
            occurrence_rows, "generation_occurrence_id"
        ),
        generated_origins=_sort_table(generated, "generated_origin_id"),
        perceptual_support_records=_sort_table(
            list(support_rows.values()), "support_id"
        ),
        explicit_dispositions=_sort_table(
            disposition_rows, "disposition_id"
        ),
        generation_bindings=_sort_table(bindings, "generation_binding_id"),
        support_space_records=[space],
        predicate_profiles=[profile],
        evidence_records=_sort_table(evidence_records, "evidence_id"),
        evidence_links=_sort_table(evidence_links, "evidence_link_id"),
        generator_manifests=[manifest],
        generator_operation_results=_sort_table(
            operations, "operation_result_id"
        ),
        environment_records=[environment],
        legacy_source_binding_projections=legacy_source,
        legacy_occurrence_binding_projections=legacy_occurrence,
    )
    snapshot = build_snapshot(tables, registry)
    snapshot_input = {
        "snapshot": snapshot,
        "execution_run_id": execution_run_id,
        "native_binding_identities": native_identities,
    }
    mapping = {
        "fact_to_binding": fact_to_binding,
        "occurrence_to_core_occurrence": native_occurrence_to_core,
        "native_fact_count": len(facts),
        "core_binding_count": len(bindings),
        "coordinate_mapping_exact": len(facts) == len(bindings),
        "adapter_mapping_sha256": canonical_hash(
            {
                "fact_to_binding": fact_to_binding,
                "occurrence_to_core_occurrence": native_occurrence_to_core,
            }
        ),
    }
    return snapshot_input, mapping


def normalize_relation_store(
    *,
    native_sidecar: dict[str, Any],
    mapping: dict[str, Any],
    require_complete: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fact_map = mapping["fact_to_binding"]
    occurrence_map = mapping["occurrence_to_core_occurrence"]
    normalized: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    evidence_ids = {
        row["evidence_id"] for row in native_sidecar.get("evidence", [])
    }
    for relation in native_sidecar["relations"]:
        endpoint_map = (
            fact_map if relation["endpoint_level"] == "fact" else occurrence_map
        )
        source = endpoint_map.get(relation["source_id"])
        target = endpoint_map.get(relation["target_id"])
        missing = []
        if source is None:
            missing.append("source")
        if target is None:
            missing.append("target")
        if missing:
            unmapped.append(
                {
                    "relation_id": relation["relation_id"],
                    "relation_type": relation["relation_type"],
                    "endpoint_level": relation["endpoint_level"],
                    "missing_endpoints": missing,
                    "source_id": relation["source_id"],
                    "target_id": relation["target_id"],
                }
            )
            continue
        if not set(relation["evidence_refs"]) <= evidence_ids:
            raise ValueError("NATIVE_RELATION_EVIDENCE_MISSING")
        normalized.append(
            {
                **relation,
                "source_id": source,
                "target_id": target,
                "native_relation": relation,
                "primitive_or_derived": "primitive",
                "rule_id": None,
                "input_relation_refs": [],
            }
        )
    audit = {
        "native_relation_count": len(native_sidecar["relations"]),
        "normalized_relation_count": len(normalized),
        "unmapped_relation_count": len(unmapped),
        "unmapped_relations": unmapped,
        "coverage_exact": not unmapped,
    }
    if require_complete and unmapped:
        raise ValueError(
            f"OCCURRENCE_ENDPOINT_WITHOUT_FACT_NODE:{len(unmapped)}"
        )
    store_id = content_id(
        "gfrstore1_",
        {
            "execution_run_id": native_sidecar["execution_run_id"],
            "native_relation_ids": sorted(
                row["relation_id"] for row in native_sidecar["relations"]
            ),
        },
    )
    return (
        {
            "relation_store_id": store_id,
            "execution_run_id": native_sidecar["execution_run_id"],
            "relations": normalized,
            "evidence": native_sidecar.get("evidence", []),
            "native_store_schema_version": native_sidecar["schema_version"],
        },
        audit,
    )
