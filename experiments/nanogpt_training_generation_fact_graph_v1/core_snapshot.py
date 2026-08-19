from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from typing import Any

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.canonical import canonical_bytes
from generation_relation_core.entities import (
    environment_record,
    evidence_link,
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
    build_snapshot,
    implementation_hashes,
    validate_snapshot,
)


DOMAIN_SCOPE = "nanogpt-training-generation-fact-graph-v1"
EVIDENCE_AUTHORITY = "actual-synchronous-pytorch-dispatch-receipt-v1"


def training_tensor_predicate(
    support_payload: dict,
    query_payload: dict,
    predicate: str,
) -> bool:
    return (
        predicate == "membership"
        and support_payload["output_ref"] == query_payload["output_ref"]
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile() -> tuple[dict, dict, PredicateRegistry]:
    space = support_space(
        domain_scope_id=DOMAIN_SCOPE,
        support_space_name="training-runtime-tensor-outcome-v1",
        support_payload_schema={
            "type": "object",
            "required": ["output_ref", "tensor"],
            "properties": {
                "output_ref": {"type": "string"},
                "tensor": {"type": "object"},
                "content_sha256": {"type": "string"},
            },
            "additionalProperties": True,
        },
        query_payload_schema={
            "type": "object",
            "required": ["output_ref"],
            "properties": {"output_ref": {"type": "string"}},
            "additionalProperties": False,
        },
        normalization_rule="canonical JSON; runtime tensor references are exact strings",
    )
    profile = predicate_profile(
        domain_scope_id=DOMAIN_SCOPE,
        support_space_id=space["support_space_id"],
        predicate_kind="exact_runtime_tensor_reference",
        supported_predicates=["membership"],
        predicate_authority="experiment-local-frozen-capture-contract",
        authorized=True,
        implementation_module=(
            "experiments.nanogpt_training_generation_fact_graph_v1.core_snapshot"
        ),
        implementation_symbol="training_tensor_predicate",
        predicate_implementation_sha256=implementation_sha256(
            training_tensor_predicate
        ),
        normalization_rule="exact UTF-8 output_ref equality",
        result_ordering_rule="support_id ascending",
    )
    registry = PredicateRegistry(
        [space],
        [profile],
        {profile["predicate_profile_id"]: training_tensor_predicate},
    )
    return space, profile, registry


def build_core_snapshot(
    capture: dict[str, Any],
    *,
    trainer_commit: str,
    torch_version: str,
    cuda_version: str,
    device_name: str,
    source_code_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    space, profile, registry = _profile()
    code_hash = _sha256_file(source_code_path)
    trainer_commit_sha256 = hashlib.sha256(
        trainer_commit.encode("utf-8")
    ).hexdigest()
    environment = environment_record(
        runtime_name="CPython+PyTorch+CUDA",
        runtime_version=f"{platform.python_version()}+torch-{torch_version}",
        operating_system=f"{platform.system()}-{platform.release()}",
        dependency_hashes={
            "capture_code": code_hash,
            "cuda_runtime": hashlib.sha256(cuda_version.encode("utf-8")).hexdigest(),
            "device": hashlib.sha256(device_name.encode("utf-8")).hexdigest(),
            "nanoGPT_commit": trainer_commit_sha256,
        },
    )
    manifest = generator_manifest(
        generator_name="nanoGPT/PyTorch actual training dispatch collector",
        generator_version=trainer_commit,
        generator_code_hash=code_hash,
        supported_support_space_ids=[space["support_space_id"]],
        supported_predicate_profile_ids=[profile["predicate_profile_id"]],
        supported_operations=[
            "aten_dispatch",
            "backward_gradient_snapshot",
            "gradient_clip",
            "optimizer_update",
        ],
        authorized_evidence_authorities=[EVIDENCE_AUTHORITY],
        dependency_hashes=[trainer_commit_sha256],
    )

    input_ref_set = {
        ref
        for event in capture["events"]
        for ref in event["input_refs"]
    }
    source_rows: list[dict] = []
    source_by_ref: dict[str, dict] = {}
    for row in capture["sources"]:
        if row["source_ref"] not in input_ref_set:
            continue
        source = source_information(
            domain_scope_id=DOMAIN_SCOPE,
            source_identity=row["source_ref"],
            source_parent_id=None,
            source_granularity="runtime_tensor_or_declared_operand",
            source_payload=row["source_payload"],
        )
        source_rows.append(source)
        source_by_ref[row["source_ref"]] = source

    occurrences: list[dict] = []
    generated: list[dict] = []
    supports: list[dict] = []
    bindings: list[dict] = []
    evidence_records: list[dict] = []
    evidence_links: list[dict] = []
    operation_results: list[dict] = []
    support_by_ref: dict[str, dict] = {}
    generated_by_ref: dict[str, dict] = {}
    facts: list[dict[str, Any]] = []

    for event in capture["events"]:
        occurrence = generation_occurrence(
            domain_scope_id=DOMAIN_SCOPE,
            generator_manifest_id=manifest["generator_manifest_id"],
            occurrence_stage=event["phase"],
            occurrence_type=event["event_kind"],
            stable_instance_key=(
                f"{capture['run_id']}:step:{event['step']}:"
                f"micro:{event['micro_step']}:event:{event['ordinal']}"
            ),
            occurrence_index=event["ordinal"],
            transform_reference=event["transform_reference"],
            occurrence_payload={
                "micro_step": event["micro_step"],
                "receipt_payload": event["receipt_payload"],
                "run_id": capture["run_id"],
                "step": event["step"],
            },
        )
        occurrences.append(occurrence)

        event_supports: list[tuple[dict[str, Any], dict]] = []
        for output in event["outputs"]:
            support = perceptual_support(
                domain_scope_id=DOMAIN_SCOPE,
                support_space_id=space["support_space_id"],
                support_payload=output,
                predicate_profile_id=profile["predicate_profile_id"],
            )
            supports.append(support)
            support_by_ref[output["output_ref"]] = support
            event_supports.append((output, support))

        event_bindings: list[dict] = []
        event_evidence: list[dict] = []
        newly_generated: list[dict] = []
        for input_ref, input_role in zip(
            event["input_refs"],
            event["input_roles"],
            strict=True,
        ):
            source = source_by_ref.get(input_ref)
            if source is not None:
                origin_reference = {
                    "kind": "registered_source",
                    "source_information_id": source["source_information_id"],
                }
                u = {
                    "kind": "registered_source",
                    "source_ref": input_ref,
                    "source_information_id": source["source_information_id"],
                }
            else:
                prior_support = support_by_ref.get(input_ref)
                if prior_support is None:
                    raise RuntimeError(f"UNRESOLVED_RUNTIME_INPUT:{input_ref}")
                origin = generated_by_ref.get(input_ref)
                if origin is None:
                    origin = generated_origin(
                        domain_scope_id=DOMAIN_SCOPE,
                        generator_manifest_id=manifest["generator_manifest_id"],
                        origin_type="prior_training_tensor_outcome",
                        origin_payload={
                            "output_ref": input_ref,
                            "support_id": prior_support["support_id"],
                        },
                    )
                    generated_by_ref[input_ref] = origin
                    generated.append(origin)
                    newly_generated.append(origin)
                origin_reference = {
                    "kind": "generated_origin",
                    "generated_origin_id": origin["generated_origin_id"],
                }
                u = {
                    "generated_origin_id": origin["generated_origin_id"],
                    "kind": "generated_origin",
                    "source_ref": input_ref,
                    "upstream_support_id": prior_support["support_id"],
                }

            for output, support in event_supports:
                material = relation_material(
                    domain_scope_id=DOMAIN_SCOPE,
                    origin_reference=origin_reference,
                    generation_occurrence_id=occurrence[
                        "generation_occurrence_id"
                    ],
                    outcome_reference={
                        "kind": "support",
                        "support_id": support["support_id"],
                    },
                    relation_role=input_role,
                )
                evidence = relation_evidence_for_material(
                    material,
                    artifact_locator=(
                        "candidate://relation_materials.jsonl#sha256="
                        f"{hashlib.sha256(canonical_bytes(material)).hexdigest()}"
                    ),
                    evidence_authority=EVIDENCE_AUTHORITY,
                    extraction_method=(
                        "synchronous completed ATen dispatch or explicit training "
                        "boundary receipt"
                    ),
                    extraction_code_hash=code_hash,
                    environment_hash=environment["environment_payload_sha256"],
                    related_record_ids=[
                        origin_reference.get(
                            "source_information_id",
                            origin_reference.get("generated_origin_id"),
                        ),
                        occurrence["generation_occurrence_id"],
                        support["support_id"],
                    ],
                )
                binding = generation_binding(
                    domain_scope_id=DOMAIN_SCOPE,
                    origin_reference=origin_reference,
                    generation_occurrence_id=occurrence[
                        "generation_occurrence_id"
                    ],
                    outcome_reference={
                        "kind": "support",
                        "support_id": support["support_id"],
                    },
                    relation_role=input_role,
                    evidence_ids=[evidence["evidence_id"]],
                )
                link = evidence_link(
                    evidence_id=evidence["evidence_id"],
                    subject_type="generation_binding",
                    subject_id=binding["generation_binding_id"],
                    evidence_role="primary_generation_relation",
                )
                evidence_records.append(evidence)
                evidence_links.append(link)
                bindings.append(binding)
                event_bindings.append(binding)
                event_evidence.append(evidence)
                facts.append(
                    {
                        "binding_id": binding["generation_binding_id"],
                        "event_ordinal": event["ordinal"],
                        "micro_step": event["micro_step"],
                        "rho": input_role,
                        "step": event["step"],
                        "tau": event["transform_reference"],
                        "u": u,
                        "omega_bar": {
                            "generation_occurrence_id": occurrence[
                                "generation_occurrence_id"
                            ],
                            "occurrence_index": event["ordinal"],
                            "phase": event["phase"],
                            "stable_instance_key": occurrence[
                                "stable_instance_key"
                            ],
                        },
                        "z": {
                            "output_ref": output["output_ref"],
                            "support_id": support["support_id"],
                            "tensor": output["tensor"],
                        },
                    }
                )

        operation_results.append(
            generator_operation_result(
                generator_manifest_id=manifest["generator_manifest_id"],
                operation_name=event["event_kind"],
                produced_entity_ids=[
                    occurrence["generation_occurrence_id"],
                    *[row["support_id"] for _output, row in event_supports],
                    *[
                        row["generated_origin_id"]
                        for row in newly_generated
                    ],
                    *[
                        row["generation_binding_id"]
                        for row in event_bindings
                    ],
                ],
                evidence_ids=[row["evidence_id"] for row in event_evidence],
            )
        )

    legacy_source, legacy_occurrence = derive_legacy_projections(
        source_rows,
        occurrences,
        bindings,
        validate_schema=False,
    )
    tables = CoreV3Tables(
        source_information_records=source_rows,
        generation_occurrences=occurrences,
        generated_origins=generated,
        perceptual_support_records=supports,
        generation_bindings=bindings,
        support_space_records=[space],
        predicate_profiles=[profile],
        evidence_records=evidence_records,
        evidence_links=evidence_links,
        generator_manifests=[manifest],
        generator_operation_results=operation_results,
        environment_records=[environment],
        legacy_source_binding_projections=legacy_source,
        legacy_occurrence_binding_projections=legacy_occurrence,
    )
    snapshot = build_snapshot(
        tables,
        registry,
        expected_implementation_hashes=implementation_hashes(),
    )
    validation = validate_snapshot(
        snapshot,
        registry,
        expected_implementation_hashes=implementation_hashes(),
    )
    snapshot_payload = {
        "record": snapshot.record,
        "tables": {
            name: getattr(snapshot.tables, name)
            for name in snapshot.tables.__dataclass_fields__
        },
    }
    graph_payload = {
        "facts": facts,
        "graph_id": hashlib.sha256(canonical_bytes(facts)).hexdigest(),
        "run_id": capture["run_id"],
        "snapshot_id": snapshot.snapshot_id,
        "validation": {
            "core_snapshot_validated": True,
            "relation_evidence_resolution_count": len(
                validation.relation_evidence
            ),
        },
    }
    return snapshot_payload, graph_payload
