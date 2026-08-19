from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3
from typing import Any

from compat.v2.projections import derive_legacy_projections
from generation_relation_core.entities import (
    environment_record,
    evidence_link,
    generation_binding,
    generation_occurrence,
    generated_origin,
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

from experiments.executable_generation_fact_graph_v2.canonical_graph import (
    content_id,
)
from experiments.executable_generation_fact_graph_v2.endpoint_registry import (
    build_core_occurrence_catalog,
)
from experiments.executable_generation_fact_graph_v2.graph_compiler import (
    compile_executable_generation_fact_graph_v2,
)
from experiments.executable_generation_fact_graph_v2.graph_validator import (
    load_contracts,
    validate_executable_generation_fact_graph_v2,
)

from .common import file_sha256, payload_sha256
from .training_capture import decode_block


AUTHORITY = "actual-synchronous-nanogpt-training-capture-v2"


def object_membership_predicate(
    support_payload: dict[str, Any],
    query_payload: dict[str, Any],
    predicate: str,
) -> bool:
    return (
        predicate == "membership"
        and support_payload["object_id"] == query_payload["object_id"]
    )


def _registry(
    domain_scope_id: str,
) -> tuple[dict[str, Any], dict[str, Any], PredicateRegistry]:
    space = support_space(
        domain_scope_id=domain_scope_id,
        support_space_name="training-object-identity-v1",
        support_payload_schema={
            "type": "object",
            "required": ["object_id", "content_sha256"],
            "properties": {
                "object_id": {"type": "string"},
                "content_sha256": {"type": "string"},
            },
            "additionalProperties": True,
        },
        query_payload_schema={
            "type": "object",
            "required": ["object_id"],
            "properties": {"object_id": {"type": "string"}},
            "additionalProperties": False,
        },
        normalization_rule="exact UTF-8 object identity",
    )
    profile = predicate_profile(
        domain_scope_id=domain_scope_id,
        support_space_id=space["support_space_id"],
        predicate_kind="exact_training_object_identity",
        supported_predicates=["membership"],
        predicate_authority="frozen-training-capture-protocol-v1",
        authorized=True,
        implementation_module=__name__,
        implementation_symbol="object_membership_predicate",
        predicate_implementation_sha256=implementation_sha256(
            object_membership_predicate
        ),
        normalization_rule="exact UTF-8 object identity",
        result_ordering_rule="support_id ascending",
    )
    registry = PredicateRegistry(
        [space],
        [profile],
        {profile["predicate_profile_id"]: object_membership_predicate},
    )
    return space, profile, registry


def _chunk_blocks(
    connection: sqlite3.Connection,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    return [
        decode_block(row[0])
        for row in connection.execute(
            """
            SELECT payload_zlib FROM graph_blocks
            WHERE block_ordinal BETWEEN ? AND ?
            ORDER BY block_ordinal
            """,
            (start, end),
        )
    ]


def _validate_chunk(
    *,
    run_id: str,
    chunk_index: int,
    blocks: list[dict[str, Any]],
    capture_code_hash: str,
) -> dict[str, Any]:
    scope = f"{run_id}:core-chunk:{chunk_index:06d}"
    space, profile, registry = _registry(scope)
    environment = environment_record(
        runtime_name="CPython+PyTorch+CUDA",
        runtime_version="frozen-nanoGPT-training-runtime-v1",
        operating_system="Windows-host/CUDA-container-independent-capture",
        dependency_hashes={"capture_code": capture_code_hash},
    )
    occurrence_types = sorted(
        {
            occurrence["occurrence_type"]
            for block in blocks
            for occurrence in block["occurrences"]
        }
    )
    manifest = generator_manifest(
        generator_name="nanoGPT actual long-training GFG capture",
        generator_version="1.0.0",
        generator_code_hash=capture_code_hash,
        supported_support_space_ids=[space["support_space_id"]],
        supported_predicate_profile_ids=[profile["predicate_profile_id"]],
        supported_operations=occurrence_types,
        authorized_evidence_authorities=[AUTHORITY],
        dependency_hashes=[capture_code_hash],
    )

    occurrence_by_native: dict[str, dict[str, Any]] = {}
    for block in blocks:
        for native in block["occurrences"]:
            occurrence_by_native[native["occurrence_id"]] = (
                generation_occurrence(
                    domain_scope_id=scope,
                    generator_manifest_id=manifest[
                        "generator_manifest_id"
                    ],
                    occurrence_stage=native["occurrence_stage"],
                    occurrence_type=native["occurrence_type"],
                    stable_instance_key=native["occurrence_id"],
                    occurrence_index=native["ordinal"],
                    transform_reference=native["transform_reference"],
                    occurrence_payload={
                        **native["payload"],
                        "native_occurrence_id": native["occurrence_id"],
                        "optimizer_step": native["optimizer_step"],
                        "run_id": run_id,
                    },
                )
            )

    source_material: dict[str, dict[str, Any]] = {}
    outcome_material: dict[str, dict[str, Any]] = {}
    for block in blocks:
        for fact_block in block["fact_blocks"]:
            for source in fact_block["sources"]:
                source_material[source["object_id"]] = source
            for outcome in fact_block["outcomes"]:
                outcome_material[outcome["object_id"]] = outcome

    generated_object_ids = set(source_material) & set(outcome_material)
    sources = {
        object_id: source_information(
            domain_scope_id=scope,
            source_identity=object_id,
            source_parent_id=None,
            source_granularity="captured_training_object_instance",
            source_payload={
                "content_sha256": row["content_sha256"],
                "native_object_id": object_id,
            },
        )
        for object_id, row in source_material.items()
        if object_id not in generated_object_ids
    }
    supports = {
        object_id: perceptual_support(
            domain_scope_id=scope,
            support_space_id=space["support_space_id"],
            support_payload={
                "content_sha256": row["content_sha256"],
                "object_id": object_id,
            },
            predicate_profile_id=profile["predicate_profile_id"],
        )
        for object_id, row in outcome_material.items()
    }
    generated_origins = {
        object_id: generated_origin(
            domain_scope_id=scope,
            generator_manifest_id=manifest["generator_manifest_id"],
            origin_type="captured_prior_training_outcome",
            origin_payload={
                "native_object_id": object_id,
                "prior_support_id": supports[object_id]["support_id"],
            },
        )
        for object_id in sorted(generated_object_ids)
    }

    bindings: list[dict[str, Any]] = []
    evidences: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    by_occurrence_bindings: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    by_occurrence_evidence: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    by_occurrence_supports: dict[str, set[str]] = defaultdict(set)
    native_binding_identities: dict[str, dict[str, Any]] = {}
    for block in blocks:
        for fact_block in block["fact_blocks"]:
            occurrence = occurrence_by_native[
                fact_block["occurrence_id"]
            ]
            occurrence_id = occurrence["generation_occurrence_id"]
            for source in fact_block["sources"]:
                native_source_id = source["object_id"]
                if native_source_id in generated_origins:
                    origin_entity = generated_origins[native_source_id]
                    origin_reference = {
                        "kind": "generated_origin",
                        "generated_origin_id": origin_entity[
                            "generated_origin_id"
                        ],
                    }
                    origin_entity_id = origin_entity["generated_origin_id"]
                else:
                    source_row = sources[native_source_id]
                    origin_reference = {
                        "kind": "registered_source",
                        "source_information_id": source_row[
                            "source_information_id"
                        ],
                    }
                    origin_entity_id = source_row["source_information_id"]
                for outcome in fact_block["outcomes"]:
                    support_row = supports[outcome["object_id"]]
                    outcome_reference = {
                        "kind": "support",
                        "support_id": support_row["support_id"],
                    }
                    material = relation_material(
                        domain_scope_id=scope,
                        origin_reference=origin_reference,
                        generation_occurrence_id=occurrence_id,
                        outcome_reference=outcome_reference,
                        relation_role=source["relation_role"],
                    )
                    evidence = relation_evidence_for_material(
                        material,
                        artifact_locator=(
                            "candidate://relation_materials.jsonl#sha256="
                            + payload_sha256(material)
                        ),
                        evidence_authority=AUTHORITY,
                        extraction_method=(
                            "synchronous actual training hook and "
                            "reversible fact-block expansion"
                        ),
                        extraction_code_hash=capture_code_hash,
                        environment_hash=environment[
                            "environment_payload_sha256"
                        ],
                        related_record_ids=[
                            origin_entity_id,
                            occurrence_id,
                            support_row["support_id"],
                        ],
                    )
                    binding = generation_binding(
                        domain_scope_id=scope,
                        origin_reference=origin_reference,
                        generation_occurrence_id=occurrence_id,
                        outcome_reference=outcome_reference,
                        relation_role=source["relation_role"],
                        evidence_ids=[evidence["evidence_id"]],
                    )
                    link = evidence_link(
                        evidence_id=evidence["evidence_id"],
                        subject_type="generation_binding",
                        subject_id=binding["generation_binding_id"],
                        evidence_role="primary_generation_relation",
                    )
                    bindings.append(binding)
                    native_fact_material = {
                        "domain_scope_id": run_id,
                        "occurrence": fact_block["occurrence_id"],
                        "origin": source["object_id"],
                        "outcome": outcome["object_id"],
                        "relation_role": source["relation_role"],
                    }
                    native_fact_id = "fact_" + payload_sha256(
                        native_fact_material
                    )
                    native_binding_identities[
                        binding["generation_binding_id"]
                    ] = {
                        "native_fact": native_fact_material,
                        "native_fact_id": native_fact_id,
                        "native_occurrence_id": fact_block["occurrence_id"],
                    }
                    evidences.append(evidence)
                    links.append(link)
                    by_occurrence_bindings[occurrence_id].append(binding)
                    by_occurrence_evidence[occurrence_id].append(evidence)
                    by_occurrence_supports[occurrence_id].add(
                        support_row["support_id"]
                    )

    operation_results = []
    for occurrence in occurrence_by_native.values():
        occurrence_id = occurrence["generation_occurrence_id"]
        occurrence_bindings = by_occurrence_bindings[occurrence_id]
        occurrence_evidence = by_occurrence_evidence[occurrence_id]
        operation_results.append(
            generator_operation_result(
                generator_manifest_id=manifest["generator_manifest_id"],
                operation_name=occurrence["occurrence_type"],
                produced_entity_ids=[
                    occurrence_id,
                    *sorted(by_occurrence_supports[occurrence_id]),
                    *[
                        row["generation_binding_id"]
                        for row in occurrence_bindings
                    ],
                ],
                evidence_ids=[
                    row["evidence_id"] for row in occurrence_evidence
                ],
            )
        )

    occurrence_rows = list(occurrence_by_native.values())
    source_rows = list(sources.values())
    support_rows = list(supports.values())
    legacy_source, legacy_occurrence = derive_legacy_projections(
        source_rows,
        occurrence_rows,
        bindings,
        validate_schema=False,
    )
    tables = CoreV3Tables(
        source_information_records=source_rows,
        generation_occurrences=occurrence_rows,
        generated_origins=list(generated_origins.values()),
        perceptual_support_records=support_rows,
        generation_bindings=bindings,
        support_space_records=[space],
        predicate_profiles=[profile],
        evidence_records=evidences,
        evidence_links=links,
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

    snapshot_input = {
        "execution_run_id": scope,
        "native_binding_identities": native_binding_identities,
        "snapshot": snapshot,
    }
    catalog = build_core_occurrence_catalog([snapshot_input])
    native_occurrence_ids = set(occurrence_by_native)
    program_order_relations = []
    for block in blocks:
        for edge in block["edges"]:
            if (
                edge["relation_type"] != "program_order"
                or edge["source_id"] not in native_occurrence_ids
                or edge["target_id"] not in native_occurrence_ids
            ):
                continue
            native_relation = {
                "endpoint_level": "occurrence",
                "execution_run_id": scope,
                "relation_id": edge["edge_id"],
                "relation_payload": edge.get("payload", {}),
                "relation_type": "program_order",
                "source_id": edge["source_id"],
                "target_id": edge["target_id"],
            }
            program_order_relations.append(
                {
                    **native_relation,
                    "authority_id": AUTHORITY,
                    "establishment_source": (
                        "synchronous_training_control_flow"
                    ),
                    "evidence_refs": [],
                    "input_relation_refs": [],
                    "native_relation": native_relation,
                    "primitive_or_derived": "primitive",
                    "rule_id": None,
                    "source_endpoint_kind": "occurrence",
                    "target_endpoint_kind": "occurrence",
                }
            )
    relation_store_material = {
        "schema_version": "validated-primitive-relation-store-v2",
        "execution_run_id": scope,
        "relations": sorted(
            program_order_relations, key=lambda row: row["relation_id"]
        ),
        "evidence": [],
    }
    relation_store = {
        **relation_store_material,
        "relation_store_id": content_id(
            "gfrs2_", relation_store_material
        ),
    }
    capture_audit_material = {
        "schema_version": "training-capture-audit-v1",
        "execution_run_id": scope,
        "status": "CAPTURE_COMPLETE",
        "concurrency_inference_allowed": False,
        "scopes": [{"scope_id": scope, "status": "PASS"}],
    }
    capture_audit = {
        **capture_audit_material,
        "capture_audit_id": content_id(
            "gfca2_", capture_audit_material
        ),
    }
    contracts = load_contracts()
    graph = compile_executable_generation_fact_graph_v2(
        [snapshot_input],
        relation_store,
        catalog,
        capture_audit,
        contracts["graph_profile"],
        contracts["relation_type_registry"],
    )
    validated_graph = validate_executable_generation_fact_graph_v2(
        graph,
        [snapshot_input],
        relation_store,
        catalog,
        capture_audit,
        contracts,
    )
    return {
        "binding_count": len(bindings),
        "chunk_index": chunk_index,
        "core_snapshot_id": snapshot.snapshot_id,
        "executable_gfg_id": validated_graph.graph_id,
        "generated_origin_count": len(generated_origins),
        "occurrence_count": len(occurrence_rows),
        "program_order_relation_count": len(program_order_relations),
        "relation_evidence_resolution_count": len(
            validation.relation_evidence
        ),
        "status": "PASS",
    }


def validate_all_core_chunks(
    database_path: Path,
) -> dict[str, Any]:
    database_path = database_path.resolve()
    connection = sqlite3.connect(database_path)
    try:
        run_id = __import__("json").loads(
            connection.execute(
                "SELECT value_json FROM metadata WHERE key='run_id'"
            ).fetchone()[0]
        )
        capture_code_hash = file_sha256(
            Path(__file__).with_name("training_capture.py")
        )
        rows = []
        for chunk in connection.execute(
            """
            SELECT chunk_index,start_block_ordinal,end_block_ordinal
            FROM chunks ORDER BY chunk_index
            """
        ):
            rows.append(
                _validate_chunk(
                    run_id=run_id,
                    chunk_index=chunk[0],
                    blocks=_chunk_blocks(
                        connection, chunk[1], chunk[2]
                    ),
                    capture_code_hash=capture_code_hash,
                )
            )
        material = {
            "chunk_count": len(rows),
            "chunks": rows,
            "core_changed_files": 0,
            "existing_executable_gfg_reused": True,
            "existing_generation_relation_core_reused": True,
            "schema": "training-gfg-core-closure-v1",
            "status": (
                "PASS"
                if rows and all(row["status"] == "PASS" for row in rows)
                else "GFG_CAPTURE_FAILURE"
            ),
        }
        return {
            **material,
            "validation_sha256": payload_sha256(material),
        }
    finally:
        connection.close()


def validate_core_representatives(
    database_path: Path,
) -> dict[str, Any]:
    """Validate every capture family and deterministic boundary blocks.

    Core v3 currently materializes every atomic relation and its authority
    closure in memory. The complete long-run expansion is independently
    validated by `validate_training_gfg`; this closure check sends one
    deterministic block from every stage plus the first and last block
    through the unmodified Core and Executable GFG v2.
    """

    database_path = database_path.resolve()
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        run_id = __import__("json").loads(
            connection.execute(
                "SELECT value_json FROM metadata WHERE key='run_id'"
            ).fetchone()[0]
        )
        rows = list(
            connection.execute(
                """
                SELECT block_ordinal,stage,payload_zlib
                FROM graph_blocks ORDER BY block_ordinal
                """
            )
        )
        if not rows:
            raise RuntimeError("EMPTY_TRAINING_GFG")
        selected: dict[int, sqlite3.Row] = {
            rows[0]["block_ordinal"]: rows[0],
            rows[-1]["block_ordinal"]: rows[-1],
        }
        for row in rows:
            if not any(
                existing["stage"] == row["stage"]
                for existing in selected.values()
            ):
                selected[row["block_ordinal"]] = row
        blocks = [
            decode_block(selected[key]["payload_zlib"])
            for key in sorted(selected)
        ]
        closure = _validate_chunk(
            run_id=run_id,
            chunk_index=0,
            blocks=blocks,
            capture_code_hash=file_sha256(
                Path(__file__).with_name("training_capture.py")
            ),
        )
        material = {
            "complete_expanded_graph_validation": (
                "validate_training_gfg"
            ),
            "core_changed_files": 0,
            "core_representative_block_count": len(blocks),
            "executable_gfg_v2_reused": True,
            "generation_relation_core_v3_reused": True,
            "representative_stages": sorted(
                {row["stage"] for row in selected.values()}
            ),
            "representative_validation": closure,
            "schema": "training-gfg-core-representative-closure-v1",
            "status": closure["status"],
        }
        return {
            **material,
            "validation_sha256": payload_sha256(material),
        }
    finally:
        connection.close()
