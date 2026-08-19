from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    canonical_bytes,
    file_sha256,
    payload_sha256,
)
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import (
    GraphRef,
    SupportGFGWriter,
)

from .analysis import array_sha256
from .runtime import COMPONENTS, COMPONENT_PAIRS, component_parameter_rows


GRAPH_SCHEMA = "nanogpt-training-learning-inference-projection-gfg-v1"
BLOCK_SCHEMA = "nanogpt-training-learning-inference-projection-block-v1"
MANIFEST_SCHEMA = "nanogpt-training-learning-inference-projection-manifest-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _origin(
    writer: SupportGFGWriter,
    row: dict[str, Any],
    source_bundle_id: str,
) -> GraphRef:
    return writer.origin(
        row,
        source_bundle_id=source_bundle_id,
        source_graph_schema="participant-safe-training-gfg-bundle-v1",
    )


def _tensor(
    writer: SupportGFGWriter,
    *,
    key: str,
    role: str,
    step: int,
    value: np.ndarray,
    representation: str,
    extra: dict[str, Any] | None = None,
) -> GraphRef:
    return writer.tensor_object(
        semantic_key=key,
        role=role,
        optimizer_step=step,
        value=value,
        representation=representation,
        extra_payload=extra,
    )


def _support_version(
    writer: SupportGFGWriter,
    *,
    entry_id: str,
    phase: str,
    step: int,
    component: str,
    rows: dict[str, dict[str, Any]],
    source_bundle_id: str,
) -> GraphRef:
    component_rows = component_parameter_rows(rows, component)
    occurrence = writer.occurrence(
        occurrence_type="component_parameter_version_collection",
        optimizer_step=step,
        transform_reference={"operation": "collect_exact_component_parameter_versions_v1"},
        payload={"entry_id": entry_id, "phase": phase, "component": component},
    )
    payload = {
        "component": component,
        "entry_id": entry_id,
        "optimizer_step": step,
        "parameter_versions": [
            {
                "name": name,
                "object_id": row["object_id"],
                "content_sha256": row["content_sha256"],
            }
            for name, row in sorted(component_rows.items())
        ],
    }
    outcome = writer.object(
        semantic_key=f"{entry_id}:{phase}:{component}:support-version",
        role="learned_component_support_version",
        optimizer_step=step,
        payload=payload,
        object_kind="content_addressed_parameter_version_collection",
    )
    writer.bind(
        occurrence,
        [
            (_origin(writer, row, source_bundle_id), f"component_parameter:{name}")
            for name, row in sorted(component_rows.items())
        ],
        outcome,
        payload={"component": component, "parameter_identity_exact": True},
    )
    return outcome


def write_run_gfg(
    *,
    record: dict[str, Any],
    ledger: dict[str, Any],
    phase_arrays: dict[str, dict[str, np.ndarray]],
    graph_root: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    entry_id = str(record["entry_id"])
    source_bundle_id = str(record["source_bundle_id"])
    run_root = graph_root / entry_id
    require(not run_root.exists(), f"TLI_GFG_RUN_ROOT_EXISTS:{run_root}")
    run_root.mkdir(parents=True)
    writer = SupportGFGWriter(
        run_root / "training_learning_inference_gfg.sqlite3",
        run_root / "tensor-objects",
        scope_id=f"nanogpt-training-learning-inference:{entry_id}",
        source_bundle_id=source_bundle_id,
        contract_sha256=protocol_sha256,
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )

    writer.start_block("target_mapping_and_query_scope", 0)
    protocol = writer.object(
        semantic_key=f"{entry_id}:protocol",
        role="frozen_experiment_protocol",
        optimizer_step=0,
        payload={"protocol_sha256": protocol_sha256, "entry_id": entry_id},
        object_kind="frozen_protocol",
    )
    mapping_occurrence = writer.occurrence(
        occurrence_type="validation_target_mapping_recovery",
        optimizer_step=0,
        transform_reference={"operation": "finite_field_rref_target_mapping_v1"},
        payload={"certificate": record["target_mapping_certificate"]},
    )
    target_ref = _tensor(
        writer,
        key=f"{entry_id}:validation-targets",
        role="derived_validation_targets",
        step=0,
        value=record["validation_targets"],
        representation="complete_validation_target_tensor",
        extra={"held_out_split": True},
    )
    writer.bind(
        mapping_occurrence,
        [
            (_origin(writer, record["training_input_source"], source_bundle_id), "training_inputs"),
            (_origin(writer, record["training_target_source"], source_bundle_id), "training_targets"),
            (_origin(writer, record["validation_input_source"], source_bundle_id), "held_out_validation_inputs"),
            (protocol, "frozen_mapping_protocol"),
        ],
        target_ref,
    )
    writer.flush_block()

    formed_supports: dict[str, GraphRef] = {}
    for phase, raw in record["phases"].items():
        step = int(raw["optimizer_step"])
        writer.start_block(f"native_inference:{phase}", step)
        validation_origin = _origin(writer, record["validation_input_source"], source_bundle_id)
        support_versions = {
            component: _support_version(
                writer,
                entry_id=entry_id,
                phase=phase,
                step=step,
                component=component,
                rows=raw["parameter_rows"],
                source_bundle_id=source_bundle_id,
            )
            for component in COMPONENTS
        }
        if phase == "formed":
            formed_supports = support_versions

        module_outputs: list[tuple[GraphRef, str]] = []
        for call in sorted(raw["calls"], key=lambda value: value.call_index):
            input_ref = _tensor(
                writer,
                key=f"{entry_id}:{phase}:{call.component}:call:{call.call_index}:input",
                role="actual_component_call_input",
                step=step,
                value=call.input_tensor,
                representation="complete_native_component_input_activation",
                extra={"component": call.component, "call_index": call.call_index},
            )
            output_ref = _tensor(
                writer,
                key=f"{entry_id}:{phase}:{call.component}:call:{call.call_index}:output",
                role="actual_component_call_output",
                step=step,
                value=call.output_tensor,
                representation="complete_native_component_output_activation",
                extra={"component": call.component, "call_index": call.call_index},
            )
            occurrence = writer.occurrence(
                occurrence_type="native_frozen_inference_component_call",
                optimizer_step=step,
                transform_reference={
                    "operation": "nanogpt_component_forward",
                    "component": call.component,
                    "model_commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
                },
                payload={"entry_id": entry_id, "phase": phase, "call_index": call.call_index},
            )
            writer.bind(
                occurrence,
                [(input_ref, "actual_call_input"), (support_versions[call.component], "exact_learned_support_version")],
                output_ref,
                payload={"component": call.component, "native_execution": True},
            )
            writer.relation(
                "calls_learned_support",
                support_versions[call.component].object_id,
                output_ref.object_id,
                {"component": call.component, "query_conditioned_by_actual_activation": True},
            )
            module_outputs.append((output_ref, f"component_output:{call.component}"))

        baseline_ref = _tensor(
            writer,
            key=f"{entry_id}:{phase}:baseline-logits",
            role="native_inference_logits",
            step=step,
            value=raw["baseline_logits"],
            representation="complete_float32_decision_logits",
            extra={"repeat_byte_exact": True, "phase": phase},
        )
        result_occurrence = writer.occurrence(
            occurrence_type="native_frozen_inference_result",
            optimizer_step=step,
            transform_reference={"operation": "frozen_nanogpt_eval_forward_v1"},
            payload={"entry_id": entry_id, "phase": phase},
        )
        writer.bind(
            result_occurrence,
            [(validation_origin, "held_out_query_inputs"), (target_ref, "evaluation_target_mapping")]
            + module_outputs
            + [(support, f"available_support:{component}") for component, support in support_versions.items()],
            baseline_ref,
            payload={"ordinary_result": True, "native_cuda_forward": True},
        )

        single_refs: dict[str, GraphRef] = {}
        for component in COMPONENTS:
            gate_spec = writer.object(
                semantic_key=f"{entry_id}:{phase}:gate:{component}",
                role="component_gate_specification",
                optimizer_step=step,
                payload={
                    "component": component,
                    "gate_value": 0.0,
                    "site": "module_output_after_projection_before_residual_addition",
                },
                object_kind="declared_causal_intervention",
            )
            gate_ref = _tensor(
                writer,
                key=f"{entry_id}:{phase}:single-gate:{component}:logits",
                role="single_component_gate_logits",
                step=step,
                value=raw["single_gate_logits"][component],
                representation="complete_float32_decision_logits",
                extra={"gate_components": [component]},
            )
            gate_occurrence = writer.occurrence(
                occurrence_type="single_component_gate_inference",
                optimizer_step=step,
                transform_reference={"operation": "zero_component_output_before_residual_addition_v1"},
                payload={"component": component},
            )
            writer.bind(
                gate_occurrence,
                [(validation_origin, "held_out_query_inputs"), (gate_spec, "causal_gate")]
                + [(support, f"component_support:{name}") for name, support in support_versions.items()],
                gate_ref,
            )
            writer.relation(
                "tests_causal_support",
                support_versions[component].object_id,
                gate_ref.object_id,
                {"intervention": "zero_component_output"},
            )
            single_refs[component] = gate_ref

        pair_refs: dict[str, GraphRef] = {}
        for pair in COMPONENT_PAIRS:
            key = "+".join(pair)
            pair_ref = _tensor(
                writer,
                key=f"{entry_id}:{phase}:pair-gate:{key}:logits",
                role="pair_component_gate_logits",
                step=step,
                value=raw["pair_gate_logits"][key],
                representation="complete_float32_decision_logits",
                extra={"gate_components": list(pair)},
            )
            pair_occurrence = writer.occurrence(
                occurrence_type="pair_component_gate_inference",
                optimizer_step=step,
                transform_reference={"operation": "zero_component_pair_outputs_before_residual_addition_v1"},
                payload={"components": list(pair)},
            )
            writer.bind(
                pair_occurrence,
                [(validation_origin, "held_out_query_inputs")]
                + [(support, f"component_support:{name}") for name, support in support_versions.items()],
                pair_ref,
            )
            pair_refs[key] = pair_ref

        profile_ref = _tensor(
            writer,
            key=f"{entry_id}:{phase}:query-conditioned-support-profile",
            role="query_conditioned_support_projection",
            step=step,
            value=phase_arrays[phase]["support_profile"],
            representation="23_target_groups_by_4_component_causal_effects",
            extra={"components": list(COMPONENTS)},
        )
        interaction_ref = _tensor(
            writer,
            key=f"{entry_id}:{phase}:pair-interaction",
            role="component_pair_nonadditive_interaction",
            step=step,
            value=phase_arrays[phase]["pair_interaction"],
            representation="23_target_groups_by_6_component_pairs",
            extra={"pairs": [list(pair) for pair in COMPONENT_PAIRS]},
        )
        analysis_occurrence = writer.occurrence(
            occurrence_type="query_conditioned_support_projection_derivation",
            optimizer_step=step,
            transform_reference={"operation": "baseline_minus_gate_group_q10_and_pair_interaction_v1"},
            payload={"entry_id": entry_id, "phase": phase, "future_information_used": False},
        )
        analysis_sources = [(baseline_ref, "native_baseline_logits"), (target_ref, "target_group_mapping")]
        analysis_sources += [(value, f"single_gate:{key}") for key, value in single_refs.items()]
        analysis_sources += [(value, f"pair_gate:{key}") for key, value in pair_refs.items()]
        writer.bind(analysis_occurrence, analysis_sources, profile_ref)
        writer.bind(analysis_occurrence, analysis_sources, interaction_ref)
        for component, support in support_versions.items():
            writer.relation(
                "projects_support_for_target_groups",
                support.object_id,
                profile_ref.object_id,
                {"component": component, "target_group_count": 23},
            )
        writer.relation(
            "combines_support_nonadditively",
            profile_ref.object_id,
            interaction_ref.object_id,
            {"component_pair_count": len(COMPONENT_PAIRS)},
        )

        if phase == "formed":
            for component in COMPONENTS:
                rollback = record["rollback"][component]
                pre_support = _support_version(
                    writer,
                    entry_id=entry_id,
                    phase="pre_formation_for_rollback",
                    step=int(record["phases"]["pre_formation"]["optimizer_step"]),
                    component=component,
                    rows=rollback["pre_parameter_rows"],
                    source_bundle_id=source_bundle_id,
                )
                rollback_ref = _tensor(
                    writer,
                    key=f"{entry_id}:formed:rollback:{component}:logits",
                    role="preformation_component_version_rollback_logits",
                    step=step,
                    value=rollback["rollback_logits"],
                    representation="complete_float32_decision_logits",
                    extra={"rolled_back_component": component},
                )
                rollback_occurrence = writer.occurrence(
                    occurrence_type="component_version_rollback_inference",
                    optimizer_step=step,
                    transform_reference={"operation": "replace_one_component_with_preformation_parameter_versions_v1"},
                    payload={"component": component, "declared_hybrid_counterfactual": True},
                )
                writer.bind(
                    rollback_occurrence,
                    [(validation_origin, "held_out_query_inputs"), (pre_support, "preformation_component_version")]
                    + [
                        (support, f"formed_component_version:{name}")
                        for name, support in formed_supports.items()
                        if name != component
                    ],
                    rollback_ref,
                )
                restored_ref = _tensor(
                    writer,
                    key=f"{entry_id}:formed:restore:{component}:logits",
                    role="restored_formed_version_logits",
                    step=step,
                    value=rollback["restored_logits"],
                    representation="complete_float32_decision_logits",
                    extra={"restored_component": component, "byte_exact_baseline": True},
                )
                restore_occurrence = writer.occurrence(
                    occurrence_type="formed_component_version_restoration",
                    optimizer_step=step,
                    transform_reference={"operation": "restore_exact_formed_component_parameter_versions_v1"},
                    payload={"component": component},
                )
                writer.bind(
                    restore_occurrence,
                    [(validation_origin, "held_out_query_inputs")]
                    + [(support, f"formed_component_version:{name}") for name, support in formed_supports.items()],
                    restored_ref,
                )
                writer.relation(
                    "rollback_tests_learned_version_dependence",
                    pre_support.object_id,
                    rollback_ref.object_id,
                    {"component": component},
                )
                writer.relation(
                    "restores_native_result",
                    formed_supports[component].object_id,
                    restored_ref.object_id,
                    {"component": component, "byte_exact": True},
                )

        writer.flush_block()

    manifest = writer.close()
    manifest.update(
        {
            "entry_id": entry_id,
            "source_bundle_id": source_bundle_id,
            "phase_count": len(record["phases"]),
            "status": "CAPTURE_CLOSED",
        }
    )
    write_json(run_root / "GFG_MANIFEST.json", manifest)
    return manifest


def validate_run_gfg(run_root: Path) -> dict[str, Any]:
    manifest = json.loads((run_root / "GFG_MANIFEST.json").read_text(encoding="utf-8"))
    database = run_root / manifest["database"]
    require(file_sha256(database) == manifest["database_sha256"], "TLI_GFG_DATABASE_HASH")
    object_ids: set[str] = set()
    occurrence_ids: set[str] = set()
    origin_ids: set[str] = set()
    fact_ids: set[str] = set()
    blocks: list[dict[str, Any]] = []
    prior = None
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        origin_ids = {str(row[0]) for row in connection.execute("SELECT origin_id FROM origin_catalog")}
        for row in connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal"):
            block = json.loads(zlib.decompress(row["payload_zlib"]))
            require(hashlib.sha256(canonical_bytes(block)).hexdigest() == row["payload_sha256"], "TLI_GFG_BLOCK_PAYLOAD_HASH")
            require(row["prior_block_sha256"] == prior, "TLI_GFG_BLOCK_CHAIN")
            prior = row["block_sha256"]
            blocks.append(block)
            object_ids.update(str(value["object_id"]) for value in block["objects"])
            occurrence_ids.update(str(value["occurrence_id"]) for value in block["occurrences"])
            fact_ids.update(str(value["fact_block_id"]) for value in block["fact_blocks"])
            for value in block["objects"]:
                payload = value.get("payload", {})
                locator = payload.get("locator")
                if locator:
                    path = run_root / locator
                    require(path.exists(), f"TLI_GFG_TENSOR_MISSING:{locator}")
                    require(file_sha256(path) == payload["file_sha256"], f"TLI_GFG_TENSOR_FILE_HASH:{locator}")
                    array = np.load(path, allow_pickle=False)
                    require(array_sha256(array) == payload["raw_tensor_sha256"], f"TLI_GFG_TENSOR_RAW_HASH:{locator}")
    known_sources = object_ids | origin_ids
    realizes: set[str] = set()
    relation_types: set[str] = set()
    for block in blocks:
        for fact in block["fact_blocks"]:
            require(fact["occurrence_id"] in occurrence_ids, "TLI_GFG_FACT_OCCURRENCE")
            require(fact["outcome"]["object_id"] in object_ids, "TLI_GFG_FACT_OUTCOME")
            require(all(source["source_id"] in known_sources for source in fact["sources"]), "TLI_GFG_FACT_SOURCE")
        for relation in block["relations"]:
            relation_types.add(str(relation["relation_type"]))
            if relation["relation_type"] == "realizes_fact":
                realizes.add(str(relation["target_id"]))
    require(realizes == fact_ids, "TLI_GFG_REALIZES_FACT_COVERAGE")
    required_relations = {
        "calls_learned_support",
        "tests_causal_support",
        "projects_support_for_target_groups",
        "combines_support_nonadditively",
        "rollback_tests_learned_version_dependence",
        "restores_native_result",
    }
    require(required_relations <= relation_types, "TLI_GFG_REQUIRED_RELATION_MISSING")
    result = {
        "schema": "nanogpt-training-learning-inference-projection-gfg-validation-v1",
        "status": "PASS",
        "entry_id": manifest["entry_id"],
        "block_count": len(blocks),
        "object_count": len(object_ids),
        "occurrence_count": len(occurrence_ids),
        "fact_count": len(fact_ids),
        "origin_count": len(origin_ids),
        "relation_types": sorted(relation_types),
        "database_sha256": file_sha256(database),
    }
    write_json(run_root / "GFG_VALIDATION.json", result)
    return result


def aggregate_manifest(graph_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for row in rows:
        run_root = graph_root / row["entry_id"]
        entries.append(
            {
                "entry_id": row["entry_id"],
                "manifest_sha256": file_sha256(run_root / "GFG_MANIFEST.json"),
                "validation_sha256": file_sha256(run_root / "GFG_VALIDATION.json"),
                "validation_status": json.loads((run_root / "GFG_VALIDATION.json").read_text())["status"],
            }
        )
    value = {
        "schema": "nanogpt-training-learning-inference-projection-graph-archive-v1",
        "status": "PASS" if all(entry["validation_status"] == "PASS" for entry in entries) else "FAIL",
        "entry_count": len(entries),
        "entries": entries,
    }
    value["archive_sha256"] = payload_sha256(value)
    write_json(graph_root / "ARCHIVE_MANIFEST.json", value)
    return value


__all__ = ["aggregate_manifest", "validate_run_gfg", "write_run_gfg"]
