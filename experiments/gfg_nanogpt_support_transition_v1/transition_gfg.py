from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator
import zlib

import numpy as np

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    canonical_bytes,
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import (
    TrainingGFG,
)
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import objects_for_stage
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import (
    GraphRef,
    SupportGFG,
    SupportGFGWriter,
)


GRAPH_SCHEMA = "nanogpt-support-transition-gfg-v1"
BLOCK_SCHEMA = "nanogpt-support-transition-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-support-transition-gfg-manifest-v1"
ALLOWED_RELATIONS = {
    "generated_origin_dependency",
    "matched_branch_of",
    "program_order",
    "reads_from",
    "realizes_fact",
}


def _is_tensor_ref(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("raw_tensor_sha256"), str)
        and isinstance(value.get("file_sha256"), str)
        and isinstance(value.get("locator"), str)
    )


def tensor_refs(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if _is_tensor_ref(value):
        yield path, value
        return
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            yield from tensor_refs(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from tensor_refs(child, path + (str(index),))


def _summary_projection(value: Any) -> Any:
    if _is_tensor_ref(value):
        return {
            "dtype": value["dtype"],
            "raw_tensor_sha256": value["raw_tensor_sha256"],
            "representation": value["representation"],
            "shape": value["shape"],
        }
    if isinstance(value, dict):
        return {key: _summary_projection(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_summary_projection(child) for child in value]
    return value


def _role(path: tuple[str, ...]) -> str:
    return "support_transition_" + "_".join(path[-4:]).replace("+", "_plus_")


def _emit_tree(
    writer: SupportGFGWriter,
    *,
    occurrence_id: str,
    optimizer_step: int,
    prefix: str,
    value: dict[str, Any],
    sources: list[tuple[GraphRef, str]],
) -> GraphRef:
    tensor_outcomes: list[GraphRef] = []
    for path, tensor in tensor_refs(value):
        semantic_path = "/".join(path)
        ref = writer.object(
            semantic_key=f"{prefix}:tensor:{semantic_path}",
            role=_role(path),
            optimizer_step=optimizer_step,
            payload=tensor,
            object_kind="content_addressed_tensor",
        )
        writer.bind(
            occurrence_id,
            sources,
            ref,
            payload={"outcome_path": list(path), "outcome_kind": "content_addressed_tensor"},
        )
        tensor_outcomes.append(ref)
    summary_payload = _summary_projection(value)
    summary = writer.object(
        semantic_key=f"{prefix}:summary",
        role="support_transition_result_summary",
        optimizer_step=optimizer_step,
        payload=summary_payload,
    )
    writer.bind(
        occurrence_id,
        sources + [(ref, "complete_tensor_outcome") for ref in tensor_outcomes],
        summary,
        payload={"outcome_kind": "complete_result_summary"},
    )
    undefined_groups = value.get("undefined_effective_support_groups")
    if isinstance(undefined_groups, list):
        for group_index in undefined_groups:
            disposition = writer.object(
                semantic_key=f"{prefix}:effective-support-disposition:{int(group_index)}",
                role="explicit_disposition",
                optimizer_step=optimizer_step,
                payload={
                    "disposition": "EFFECTIVE_SUPPORT_UNDEFINED_ZERO_TOTAL_NECESSITY",
                    "target_group_index": int(group_index),
                },
                object_kind="ExplicitDisposition",
            )
            writer.bind(
                occurrence_id,
                sources,
                disposition,
                payload={
                    "outcome_kind": "ExplicitDisposition",
                    "outcome_path": ["effective_support", str(int(group_index))],
                },
            )
    for path, child in _dispositions(value):
        disposition = writer.object(
            semantic_key=f"{prefix}:disposition:{'/'.join(path)}",
            role="explicit_disposition",
            optimizer_step=optimizer_step,
            payload=child,
            object_kind="ExplicitDisposition",
        )
        writer.bind(
            occurrence_id,
            sources,
            disposition,
            payload={"outcome_path": list(path), "outcome_kind": "ExplicitDisposition"},
        )
    return summary


def _dispositions(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(value, dict):
        if "disposition" in value:
            yield path, value
            return
        for key, child in sorted(value.items()):
            yield from _dispositions(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _dispositions(child, path + (str(index),))


def _csrg_objects(graph: SupportGFG) -> tuple[dict[int, dict[str, str]], dict[str, dict[str, Any]]]:
    checkpoint_ids: dict[int, dict[str, str]] = {}
    for row in graph.checkpoints():
        checkpoint_ids[int(row["optimizer_step"])] = json.loads(row["derived_object_ids_json"])
    required_ids = {
        object_id
        for values in checkpoint_ids.values()
        for object_id in values.values()
    }
    objects: dict[str, dict[str, Any]] = {}
    rows = graph.connection.execute(
        "SELECT payload_zlib FROM graph_blocks WHERE stage='support_redundancy_derivations' ORDER BY block_ordinal"
    )
    for row in rows:
        payload = json.loads(zlib.decompress(row["payload_zlib"]))
        for child in payload["objects"]:
            if child["object_id"] in required_ids:
                objects[child["object_id"]] = child
    require(set(objects) == required_ids, "CST_SOURCE_CSRG_OBJECT_CATALOG_INCOMPLETE")
    return checkpoint_ids, objects


def _training_origins(
    writer: SupportGFGWriter,
    graph: TrainingGFG,
    result: dict[str, Any],
) -> list[tuple[GraphRef, str]]:
    ids = result["prestate"]["source_objects"]
    optimizer_step = int(result["optimizer_step"])
    source_rows = {
        row["object_id"]: row
        for row in objects_for_stage(graph, optimizer_step, "after_optimizer_step")
    }
    values: list[tuple[GraphRef, str]] = []
    for group in ("parameter_object_ids", "optimizer_object_ids"):
        for name, object_id in sorted(ids[group].items()):
            require(object_id in source_rows, f"CST_SOURCE_OBJECT_NOT_IN_PRESTATE_BLOCK:{object_id}")
            row = source_rows[object_id]
            values.append((writer.origin(row), f"source_{group}:{name}"))
    for row in objects_for_stage(graph, 0, "before_batch"):
        if row["role"] in {"training_batch_inputs", "training_batch_targets"}:
            values.append((writer.origin(row), row["role"]))
    return values


def _csrg_origins(
    writer: SupportGFGWriter,
    *,
    bundle_id: str,
    optimizer_step: int,
    checkpoint_ids: dict[int, dict[str, str]],
    object_catalog: dict[str, dict[str, Any]],
) -> list[tuple[GraphRef, str]]:
    result: list[tuple[GraphRef, str]] = []
    for name, object_id in sorted(checkpoint_ids[optimizer_step].items()):
        row = object_catalog[object_id]
        result.append(
            (
                writer.origin(
                    row,
                    source_bundle_id=bundle_id + "::csrg-v2",
                    source_graph_schema="nanogpt-support-redundancy-gfg-v1",
                ),
                f"source_prestate_csrg:{name}",
            )
        )
    return result


def _occurrence(
    writer: SupportGFGWriter,
    *,
    occurrence_type: str,
    optimizer_step: int,
    operation: str,
    payload: dict[str, Any],
) -> str:
    return writer.occurrence(
        occurrence_type=occurrence_type,
        optimizer_step=optimizer_step,
        transform_reference={"operation": operation, "contract_version": "v1"},
        payload=payload,
    )


def _emit_current_runtime_rng(
    writer: SupportGFGWriter,
    *,
    occurrence_id: str,
    optimizer_step: int,
    prefix: str,
    rng_before: dict[str, Any],
    contract_ref: GraphRef,
) -> GraphRef:
    return _emit_tree(
        writer,
        occurrence_id=occurrence_id,
        optimizer_step=optimizer_step,
        prefix=prefix,
        value={
            "current_runtime_rng": rng_before,
            "historical_rng": {
                "disposition": "HISTORICAL_CPU_AND_CUDA_RNG_PAYLOAD_NOT_MATERIALIZED",
                "replacement_rule": "content_bound_current_runtime_seed_shared_by_matched_branches",
            },
        },
        sources=[(contract_ref, "frozen_rng_restoration_rule")],
    )


def _compile_scan(
    writer: SupportGFGWriter,
    *,
    result: dict[str, Any],
    training_graph: TrainingGFG,
    csrg_checkpoint_ids: dict[int, dict[str, str]],
    csrg_objects: dict[str, dict[str, Any]],
    bundle_id: str,
    contract_ref: GraphRef,
) -> None:
    step = int(result["optimizer_step"])
    entry_id = str(result["entry_id"])
    writer.start_block("single_step_causal_scan", step)
    origins = _training_origins(writer, training_graph, result)
    origins += _csrg_origins(
        writer,
        bundle_id=bundle_id,
        optimizer_step=step,
        checkpoint_ids=csrg_checkpoint_ids,
        object_catalog=csrg_objects,
    )
    origins.append((contract_ref, "frozen_transition_contract"))
    restore_occ = _occurrence(
        writer,
        occurrence_type="historical_prestate_restore",
        optimizer_step=step,
        operation="restore_content_bound_parameter_adam_and_batch",
        payload={"entry_id": entry_id, "optimizer_step": step},
    )
    prestate_ref = _emit_tree(
        writer,
        occurrence_id=restore_occ,
        optimizer_step=step,
        prefix=f"scan:{entry_id}:{step}:prestate",
        value={"state": result["prestate"]["state"]},
        sources=origins,
    )
    rng_occ = _occurrence(
        writer,
        occurrence_type="current_runtime_rng_restore",
        optimizer_step=step,
        operation="restore_content_bound_current_runtime_rng_and_record_historical_rng_disposition",
        payload={"entry_id": entry_id, "optimizer_step": step},
    )
    rng_ref = _emit_current_runtime_rng(
        writer,
        occurrence_id=rng_occ,
        optimizer_step=step,
        prefix=f"scan:{entry_id}:{step}:rng",
        rng_before=result["full"]["step"]["rng_before"],
        contract_ref=contract_ref,
    )
    baseline_occ = _occurrence(
        writer,
        occurrence_type="prestate_ungated_repeat",
        optimizer_step=step,
        operation="two_ungated_forwards_require_byte_identity",
        payload={"entry_id": entry_id, "repetitions": 2},
    )
    _emit_tree(
        writer,
        occurrence_id=baseline_occ,
        optimizer_step=step,
        prefix=f"scan:{entry_id}:{step}:prebaseline",
        value={"baseline_logits": result["prestate"]["baseline_logits"]},
        sources=[(prestate_ref, "restored_prestate"), (rng_ref, "current_runtime_rng_state")],
    )
    branch_occurrences: dict[str, str] = {}
    branch_refs: dict[str, GraphRef] = {}
    probe_refs: dict[str, GraphRef] = {}
    for branch in ("full", "skip"):
        occurrence_id = _occurrence(
            writer,
            occurrence_type=branch + "_step_occurrence",
            optimizer_step=step,
            operation="forward_backward_clip_adamw_step" if branch == "full" else "forward_backward_clip_skip_optimizer_step",
            payload={
                "branch": branch,
                "entry_id": entry_id,
                "matched_prestate_state_sha256": result["prestate"]["state"]["commitment"]["state_sha256"],
                "optimizer_step": step,
            },
        )
        branch_occurrences[branch] = occurrence_id
        branch_refs[branch] = _emit_tree(
            writer,
            occurrence_id=occurrence_id,
            optimizer_step=step,
            prefix=f"scan:{entry_id}:{step}:{branch}:training",
            value={"state": result[branch]["state"], "step": result[branch]["step"]},
            sources=[
                (prestate_ref, "immutable_matched_prestate"),
                (rng_ref, "matched_current_runtime_rng_state"),
                (contract_ref, "frozen_branch_rule"),
            ],
        )
        probe_occ = _occurrence(
            writer,
            occurrence_type=branch + "_support_probe_occurrence",
            optimizer_step=step,
            operation="complete_12_forward_four_component_csrg_probe",
            payload={"branch": branch, "entry_id": entry_id, "forward_count": 12},
        )
        probe_refs[branch] = _emit_tree(
            writer,
            occurrence_id=probe_occ,
            optimizer_step=step,
            prefix=f"scan:{entry_id}:{step}:{branch}:probe",
            value=result[branch]["probe"],
            sources=[(branch_refs[branch], "branch_poststate"), (contract_ref, "frozen_probe_rule")],
        )
    writer.relation(
        "matched_branch_of",
        branch_occurrences["full"],
        branch_occurrences["skip"],
        {"matched_prestate_state_sha256": result["prestate"]["state"]["commitment"]["state_sha256"]},
    )
    compare_occ = _occurrence(
        writer,
        occurrence_type="single_step_support_transition_comparison",
        optimizer_step=step,
        operation="S_after_full_step_minus_S_after_skip_step",
        payload={"entry_id": entry_id, "optimizer_step": step},
    )
    _emit_tree(
        writer,
        occurrence_id=compare_occ,
        optimizer_step=step,
        prefix=f"scan:{entry_id}:{step}:effect",
        value={
            "effect": result["effect"],
            "historical_next_comparison": result["historical_next_comparison"],
        },
        sources=[
            (probe_refs["full"], "full_branch_support_state"),
            (probe_refs["skip"], "skip_branch_support_state"),
        ],
    )
    writer.flush_block()


def _compile_anchor(
    writer: SupportGFGWriter,
    *,
    result: dict[str, Any],
    training_graph: TrainingGFG,
    csrg_checkpoint_ids: dict[int, dict[str, str]],
    csrg_objects: dict[str, dict[str, Any]],
    bundle_id: str,
    contract_ref: GraphRef,
    selection_ref: GraphRef,
) -> None:
    step = int(result["optimizer_step"])
    entry_id = str(result["entry_id"])
    anchor_id = "anchor-" + payload_sha256({"entry_id": entry_id, "optimizer_step": step})[:16]
    writer.start_block("deep_four_branch_anchor", step)
    origins = _training_origins(writer, training_graph, result)
    origins += _csrg_origins(
        writer,
        bundle_id=bundle_id,
        optimizer_step=step,
        checkpoint_ids=csrg_checkpoint_ids,
        object_catalog=csrg_objects,
    )
    origins += [(contract_ref, "frozen_transition_contract"), (selection_ref, "frozen_anchor_selection")]
    restore_occ = _occurrence(
        writer,
        occurrence_type="anchor_prestate_restore",
        optimizer_step=step,
        operation="restore_content_bound_anchor_parameter_adam_and_batch",
        payload={"anchor_id": anchor_id, "entry_id": entry_id, "optimizer_step": step},
    )
    prestate_ref = _emit_tree(
        writer,
        occurrence_id=restore_occ,
        optimizer_step=step,
        prefix=f"anchor:{entry_id}:{anchor_id}:{step}:prestate",
        value={"state": result["prestate"]["state"]},
        sources=origins,
    )
    rng_occ = _occurrence(
        writer,
        occurrence_type="current_runtime_rng_restore",
        optimizer_step=step,
        operation="restore_content_bound_current_runtime_rng_and_record_historical_rng_disposition",
        payload={"anchor_id": anchor_id, "entry_id": entry_id, "optimizer_step": step},
    )
    rng_ref = _emit_current_runtime_rng(
        writer,
        occurrence_id=rng_occ,
        optimizer_step=step,
        prefix=f"anchor:{entry_id}:{anchor_id}:{step}:rng",
        rng_before=result["branches"]["full_update"]["initial_step"]["rng_before"],
        contract_ref=contract_ref,
    )
    baseline_occ = _occurrence(
        writer,
        occurrence_type="prestate_ungated_repeat",
        optimizer_step=step,
        operation="two_ungated_forwards_require_byte_identity",
        payload={"anchor_id": anchor_id, "entry_id": entry_id, "repetitions": 2},
    )
    _emit_tree(
        writer,
        occurrence_id=baseline_occ,
        optimizer_step=step,
        prefix=f"anchor:{entry_id}:{anchor_id}:{step}:prebaseline",
        value={"baseline_logits": result["prestate"]["baseline_logits"]},
        sources=[(prestate_ref, "restored_prestate"), (rng_ref, "current_runtime_rng_state")],
    )
    branch_occurrences: dict[str, str] = {}
    prior_state_refs: dict[str, GraphRef] = {}
    horizon_probe_refs: dict[int, dict[str, GraphRef]] = {}

    # The runtime executes three independent full-step basis replays before the
    # skip branch.  Preserve that control flow instead of inventing a
    # horizon-major order during graph compilation.
    basis_refs: dict[str, GraphRef] = {}
    for branch in ("full_update", "parameter_only", "optimizer_state_only"):
        branch_payload = result["branches"][branch]
        initial_occ = _occurrence(
            writer,
            occurrence_type=(
                "full_update_occurrence"
                if branch == "full_update"
                else branch + "_basis_full_replay_occurrence"
            ),
            optimizer_step=step,
            operation="forward_backward_clip_adamw_step",
            payload={
                "anchor_id": anchor_id,
                "branch": branch,
                "entry_id": entry_id,
                "matched_prestate_state_sha256": result["prestate"]["state"]["commitment"]["state_sha256"],
            },
        )
        initial_value: dict[str, Any] = {"initial_step": branch_payload["initial_step"]}
        if branch == "full_update":
            initial_value["initial_state"] = branch_payload["horizons"]["1"]["state"]
        initial_ref = _emit_tree(
            writer,
            occurrence_id=initial_occ,
            optimizer_step=step,
            prefix=f"anchor:{entry_id}:{anchor_id}:{step}:{branch}:initial",
            value=initial_value,
            sources=[
                (prestate_ref, "immutable_matched_prestate"),
                (rng_ref, "matched_current_runtime_rng_state"),
                (contract_ref, "frozen_branch_rule"),
            ],
        )
        basis_refs[branch] = initial_ref
        if branch == "full_update":
            branch_occurrences[branch] = initial_occ
            prior_state_refs[branch] = initial_ref

    skip_payload = result["branches"]["skip_update"]
    skip_occ = _occurrence(
        writer,
        occurrence_type="skip_update_occurrence",
        optimizer_step=step,
        operation="forward_backward_clip_skip_optimizer_step",
        payload={
            "anchor_id": anchor_id,
            "branch": "skip_update",
            "entry_id": entry_id,
            "matched_prestate_state_sha256": result["prestate"]["state"]["commitment"]["state_sha256"],
        },
    )
    skip_ref = _emit_tree(
        writer,
        occurrence_id=skip_occ,
        optimizer_step=step,
        prefix=f"anchor:{entry_id}:{anchor_id}:{step}:skip_update:initial",
        value={
            "initial_state": skip_payload["horizons"]["1"]["state"],
            "initial_step": skip_payload["initial_step"],
        },
        sources=[
            (prestate_ref, "immutable_matched_prestate"),
            (rng_ref, "matched_current_runtime_rng_state"),
            (contract_ref, "frozen_branch_rule"),
        ],
    )
    branch_occurrences["skip_update"] = skip_occ
    prior_state_refs["skip_update"] = skip_ref

    for branch, operation in (
        ("parameter_only", "keep_full_replay_parameter_restore_adam_prestate"),
        ("optimizer_state_only", "keep_full_replay_adam_restore_parameter_prestate"),
    ):
        assembly_occ = _occurrence(
            writer,
            occurrence_type=branch + "_occurrence",
            optimizer_step=step,
            operation=operation,
            payload={
                "anchor_id": anchor_id,
                "branch": branch,
                "entry_id": entry_id,
                "matched_prestate_state_sha256": result["prestate"]["state"]["commitment"]["state_sha256"],
            },
        )
        assembly_ref = _emit_tree(
            writer,
            occurrence_id=assembly_occ,
            optimizer_step=step,
            prefix=f"anchor:{entry_id}:{anchor_id}:{step}:{branch}:assembled",
            value={"initial_state": result["branches"][branch]["horizons"]["1"]["state"]},
            sources=[
                (basis_refs[branch], "independent_full_replay_basis"),
                (prestate_ref, "immutable_matched_prestate"),
                (contract_ref, "frozen_hybrid_branch_rule"),
            ],
        )
        branch_occurrences[branch] = assembly_occ
        prior_state_refs[branch] = assembly_ref

    for branch in branch_occurrences:
        if branch != "full_update":
            writer.relation(
                "matched_branch_of",
                branch_occurrences["full_update"],
                branch_occurrences[branch],
                {"matched_prestate_state_sha256": result["prestate"]["state"]["commitment"]["state_sha256"]},
            )
    for horizon in (1, 5, 20, 100):
        horizon_probe_refs[horizon] = {}
    for branch in ("full_update", "skip_update", "parameter_only", "optimizer_state_only"):
        for horizon in (1, 5, 20, 100):
            row = result["branches"][branch]["horizons"][str(horizon)]
            if horizon == 1:
                state_ref = prior_state_refs[branch]
            else:
                continuation_occ = _occurrence(
                    writer,
                    occurrence_type="continuation_occurrence",
                    optimizer_step=step,
                    operation="continue_branch_to_aligned_training_opportunity_horizon",
                    payload={
                        "branch": branch,
                        "entry_id": entry_id,
                        "horizon": horizon,
                        "training_opportunities": row["training_opportunities"],
                    },
                )
                state_ref = _emit_tree(
                    writer,
                    occurrence_id=continuation_occ,
                    optimizer_step=step,
                    prefix=f"anchor:{entry_id}:{anchor_id}:{step}:{branch}:h{horizon}:continuation",
                    value={"state": row["state"], "terminal_step": row["terminal_step"]},
                    sources=[
                        (prior_state_refs[branch], "prior_measured_branch_state"),
                        (contract_ref, "frozen_continuation_rule"),
                    ],
                )
                prior_state_refs[branch] = state_ref
            probe_occ = _occurrence(
                writer,
                occurrence_type="support_probe_occurrence",
                optimizer_step=step,
                operation="complete_12_forward_four_component_csrg_probe",
                payload={"branch": branch, "entry_id": entry_id, "forward_count": 12, "horizon": horizon},
            )
            horizon_probe_refs[horizon][branch] = _emit_tree(
                writer,
                occurrence_id=probe_occ,
                optimizer_step=step,
                prefix=f"anchor:{entry_id}:{anchor_id}:{step}:{branch}:h{horizon}:probe",
                value=row["probe"],
                sources=[(state_ref, "branch_state_at_horizon"), (contract_ref, "frozen_probe_rule")],
            )
    for horizon in (1, 5, 20, 100):
        compare_occ = _occurrence(
            writer,
            occurrence_type="support_transition_comparison_occurrence",
            optimizer_step=step,
            operation="parameter_optimizer_full_and_interaction_effects_against_skip",
            payload={"entry_id": entry_id, "horizon": horizon},
        )
        _emit_tree(
            writer,
            occurrence_id=compare_occ,
            optimizer_step=step,
            prefix=f"anchor:{entry_id}:{anchor_id}:{step}:h{horizon}:effects",
            value=result["effects"][str(horizon)],
            sources=[
                (horizon_probe_refs[horizon][branch], f"{branch}_support_state")
                for branch in branch_occurrences
            ],
        )
    writer.flush_block()


def compile_entry_gfg(
    *,
    entry_directory: Path,
    source_training_bundle: Path,
    source_csrg_bundle: Path,
    contract_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    entry_directory = entry_directory.resolve()
    output_database = entry_directory / "support_transition_gfg.sqlite3"
    output_manifest = entry_directory / "support_transition_gfg_manifest.json"
    if output_database.exists() and output_manifest.exists():
        existing = read_json(output_manifest)
        require(existing["schema"] == MANIFEST_SCHEMA, "CST_EXISTING_GFG_MANIFEST_SCHEMA_INVALID")
        require(existing["database_sha256"] == file_sha256(output_database), "CST_EXISTING_GFG_DATABASE_HASH_INVALID")
        return existing
    require(not output_database.exists() and not output_manifest.exists(), "CST_PARTIAL_GFG_ALREADY_EXISTS")
    training = TrainingGFG(source_training_bundle / "participant_gfg.sqlite3")
    csrg = SupportGFG(source_csrg_bundle / "support_gfg.sqlite3")
    checkpoint_ids, csrg_object_catalog = _csrg_objects(csrg)
    entry_id = entry_directory.name
    bundle_id = source_training_bundle.name
    writer = SupportGFGWriter(
        output_database,
        entry_directory / "tensor-objects",
        scope_id=f"nanogpt-support-transition-v1:{entry_id}",
        source_bundle_id=bundle_id,
        contract_sha256=file_sha256(contract_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    try:
        writer.start_block("frozen_protocol", 0)
        contract = read_json(contract_path)
        contract_ref = writer.object(
            semantic_key="support-transition:frozen-contract",
            role="capture_contract",
            optimizer_step=0,
            payload={"contract": contract, "contract_sha256": file_sha256(contract_path)},
            object_kind="declared_execution_source",
        )
        selection = read_json(selection_path)
        selection_ref = writer.object(
            semantic_key="support-transition:frozen-anchor-selection",
            role="anchor_selection_receipt",
            optimizer_step=0,
            payload={
                "selection_sha256": selection["selection_sha256"],
                "entry_anchor_count": sum(row["entry_id"] == entry_id for row in selection["anchors"]),
                "selection_frozen_before_branch_execution": True,
            },
            object_kind="declared_execution_source",
        )
        protocol_occ = _occurrence(
            writer,
            occurrence_type="protocol_freeze_occurrence",
            optimizer_step=0,
            operation="freeze_contract_and_anchor_selection_before_causal_results",
            payload={"entry_id": entry_id},
        )
        protocol_outcome = writer.object(
            semantic_key="support-transition:protocol-freeze-receipt",
            role="protocol_freeze_receipt",
            optimizer_step=0,
            payload={"contract_sha256": file_sha256(contract_path), "selection_sha256": selection["selection_sha256"]},
        )
        writer.bind(protocol_occ, [(contract_ref, "frozen_contract"), (selection_ref, "frozen_selection")], protocol_outcome)
        writer.flush_block()
        for path in sorted((entry_directory / "scan").glob("step-*.json")):
            _compile_scan(
                writer,
                result=read_json(path),
                training_graph=training,
                csrg_checkpoint_ids=checkpoint_ids,
                csrg_objects=csrg_object_catalog,
                bundle_id=bundle_id,
                contract_ref=contract_ref,
            )
        for path in sorted((entry_directory / "anchors").glob("*.json")):
            _compile_anchor(
                writer,
                result=read_json(path),
                training_graph=training,
                csrg_checkpoint_ids=checkpoint_ids,
                csrg_objects=csrg_object_catalog,
                bundle_id=bundle_id,
                contract_ref=contract_ref,
                selection_ref=selection_ref,
            )
        manifest = writer.close()
        write_json(output_manifest, manifest)
        return manifest
    except Exception:
        writer.connection.close()
        raise
    finally:
        training.close()
        csrg.close()


def _verify_tensor_ref(entry_directory: Path, value: dict[str, Any]) -> None:
    locator = str(value["locator"])
    require(locator.startswith("tensor-objects/"), "CST_TENSOR_LOCATOR_OUTSIDE_ENTRY")
    path = entry_directory / locator
    require(path.is_file(), f"CST_TENSOR_PAYLOAD_MISSING:{locator}")
    require(file_sha256(path) == value["file_sha256"], f"CST_TENSOR_FILE_HASH_MISMATCH:{locator}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(array.shape) == value["shape"], f"CST_TENSOR_SHAPE_MISMATCH:{locator}")
    require(str(array.dtype) == value["dtype"], f"CST_TENSOR_DTYPE_MISMATCH:{locator}")
    raw_sha = hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()
    require(raw_sha == value["raw_tensor_sha256"], f"CST_TENSOR_RAW_HASH_MISMATCH:{locator}")


def validate_entry_results(entry_directory: Path) -> dict[str, Any]:
    scan = sorted((entry_directory / "scan").glob("step-*.json"))
    anchors = sorted((entry_directory / "anchors").glob("*.json"))
    require(len(scan) == 100, "CST_RESULT_SCAN_COUNT_INVALID")
    require(len(anchors) == 4, "CST_RESULT_ANCHOR_COUNT_INVALID")
    tensor_ref_count = 0
    verified_tensor_payloads: set[tuple[str, str, str]] = set()
    historical_exact_count = 0
    historical_comparable_count = 0
    for path in [*scan, *anchors]:
        value = read_json(path)
        material = {key: child for key, child in value.items() if key != "result_sha256"}
        require(payload_sha256(material) == value["result_sha256"], f"CST_RESULT_HASH_MISMATCH:{path.name}")
        for _key, tensor in tensor_refs(value):
            identity = (
                tensor["locator"],
                tensor["file_sha256"],
                tensor["raw_tensor_sha256"],
            )
            if identity not in verified_tensor_payloads:
                _verify_tensor_ref(entry_directory, tensor)
                verified_tensor_payloads.add(identity)
            tensor_ref_count += 1
        if value["schema"] == "nanogpt-support-transition-scan-checkpoint-v1":
            admitted_feature_material = json.dumps(
                {key: value[key] for key in ("effect", "full", "prestate", "skip")},
                ensure_ascii=False,
                sort_keys=True,
            ).lower()
            require(
                not any(token in admitted_feature_material for token in ("future_decline", "stability_interval", "recovery_label")),
                "CST_RESULT_SCAN_FUTURE_LABEL_LEAKAGE",
            )
            require(
                value["full"]["step"]["raw_gradients"]["raw_tensor_sha256"]
                == value["skip"]["step"]["raw_gradients"]["raw_tensor_sha256"],
                "CST_RESULT_MATCHED_RAW_GRADIENT_MISMATCH",
            )
            require(
                value["full"]["step"]["clipped_gradients"]["raw_tensor_sha256"]
                == value["skip"]["step"]["clipped_gradients"]["raw_tensor_sha256"],
                "CST_RESULT_MATCHED_CLIPPED_GRADIENT_MISMATCH",
            )
            require(
                value["skip"]["state"]["commitment"]["state_sha256"]
                == value["prestate"]["state"]["commitment"]["state_sha256"],
                "CST_RESULT_SKIP_PRESTATE_MISMATCH",
            )
            comparison = value["historical_next_comparison"]
            if comparison["all_available_object_hashes_exact"] is not None:
                historical_comparable_count += 1
                historical_exact_count += int(comparison["all_available_object_hashes_exact"])
        else:
            admitted_feature_material = json.dumps(
                {key: value[key] for key in ("branches", "effects", "prestate")},
                ensure_ascii=False,
                sort_keys=True,
            ).lower()
            require(
                not any(token in admitted_feature_material for token in ("future_decline", "stability_interval", "recovery_label")),
                "CST_RESULT_ANCHOR_FUTURE_LABEL_LEAKAGE",
            )
            require(len(set(value["full_basis_replay_state_sha256"])) == 1, "CST_RESULT_FULL_REPLAY_MISMATCH")
    material = {
        "anchor_result_count": len(anchors),
        "entry_id": entry_directory.name,
        "future_leakage_audit": {
            "absolute_step_admitted_as_model_feature": False,
            "anchor_outcome_labels_confined_to_selection_metadata": True,
            "new_branch_results_used_for_selection": False,
            "run_identity_admitted_as_model_feature": False,
            "status": "PASS",
        },
        "historical_comparable_scan_count": historical_comparable_count,
        "historical_exact_scan_count": historical_exact_count,
        "scan_result_count": len(scan),
        "schema": "nanogpt-support-transition-result-validation-v1",
        "status": "PASS",
        "tensor_reference_checks": tensor_ref_count,
        "unique_tensor_payload_checks": len(verified_tensor_payloads),
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(entry_directory / "result_validation.json", result)
    return result


def validate_entry_gfg(
    *,
    entry_directory: Path,
    source_training_bundle: Path,
    source_csrg_bundle: Path,
) -> dict[str, Any]:
    database = entry_directory / "support_transition_gfg.sqlite3"
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    metadata = {
        row["key"]: json.loads(row["value_json"])
        for row in connection.execute("SELECT key,value_json FROM metadata")
    }
    require(metadata["schema"] == GRAPH_SCHEMA, "CST_GFG_SCHEMA_INVALID")
    origin_rows = list(connection.execute("SELECT * FROM origin_catalog ORDER BY origin_id"))
    origin_ids = {row["origin_id"] for row in origin_rows}
    blocks = list(connection.execute("SELECT * FROM graph_blocks ORDER BY block_ordinal"))
    require(len(blocks) == 105, "CST_GFG_BLOCK_COUNT_INVALID")
    prior = None
    object_ids: set[str] = set()
    occurrence_ids: set[str] = set()
    fact_ids: set[str] = set()
    relations: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    tensor_objects: list[dict[str, Any]] = []
    future_label_tokens = {
        "anchor_category",
        "boundary_distance",
        "decline_minimum",
        "decline_start",
        "forward_stable_steps",
        "immediate_pre_severe_decline",
        "matched_non_declining_control",
        "matched_to_step",
        "post_formation_sustained_stable",
        "recovery_after_same_severe_decline",
        "standardized_rms_distance",
    }
    for ordinal, row in enumerate(blocks):
        require(int(row["block_ordinal"]) == ordinal, "CST_GFG_BLOCK_ORDINAL_INVALID")
        payload_raw = zlib.decompress(row["payload_zlib"])
        require(hashlib.sha256(payload_raw).hexdigest() == row["payload_sha256"], "CST_GFG_BLOCK_PAYLOAD_HASH_INVALID")
        payload = json.loads(payload_raw)
        require(payload["schema"] == BLOCK_SCHEMA, "CST_GFG_BLOCK_SCHEMA_INVALID")
        require(row["prior_block_sha256"] == prior, "CST_GFG_BLOCK_CHAIN_INVALID")
        block_material = {
            "block_ordinal": ordinal,
            "optimizer_step": int(row["optimizer_step"]),
            "payload_sha256": row["payload_sha256"],
            "prior_block_sha256": prior,
            "scope_id": metadata["scope_id"],
            "stage": row["stage"],
        }
        require(payload_sha256(block_material) == row["block_sha256"], "CST_GFG_BLOCK_HASH_INVALID")
        prior = row["block_sha256"]
        require(len(payload["objects"]) == row["object_count"], "CST_GFG_OBJECT_COUNT_INVALID")
        require(len(payload["occurrences"]) == row["occurrence_count"], "CST_GFG_OCCURRENCE_COUNT_INVALID")
        require(len(payload["relations"]) == row["relation_count"], "CST_GFG_RELATION_COUNT_INVALID")
        if ordinal > 0:
            participant_material = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
            require(
                not any(token in participant_material for token in future_label_tokens),
                "CST_GFG_ANCHOR_FUTURE_LABEL_LEAKAGE",
            )
        occurrence_types = [child["occurrence_type"] for child in payload["occurrences"]]
        if row["stage"] == "single_step_causal_scan":
            require(
                occurrence_types
                == [
                    "historical_prestate_restore",
                    "current_runtime_rng_restore",
                    "prestate_ungated_repeat",
                    "full_step_occurrence",
                    "full_support_probe_occurrence",
                    "skip_step_occurrence",
                    "skip_support_probe_occurrence",
                    "single_step_support_transition_comparison",
                ],
                "CST_GFG_SCAN_PROGRAM_ORDER_INVALID",
            )
        elif row["stage"] == "deep_four_branch_anchor":
            expected_anchor_order = [
                "anchor_prestate_restore",
                "current_runtime_rng_restore",
                "prestate_ungated_repeat",
                "full_update_occurrence",
                "parameter_only_basis_full_replay_occurrence",
                "optimizer_state_only_basis_full_replay_occurrence",
                "skip_update_occurrence",
                "parameter_only_occurrence",
                "optimizer_state_only_occurrence",
            ]
            for _branch in range(4):
                expected_anchor_order.extend(
                    [
                        "support_probe_occurrence",
                        "continuation_occurrence",
                        "support_probe_occurrence",
                        "continuation_occurrence",
                        "support_probe_occurrence",
                        "continuation_occurrence",
                        "support_probe_occurrence",
                    ]
                )
            expected_anchor_order.extend(["support_transition_comparison_occurrence"] * 4)
            require(occurrence_types == expected_anchor_order, "CST_GFG_ANCHOR_PROGRAM_ORDER_INVALID")
        object_ids.update(child["object_id"] for child in payload["objects"])
        occurrence_ids.update(child["occurrence_id"] for child in payload["occurrences"])
        fact_ids.update(child["fact_block_id"] for child in payload["fact_blocks"])
        tensor_objects.extend(child for child in payload["objects"] if child["object_kind"] == "content_addressed_tensor")
        relations.extend(payload["relations"])
        facts.extend(payload["fact_blocks"])
    all_source_ids = object_ids | origin_ids
    for fact in facts:
        require(fact["occurrence_id"] in occurrence_ids, "CST_GFG_FACT_OCCURRENCE_UNRESOLVED")
        require(fact["outcome"]["object_id"] in object_ids, "CST_GFG_FACT_OUTCOME_UNRESOLVED")
        require(bool(fact["sources"]), "CST_GFG_FACT_WITHOUT_SOURCE")
        require(all(source["source_id"] in all_source_ids for source in fact["sources"]), "CST_GFG_FACT_SOURCE_UNRESOLVED")
        require(all(bool(source["relation_role"]) for source in fact["sources"]), "CST_GFG_FACT_ROLE_EMPTY")
    require(set(row["relation_type"] for row in relations) <= ALLOWED_RELATIONS, "CST_GFG_RELATION_TYPE_UNREGISTERED")
    realizes = [row for row in relations if row["relation_type"] == "realizes_fact"]
    realized_counts = {fact_id: 0 for fact_id in fact_ids}
    for row in realizes:
        require(row["source_id"] in occurrence_ids and row["target_id"] in fact_ids, "CST_GFG_INCIDENCE_ENDPOINT_INVALID")
        realized_counts[row["target_id"]] += 1
    require(all(value == 1 for value in realized_counts.values()), "CST_GFG_INCIDENCE_NOT_EXACT")
    relation_ids = [row["relation_id"] for row in relations]
    require(len(relation_ids) == len(set(relation_ids)), "CST_GFG_RELATION_ID_DUPLICATE")
    reads_index = {
        (
            row["source_id"],
            row["target_id"],
            row["payload"].get("fact_block_id"),
            row["payload"].get("relation_role"),
        )
        for row in relations
        if row["relation_type"] == "reads_from"
    }
    origin_dependency_index = {
        (row["source_id"], row["target_id"], row["payload"].get("fact_block_id"))
        for row in relations
        if row["relation_type"] == "generated_origin_dependency"
    }
    for fact in facts:
        for source in fact["sources"]:
            require(
                (
                    source["source_id"],
                    fact["occurrence_id"],
                    fact["fact_block_id"],
                    source["relation_role"],
                )
                in reads_index,
                "CST_GFG_READS_FROM_NOT_EXACT",
            )
            if source["source_kind"] == "generated_origin":
                require(
                    (
                        source["source_id"],
                        fact["outcome"]["object_id"],
                        fact["fact_block_id"],
                    )
                    in origin_dependency_index,
                    "CST_GFG_GENERATED_ORIGIN_DEPENDENCY_MISSING",
                )
    for relation in relations:
        if relation["relation_type"] == "program_order":
            require(
                relation["source_id"] in occurrence_ids and relation["target_id"] in occurrence_ids,
                "CST_GFG_PROGRAM_ORDER_ENDPOINT_INVALID",
            )
    verified_tensor_payloads: set[tuple[str, str, str]] = set()
    for row in tensor_objects:
        value = row["payload"]
        identity = (value["locator"], value["file_sha256"], value["raw_tensor_sha256"])
        if identity not in verified_tensor_payloads:
            _verify_tensor_ref(entry_directory, value)
            verified_tensor_payloads.add(identity)
    training = TrainingGFG(source_training_bundle / "participant_gfg.sqlite3")
    csrg = SupportGFG(source_csrg_bundle / "support_gfg.sqlite3")
    _checkpoint_ids, csrg_catalog = _csrg_objects(csrg)
    try:
        origin_payloads = [json.loads(row["payload_json"]) for row in origin_rows]
        training_needed = {
            payload["source_object_id"]: int(payload["source_optimizer_step"])
            for payload in origin_payloads
            if payload["source_graph_schema"] == "participant-safe-training-gfg-bundle-v1"
        }
        training_catalog: dict[str, dict[str, Any]] = {}
        for step in sorted(set(training_needed.values())):
            stages = ("before_batch",) if step == 0 else ("after_optimizer_step",)
            for stage in stages:
                for source in objects_for_stage(training, step, stage):
                    if source["object_id"] in training_needed:
                        training_catalog[source["object_id"]] = source
        require(set(training_catalog) == set(training_needed), "CST_GFG_TRAINING_ORIGIN_CATALOG_INCOMPLETE")
        for row in origin_rows:
            payload = json.loads(row["payload_json"])
            schema = payload["source_graph_schema"]
            if schema == "participant-safe-training-gfg-bundle-v1":
                source = training_catalog[payload["source_object_id"]]
            elif schema == "nanogpt-support-redundancy-gfg-v1":
                source = csrg_catalog[payload["source_object_id"]]
            else:
                raise RuntimeError(f"CST_GFG_ORIGIN_SCHEMA_UNKNOWN:{schema}")
            require(source["content_sha256"] == payload["source_content_sha256"], "CST_GFG_ORIGIN_CONTENT_MISMATCH")
    finally:
        training.close()
        csrg.close()
        connection.close()
    matched = [row for row in relations if row["relation_type"] == "matched_branch_of"]
    require(len(matched) == 112, "CST_GFG_MATCHED_BRANCH_RELATION_COUNT_INVALID")
    material = {
        "block_count": len(blocks),
        "entry_id": entry_directory.name,
        "fact_block_count": len(facts),
        "five_coordinate_completeness": "PASS",
        "generated_origin_count": len(origin_rows),
        "generated_origin_resolution": "PASS",
        "incidence_exactness": "PASS",
        "participant_future_label_absence": "PASS",
        "matched_branch_relation_count": len(matched),
        "relation_count": len(relations),
        "schema": "nanogpt-support-transition-gfg-validation-v1",
        "status": "PASS",
        "tensor_payload_count": len(tensor_objects),
        "unique_tensor_payload_count": len(verified_tensor_payloads),
    }
    result = {**material, "validation_sha256": payload_sha256(material)}
    write_json(entry_directory / "support_transition_gfg_validation.json", result)
    return result
