from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import TrainingGFG
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import objects_for_stage
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import (
    GraphRef,
    SupportGFG,
    SupportGFGWriter,
)

from .contracts import ComponentRegistry, ProbeContract
from .stepwise_gfg import (
    _bind_all,
    _emit_probe,
    _external_tensor,
    _object,
    _occurrence,
    _source_row,
    _validation_probe_sources,
)


BRANCH_GRAPH_SCHEMA = "nanogpt-stepwise-causal-branch-gfg-v1"
BRANCH_BLOCK_SCHEMA = "nanogpt-stepwise-causal-branch-gfg-block-v1"
BRANCH_MANIFEST_SCHEMA = "nanogpt-stepwise-causal-branch-gfg-manifest-v1"
BRANCHES = ("full_step", "skip_step", "parameter_only", "optimizer_state_only")


def _branch_batch_sources(
    writer: SupportGFGWriter,
    *,
    batch: dict[str, Any],
    source_bundle_id: str,
    branch_contract_ref: GraphRef,
    semantic_prefix: str,
    optimizer_step: int,
) -> list[tuple[GraphRef, str]]:
    values = [
        (
            writer.origin(_source_row(reference), source_bundle_id=source_bundle_id),
            role,
        )
        for role, reference in sorted(batch["source_training_gfg_objects"].items())
    ]
    availability = batch.get("batch_selection_order_availability")
    if isinstance(availability, dict) and availability.get("outcome_kind") == "ExplicitDisposition":
        occurrence = _occurrence(
            writer,
            occurrence_type="batch_selection_order_availability_occurrence",
            optimizer_step=optimizer_step,
            operation="adjudicate_source_batch_selection_order_availability",
            contract_id="STEPWISE-FOUR-BRANCH-v1",
            payload={"semantic_prefix": semantic_prefix, "reconstruction_or_guess_used": False},
        )
        disposition = _object(
            writer,
            semantic_key=f"{semantic_prefix}:batch-selection-order-disposition",
            role="explicit_disposition",
            optimizer_step=optimizer_step,
            payload=availability,
            object_kind="ExplicitDisposition",
        )
        writer.bind(
            occurrence,
            values + [(branch_contract_ref, "frozen_batch_identity_availability_contract")],
            disposition,
            payload={"outcome_kind": "ExplicitDisposition"},
        )
    return values


def _main_state_origin(
    writer: SupportGFGWriter,
    *,
    main_manifest: dict[str, Any],
    main_objects: dict[str, dict[str, Any]],
    window_id: str,
    state_id: str,
    optimizer_step: int,
) -> GraphRef:
    key = f"{window_id}:{state_id}"
    require(key in main_manifest["state_catalog"], f"SST_BRANCH_GFG_MAIN_STATE_MISSING:{key}")
    source_object_id = str(main_manifest["state_catalog"][key])
    require(source_object_id in main_objects, f"SST_BRANCH_GFG_MAIN_STATE_OBJECT_MISSING:{source_object_id}")
    source_object = main_objects[source_object_id]
    require(
        int(source_object["optimizer_step"]) == optimizer_step,
        f"SST_BRANCH_GFG_MAIN_STATE_STEP_MISMATCH:{source_object_id}",
    )
    created = _object(
        writer,
        semantic_key=f"main-stepwise-gfg-state-origin:{key}",
        role="generated_main_training_state_origin",
        optimizer_step=optimizer_step,
        payload={
            "origin_kind": "GeneratedOrigin",
            "source_graph_schema": main_manifest["schema"],
            "source_graph_manifest_sha256": main_manifest["manifest_sha256"],
            "source_graph_database_sha256": main_manifest["database_sha256"],
            "source_object_id": source_object_id,
            "source_content_sha256": source_object["content_sha256"],
            "source_role": source_object["role"],
            "source_optimizer_step": source_object["optimizer_step"],
            "source_state_id": state_id,
            "source_window_id": window_id,
        },
        object_kind="GeneratedOrigin",
    )
    return GraphRef(created.object_id, created.content_sha256, created.role, "generated_origin")


def _main_object_index(database_path: Path, required_object_ids: set[str]) -> dict[str, dict[str, Any]]:
    graph = SupportGFG(database_path)
    found: dict[str, dict[str, Any]] = {}
    try:
        for _row, block in graph.blocks():
            for value in block["objects"]:
                object_id = str(value["object_id"])
                if object_id in required_object_ids:
                    found[object_id] = value
            if len(found) == len(required_object_ids):
                break
    finally:
        graph.close()
    require(set(found) == required_object_ids, "SST_BRANCH_GFG_MAIN_STATE_INDEX_INCOMPLETE")
    return found


def _tensor_refs(value: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict) and {"locator", "file_sha256", "raw_tensor_sha256", "shape", "dtype"} <= set(value):
        return [(prefix, value)]
    result: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            result.extend(_tensor_refs(child, f"{prefix}/{key}" if prefix else str(key)))
    return result


def _emit_seed_branch(
    writer: SupportGFGWriter,
    *,
    seed: dict[str, Any],
    branch: str,
    main_origin: GraphRef,
    batch_sources: list[tuple[GraphRef, str]],
    branch_contract_ref: GraphRef,
    full_step_ref: GraphRef | None = None,
) -> GraphRef:
    step = int(seed["optimizer_step"])
    if branch in ("full_step", "skip_step"):
        evidence = seed[branch]
        occurrence = _occurrence(
            writer,
            occurrence_type="actual_full_training_step_occurrence" if branch == "full_step" else "actual_forward_backward_clip_without_optimizer_occurrence",
            optimizer_step=step,
            operation="native_training_step" if branch == "full_step" else "native_forward_backward_clip_with_optimizer_skipped",
            contract_id="STEPWISE-FOUR-BRANCH-v1",
            payload={
                "seed_id": seed["seed_id"],
                "branch": branch,
                "execute_optimizer": evidence["execute_optimizer"],
                "loss": evidence["loss"],
                "total_gradient_norm": evidence["total_gradient_norm"],
                "step_evidence_sha256": evidence["step_evidence_sha256"],
            },
        )
        sources = [(main_origin, "immutable_prebranch_state"), *batch_sources, (branch_contract_ref, "frozen_branch_contract")]
        outcomes: list[tuple[GraphRef, str]] = []
        for name, reference in _tensor_refs(evidence):
            outcome = _external_tensor(
                writer,
                reference=reference,
                semantic_key=f"seed:{seed['seed_id']}:{branch}:{name}",
                role=f"{branch}:{name}",
                optimizer_step=step,
            )
            outcomes.append((outcome, name))
        _bind_all(writer, occurrence, sources, outcomes)
        summary = _object(
            writer,
            semantic_key=f"seed:{seed['seed_id']}:{branch}:summary",
            role=f"{branch}_seed_transition_summary",
            optimizer_step=step,
            payload={
                "seed_id": seed["seed_id"],
                "branch": branch,
                "execute_optimizer": evidence["execute_optimizer"],
                "step_evidence_sha256": evidence["step_evidence_sha256"],
            },
        )
        writer.bind(occurrence, sources + [(ref, role) for ref, role in outcomes], summary, payload={"outcome_kind": "branch_seed_transition"})
        if branch == "skip_step":
            disposition = _object(
                writer,
                semantic_key=f"seed:{seed['seed_id']}:skip:optimizer-disposition",
                role="explicit_disposition",
                optimizer_step=step,
                payload={
                    "outcome_kind": "ExplicitDisposition",
                    "disposition": "OPTIMIZER_STEP_SKIPPED_BY_FROZEN_BRANCH",
                },
                object_kind="ExplicitDisposition",
            )
            writer.bind(occurrence, sources, disposition, payload={"outcome_kind": "ExplicitDisposition"})
        return summary
    composition_key = "parameter_only_composition" if branch == "parameter_only" else "optimizer_state_only_composition"
    occurrence = _occurrence(
        writer,
        occurrence_type="explicit_branch_state_composition_occurrence",
        optimizer_step=step,
        operation=composition_key,
        contract_id="STEPWISE-FOUR-BRANCH-v1",
        payload={"seed_id": seed["seed_id"], "branch": branch, **seed[composition_key]},
    )
    outcome = _object(
        writer,
        semantic_key=f"seed:{seed['seed_id']}:{branch}:summary",
        role=f"{branch}_composed_state_summary",
        optimizer_step=step,
        payload={"seed_id": seed["seed_id"], "branch": branch, **seed[composition_key]},
    )
    require(full_step_ref is not None, f"SST_BRANCH_GFG_FULL_STEP_SOURCE_MISSING:{branch}")
    composition_sources = [
        (main_origin, "immutable_prebranch_state"),
        (
            full_step_ref,
            "actual_full_step_updated_parameters"
            if branch == "parameter_only"
            else "actual_full_step_updated_optimizer_state",
        ),
        (branch_contract_ref, "frozen_branch_composition_contract"),
    ]
    writer.bind(
        occurrence,
        composition_sources,
        outcome,
        payload={"outcome_kind": "explicit_analysis_state_composition"},
    )
    return outcome


def _emit_branch_state(
    writer: SupportGFGWriter,
    *,
    seed_id: str,
    branch: str,
    horizon: int,
    state: dict[str, Any],
    branch_prestate_ref: GraphRef,
    branch_contract_ref: GraphRef,
) -> GraphRef:
    step = int(state["physical_optimizer_opportunity"])
    occurrence = _occurrence(
        writer,
        occurrence_type="branch_horizon_state_materialization_occurrence",
        optimizer_step=step,
        operation="materialize_registered_branch_horizon_state",
        contract_id="STEPWISE-FOUR-BRANCH-v1",
        payload={"seed_id": seed_id, "branch": branch, "horizon": horizon},
    )
    outcome = _object(
        writer,
        semantic_key=f"seed:{seed_id}:horizon:{horizon}:{branch}:complete-state",
        role=f"{branch}_restorable_state_at_horizon",
        optimizer_step=step,
        payload={
            "seed_id": seed_id,
            "branch": branch,
            "horizon": horizon,
            "state": state["state"],
            "state_summary": state["state_summary"],
            "state_result_sha256": state["result_sha256"],
        },
        object_kind="restorable_branch_state",
    )
    writer.bind(
        occurrence,
        [(branch_prestate_ref, "actual_branch_state_at_registered_horizon"), (branch_contract_ref, "aligned_continuation_contract")],
        outcome,
        payload={"outcome_kind": "restorable_branch_state"},
    )
    return outcome


def _emit_branch_continuation(
    writer: SupportGFGWriter,
    *,
    seed_id: str,
    branch: str,
    continuation: dict[str, Any],
    branch_prestate_ref: GraphRef,
    batch_sources: list[tuple[GraphRef, str]],
    branch_contract_ref: GraphRef,
) -> GraphRef:
    step = int(continuation["physical_optimizer_step"])
    branch_step = continuation["branches"][branch]
    occurrence = _occurrence(
        writer,
        occurrence_type="actual_aligned_branch_training_step_occurrence",
        optimizer_step=step,
        operation="native_training_step_under_aligned_branch_continuation",
        contract_id="STEPWISE-FOUR-BRANCH-v1",
        payload={
            "seed_id": seed_id,
            "branch": branch,
            "from_state_sha256": branch_step["from_state_sha256"],
            "to_state_sha256": branch_step["to_state_sha256"],
            "same_external_rng_opportunity_all_branches": continuation[
                "same_external_rng_opportunity_all_branches"
            ],
            "training_opportunity_alignment": continuation["training_opportunity_alignment"],
        },
    )
    sources = [
        (branch_prestate_ref, "actual_branch_prestate"),
        *batch_sources,
        (branch_contract_ref, "frozen_aligned_continuation_contract"),
    ]
    outcome = _object(
        writer,
        semantic_key=f"seed:{seed_id}:continuation:{step}:{branch}:state-commitment",
        role=f"{branch}_actual_continuation_state_commitment",
        optimizer_step=step + 1,
        payload={
            "seed_id": seed_id,
            "branch": branch,
            "from_state_sha256": branch_step["from_state_sha256"],
            "to_state_sha256": branch_step["to_state_sha256"],
            "step": branch_step["step"],
        },
        object_kind="branch_state_commitment",
    )
    writer.bind(occurrence, sources, outcome, payload={"outcome_kind": "branch_state_commitment"})
    disposition = _object(
        writer,
        semantic_key=f"seed:{seed_id}:continuation:{step}:{branch}:intermediate-payload-disposition",
        role="explicit_disposition",
        optimizer_step=step,
        payload={
            "outcome_kind": "ExplicitDisposition",
            "disposition": "BRANCH_INTERMEDIATE_TENSOR_PAYLOAD_NOT_MATERIALIZED_UNDER_FROZEN_PROFILE",
            "branch": branch,
            "physical_optimizer_step": step,
            "complete_tensor_commitments_preserved_in": outcome.object_id,
        },
        object_kind="ExplicitDisposition",
    )
    writer.bind(occurrence, sources, disposition, payload={"outcome_kind": "ExplicitDisposition"})
    return outcome


def _emit_effects(
    writer: SupportGFGWriter,
    *,
    seed_id: str,
    horizon: int,
    effects: dict[str, Any],
    probe_summaries: dict[str, GraphRef],
    branch_contract_ref: GraphRef,
    optimizer_step: int,
) -> None:
    occurrence = _occurrence(
        writer,
        occurrence_type="matched_four_branch_causal_contrast_occurrence",
        optimizer_step=optimizer_step,
        operation="compute_full_skip_parameter_optimizer_causal_contrasts",
        contract_id="STEPWISE-FOUR-BRANCH-v1",
        payload={
            "seed_id": seed_id,
            "horizon": horizon,
            "trajectory_finite_difference_conflated": False,
            "Phi_full_is_D_U_S_k": horizon == 1,
        },
    )
    sources = [
        *[(probe_summaries[branch], f"{branch}_support_observation") for branch in BRANCHES],
        (branch_contract_ref, "frozen_causal_contrast_contract"),
    ]
    outcomes: list[tuple[GraphRef, str]] = []
    tensor_effects = _tensor_refs(effects["numeric_effects"], "numeric_effects") + _tensor_refs(
        effects["categorical_effects"], "categorical_effects"
    )
    for name, reference in tensor_effects:
        is_full_minus_skip = "/Phi_full" in name or "/full_step/changed_mask" in name
        role = "actual_update_causal_difference_D_U_S_k" if horizon == 1 and is_full_minus_skip else "branch_causal_contrast"
        outcome = _external_tensor(
            writer,
            reference=reference,
            semantic_key=f"seed:{seed_id}:horizon:{horizon}:effect:{name}",
            role=f"{role}:{name}",
            optimizer_step=optimizer_step,
        )
        outcomes.append((outcome, role))
    _bind_all(writer, occurrence, sources, outcomes)
    summary = _object(
        writer,
        semantic_key=f"seed:{seed_id}:horizon:{horizon}:causal-contrast-summary",
        role="matched_four_branch_causal_contrast_summary",
        optimizer_step=optimizer_step,
        payload={
            "seed_id": seed_id,
            "horizon": horizon,
            "effect_sha256": effects["effect_sha256"],
            "causal_difference_semantics": effects["causal_difference_semantics"],
            "branch_probe_observation_ids": effects["branch_probe_observation_ids"],
            "numeric_effect_tensor_count": len(_tensor_refs(effects["numeric_effects"])),
            "categorical_effect_tensor_count": len(_tensor_refs(effects["categorical_effects"])),
        },
    )
    writer.bind(occurrence, sources + outcomes, summary, payload={"outcome_kind": "causal_contrast_summary"})


def build_entry_branch_gfg(
    *,
    entry_id: str,
    branch_root: Path,
    formal_root: Path,
    source_bundle: Path,
    branch_profile_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
) -> dict[str, Any]:
    branch_entry_root = branch_root / entry_id
    seed_roots = sorted((branch_entry_root / "branch-seeds").glob("branch-seed-*"))
    require(bool(seed_roots), f"SST_BRANCH_GFG_SEEDS_EMPTY:{entry_id}")
    main_manifest = read_json(formal_root / entry_id / "stepwise_support_transition_gfg_manifest.json")
    main_database_path = formal_root / entry_id / str(main_manifest["database"])
    require(file_sha256(main_database_path) == main_manifest["database_sha256"], "SST_BRANCH_GFG_MAIN_DATABASE_HASH_MISMATCH")
    required_main_object_ids: set[str] = set()
    for seed_root in seed_roots:
        seed = read_json(seed_root / "seed_result.json")
        state_key = f"{seed['window_id']}:{seed['immutable_prestate_id']}"
        require(state_key in main_manifest["state_catalog"], f"SST_BRANCH_GFG_MAIN_STATE_MISSING:{state_key}")
        required_main_object_ids.add(str(main_manifest["state_catalog"][state_key]))
    main_objects = _main_object_index(main_database_path, required_main_object_ids)
    source_manifest = read_json(source_bundle / "manifest.json")
    source_bundle_id = str(source_manifest["bundle_manifest_sha256"])
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    database_path = branch_entry_root / "stepwise_causal_branch_gfg.sqlite3"
    require(not database_path.exists(), f"SST_BRANCH_GFG_DATABASE_ALREADY_EXISTS:{database_path}")
    writer = SupportGFGWriter(
        database_path,
        branch_entry_root / "tensor-objects",
        scope_id=f"stepwise-causal-branches:{entry_id}",
        source_bundle_id=source_bundle_id,
        contract_sha256=file_sha256(branch_profile_path),
        graph_schema=BRANCH_GRAPH_SCHEMA,
        block_schema=BRANCH_BLOCK_SCHEMA,
        manifest_schema=BRANCH_MANIFEST_SCHEMA,
    )
    source_graph = TrainingGFG(source_bundle / "participant_gfg.sqlite3")
    seed_catalog: dict[str, dict[str, str]] = {}
    try:
        writer.start_block("branch_graph_contracts", 0)
        branch_contract_ref = _object(
            writer,
            semantic_key="contract:stepwise-four-branch",
            role="frozen_four_branch_causal_contract",
            optimizer_step=0,
            payload={"contract": read_json(branch_profile_path), "file_sha256": file_sha256(branch_profile_path)},
            object_kind="frozen_contract",
        )
        registry_ref = _object(
            writer,
            semantic_key=f"component-registry:{registry.registry_id}",
            role="versioned_component_registry",
            optimizer_step=0,
            payload={"registry": read_json(component_registry_path), "file_sha256": registry.source_sha256},
            object_kind="frozen_contract",
        )
        probe_contract_ref = _object(
            writer,
            semantic_key=f"probe-contract:{probe_contract.probe_contract_id}",
            role="versioned_probe_contract",
            optimizer_step=0,
            payload={"contract": read_json(probe_contract_path), "file_sha256": probe_contract.source_sha256},
            object_kind="frozen_contract",
        )
        validation_sources = _validation_probe_sources(
            writer,
            source_graph=source_graph,
            source_bundle=source_bundle,
            source_bundle_id=source_bundle_id,
            protocol_ref=branch_contract_ref,
        )
        writer.flush_block()
        for ordinal, seed_root in enumerate(seed_roots, start=1):
            seed = read_json(seed_root / "seed_result.json")
            receipt = read_json(seed_root / "branch_receipt.json")
            writer._last_occurrence_id = None
            writer.start_block(f"seed:{seed['seed_id']}:initial", int(seed["optimizer_step"]))
            main_origin = _main_state_origin(
                writer,
                main_manifest=main_manifest,
                main_objects=main_objects,
                window_id=str(seed["window_id"]),
                state_id=str(seed["immutable_prestate_id"]),
                optimizer_step=int(seed["optimizer_step"]),
            )
            batch_sources = _branch_batch_sources(
                writer,
                batch=seed["batch"],
                source_bundle_id=source_bundle_id,
                branch_contract_ref=branch_contract_ref,
                semantic_prefix=f"seed:{seed['seed_id']}:initial",
                optimizer_step=int(seed["optimizer_step"]),
            )
            full_step_ref = _emit_seed_branch(
                writer,
                seed=seed,
                branch="full_step",
                main_origin=main_origin,
                batch_sources=batch_sources,
                branch_contract_ref=branch_contract_ref,
            )
            skip_step_ref = _emit_seed_branch(
                writer,
                seed=seed,
                branch="skip_step",
                main_origin=main_origin,
                batch_sources=batch_sources,
                branch_contract_ref=branch_contract_ref,
            )
            branch_refs = {
                "full_step": full_step_ref,
                "skip_step": skip_step_ref,
                "parameter_only": _emit_seed_branch(
                    writer,
                    seed=seed,
                    branch="parameter_only",
                    main_origin=main_origin,
                    batch_sources=batch_sources,
                    branch_contract_ref=branch_contract_ref,
                    full_step_ref=full_step_ref,
                ),
                "optimizer_state_only": _emit_seed_branch(
                    writer,
                    seed=seed,
                    branch="optimizer_state_only",
                    main_origin=main_origin,
                    batch_sources=batch_sources,
                    branch_contract_ref=branch_contract_ref,
                    full_step_ref=full_step_ref,
                ),
            }
            writer.flush_block()
            horizon_catalog: dict[str, str] = {}
            horizon_rows = {int(row["horizon"]): row for row in receipt["horizon_results"]}
            require(set(horizon_rows) == set(receipt["legal_horizons"]), "SST_BRANCH_GFG_HORIZON_SET_MISMATCH")
            max_horizon = max(horizon_rows)
            continuation_rows = {
                int(row["physical_optimizer_step"]): read_json(
                    seed_root
                    / "continuations"
                    / f"step-{int(row['physical_optimizer_step']):05d}-to-{int(row['physical_optimizer_step']) + 1:05d}.json"
                )
                for row in receipt["continuation_results"]
            }
            current_branch_refs = dict(branch_refs)
            for horizon in range(1, max_horizon + 1):
                if horizon in horizon_rows:
                    writer.start_block(f"seed:{seed['seed_id']}:horizon", int(seed["optimizer_step"]) + horizon)
                    probe_summaries: dict[str, GraphRef] = {}
                    horizon_state_refs: dict[str, GraphRef] = {}
                    for branch in BRANCHES:
                        state_path = seed_root / "horizons" / f"h-{horizon:03d}" / f"{branch}-state.json"
                        state = read_json(state_path)
                        state_ref = _emit_branch_state(
                            writer,
                            seed_id=str(seed["seed_id"]),
                            branch=branch,
                            horizon=horizon,
                            state=state,
                            branch_prestate_ref=current_branch_refs[branch],
                            branch_contract_ref=branch_contract_ref,
                        )
                        horizon_state_refs[branch] = state_ref
                        observation = read_json(
                            branch_entry_root
                            / "probe-observations"
                            / probe_contract.probe_contract_id
                            / f"{state['state']['state_id']}.json"
                        )
                        probe_summaries[branch] = _emit_probe(
                            writer,
                            observation=observation,
                            state_origin=state_ref,
                            validation_sources=validation_sources,
                            contract_ref=probe_contract_ref,
                            registry_ref=registry_ref,
                            semantic_prefix=f"seed:{seed['seed_id']}:horizon:{horizon}:{branch}",
                            optimizer_step=int(seed["optimizer_step"]) + horizon,
                            probe_contract=probe_contract,
                        )
                        horizon_catalog[f"{horizon}:{branch}"] = state_ref.object_id
                    current_branch_refs = horizon_state_refs
                    effects = read_json(seed_root / "horizons" / f"h-{horizon:03d}" / "effects.json")
                    _emit_effects(
                        writer,
                        seed_id=str(seed["seed_id"]),
                        horizon=horizon,
                        effects=effects,
                        probe_summaries=probe_summaries,
                        branch_contract_ref=branch_contract_ref,
                        optimizer_step=int(seed["optimizer_step"]) + horizon,
                    )
                    writer.flush_block()
                if horizon == max_horizon:
                    continue
                physical_step = int(seed["optimizer_step"]) + horizon
                require(physical_step in continuation_rows, f"SST_BRANCH_GFG_CONTINUATION_MISSING:{physical_step}")
                continuation = continuation_rows[physical_step]
                writer.start_block(f"seed:{seed['seed_id']}:continuation", physical_step)
                continuation_batch_sources = _branch_batch_sources(
                    writer,
                    batch=continuation["same_batch_all_branches"],
                    source_bundle_id=source_bundle_id,
                    branch_contract_ref=branch_contract_ref,
                    semantic_prefix=f"seed:{seed['seed_id']}:continuation:{physical_step}",
                    optimizer_step=physical_step,
                )
                current_branch_refs = {
                    branch: _emit_branch_continuation(
                        writer,
                        seed_id=str(seed["seed_id"]),
                        branch=branch,
                        continuation=continuation,
                        branch_prestate_ref=current_branch_refs[branch],
                        batch_sources=continuation_batch_sources,
                        branch_contract_ref=branch_contract_ref,
                    )
                    for branch in BRANCHES
                }
                writer.flush_block()
            seed_catalog[str(seed["seed_id"])] = horizon_catalog
            print({"event": "SST_BRANCH_GFG_SEED_COMPLETE", "ordinal": ordinal, "seed_count": len(seed_roots), "seed_id": seed["seed_id"]}, flush=True)
        manifest = writer.close()
    finally:
        source_graph.close()
    material = {
        **manifest,
        "entry_id": entry_id,
        "source_bundle_manifest_sha256": source_bundle_id,
        "source_gfg_database_sha256": source_manifest["gfg_database_sha256"],
        "main_stepwise_gfg_manifest_sha256": main_manifest["manifest_sha256"],
        "branch_profile_sha256": file_sha256(branch_profile_path),
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "seed_count": len(seed_roots),
        "seed_catalog": seed_catalog,
    }
    result = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(branch_entry_root / "stepwise_causal_branch_gfg_manifest.json", result)
    return result
