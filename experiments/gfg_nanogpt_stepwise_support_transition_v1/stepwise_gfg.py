from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    read_json,
    require,
    write_json,
)
from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.training_gfg import TrainingGFG
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import objects_for_stage
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import load_tensor
from experiments.gfg_nanogpt_support_redundancy_v1.support_gfg import (
    GraphRef,
    SupportGFGWriter,
)
from experiments.gfg_nanogpt_support_redundancy_v1.task_mapping import (
    recover_cyclic_target_mapping,
)

from .contracts import ComponentRegistry, ProbeContract


GRAPH_SCHEMA = "nanogpt-stepwise-support-transition-gfg-v1"
BLOCK_SCHEMA = "nanogpt-stepwise-support-transition-gfg-block-v1"
MANIFEST_SCHEMA = "nanogpt-stepwise-support-transition-gfg-manifest-v1"


def _external_tensor(
    writer: SupportGFGWriter,
    *,
    reference: dict[str, Any],
    semantic_key: str,
    role: str,
    optimizer_step: int,
) -> GraphRef:
    return writer.object(
        semantic_key=semantic_key,
        role=role,
        optimizer_step=optimizer_step,
        payload=reference,
        object_kind="content_addressed_tensor",
    )


def _object(
    writer: SupportGFGWriter,
    *,
    semantic_key: str,
    role: str,
    optimizer_step: int,
    payload: dict[str, Any],
    object_kind: str = "analysis_result",
) -> GraphRef:
    return writer.object(
        semantic_key=semantic_key,
        role=role,
        optimizer_step=optimizer_step,
        payload=payload,
        object_kind=object_kind,
    )


def _generated_origin(
    writer: SupportGFGWriter,
    *,
    outcome: GraphRef,
    semantic_key: str,
    optimizer_step: int,
    role: str = "generated_training_state_origin",
) -> GraphRef:
    payload = {
        "origin_kind": "GeneratedOrigin",
        "source_graph_schema": GRAPH_SCHEMA,
        "source_object_id": outcome.object_id,
        "source_content_sha256": outcome.content_sha256,
        "source_role": outcome.role,
    }
    created = _object(
        writer,
        semantic_key=semantic_key,
        role=role,
        optimizer_step=optimizer_step,
        payload=payload,
        object_kind="GeneratedOrigin",
    )
    writer.relation(
        "generated_origin_dependency",
        outcome.object_id,
        created.object_id,
        {"basis": "same_graph_generated_result_continuity"},
    )
    return GraphRef(created.object_id, created.content_sha256, created.role, "generated_origin")


def _occurrence(
    writer: SupportGFGWriter,
    *,
    occurrence_type: str,
    optimizer_step: int,
    operation: str,
    contract_id: str,
    payload: dict[str, Any],
) -> str:
    return writer.occurrence(
        occurrence_type=occurrence_type,
        optimizer_step=optimizer_step,
        transform_reference={"operation": operation, "contract_id": contract_id},
        payload=payload,
    )


def _bind_all(
    writer: SupportGFGWriter,
    occurrence_id: str,
    sources: list[tuple[GraphRef, str]],
    outcomes: Iterable[tuple[GraphRef, str]],
) -> None:
    for outcome, outcome_kind in outcomes:
        writer.bind(
            occurrence_id,
            sources,
            outcome,
            payload={"outcome_kind": outcome_kind},
        )


def _source_row(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": reference["object_id"],
        "content_sha256": reference["content_sha256"],
        "optimizer_step": int(reference["optimizer_step"]),
        "role": reference["role"],
        "semantic_key": reference["semantic_key"],
    }


def _initial_state_sources(
    writer: SupportGFGWriter,
    *,
    source: dict[str, Any],
    optimizer_step: int,
    source_bundle_id: str,
) -> list[tuple[GraphRef, str]]:
    if "source_checkpoint" in source:
        source = source["source_checkpoint"]
    values: list[tuple[GraphRef, str]] = []
    for group, role in (
        ("parameter_object_ids", "restored_parameter_version"),
        ("optimizer_object_ids", "restored_optimizer_state"),
    ):
        hashes = source[group.replace("object_ids", "sha256")]
        for semantic_name, object_id in sorted(source[group].items()):
            row = {
                "object_id": object_id,
                "content_sha256": hashes[semantic_name],
                "optimizer_step": optimizer_step,
                "role": "parameter_version" if group.startswith("parameter") else "optimizer_state",
                "semantic_key": semantic_name,
            }
            values.append(
                (
                    writer.origin(
                        row,
                        source_bundle_id=source_bundle_id,
                        source_graph_schema="participant-safe-training-gfg-bundle-v1",
                    ),
                    role + ":" + semantic_name,
                )
            )
    return values


def _batch_sources(
    writer: SupportGFGWriter,
    *,
    batch: dict[str, Any],
    source_bundle_id: str,
    protocol_ref: GraphRef,
    semantic_prefix: str,
    optimizer_step: int,
    relation_role_prefix: str = "",
) -> list[tuple[GraphRef, str]]:
    values: list[tuple[GraphRef, str]] = []
    for role, reference in sorted(batch["source_training_gfg_objects"].items()):
        values.append(
            (
                writer.origin(
                    _source_row(reference),
                    source_bundle_id=source_bundle_id,
                    source_graph_schema="participant-safe-training-gfg-bundle-v1",
                ),
                relation_role_prefix + role,
            )
        )
    availability = batch.get("batch_selection_order_availability")
    if isinstance(availability, dict) and availability.get("outcome_kind") == "ExplicitDisposition":
        occurrence = _occurrence(
            writer,
            occurrence_type="batch_selection_order_availability_occurrence",
            optimizer_step=optimizer_step,
            operation="adjudicate_source_batch_selection_order_availability",
            contract_id="STEPWISE-SUPPORT-TRANSITION-PHASE-v3",
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
            values + [(protocol_ref, "frozen_batch_identity_availability_contract")],
            disposition,
            payload={"outcome_kind": "ExplicitDisposition"},
        )
    return values


def _warmup_batch_sources(
    writer: SupportGFGWriter,
    *,
    source: dict[str, Any],
    source_bundle_id: str,
    protocol_ref: GraphRef,
    window_id: str,
) -> list[tuple[GraphRef, str]]:
    warmup = source.get("replay_warmup")
    if not isinstance(warmup, dict):
        return []
    values: list[tuple[GraphRef, str]] = []
    for step in warmup["steps"]:
        optimizer_step = int(step["optimizer_step"])
        values.extend(
            _batch_sources(
                writer,
                batch=step["batch"],
                source_bundle_id=source_bundle_id,
                protocol_ref=protocol_ref,
                semantic_prefix=f"window:{window_id}:warmup:{optimizer_step}",
                optimizer_step=optimizer_step,
                relation_role_prefix=f"warmup_step_{optimizer_step}_",
            )
        )
    return values


def _emit_state(
    writer: SupportGFGWriter,
    *,
    occurrence_id: str,
    sources: list[tuple[GraphRef, str]],
    state_record: dict[str, Any],
    semantic_prefix: str,
    optimizer_step: int,
) -> tuple[GraphRef, GraphRef]:
    manifest = state_record["state"]
    tensor_outcomes = [
        (
            _external_tensor(
                writer,
                reference=manifest[key],
                semantic_key=f"{semantic_prefix}:state:{key}",
                role="restorable_" + key,
                optimizer_step=optimizer_step,
            ),
            key,
        )
        for key in ("parameters", "optimizer_exp_avg", "optimizer_exp_avg_sq", "optimizer_steps")
    ]
    _bind_all(writer, occurrence_id, sources, tensor_outcomes)
    summary = _object(
        writer,
        semantic_key=f"{semantic_prefix}:state:summary",
        role="complete_restorable_training_state",
        optimizer_step=optimizer_step,
        payload={
            "state_id": manifest["state_id"],
            "entry_id": manifest["entry_id"],
            "optimizer_step": optimizer_step,
            "commitment": manifest["commitment"],
            "state_summary": state_record["state_summary"],
            "restorable_without_training_reexecution": manifest["restorable_without_training_reexecution"],
            "tensor_outcome_object_ids": [ref.object_id for ref, _kind in tensor_outcomes],
        },
        object_kind="restorable_training_state",
    )
    writer.bind(
        occurrence_id,
        sources + [(ref, "complete_state_tensor") for ref, _kind in tensor_outcomes],
        summary,
        payload={"outcome_kind": "complete_restorable_training_state"},
    )
    origin = _generated_origin(
        writer,
        outcome=summary,
        semantic_key=f"{semantic_prefix}:state:generated-origin",
        optimizer_step=optimizer_step,
    )
    return summary, origin


def _emit_probe(
    writer: SupportGFGWriter,
    *,
    observation: dict[str, Any],
    state_origin: GraphRef,
    validation_sources: list[tuple[GraphRef, str]],
    contract_ref: GraphRef,
    registry_ref: GraphRef,
    semantic_prefix: str,
    optimizer_step: int,
    probe_contract: ProbeContract,
) -> GraphRef:
    plans = [()] * probe_contract.baseline_repetitions + list(probe_contract.gate_sets)
    require(len(plans) == len(observation["forwards"]), "SST_GFG_PROBE_FORWARD_COUNT_MISMATCH")
    forward_outcomes: list[tuple[GraphRef, str]] = []
    for index, (plan, row) in enumerate(zip(plans, observation["forwards"])):
        require(tuple(row["gate_components"]) == tuple(plan), "SST_GFG_PROBE_GATE_PLAN_MISMATCH")
        occurrence = _occurrence(
            writer,
            occurrence_type="baseline_probe_occurrence" if not plan else "gated_support_probe_occurrence",
            optimizer_step=optimizer_step,
            operation="model.forward" if not plan else "model.forward_with_registered_component_gates",
            contract_id=probe_contract.probe_contract_id,
            payload={
                "probe_contract_id": observation["probe_contract_id"],
                "component_registry_id": observation["component_registry_id"],
                "forward_ordinal": index,
                "gate_components": list(plan),
            },
        )
        sources = [
            (state_origin, "complete_preprobe_training_state"),
            *validation_sources,
            (contract_ref, "versioned_probe_contract"),
            (registry_ref, "versioned_component_registry"),
        ]
        outputs: list[tuple[GraphRef, str]] = []
        for key, role in (
            ("logits", "complete_decision_logits"),
            ("margins", "complete_per_example_margins"),
            ("predictions", "complete_predictions"),
            ("group_membership", "complete_group_membership"),
            ("group_q10_margin", "complete_target_group_q10_margins"),
        ):
            ref = _external_tensor(
                writer,
                reference=row[key],
                semantic_key=f"{semantic_prefix}:probe:{observation['probe_contract_id']}:forward:{index}:{key}",
                role=role,
                optimizer_step=optimizer_step,
            )
            outputs.append((ref, role))
            forward_outcomes.append((ref, f"forward_{index}_{role}"))
        _bind_all(writer, occurrence, sources, outputs)

    derive = _occurrence(
        writer,
        occurrence_type="support_metric_derivation_occurrence",
        optimizer_step=optimizer_step,
        operation="derive_contract_registered_support_metrics",
        contract_id=probe_contract.probe_contract_id,
        payload={
            "probe_contract_id": observation["probe_contract_id"],
            "component_registry_id": observation["component_registry_id"],
            "raw_forward_count": len(forward_outcomes) // 5,
        },
    )
    metric_sources = [
        (state_origin, "observed_training_state"),
        (contract_ref, "versioned_probe_contract"),
        (registry_ref, "versioned_component_registry"),
        *forward_outcomes,
    ]
    metric_outcomes: list[tuple[GraphRef, str]] = []
    for key in (
        "necessity",
        "pair_backup",
        "single_failure_slack",
        "double_failure_slack",
        "support_allocation",
        "support_concentration",
        "effective_support",
    ):
        ref = _external_tensor(
            writer,
            reference=observation[key],
            semantic_key=f"{semantic_prefix}:probe:{observation['probe_contract_id']}:metric:{key}",
            role="support_metric_" + key,
            optimizer_step=optimizer_step,
        )
        metric_outcomes.append((ref, key))
    _bind_all(writer, derive, metric_sources, metric_outcomes)
    summary = _object(
        writer,
        semantic_key=f"{semantic_prefix}:probe:{observation['probe_contract_id']}:summary",
        role="complete_probe_observation",
        optimizer_step=optimizer_step,
        payload={
            key: observation[key]
            for key in (
                "probe_observation_id",
                "observed_state_id",
                "probe_contract_id",
                "probe_contract_sha256",
                "component_registry_id",
                "component_registry_sha256",
                "component_ids",
                "pair_ids",
                "actual_forward_count",
                "baseline_byte_exact",
                "capability_accuracy",
                "component_loads",
                "undefined_effective_support_groups",
                "append_only_observation_layer",
            )
        },
    )
    writer.bind(
        derive,
        metric_sources + [(ref, "derived_support_metric") for ref, _kind in metric_outcomes],
        summary,
        payload={"outcome_kind": "complete_probe_observation"},
    )
    for group_index in observation["undefined_effective_support_groups"]:
        disposition = _object(
            writer,
            semantic_key=f"{semantic_prefix}:probe:{observation['probe_contract_id']}:effective-support-disposition:{group_index}",
            role="explicit_disposition",
            optimizer_step=optimizer_step,
            payload={
                "outcome_kind": "ExplicitDisposition",
                "disposition": "EFFECTIVE_SUPPORT_UNDEFINED_ZERO_TOTAL_NECESSITY",
                "target_group_index": int(group_index),
            },
            object_kind="ExplicitDisposition",
        )
        writer.bind(
            derive,
            metric_sources,
            disposition,
            payload={"outcome_kind": "ExplicitDisposition"},
        )
    return summary


def _emit_training_step(
    writer: SupportGFGWriter,
    *,
    transition: dict[str, Any],
    prestate_origin: GraphRef,
    batch_sources: list[tuple[GraphRef, str]],
    protocol_ref: GraphRef,
    poststate_record: dict[str, Any],
    semantic_prefix: str,
    optimizer_step: int,
    contract_id: str,
) -> tuple[GraphRef, GraphRef]:
    step = transition["step"]
    rng_occurrence = _occurrence(
        writer,
        occurrence_type="runtime_rng_configuration_occurrence",
        optimizer_step=optimizer_step,
        operation="set_content_derived_runtime_rng",
        contract_id=contract_id,
        payload={"historical_rng_restoration_claimed": False},
    )
    rng = _object(
        writer,
        semantic_key=f"{semantic_prefix}:rng-before",
        role="current_runtime_rng_state",
        optimizer_step=optimizer_step,
        payload=step["rng_before"],
    )
    writer.bind(rng_occurrence, [(protocol_ref, "frozen_rng_policy")], rng)

    forward = _occurrence(
        writer,
        occurrence_type="forward_occurrence",
        optimizer_step=optimizer_step,
        operation="native_nanogpt_model_forward",
        contract_id=contract_id,
        payload={"loss_is_forward_outcome": True},
    )
    forward_sources = [
        (prestate_origin, "complete_pretraining_state"),
        *batch_sources,
        (rng, "current_runtime_rng_state"),
        (protocol_ref, "frozen_stepwise_capture_protocol"),
    ]
    logits = _external_tensor(
        writer,
        reference=step["training_logits"],
        semantic_key=f"{semantic_prefix}:forward:logits",
        role="training_forward_logits",
        optimizer_step=optimizer_step,
    )
    activations = _external_tensor(
        writer,
        reference=step["registered_component_activations"],
        semantic_key=f"{semantic_prefix}:forward:registered-component-activations",
        role="registered_component_activations",
        optimizer_step=optimizer_step,
    )
    loss = _object(
        writer,
        semantic_key=f"{semantic_prefix}:forward:loss",
        role="training_loss",
        optimizer_step=optimizer_step,
        payload={"loss": step["loss"]},
    )
    _bind_all(writer, forward, forward_sources, [(logits, "logits"), (activations, "activations"), (loss, "loss")])

    backward = _occurrence(
        writer,
        occurrence_type="backward_occurrence",
        optimizer_step=optimizer_step,
        operation="native_torch_autograd_backward",
        contract_id=contract_id,
        payload={},
    )
    raw_gradients = _external_tensor(
        writer,
        reference=step["raw_gradients"],
        semantic_key=f"{semantic_prefix}:backward:raw-gradients",
        role="complete_named_raw_gradients",
        optimizer_step=optimizer_step,
    )
    writer.bind(backward, [(loss, "loss_source"), (prestate_origin, "differentiated_parameter_state")], raw_gradients)

    clipping = _occurrence(
        writer,
        occurrence_type="gradient_clip_occurrence",
        optimizer_step=optimizer_step,
        operation="torch_clip_grad_norm",
        contract_id=contract_id,
        payload={"gradient_clip": step["optimizer_config"]["gradient_clip"]},
    )
    clipped = _external_tensor(
        writer,
        reference=step["clipped_gradients"],
        semantic_key=f"{semantic_prefix}:gradient-clip:clipped-gradients",
        role="complete_named_clipped_gradients",
        optimizer_step=optimizer_step,
    )
    gradient_norm = _object(
        writer,
        semantic_key=f"{semantic_prefix}:gradient-clip:total-norm",
        role="total_gradient_norm",
        optimizer_step=optimizer_step,
        payload={"total_gradient_norm": step["total_gradient_norm"]},
    )
    _bind_all(writer, clipping, [(raw_gradients, "unclipped_gradient")], [(clipped, "clipped_gradient"), (gradient_norm, "total_gradient_norm")])

    optimizer = _occurrence(
        writer,
        occurrence_type="optimizer_step_occurrence",
        optimizer_step=optimizer_step,
        operation="native_fused_adamw_step",
        contract_id=contract_id,
        payload={
            "adam_and_parameter_changes_are_joint_outcomes": True,
            "optimizer_config": step["optimizer_config"],
        },
    )
    optimizer_sources = [
        (prestate_origin, "complete_preoptimizer_state"),
        (clipped, "clipped_gradient"),
        (gradient_norm, "gradient_norm"),
        (protocol_ref, "frozen_optimizer_profile"),
    ]
    optimizer_outcomes: list[tuple[GraphRef, str]] = []
    for key, reference, role in (
        ("parameter-update", step["parameter_update"], "complete_named_parameter_update"),
        ("nominal-weight-decay", step["nominal_weight_decay_update"], "nominal_weight_decay_update"),
        ("adaptive-update-residual", step["adaptive_update_residual"], "adaptive_update_residual"),
        ("exp-avg-delta", step["optimizer_deltas"]["exp_avg"], "complete_named_exp_avg_delta"),
        ("exp-avg-sq-delta", step["optimizer_deltas"]["exp_avg_sq"], "complete_named_exp_avg_sq_delta"),
        ("adam-step-delta", step["optimizer_deltas"]["adam_step"], "complete_named_adam_step_delta"),
        ("preconditioned-direction", step["optimizer_deltas"]["post_preconditioned_direction"], "complete_named_post_preconditioned_direction"),
    ):
        outcome = _external_tensor(
            writer,
            reference=reference,
            semantic_key=f"{semantic_prefix}:optimizer:{key}",
            role=role,
            optimizer_step=optimizer_step,
        )
        optimizer_outcomes.append((outcome, role))
    _bind_all(writer, optimizer, optimizer_sources, optimizer_outcomes)
    return _emit_state(
        writer,
        occurrence_id=optimizer,
        sources=optimizer_sources + [(ref, role) for ref, role in optimizer_outcomes],
        state_record=poststate_record,
        semantic_prefix=f"{semantic_prefix}:post",
        optimizer_step=optimizer_step + 1,
    )


def _emit_phase_tensor_group(
    writer: SupportGFGWriter,
    *,
    references: dict[str, Any],
    sources: list[tuple[GraphRef, str]],
    semantic_prefix: str,
    optimizer_step: int,
    occurrence_type: str,
    operation: str,
    contract_id: str,
    temporal_role: str,
) -> None:
    occurrence = _occurrence(
        writer,
        occurrence_type=occurrence_type,
        optimizer_step=optimizer_step,
        operation=operation,
        contract_id=contract_id,
        payload={"temporal_role": temporal_role, "result_field_count": len(references)},
    )
    outcomes: list[tuple[GraphRef, str]] = []
    for name, reference in sorted(references.items()):
        require(reference["temporal_role"] == temporal_role, "SST_GFG_PHASE_TEMPORAL_ROLE_MISMATCH")
        outcome = _external_tensor(
            writer,
            reference=reference,
            semantic_key=f"{semantic_prefix}:{name}",
            role=f"{temporal_role}:{operation}:{name}",
            optimizer_step=optimizer_step,
        )
        outcomes.append((outcome, name))
    _bind_all(writer, occurrence, sources, outcomes)


def _emit_phase_window_layer(
    writer: SupportGFGWriter,
    *,
    entry_root: Path,
    window: dict[str, Any],
    probe_summary_catalog: dict[int, GraphRef],
    phase_contract_ref: GraphRef,
    phase_protocol: dict[str, Any],
    phase_protocol_sha256: str,
) -> None:
    window_id = str(window["window_id"])
    scientific_start = int(window["scientific_start_optimizer_step"])
    scientific_end = int(window["scientific_end_optimizer_step"])
    origin_cache: dict[int, GraphRef] = {}

    def origin(step: int) -> GraphRef:
        require(step in probe_summary_catalog, f"SST_GFG_PHASE_PROBE_SUMMARY_MISSING:{window_id}:{step}")
        if step not in origin_cache:
            origin_cache[step] = _generated_origin(
                writer,
                outcome=probe_summary_catalog[step],
                semantic_key=f"window:{window_id}:step:{step}:probe-generated-origin",
                optimizer_step=step,
                role="generated_support_observation_origin",
            )
        return origin_cache[step]

    for step in range(scientific_start, scientific_end + 1):
        writer.start_block(f"window:{window_id}:phase", step)
        record_path = (
            entry_root
            / "derived"
            / "support-phase-finite-difference-v1"
            / window_id
            / "states"
            / f"step-{step:05d}.json"
        )
        record = read_json(record_path)
        require(record["phase_protocol_sha256"] == phase_protocol_sha256, "SST_GFG_PHASE_PROTOCOL_DRIFT")
        for scale_text, references in sorted(record["left_rates"].items(), key=lambda row: int(row[0])):
            scale = int(scale_text)
            _emit_phase_tensor_group(
                writer,
                references=references,
                sources=[
                    (origin(step - scale), "earlier_support_observation"),
                    (origin(step), "current_support_observation"),
                    (phase_contract_ref, "frozen_finite_difference_contract"),
                ],
                semantic_prefix=f"window:{window_id}:step:{step}:V-minus:m={scale}",
                optimizer_step=step,
                occurrence_type="support_left_finite_difference_occurrence",
                operation="finite_difference_left",
                contract_id=str(phase_protocol["protocol_id"]),
                temporal_role="input_available_at_cut",
            )
            _emit_phase_tensor_group(
                writer,
                references=record["left_prediction_change_masks"][scale_text],
                sources=[
                    (origin(step - scale), "earlier_support_observation"),
                    (origin(step), "current_support_observation"),
                    (phase_contract_ref, "frozen_categorical_change_contract"),
                ],
                semantic_prefix=f"window:{window_id}:step:{step}:categorical-left-change:m={scale}",
                optimizer_step=step,
                occurrence_type="support_left_categorical_change_occurrence",
                operation="identity_aligned_categorical_change",
                contract_id=str(phase_protocol["protocol_id"]),
                temporal_role="input_available_at_cut",
            )
        _emit_phase_tensor_group(
            writer,
            references=record["left_acceleration"],
            sources=[
                (origin(step - 2), "support_observation_k_minus_2"),
                (origin(step - 1), "support_observation_k_minus_1"),
                (origin(step), "support_observation_k"),
                (phase_contract_ref, "frozen_finite_difference_contract"),
            ],
            semantic_prefix=f"window:{window_id}:step:{step}:A-minus",
            optimizer_step=step,
            occurrence_type="support_left_acceleration_finite_difference_occurrence",
            operation="finite_difference_left_acceleration",
            contract_id=str(phase_protocol["protocol_id"]),
            temporal_role="input_available_at_cut",
        )
        categorical_disposition_occurrence = _occurrence(
            writer,
            occurrence_type="categorical_acceleration_disposition_occurrence",
            optimizer_step=step,
            operation="adjudicate_categorical_acceleration",
            contract_id=str(phase_protocol["protocol_id"]),
            payload=record["categorical_acceleration_disposition"],
        )
        categorical_disposition = _object(
            writer,
            semantic_key=f"window:{window_id}:step:{step}:categorical-acceleration-disposition",
            role="explicit_disposition",
            optimizer_step=step,
            payload={"outcome_kind": "ExplicitDisposition", **record["categorical_acceleration_disposition"]},
            object_kind="ExplicitDisposition",
        )
        writer.bind(
            categorical_disposition_occurrence,
            [(origin(step - 1), "prior_categorical_support_state"), (origin(step), "current_categorical_support_state")],
            categorical_disposition,
            payload={"outcome_kind": "ExplicitDisposition"},
        )
        if record["right_rate_target_only"] is not None:
            right_sources = [
                (origin(step - 1), "prior_support_observation"),
                (origin(step), "current_support_observation"),
                (origin(step + 1), "future_support_observation_target_only"),
                (phase_contract_ref, "frozen_finite_difference_contract"),
            ]
            for key, occurrence_type, operation in (
                ("right_rate_target_only", "support_right_finite_difference_occurrence", "finite_difference_right"),
                ("law_break_target_only", "support_law_break_finite_difference_occurrence", "left_right_discrete_rate_difference"),
                ("right_prediction_change_target_only", "support_right_categorical_change_occurrence", "identity_aligned_categorical_change"),
                ("categorical_law_break_target_only", "support_categorical_law_break_occurrence", "categorical_transition_change"),
            ):
                _emit_phase_tensor_group(
                    writer,
                    references=record[key],
                    sources=right_sources,
                    semantic_prefix=f"window:{window_id}:step:{step}:{key}",
                    optimizer_step=step,
                    occurrence_type=occurrence_type,
                    operation=operation,
                    contract_id=str(phase_protocol["protocol_id"]),
                    temporal_role="target_only_after_cut",
                )
        else:
            disposition_occurrence = _occurrence(
                writer,
                occurrence_type="right_difference_disposition_occurrence",
                optimizer_step=step,
                operation="adjudicate_right_difference_scope",
                contract_id=str(phase_protocol["protocol_id"]),
                payload=record["right_difference_disposition"],
            )
            disposition = _object(
                writer,
                semantic_key=f"window:{window_id}:step:{step}:right-difference-disposition",
                role="explicit_disposition",
                optimizer_step=step,
                payload={"outcome_kind": "ExplicitDisposition", **record["right_difference_disposition"]},
                object_kind="ExplicitDisposition",
            )
            writer.bind(
                disposition_occurrence,
                [(origin(step), "terminal_support_observation"), (phase_contract_ref, "frozen_window_scope")],
                disposition,
                payload={"outcome_kind": "ExplicitDisposition"},
            )
        writer.flush_block()


def _validation_probe_sources(
    writer: SupportGFGWriter,
    *,
    source_graph: TrainingGFG,
    source_bundle: Path,
    source_bundle_id: str,
    protocol_ref: GraphRef,
) -> list[tuple[GraphRef, str]]:
    zero_rows = objects_for_stage(source_graph, 0, "before_batch")
    native_inputs = [row for row in zero_rows if row["role"] == "validation_dataset_inputs"]
    native_targets = [row for row in zero_rows if row["role"] == "validation_dataset_targets"]
    require(len(native_inputs) <= 1, "SST_GFG_VALIDATION_INPUT_MULTIPLE")
    require(len(native_targets) <= 1, "SST_GFG_VALIDATION_TARGET_MULTIPLE")

    if native_inputs:
        input_row = native_inputs[0]
    else:
        embedding_rows = objects_for_stage(source_graph, 100, "evaluation_validation:token_embedding")
        candidates = [
            row
            for row in embedding_rows
            if row["role"] == "layer_input" and row.get("name") == "token_embedding.input.0"
        ]
        require(len(candidates) == 1, "SST_GFG_VALIDATION_INPUT_RECOVERY_SOURCE_NOT_UNIQUE")
        input_row = candidates[0]
    input_ref = writer.origin(input_row, source_bundle_id=source_bundle_id)

    if native_targets:
        target_ref = writer.origin(native_targets[0], source_bundle_id=source_bundle_id)
        return [(input_ref, "validation_dataset_inputs"), (target_ref, "validation_dataset_targets")]

    evaluation_rows = objects_for_stage(source_graph, 100, "evaluation")
    explicit_targets = [row for row in evaluation_rows if row["role"] == "evaluation_validation_targets"]
    require(len(explicit_targets) <= 1, "SST_GFG_EXPLICIT_VALIDATION_TARGET_MULTIPLE")
    if explicit_targets:
        target_ref = writer.origin(explicit_targets[0], source_bundle_id=source_bundle_id)
        return [(input_ref, "validation_dataset_inputs"), (target_ref, "validation_dataset_targets")]

    training_inputs = [row for row in zero_rows if row["role"] == "training_batch_inputs"]
    training_targets = [row for row in zero_rows if row["role"] == "training_batch_targets"]
    require(len(training_inputs) == len(training_targets) == 1, "SST_GFG_TARGET_RECOVERY_TRAINING_SOURCE_NOT_UNIQUE")
    recovered_targets, certificate = recover_cyclic_target_mapping(
        load_tensor(source_bundle, training_inputs[0]).numpy(),
        load_tensor(source_bundle, training_targets[0]).numpy(),
        load_tensor(source_bundle, input_row).numpy(),
    )
    recovery = _occurrence(
        writer,
        occurrence_type="validation_target_recovery_occurrence",
        optimizer_step=0,
        operation="recover_cyclic_target_mapping",
        contract_id="STEPWISE-SUPPORT-TRANSITION-PHASE-v3",
        payload={
            "explicit_native_target_present": False,
            "mapping_certificate": certificate,
            "future_training_fact_used": False,
        },
    )
    target_ref = writer.tensor_object(
        semantic_key="derived-validation-dataset-targets",
        role="derived_validation_dataset_targets",
        optimizer_step=0,
        value=recovered_targets,
        representation="complete_validation_targets_recovered_from_cyclic_training_task_mapping",
        extra_payload={"mapping_certificate": certificate},
    )
    recovery_sources = [
        (writer.origin(training_inputs[0], source_bundle_id=source_bundle_id), "task_mapping_training_inputs"),
        (writer.origin(training_targets[0], source_bundle_id=source_bundle_id), "task_mapping_training_targets"),
        (input_ref, "validation_inputs_to_be_mapped"),
        (protocol_ref, "frozen_validation_target_recovery_contract"),
    ]
    writer.bind(recovery, recovery_sources, target_ref, payload={"outcome_kind": "derived_validation_dataset_targets"})
    return [(input_ref, "validation_dataset_inputs"), (target_ref, "validation_dataset_targets")]


def build_entry_stepwise_gfg(
    *,
    entry_id: str,
    formal_root: Path,
    source_bundle: Path,
    selection_path: Path,
    component_registry_path: Path,
    probe_contract_path: Path,
    protocol_path: Path,
    phase_protocol_path: Path | None = None,
    max_windows: int | None = None,
) -> dict[str, Any]:
    selection = read_json(selection_path)
    windows = [row for row in selection["windows"] if str(row["entry_id"]) == entry_id]
    if max_windows is not None:
        windows = windows[:max_windows]
    require(bool(windows), f"SST_GFG_ENTRY_WINDOWS_EMPTY:{entry_id}")
    entry_root = formal_root / entry_id
    entry_receipt = _read_entry_receipt(entry_root)
    source_manifest = read_json(source_bundle / "manifest.json")
    source_bundle_id = str(source_manifest["bundle_manifest_sha256"])
    registry = ComponentRegistry.load(component_registry_path)
    probe_contract = ProbeContract.load(probe_contract_path, registry)
    protocol = read_json(protocol_path)
    phase_protocol = read_json(phase_protocol_path) if phase_protocol_path is not None else None
    database_path = entry_root / "stepwise_support_transition_gfg.sqlite3"
    require(not database_path.exists(), f"SST_GFG_DATABASE_ALREADY_EXISTS:{database_path}")
    writer = SupportGFGWriter(
        database_path,
        entry_root / "tensor-objects",
        scope_id=f"stepwise-support-transition:{entry_id}",
        source_bundle_id=source_bundle_id,
        contract_sha256=file_sha256(protocol_path),
        graph_schema=GRAPH_SCHEMA,
        block_schema=BLOCK_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
    )
    source_graph = TrainingGFG(source_bundle / "participant_gfg.sqlite3")
    state_catalog: dict[str, str] = {}
    try:
        writer.start_block("graph_contracts", 0)
        protocol_ref = _object(
            writer,
            semantic_key="contract:stepwise-support-transition",
            role="frozen_stepwise_capture_protocol",
            optimizer_step=0,
            payload={"protocol": protocol, "file_sha256": file_sha256(protocol_path)},
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
            protocol_ref=protocol_ref,
        )
        phase_contract_ref = None
        if phase_protocol_path is not None:
            phase_contract_ref = _object(
                writer,
                semantic_key=f"phase-contract:{phase_protocol['protocol_id']}",
                role="frozen_support_phase_finite_difference_contract",
                optimizer_step=0,
                payload={"contract": phase_protocol, "file_sha256": file_sha256(phase_protocol_path)},
                object_kind="frozen_contract",
            )
        writer.flush_block()

        for window_ordinal, window in enumerate(windows):
            writer._last_occurrence_id = None  # Deliberately start a new represented replay execution.
            start = int(window.get("capture_start_optimizer_step", window["start_optimizer_step"]))
            end = int(window.get("capture_end_optimizer_step", window["end_optimizer_step"]))
            window_id = str(window["window_id"])
            current_origin: GraphRef | None = None
            probe_summary_catalog: dict[int, GraphRef] = {}
            for step in range(start, end + 1):
                writer.start_block(f"window:{window_id}:step", step)
                state_path = entry_root / "windows" / window_id / "states" / f"step-{step:05d}.json"
                state_record = read_json(state_path)
                if step == start:
                    initial_sources = _initial_state_sources(
                        writer,
                        source=state_record["source_start_state_objects"],
                        optimizer_step=int(window.get("restore_optimizer_step", step)),
                        source_bundle_id=source_bundle_id,
                    ) + _warmup_batch_sources(
                        writer,
                        source=state_record["source_start_state_objects"],
                        source_bundle_id=source_bundle_id,
                        protocol_ref=protocol_ref,
                        window_id=window_id,
                    ) + [(protocol_ref, "frozen_restore_protocol")]
                    restore = _occurrence(
                        writer,
                        occurrence_type="state_restore_occurrence",
                        optimizer_step=step,
                        operation=(
                            "restore_checkpoint_and_reexecute_uncaptured_warmup"
                            if "source_checkpoint" in state_record["source_start_state_objects"]
                            else "restore_complete_parameter_and_adam_prestate"
                        ),
                        contract_id=str(protocol["protocol_id"]),
                        payload={
                            "window_id": window_id,
                            "historical_rng_restoration_claimed": False,
                            "source_start_state_objects": state_record["source_start_state_objects"],
                            "capture_start_optimizer_step": start,
                            "restore_optimizer_step": int(window.get("restore_optimizer_step", start)),
                        },
                    )
                    summary, current_origin = _emit_state(
                        writer,
                        occurrence_id=restore,
                        sources=initial_sources,
                        state_record=state_record,
                        semantic_prefix=f"window:{window_id}:step:{step}:pre",
                        optimizer_step=step,
                    )
                    state_catalog[f"{window_id}:{state_record['state']['state_id']}"] = summary.object_id
                require(current_origin is not None, "SST_GFG_CURRENT_STATE_ORIGIN_MISSING")
                observation_path = (
                    entry_root
                    / "probe-observations"
                    / probe_contract.probe_contract_id
                    / f"{state_record['state']['state_id']}.json"
                )
                observation = read_json(observation_path)
                probe_summary_catalog[step] = _emit_probe(
                    writer,
                    observation=observation,
                    state_origin=current_origin,
                    validation_sources=validation_sources,
                    contract_ref=probe_contract_ref,
                    registry_ref=registry_ref,
                    semantic_prefix=f"window:{window_id}:step:{step}",
                    optimizer_step=step,
                    probe_contract=probe_contract,
                )
                if step < end:
                    transition_path = entry_root / "windows" / window_id / "transitions" / f"step-{step:05d}-to-{step + 1:05d}.json"
                    transition = read_json(transition_path)
                    batch_sources = _batch_sources(
                        writer,
                        batch=transition["batch"],
                        source_bundle_id=source_bundle_id,
                        protocol_ref=protocol_ref,
                        semantic_prefix=f"window:{window_id}:step:{step}",
                        optimizer_step=step,
                    )
                    post_state_path = entry_root / "windows" / window_id / "states" / f"step-{step + 1:05d}.json"
                    post_state = read_json(post_state_path)
                    post_summary, current_origin = _emit_training_step(
                        writer,
                        transition=transition,
                        prestate_origin=current_origin,
                        batch_sources=batch_sources,
                        protocol_ref=protocol_ref,
                        poststate_record=post_state,
                        semantic_prefix=f"window:{window_id}:step:{step}",
                        optimizer_step=step,
                        contract_id=str(protocol["protocol_id"]),
                    )
                    state_catalog[f"{window_id}:{post_state['state']['state_id']}"] = post_summary.object_id
                writer.flush_block()
            if phase_contract_ref is not None:
                _emit_phase_window_layer(
                    writer,
                    entry_root=entry_root,
                    window=window,
                    probe_summary_catalog=probe_summary_catalog,
                    phase_contract_ref=phase_contract_ref,
                    phase_protocol=phase_protocol,
                    phase_protocol_sha256=file_sha256(phase_protocol_path),
                )
            print({"event": "SST_GFG_WINDOW_COMPLETE", "entry_id": entry_id, "ordinal": window_ordinal + 1, "window_count": len(windows), "window_id": window_id}, flush=True)
        manifest = writer.close()
    finally:
        source_graph.close()
    material = {
        **manifest,
        "entry_id": entry_id,
        "source_bundle_manifest_sha256": source_bundle_id,
        "source_gfg_database_sha256": source_manifest["gfg_database_sha256"],
        "entry_execution_receipt_sha256": entry_receipt["result_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "protocol_sha256": file_sha256(protocol_path),
        "component_registry_sha256": registry.source_sha256,
        "probe_contract_sha256": probe_contract.source_sha256,
        "phase_protocol_sha256": file_sha256(phase_protocol_path) if phase_protocol_path is not None else None,
        "state_catalog": state_catalog,
        "window_count": len(windows),
    }
    result = {**material, "manifest_sha256": payload_sha256(material)}
    write_json(entry_root / "stepwise_support_transition_gfg_manifest.json", result)
    return result


def _read_entry_receipt(entry_root: Path) -> dict[str, Any]:
    value = read_json(entry_root / "entry_receipt.json")
    require(value["schema"] == "nanogpt-stepwise-entry-receipt-v1", "SST_GFG_ENTRY_RECEIPT_SCHEMA_INVALID")
    material = {key: child for key, child in value.items() if key != "result_sha256"}
    require(payload_sha256(material) == value["result_sha256"], "SST_GFG_ENTRY_RECEIPT_HASH_INVALID")
    return value
