from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import (
    file_sha256,
    payload_sha256,
    require,
    write_json,
)

from .runtime import (
    COMPONENTS,
    COMPONENT_PAIRS,
    HistoricalRunRuntime,
    load_tensor,
    objects_for_stage,
    tensor_sha256,
    unique_role_objects,
)
from .support_gfg import GraphRef, SupportGFGWriter, validate_support_gfg


def decision_outputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, np.ndarray]:
    logits_np = logits.detach().contiguous().cpu().numpy().astype(np.float32, copy=False)
    target_tokens = targets[:, -1].detach().contiguous().cpu().numpy().astype(np.int64, copy=False)
    require(logits_np.shape == (212, 24), "CSRG_DECISION_LOGIT_SHAPE_INVALID")
    require(target_tokens.shape == (212,), "CSRG_TARGET_TOKEN_SHAPE_INVALID")
    require(set(np.unique(target_tokens).tolist()) == set(range(23)), "CSRG_TARGET_GROUP_SET_INVALID")
    row = np.arange(logits_np.shape[0])
    target_logits = logits_np[row, target_tokens]
    masked = logits_np.copy()
    masked[row, target_tokens] = -np.inf
    competitor_logits = masked.max(axis=1)
    margins = (target_logits - competitor_logits).astype(np.float32)
    predictions = logits_np.argmax(axis=1).astype(np.int64)
    groups = target_tokens.copy()
    group_q10 = np.asarray(
        [
            np.quantile(margins[groups == token], 0.10, method="linear")
            for token in range(23)
        ],
        dtype=np.float64,
    )
    return {
        "group_membership": groups,
        "group_q10_margin": group_q10,
        "margins": margins,
        "predictions": predictions,
    }


def _rms(rows: Iterable[torch.Tensor]) -> float:
    total = 0.0
    count = 0
    for value in rows:
        array = value.detach().to(torch.float64)
        total += float(torch.sum(array * array))
        count += array.numel()
    require(count > 0, "CSRG_COMPONENT_SCALAR_WITHOUT_VALUES")
    return math.sqrt(total / count)


def component_optimizer_loads(
    runtime: HistoricalRunRuntime,
    optimizer_step: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    checkpoint = objects_for_stage(runtime.graph, optimizer_step, "after_optimizer_step")
    # Parameter version t is formed by the optimizer update executed with the
    # clipped gradients captured at training loop index t-1.  The source GFG
    # deliberately keeps those two identities distinct.
    gradients = objects_for_stage(
        runtime.graph,
        optimizer_step - 1,
        "after_gradient_clip",
    )
    parameters = unique_role_objects(checkpoint, "parameter_version")
    optimizer = unique_role_objects(checkpoint, "optimizer_state")
    clipped = unique_role_objects(gradients, "clipped_parameter_gradient")
    results: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    prefixes = {
        "h0.attn": "transformer.h.0.attn.",
        "h0.mlp": "transformer.h.0.mlp.",
        "h1.attn": "transformer.h.1.attn.",
        "h1.mlp": "transformer.h.1.mlp.",
    }
    for component, prefix in prefixes.items():
        names = sorted(name for name in parameters if name.startswith(prefix))
        require(bool(names), f"CSRG_COMPONENT_PARAMETER_SET_EMPTY:{component}")
        parameter_rows = [parameters[name] for name in names]
        gradient_rows = [clipped[name] for name in names]
        exp_avg_rows = [optimizer[name + ".exp_avg"] for name in names]
        exp_avg_sq_rows = [optimizer[name + ".exp_avg_sq"] for name in names]
        gradient_payload_available = all(bool(row["materialized"]) for row in gradient_rows)
        results[component] = {
            "clipped_gradient_payload_available": gradient_payload_available,
            "clipped_gradient_rms": _rms(
                load_tensor(runtime.bundle, row) for row in gradient_rows
            )
            if gradient_payload_available
            else None,
            "clipped_gradient_unmaterialized_object_ids": [
                row["object_id"] for row in gradient_rows if not row["materialized"]
            ],
            "exp_avg_rms": _rms(load_tensor(runtime.bundle, row) for row in exp_avg_rows),
            "exp_avg_sq_sqrt_mean": math.sqrt(
                sum(float(load_tensor(runtime.bundle, row).to(torch.float64).sum()) for row in exp_avg_sq_rows)
                / sum(int(np.prod(row["shape"], dtype=np.int64)) for row in exp_avg_sq_rows)
            ),
            "parameter_rms": _rms(load_tensor(runtime.bundle, row) for row in parameter_rows),
            "parameter_version": optimizer_step,
            "producing_gradient_loop_index": optimizer_step - 1,
        }
        sources[component] = parameter_rows + gradient_rows + exp_avg_rows + exp_avg_sq_rows
    return results, sources


def _runtime_payload(trainer_root: Path) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "model_commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        "model_py_sha256": file_sha256(trainer_root / "model.py"),
        "python_torch_version": torch.__version__,
        "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }


def _bind_outputs(
    writer: SupportGFGWriter,
    *,
    occurrence_id: str,
    sources: list[tuple[GraphRef, str]],
    refs: dict[str, GraphRef],
    execution_kind: str,
) -> None:
    for role, ref in refs.items():
        writer.bind(
            occurrence_id,
            sources,
            ref,
            payload={"execution_kind": execution_kind, "outcome_role": role},
        )


def _forward_objects(
    writer: SupportGFGWriter,
    *,
    prefix: str,
    optimizer_step: int,
    logits: torch.Tensor,
    targets: torch.Tensor,
    gate_components: tuple[str, ...],
) -> tuple[dict[str, GraphRef], dict[str, np.ndarray]]:
    values = decision_outputs(logits, targets)
    gate_payload = {"gate_components": list(gate_components)}
    refs = {
        "decision_logits": writer.tensor_object(
            semantic_key=f"{prefix}:decision_logits",
            role="decision_logits",
            optimizer_step=optimizer_step,
            value=logits,
            representation="complete_float32_decision_logits",
            extra_payload=gate_payload,
        ),
        "predictions": writer.tensor_object(
            semantic_key=f"{prefix}:predictions",
            role="predictions",
            optimizer_step=optimizer_step,
            value=values["predictions"],
            representation="complete_argmax_prediction_tokens",
            extra_payload=gate_payload,
        ),
        "per_example_margins": writer.tensor_object(
            semantic_key=f"{prefix}:per_example_margins",
            role="per_example_true_target_margin",
            optimizer_step=optimizer_step,
            value=values["margins"],
            representation="complete_row_order_true_target_minus_max_non_target_margin",
            extra_payload=gate_payload,
        ),
        "target_group_q10_margins": writer.tensor_object(
            semantic_key=f"{prefix}:target_group_q10_margins",
            role="target_group_q10_margin",
            optimizer_step=optimizer_step,
            value=values["group_q10_margin"],
            representation="all_23_target_groups_linear_q10_margin",
            extra_payload=gate_payload,
        ),
    }
    return refs, values


def _parameter_state_exact(
    runtime: HistoricalRunRuntime,
    parameter_rows: dict[str, dict[str, Any]],
) -> bool:
    for name, value in runtime.model.named_parameters():
        if tensor_sha256(value) != parameter_rows[name]["content_sha256"]:
            return False
    return True


def build_support_gfg(
    source_bundle: Path,
    output_directory: Path,
    trainer_root: Path,
    *,
    entry_id: str,
    contract_path: Path,
    max_checkpoints: int | None = None,
) -> dict[str, Any]:
    source_bundle = source_bundle.resolve()
    output_directory = output_directory.resolve()
    trainer_root = trainer_root.resolve()
    contract_path = contract_path.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    source_bundle_id = source_bundle.name
    contract_sha = file_sha256(contract_path)
    scope_id = f"nanogpt-csrg-v2:{entry_id}"
    database_path = output_directory / "support_gfg.sqlite3"
    tensor_directory = output_directory / "tensor-objects"
    writer = SupportGFGWriter(
        database_path,
        tensor_directory,
        scope_id=scope_id,
        source_bundle_id=source_bundle_id,
        contract_sha256=contract_sha,
    )
    runtime = HistoricalRunRuntime.open(
        source_bundle,
        trainer_root,
        device="cuda",
        reference_step=100,
    )
    try:
        writer.start_block("analysis_protocol_and_target_mapping", 0)
        runtime_ref = writer.object(
            semantic_key="runtime:authoritative-v2-analysis-execution",
            role="analysis_runtime_profile",
            optimizer_step=0,
            payload=_runtime_payload(trainer_root),
            object_kind="declared_execution_source",
        )
        contract_ref = writer.object(
            semantic_key="contract:csrg-current-runtime-v2",
            role="capture_contract",
            optimizer_step=0,
            payload={"contract_sha256": contract_sha, "schema": "nanogpt-csrg-current-runtime-capture-contract-v2"},
            object_kind="declared_execution_source",
        )
        gate_refs: dict[str, GraphRef] = {}
        for component in COMPONENTS:
            gate_refs[component] = writer.object(
                semantic_key=f"gate:{component}:zero",
                role="component_gate_specification",
                optimizer_step=0,
                payload={
                    "component": component,
                    "gate_value": 0.0,
                    "site": "module_output_after_projection_before_residual_addition",
                },
                object_kind="declared_execution_source",
            )
        mapping_occurrence = writer.occurrence(
            occurrence_type="opaque_cyclic_target_mapping_derivation",
            optimizer_step=0,
            transform_reference={"operation": "finite_field_rref_nullspace_and_automorphism_invariance_v1"},
            payload={"field_modulus": 23},
        )
        mapping_ref = writer.object(
            semantic_key="target_mapping:certificate",
            role="derived_target_mapping_certificate",
            optimizer_step=0,
            payload=runtime.target_mapping_certificate,
        )
        target_ref = writer.tensor_object(
            semantic_key="target_mapping:validation_targets",
            role="derived_validation_targets",
            optimizer_step=0,
            value=runtime.validation_targets,
            representation="derived_complete_validation_target_tensor",
            extra_payload={"not_native_training_fact": True},
        )
        mapping_sources = [
            (writer.origin(runtime.source_training_inputs), "exact_training_inputs"),
            (writer.origin(runtime.source_training_targets), "exact_training_targets"),
            (writer.origin(runtime.source_validation_inputs), "exact_validation_inputs"),
            (contract_ref, "frozen_derivation_contract"),
        ]
        writer.bind(mapping_occurrence, mapping_sources, mapping_ref)
        writer.bind(mapping_occurrence, mapping_sources + [(mapping_ref, "mapping_certificate")], target_ref)
        writer.flush_block()

        previous_necessity: np.ndarray | None = None
        steps = list(range(100, 10001, 100))
        if max_checkpoints is not None:
            steps = steps[:max_checkpoints]
        for checkpoint_ordinal, step in enumerate(steps, start=1):
            parameter_rows = runtime.load_checkpoint(step)
            require(_parameter_state_exact(runtime, parameter_rows), "CSRG_PARAMETER_LOAD_NOT_EXACT")
            validation_rows = objects_for_stage(runtime.graph, step, "evaluation_validation:token_embedding")
            validation_input_row = next(
                row for row in validation_rows
                if row["role"] == "layer_input" and row["name"] == "token_embedding.input.0"
            )
            evaluation_rows = objects_for_stage(runtime.graph, step, "evaluation")
            historical_logits_row = next(row for row in evaluation_rows if row["role"] == "validation_logits")
            historical_predictions_row = next(row for row in evaluation_rows if row["role"] == "validation_predictions")
            capability_row = next(row for row in evaluation_rows if row["role"] == "capability_evaluation")
            historical_logits = load_tensor(source_bundle, historical_logits_row)
            historical_predictions = load_tensor(source_bundle, historical_predictions_row)

            writer.start_block("current_runtime_baseline_and_component_gates", step)
            parameter_sources = [
                (writer.origin(row), f"checkpoint_parameter:{name}")
                for name, row in sorted(parameter_rows.items())
            ]
            validation_origin = writer.origin(validation_input_row)
            common_sources = parameter_sources + [
                (validation_origin, "validation_inputs"),
                (target_ref, "derived_validation_targets"),
                (runtime_ref, "bound_execution_runtime"),
                (contract_ref, "frozen_capture_contract"),
            ]

            baseline_first = runtime.forward()
            baseline_second = runtime.forward()
            require(torch.equal(baseline_first, baseline_second), "CSRG_CURRENT_BASELINE_REPEAT_NOT_BYTE_EXACT")
            baseline_values = decision_outputs(baseline_first, runtime.validation_targets)
            historical_prediction_exact = np.array_equal(
                baseline_values["predictions"],
                historical_predictions.detach().cpu().numpy(),
            )
            historical_capability = float(capability_row["literal_payload"]["validation_accuracy"])
            current_capability = runtime.accuracy(baseline_first)
            historical_capability_exact = current_capability == historical_capability
            historical_raw_exact = tensor_sha256(baseline_first) == historical_logits_row["content_sha256"]
            historical_max_error = float((baseline_first - historical_logits).abs().max())

            baseline_refs: dict[str, GraphRef] | None = None
            for repeat, logits in ((1, baseline_first), (2, baseline_second)):
                occurrence = writer.occurrence(
                    occurrence_type="current_runtime_baseline_replay",
                    optimizer_step=step,
                    transform_reference={"operation": "frozen_nanogpt_eval_forward_no_component_gate_v2"},
                    payload={"repeat": repeat},
                )
                refs, _values = _forward_objects(
                    writer,
                    prefix=f"step:{step}:baseline:repeat:{repeat}",
                    optimizer_step=step,
                    logits=logits,
                    targets=runtime.validation_targets,
                    gate_components=(),
                )
                _bind_outputs(
                    writer,
                    occurrence_id=occurrence,
                    sources=common_sources,
                    refs=refs,
                    execution_kind="current_runtime_baseline",
                )
                if repeat == 1:
                    baseline_refs = refs
            require(baseline_refs is not None, "CSRG_BASELINE_REFS_MISSING")

            single_group_margins: dict[str, np.ndarray] = {}
            single_refs: dict[str, dict[str, GraphRef]] = {}
            single_hashes: dict[str, str] = {}
            for component in COMPONENTS:
                logits = runtime.forward((component,))
                occurrence = writer.occurrence(
                    occurrence_type="single_component_gate_evaluation",
                    optimizer_step=step,
                    transform_reference={"operation": "zero_component_output_before_residual_addition_v1", "components": [component]},
                    payload={"gate_components": [component]},
                )
                refs, values = _forward_objects(
                    writer,
                    prefix=f"step:{step}:single:{component}",
                    optimizer_step=step,
                    logits=logits,
                    targets=runtime.validation_targets,
                    gate_components=(component,),
                )
                _bind_outputs(
                    writer,
                    occurrence_id=occurrence,
                    sources=common_sources + [(gate_refs[component], "active_zero_gate")],
                    refs=refs,
                    execution_kind="single_component_gate",
                )
                if torch.equal(logits, baseline_first):
                    disposition = writer.object(
                        semantic_key=f"step:{step}:single:{component}:zero_effect",
                        role="explicit_disposition",
                        optimizer_step=step,
                        object_kind="ExplicitDisposition",
                        payload={"disposition": "ZERO_EFFECT_OBSERVED", "equal_output_is_not_equal_occurrence": True},
                    )
                    writer.bind(
                        occurrence,
                        [(baseline_refs["decision_logits"], "ungated_baseline"), (refs["decision_logits"], "gated_output")],
                        disposition,
                    )
                single_group_margins[component] = values["group_q10_margin"]
                single_refs[component] = refs
                single_hashes[component] = tensor_sha256(logits)

            pair_group_margins: dict[tuple[str, str], np.ndarray] = {}
            pair_refs: dict[tuple[str, str], dict[str, GraphRef]] = {}
            pair_hashes: dict[str, str] = {}
            for left, right in COMPONENT_PAIRS:
                pair = (left, right)
                logits = runtime.forward(pair)
                occurrence = writer.occurrence(
                    occurrence_type="pair_component_gate_evaluation",
                    optimizer_step=step,
                    transform_reference={"operation": "zero_component_outputs_before_residual_addition_v1", "components": list(pair)},
                    payload={"gate_components": list(pair)},
                )
                refs, values = _forward_objects(
                    writer,
                    prefix=f"step:{step}:pair:{left}+{right}",
                    optimizer_step=step,
                    logits=logits,
                    targets=runtime.validation_targets,
                    gate_components=pair,
                )
                _bind_outputs(
                    writer,
                    occurrence_id=occurrence,
                    sources=common_sources + [(gate_refs[left], "active_zero_gate_1"), (gate_refs[right], "active_zero_gate_2")],
                    refs=refs,
                    execution_kind="pair_component_gate",
                )
                if torch.equal(logits, baseline_first):
                    disposition = writer.object(
                        semantic_key=f"step:{step}:pair:{left}+{right}:zero_effect",
                        role="explicit_disposition",
                        optimizer_step=step,
                        object_kind="ExplicitDisposition",
                        payload={"disposition": "ZERO_EFFECT_OBSERVED", "equal_output_is_not_equal_occurrence": True},
                    )
                    writer.bind(
                        occurrence,
                        [(baseline_refs["decision_logits"], "ungated_baseline"), (refs["decision_logits"], "gated_output")],
                        disposition,
                    )
                pair_group_margins[pair] = values["group_q10_margin"]
                pair_refs[pair] = refs
                pair_hashes[left + "+" + right] = tensor_sha256(logits)

            require(_parameter_state_exact(runtime, parameter_rows), "CSRG_PARAMETER_MUTATED_BY_PROBE")
            writer.flush_block()

            writer.start_block("support_redundancy_derivations", step)
            baseline_group = baseline_values["group_q10_margin"]
            necessity = np.stack(
                [np.maximum(0.0, baseline_group - single_group_margins[name]) for name in COMPONENTS]
            )
            necessity_occurrence = writer.occurrence(
                occurrence_type="component_necessity_derivation",
                optimizer_step=step,
                transform_reference={"operation": "N_jkt=max(0,m_kt-m_minus_j_kt)"},
                payload={"component_order": list(COMPONENTS), "target_group_order": list(range(23))},
            )
            necessity_ref = writer.tensor_object(
                semantic_key=f"step:{step}:necessity",
                role="component_target_group_necessity",
                optimizer_step=step,
                value=necessity,
                representation="complete_4_by_23_component_necessity",
            )
            writer.bind(
                necessity_occurrence,
                [(baseline_refs["target_group_q10_margins"], "baseline_group_margin")]
                + [(single_refs[name]["target_group_q10_margins"], f"single_gate_group_margin:{name}") for name in COMPONENTS],
                necessity_ref,
            )

            backups = np.stack(
                [
                    np.maximum(
                        0.0,
                        baseline_group
                        - pair_group_margins[pair]
                        - necessity[COMPONENTS.index(pair[0])]
                        - necessity[COMPONENTS.index(pair[1])],
                    )
                    for pair in COMPONENT_PAIRS
                ]
            )
            backup_occurrence = writer.occurrence(
                occurrence_type="pair_backup_derivation",
                optimizer_step=step,
                transform_reference={"operation": "R_ijkt=max(0,(m_kt-m_minus_ij_kt)-N_ikt-N_jkt)"},
                payload={"pair_order": [list(pair) for pair in COMPONENT_PAIRS]},
            )
            backup_ref = writer.tensor_object(
                semantic_key=f"step:{step}:pair_backup",
                role="pair_target_group_backup",
                optimizer_step=step,
                value=backups,
                representation="complete_6_by_23_pair_backup",
            )
            writer.bind(
                backup_occurrence,
                [(necessity_ref, "component_necessity")]
                + [(pair_refs[pair]["target_group_q10_margins"], f"pair_gate_group_margin:{pair[0]}+{pair[1]}") for pair in COMPONENT_PAIRS],
                backup_ref,
            )

            single_slack = np.min(np.stack([single_group_margins[name] for name in COMPONENTS]), axis=0)
            double_slack = np.min(np.stack([pair_group_margins[pair] for pair in COMPONENT_PAIRS]), axis=0)
            total_necessity = necessity.sum(axis=0)
            defined = total_necessity > 0.0
            effective_support = np.full(23, np.nan, dtype=np.float64)
            weights = np.zeros_like(necessity, dtype=np.float64)
            weights[:, defined] = necessity[:, defined] / total_necessity[defined]
            effective_support[defined] = 1.0 / np.sum(weights[:, defined] ** 2, axis=0)
            scalar_refs: dict[str, GraphRef] = {}
            scalar_specs = {
                "single_failure_slack": (single_slack, "S_kt=min_j(m_minus_j_kt)", [single_refs[name]["target_group_q10_margins"] for name in COMPONENTS]),
                "double_failure_slack": (double_slack, "P_kt=min_i_less_j(m_minus_ij_kt)", [pair_refs[pair]["target_group_q10_margins"] for pair in COMPONENT_PAIRS]),
                "effective_support": (effective_support, "Q_kt=1/sum_j(w_jkt^2)", [necessity_ref]),
            }
            for name, (value, operation, source_refs) in scalar_specs.items():
                occurrence = writer.occurrence(
                    occurrence_type=name + "_derivation",
                    optimizer_step=step,
                    transform_reference={"operation": operation},
                    payload={"target_group_order": list(range(23))},
                )
                ref = writer.tensor_object(
                    semantic_key=f"step:{step}:{name}",
                    role=name,
                    optimizer_step=step,
                    value=value,
                    representation=f"complete_23_target_group_{name}",
                    extra_payload={"undefined_group_mask": (~defined).tolist()} if name == "effective_support" else None,
                )
                writer.bind(occurrence, [(source_ref, "direct_numeric_source") for source_ref in source_refs], ref)
                scalar_refs[name] = ref
                if name == "effective_support" and not bool(np.all(defined)):
                    disposition = writer.object(
                        semantic_key=f"step:{step}:effective_support:undefined",
                        role="explicit_disposition",
                        optimizer_step=step,
                        object_kind="ExplicitDisposition",
                        payload={"disposition": "EFFECTIVE_SUPPORT_UNDEFINED_ZERO_TOTAL_NECESSITY", "target_groups": np.flatnonzero(~defined).tolist()},
                    )
                    writer.bind(occurrence, [(necessity_ref, "zero_total_necessity_source")], disposition)

            turnover_occurrence = writer.occurrence(
                occurrence_type="support_turnover_derivation",
                optimizer_step=step,
                transform_reference={"operation": "U_kt=RMS_j(N_jkt-N_jk_previous_checkpoint)"},
                payload={"previous_checkpoint": step - 100 if previous_necessity is not None else None},
            )
            if previous_necessity is None:
                turnover_ref = writer.object(
                    semantic_key=f"step:{step}:support_turnover:disposition",
                    role="explicit_disposition",
                    optimizer_step=step,
                    object_kind="ExplicitDisposition",
                    payload={"disposition": "SUPPORT_TURNOVER_UNDEFINED_NO_PREVIOUS_CHECKPOINT"},
                )
                writer.bind(turnover_occurrence, [(necessity_ref, "current_necessity")], turnover_ref)
            else:
                turnover = np.sqrt(np.mean((necessity - previous_necessity) ** 2, axis=0))
                turnover_ref = writer.tensor_object(
                    semantic_key=f"step:{step}:support_turnover",
                    role="support_turnover",
                    optimizer_step=step,
                    value=turnover,
                    representation="complete_23_target_group_support_turnover",
                )
                writer.bind(turnover_occurrence, [(necessity_ref, "current_necessity")], turnover_ref, payload={"previous_necessity_semantic_key": f"step:{step-100}:necessity"})
            previous_necessity = necessity.copy()

            loads, load_sources = component_optimizer_loads(runtime, step)
            load_occurrence = writer.occurrence(
                occurrence_type="component_optimizer_load_derivation",
                optimizer_step=step,
                transform_reference={"operation": "component_rms_scalar_projection_v1", "directional_statistics": False},
                payload={"component_order": list(COMPONENTS)},
            )
            load_ref = writer.object(
                semantic_key=f"step:{step}:component_optimizer_loads",
                role="component_optimizer_loads",
                optimizer_step=step,
                payload={"components": loads, "directional_statistics": False},
            )
            load_fact_sources: list[tuple[GraphRef, str]] = []
            for component in COMPONENTS:
                for row in load_sources[component]:
                    load_fact_sources.append((writer.origin(row), f"{component}:{row['role']}:{row['name']}"))
            writer.bind(load_occurrence, load_fact_sources, load_ref)
            unavailable_components = [
                component
                for component in COMPONENTS
                if not loads[component]["clipped_gradient_payload_available"]
            ]
            if unavailable_components:
                load_disposition = writer.object(
                    semantic_key=f"step:{step}:component_optimizer_loads:gradient_disposition",
                    role="explicit_disposition",
                    optimizer_step=step,
                    object_kind="ExplicitDisposition",
                    payload={
                        "components": unavailable_components,
                        "disposition": "PRODUCING_CLIPPED_GRADIENT_PAYLOAD_NOT_MATERIALIZED",
                        "optimizer_and_parameter_loads_still_formed": True,
                        "producing_gradient_loop_index": step - 1,
                    },
                )
                writer.bind(
                    load_occurrence,
                    [(load_ref, "partial_component_optimizer_load_result")]
                    + [
                        (writer.origin(row), f"unmaterialized_gradient_identity:{component}:{row['name']}")
                        for component in unavailable_components
                        for row in load_sources[component]
                        if row["role"] == "clipped_parameter_gradient"
                        and not row["materialized"]
                    ],
                    load_disposition,
                )
            writer.flush_block()

            derived_ids = {
                "component_optimizer_loads": load_ref.object_id,
                "double_failure_slack": scalar_refs["double_failure_slack"].object_id,
                "effective_support": scalar_refs["effective_support"].object_id,
                "necessity": necessity_ref.object_id,
                "pair_backup": backup_ref.object_id,
                "single_failure_slack": scalar_refs["single_failure_slack"].object_id,
                "support_turnover": turnover_ref.object_id,
            }
            writer.add_checkpoint(
                {
                    "optimizer_step": step,
                    "current_baseline_logits_sha256": tensor_sha256(baseline_first),
                    "historical_baseline_logits_sha256": historical_logits_row["content_sha256"],
                    "historical_raw_logits_exact": int(historical_raw_exact),
                    "historical_predictions_exact": int(historical_prediction_exact),
                    "historical_capability_exact": int(historical_capability_exact),
                    "historical_max_abs_logit_error": historical_max_error,
                    "single_gate_hashes_json": single_hashes,
                    "pair_gate_hashes_json": pair_hashes,
                    "derived_object_ids_json": derived_ids,
                    "actual_forward_count": 12,
                    "status": "PASS",
                }
            )
            writer.connection.commit()
            print(
                "CSRG_CHECKPOINT_COMPLETE "
                + json.dumps(
                    {
                        "checkpoint_ordinal": checkpoint_ordinal,
                        "entry_id": entry_id,
                        "historical_max_abs_logit_error": historical_max_error,
                        "historical_predictions_exact": historical_prediction_exact,
                        "optimizer_step": step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        manifest = writer.close()
        write_json(output_directory / "manifest.json", manifest)
        validation = validate_support_gfg(
            database_path,
            source_database_path=source_bundle / "participant_gfg.sqlite3",
            tensor_directory=tensor_directory,
            report_path=output_directory / "validation.json",
        )
        require(validation["status"] == "PASS", "CSRG_POST_CAPTURE_VALIDATION_FAILED")
        receipt = {
            "contract_sha256": contract_sha,
            "entry_id": entry_id,
            "manifest_sha256": file_sha256(output_directory / "manifest.json"),
            "schema": "nanogpt-support-redundancy-build-receipt-v1",
            "source_bundle_id": source_bundle_id,
            "status": "PASS",
            "validation_sha256": validation["validation_sha256"],
        }
        receipt["receipt_sha256"] = payload_sha256(receipt)
        write_json(output_directory / "build_receipt.json", receipt)
        return receipt
    except Exception:
        writer.connection.close()
        raise
    finally:
        runtime.close()
