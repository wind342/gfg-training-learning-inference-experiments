from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any
import zlib

import numpy as np
import torch

from experiments.gfg_nanogpt_support_redundancy_v1.builder import decision_outputs
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import (
    COMPONENTS,
    COMPONENT_PAIRS,
    HistoricalRunRuntime,
    load_tensor,
    objects_for_stage,
    tensor_sha256,
    unique_role_objects,
)


COMPONENT_PREFIXES = {
    "h0.attn": "transformer.h.0.attn.",
    "h0.mlp": "transformer.h.0.mlp.",
    "h1.attn": "transformer.h.1.attn.",
    "h1.mlp": "transformer.h.1.mlp.",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _blocks(database: Path, optimizer_step: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(
            "SELECT payload_zlib FROM graph_blocks WHERE optimizer_step=? ORDER BY block_ordinal",
            (optimizer_step,),
        ):
            result.append(json.loads(zlib.decompress(row["payload_zlib"])))
    return result


def _tensor_from_object(bundle: Path, value: dict[str, Any]) -> np.ndarray:
    return np.load(bundle / value["payload"]["locator"], allow_pickle=False)


def validated_accuracy_curve(support_bundle: Path) -> list[dict[str, Any]]:
    database = support_bundle / "support_gfg.sqlite3"
    target = next(
        value
        for block in _blocks(database, 0)
        for value in block["objects"]
        if value["role"] == "derived_validation_targets"
    )
    targets = _tensor_from_object(support_bundle, target)[:, -1]
    rows: list[dict[str, Any]] = []
    for step in range(100, 10001, 100):
        candidates = [
            value
            for block in _blocks(database, step)
            for value in block["objects"]
            if value["role"] == "predictions"
            and value["payload"].get("gate_components") == []
        ]
        require(bool(candidates), f"TLI_BASELINE_PREDICTION_MISSING:{step}")
        predictions = _tensor_from_object(support_bundle, candidates[0])
        rows.append(
            {
                "optimizer_step": step,
                "validation_accuracy": float(np.mean(predictions == targets)),
                "prediction_sha256": candidates[0]["payload"]["raw_tensor_sha256"],
            }
        )
    return rows


def select_phases(curve: list[dict[str, Any]]) -> dict[str, Any]:
    formed_index = next(
        index for index, row in enumerate(curve) if row["validation_accuracy"] >= 0.90
    )
    require(formed_index > 0, "TLI_NO_PREFORMATION_CHECKPOINT")
    drops = [
        (
            curve[index - 1]["validation_accuracy"] - curve[index]["validation_accuracy"],
            index,
        )
        for index in range(formed_index + 1, len(curve))
    ]
    decline_amount, decline_index = max(drops, key=lambda value: (value[0], -value[1]))
    require(decline_amount > 0.0, "TLI_NO_POSTFORMATION_DECLINE")
    recovery_index = next(
        (
            index
            for index in range(decline_index + 1, len(curve))
            if curve[index]["validation_accuracy"] >= 0.90
        ),
        len(curve) - 1,
    )
    recovered = curve[recovery_index]["validation_accuracy"] >= 0.90
    selected = {
        "pre_formation": curve[formed_index - 1],
        "formed": curve[formed_index],
        "decline": {
            **curve[decline_index],
            "prior_optimizer_step": curve[decline_index - 1]["optimizer_step"],
            "prior_validation_accuracy": curve[decline_index - 1]["validation_accuracy"],
            "adjacent_drop": decline_amount,
        },
        "recovered" if recovered else "post_decline_not_recovered": curve[recovery_index],
    }
    return selected


@dataclass
class CapturedCall:
    component: str
    call_index: int
    input_tensor: np.ndarray
    output_tensor: np.ndarray


def forward_with_calls(runtime: HistoricalRunRuntime) -> tuple[torch.Tensor, list[CapturedCall]]:
    calls: list[CapturedCall] = []
    handles = []

    def hook_for(component: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            require(len(inputs) >= 1 and isinstance(inputs[0], torch.Tensor), "TLI_CALL_INPUT_INVALID")
            require(isinstance(output, torch.Tensor), "TLI_CALL_OUTPUT_INVALID")
            calls.append(
                CapturedCall(
                    component=component,
                    call_index=len(calls),
                    input_tensor=inputs[0].detach().contiguous().cpu().numpy().copy(),
                    output_tensor=output.detach().contiguous().cpu().numpy().copy(),
                )
            )

        return hook

    for component, module in runtime.component_modules().items():
        handles.append(module.register_forward_hook(hook_for(component)))
    try:
        logits = runtime.forward()
    finally:
        for handle in handles:
            handle.remove()
    require(tuple(call.component for call in calls) == COMPONENTS, "TLI_NATIVE_CALL_ORDER_MISMATCH")
    return logits, calls


def component_parameter_rows(
    rows: dict[str, dict[str, Any]], component: str
) -> dict[str, dict[str, Any]]:
    prefix = COMPONENT_PREFIXES[component]
    selected = {name: row for name, row in rows.items() if name.startswith(prefix)}
    require(bool(selected), f"TLI_COMPONENT_PARAMETER_SET_EMPTY:{component}")
    return selected


def _json_decisions(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    values = decision_outputs(logits, targets)
    return {
        "accuracy": float(np.mean(values["predictions"] == targets[:, -1].numpy())),
        "logits_sha256": tensor_sha256(logits),
        "predictions_sha256": tensor_sha256(values["predictions"]),
        "margins_sha256": tensor_sha256(values["margins"]),
        "group_q10_margin": values["group_q10_margin"].tolist(),
    }


def execute_run(
    *,
    entry_id: str,
    source_bundle_id: str,
    source_bundle: Path,
    trainer_root: Path,
    phase_selection: dict[str, Any],
) -> dict[str, Any]:
    runtime = HistoricalRunRuntime.open(source_bundle, trainer_root, device="cuda", reference_step=100)
    result: dict[str, Any] = {
        "entry_id": entry_id,
        "source_bundle_id": source_bundle_id,
        "phase_selection": phase_selection,
        "phases": {},
        "validation_input_source": runtime.source_validation_inputs,
        "training_input_source": runtime.source_training_inputs,
        "training_target_source": runtime.source_training_targets,
        "validation_targets": runtime.validation_targets.detach().cpu().numpy().copy(),
        "target_mapping_certificate": runtime.target_mapping_certificate,
    }
    try:
        for phase, selection in phase_selection.items():
            step = int(selection["optimizer_step"])
            parameter_rows = runtime.load_checkpoint(step)
            for name, parameter in runtime.model.named_parameters():
                require(tensor_sha256(parameter) == parameter_rows[name]["content_sha256"], f"TLI_PARAMETER_IDENTITY:{entry_id}:{phase}:{name}")
            baseline, calls = forward_with_calls(runtime)
            repeat = runtime.forward()
            require(torch.equal(baseline, repeat), f"TLI_BASELINE_REPEAT:{entry_id}:{phase}")
            singles = {component: runtime.forward((component,)) for component in COMPONENTS}
            pairs = {"+".join(pair): runtime.forward(pair) for pair in COMPONENT_PAIRS}
            phase_result: dict[str, Any] = {
                "optimizer_step": step,
                "selection": selection,
                "parameter_rows": parameter_rows,
                "calls": calls,
                "baseline_logits": baseline.numpy().copy(),
                "baseline_repeat_logits": repeat.numpy().copy(),
                "single_gate_logits": {key: value.numpy().copy() for key, value in singles.items()},
                "pair_gate_logits": {key: value.numpy().copy() for key, value in pairs.items()},
                "baseline": _json_decisions(baseline, runtime.validation_targets),
                "single_gate": {key: _json_decisions(value, runtime.validation_targets) for key, value in singles.items()},
                "pair_gate": {key: _json_decisions(value, runtime.validation_targets) for key, value in pairs.items()},
            }
            result["phases"][phase] = phase_result

        formed = result["phases"]["formed"]
        pre = result["phases"]["pre_formation"]
        formed_step = int(formed["optimizer_step"])
        pre_step = int(pre["optimizer_step"])
        formed_rows = runtime.load_checkpoint(formed_step)
        pre_rows = unique_role_objects(
            objects_for_stage(runtime.graph, pre_step, "after_optimizer_step"),
            "parameter_version",
        )
        named = dict(runtime.model.named_parameters())
        rollback: dict[str, Any] = {}
        for component in COMPONENTS:
            formed_component = component_parameter_rows(formed_rows, component)
            pre_component = component_parameter_rows(pre_rows, component)
            require(set(formed_component) == set(pre_component), f"TLI_ROLLBACK_PARAMETER_SET:{component}")
            with torch.no_grad():
                for name, row in pre_component.items():
                    named[name].copy_(load_tensor(source_bundle, row).to(runtime.device))
            rollback_logits = runtime.forward()
            with torch.no_grad():
                for name, row in formed_component.items():
                    named[name].copy_(load_tensor(source_bundle, row).to(runtime.device))
            for name, row in formed_component.items():
                require(tensor_sha256(named[name]) == row["content_sha256"], f"TLI_RESTORE_PARAMETER_HASH:{entry_id}:{component}:{name}")
            restored_logits = runtime.forward()
            require(torch.equal(restored_logits, torch.from_numpy(formed["baseline_logits"])), f"TLI_RESTORE_LOGITS:{entry_id}:{component}")
            rollback[component] = {
                "pre_parameter_rows": pre_component,
                "formed_parameter_rows": formed_component,
                "rollback_logits": rollback_logits.numpy().copy(),
                "restored_logits": restored_logits.numpy().copy(),
                "rollback": _json_decisions(rollback_logits, runtime.validation_targets),
                "restored": _json_decisions(restored_logits, runtime.validation_targets),
            }
        result["rollback"] = rollback
        return result
    finally:
        runtime.close()


__all__ = [
    "COMPONENTS",
    "COMPONENT_PAIRS",
    "execute_run",
    "select_phases",
    "validated_accuracy_curve",
]
