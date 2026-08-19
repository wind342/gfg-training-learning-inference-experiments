from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch

from .common import canonical_bytes, payload_sha256


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _model_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: _tensor_hash(parameter)
        for name, parameter in model.named_parameters()
    }


def _optimizer_state_hashes(
    optimizer: torch.optim.Optimizer,
) -> dict[str, str]:
    rows: dict[str, str] = {}
    for index, (_parameter, state) in enumerate(
        optimizer.state.items()
    ):
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                rows[f"{index}:{name}"] = _tensor_hash(value)
    return rows


def _gradient_hashes(
    model: torch.nn.Module,
) -> dict[str, str | None]:
    return {
        name: (
            None if parameter.grad is None else _tensor_hash(parameter.grad)
        )
        for name, parameter in model.named_parameters()
    }


class AuditedIntervention:
    def __init__(
        self,
        *,
        submission: Path,
        mechanism_state: dict[str, Any],
        forecast: dict[str, Any],
    ) -> None:
        path = submission / "intervention.py"
        spec = importlib.util.spec_from_file_location(
            "sealed_training_intervention_runtime", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("SEALED_INTERVENTION_LOAD_FAILED")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.intervention = module.TrainingIntervention()
        self.state = self.intervention.initialize(
            mechanism_state, forecast
        )
        canonical_bytes(self.state)
        self.events: list[dict[str, Any]] = []

    def __call__(
        self,
        stage: str,
        context: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        model = context.get("model")
        optimizer = context.get("optimizer")
        parameter_before = (
            _model_hashes(model) if model is not None else {}
        )
        gradient_before = (
            _gradient_hashes(model) if model is not None else {}
        )
        optimizer_state_before = (
            _optimizer_state_hashes(optimizer)
            if optimizer is not None
            else {}
        )
        groups_before = (
            [
                {
                    key: value
                    for key, value in group.items()
                    if key != "params"
                }
                for group in optimizer.param_groups
            ]
            if optimizer is not None
            else []
        )
        clip_control_before = context.get("max_norm")
        updated = self.intervention.apply(stage, context, self.state)
        if updated is not None:
            canonical_bytes(updated)
            self.state = updated
        parameter_after = (
            _model_hashes(model) if model is not None else {}
        )
        gradient_after = (
            _gradient_hashes(model) if model is not None else {}
        )
        optimizer_state_after = (
            _optimizer_state_hashes(optimizer)
            if optimizer is not None
            else {}
        )
        if parameter_before != parameter_after:
            raise RuntimeError("DIRECT_PARAMETER_VALUE_MUTATION_FORBIDDEN")
        if optimizer_state_before != optimizer_state_after:
            raise RuntimeError("DIRECT_OPTIMIZER_STATE_MUTATION_FORBIDDEN")
        if gradient_before != gradient_after:
            if stage not in {
                "after_backward",
                "before_gradient_clip",
                "after_gradient_clip",
                "before_optimizer_step",
            }:
                raise RuntimeError("GRADIENT_MUTATION_AT_INVALID_HOOK")
            self.events.append(
                {
                    "change": "current_gradients",
                    "stage": stage,
                    "step": int(context["step"]),
                    "after_sha256": payload_sha256(gradient_after),
                    "before_sha256": payload_sha256(gradient_before),
                }
            )
        groups_after = (
            [
                {
                    key: value
                    for key, value in group.items()
                    if key != "params"
                }
                for group in optimizer.param_groups
            ]
            if optimizer is not None
            else []
        )
        if groups_before != groups_after:
            self.events.append(
                {
                    "change": "optimizer_group_hyperparameters",
                    "stage": stage,
                    "step": int(context["step"]),
                    "after_sha256": payload_sha256(groups_after),
                    "before_sha256": payload_sha256(groups_before),
                }
            )
        clip_control_after = context.get("max_norm")
        if clip_control_before != clip_control_after:
            if stage != "before_gradient_clip":
                raise RuntimeError("GRADIENT_CLIP_CONTROL_AT_INVALID_HOOK")
            if (
                not isinstance(clip_control_after, (int, float))
                or isinstance(clip_control_after, bool)
            ):
                raise RuntimeError("GRADIENT_CLIP_CONTROL_INVALID")
            self.events.append(
                {
                    "after": float(clip_control_after),
                    "before": float(clip_control_before),
                    "change": "gradient_clipping_control",
                    "stage": stage,
                    "step": int(context["step"]),
                }
            )
        return self.state

    def receipt(self) -> dict[str, Any]:
        material = {
            "event_count": len(self.events),
            "events": self.events,
            "final_state_sha256": payload_sha256(self.state),
            "schema": "audited-training-intervention-receipt-v1",
        }
        return {
            **material,
            "receipt_sha256": payload_sha256(material),
        }
