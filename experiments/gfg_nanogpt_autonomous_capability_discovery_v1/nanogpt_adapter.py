from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from .common import NANOGPT_COMMIT, payload_sha256, require
from .task_generator import TokenizedTask


@dataclass(frozen=True)
class TrainingConfig:
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False
    learning_rate: float = 0.001
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    gradient_clip: float = 1.0
    max_steps: int = 6000
    evaluation_interval: int = 50
    seed: int = 1729
    data_order_seed: int = 2718
    device: str = "cuda"


@dataclass
class TrainingCheckpoint:
    step: int
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_state: list[torch.Tensor]
    numpy_rng_state: tuple[Any, ...]
    python_rng_state: tuple[Any, ...]
    data_generator_state: torch.Tensor


def _load_model_module(trainer_root: Path) -> Any:
    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(trainer_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    require(commit == NANOGPT_COMMIT, f"NANOGPT_COMMIT_DRIFT:{commit}")
    path = trainer_root / "model.py"
    spec = importlib.util.spec_from_file_location(
        "capability_discovery_frozen_nanogpt_model",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("NANOGPT_MODEL_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_model_and_optimizer(
    trainer_root: Path,
    task: TokenizedTask,
    config: TrainingConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    require(config.device == "cpu" or torch.cuda.is_available(), "CUDA_REQUIRED")
    configure_determinism(config.seed)
    module = _load_model_module(trainer_root)
    model_config = module.GPTConfig(
        block_size=int(task.train_inputs.shape[1]),
        vocab_size=task.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias,
    )
    model = module.GPT(model_config).to(config.device)
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        config.device,
    )
    return model, optimizer


@torch.no_grad()
def evaluation_details(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: str,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    logits, _loss = model(inputs.to(device), targets.to(device))
    predicted = logits[:, -1, :].argmax(dim=-1).cpu()
    expected = targets[:, -1]
    value = float((predicted == expected).float().mean())
    model.train()
    return value, predicted, logits[:, -1, :].detach().cpu()


def accuracy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: str,
) -> float:
    value, _predicted, _logits = evaluation_details(
        model,
        inputs,
        targets,
        device,
    )
    return value


def capture_checkpoint(
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        step=step,
        model_state={
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        optimizer_state=_cpu_clone(optimizer.state_dict()),
        torch_cpu_rng_state=torch.get_rng_state().clone(),
        torch_cuda_rng_state=[row.clone() for row in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        numpy_rng_state=np.random.get_state(),
        python_rng_state=random.getstate(),
        data_generator_state=data_generator.get_state().clone(),
    )


def restore_checkpoint(
    checkpoint: TrainingCheckpoint,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    device: str,
) -> None:
    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    torch.set_rng_state(checkpoint.torch_cpu_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint.torch_cuda_rng_state)
    np.random.set_state(checkpoint.numpy_rng_state)
    random.setstate(checkpoint.python_rng_state)
    data_generator.set_state(checkpoint.data_generator_state)


def checkpoint_commitment(checkpoint: TrainingCheckpoint) -> str:
    rows: dict[str, Any] = {"step": checkpoint.step, "model": {}, "optimizer": {}}
    for key, tensor in sorted(checkpoint.model_state.items()):
        rows["model"][key] = _tensor_commitment(tensor)
    state = checkpoint.optimizer_state
    rows["optimizer"]["param_groups"] = state["param_groups"]
    rows["optimizer"]["state"] = {
        str(index): {
            key: _tensor_commitment(value) if isinstance(value, torch.Tensor) else value
            for key, value in sorted(values.items())
        }
        for index, values in sorted(state["state"].items())
    }
    rows["rng"] = {
        "cpu": _tensor_commitment(checkpoint.torch_cpu_rng_state),
        "cuda": [
            _tensor_commitment(value) for value in checkpoint.torch_cuda_rng_state
        ],
        "data": _tensor_commitment(checkpoint.data_generator_state),
    }
    return payload_sha256(rows)


def _tensor_commitment(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().contiguous().cpu()
    return {
        "dtype": str(array.dtype),
        "sha256": payload_sha256(
            {
                "bytes": array.numpy().tobytes().hex(),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        ),
        "shape": list(array.shape),
    }


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(child) for child in value)
    return value


def train_plain(
    trainer_root: Path,
    task: TokenizedTask,
    config: TrainingConfig,
    *,
    stop_step: int | None = None,
    initial_checkpoint: TrainingCheckpoint | None = None,
    hook: Callable[[str, dict[str, Any]], None] | None = None,
    intervention_hook: Callable[
        [str, dict[str, Any], dict[str, Any]], dict[str, Any] | None
    ]
    | None = None,
    intervention_state: dict[str, Any] | None = None,
    stop_when: Callable[[dict[str, Any], list[dict[str, Any]]], bool] | None = None,
) -> dict[str, Any]:
    model, optimizer = build_model_and_optimizer(trainer_root, task, config)
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(config.data_order_seed)
    fixed_batch_order = torch.randperm(
        task.train_inputs.shape[0],
        generator=data_generator,
    )
    start_step = 0
    if initial_checkpoint is not None:
        restore_checkpoint(
            initial_checkpoint,
            model,
            optimizer,
            data_generator,
            config.device,
        )
        start_step = initial_checkpoint.step
    end_step = config.max_steps if stop_step is None else stop_step
    require(end_step >= start_step, "INVALID_TRAINING_RANGE")
    train_x = task.train_inputs
    train_y = task.train_targets
    metrics: list[dict[str, Any]] = []
    losses: list[float] = []
    started = time.perf_counter()
    checkpoint: TrainingCheckpoint | None = None
    active_intervention_state = {} if intervention_state is None else intervention_state
    actual_end_step = start_step

    def emit(stage: str, payload: dict[str, Any]) -> None:
        nonlocal active_intervention_state
        # Capture the native stage result before an allowed intervention
        # mutates the live training objects exposed at that boundary.
        if hook is not None:
            hook(stage, payload)
        if intervention_hook is not None and stage in {
            "before_batch",
            "after_forward",
            "after_backward",
            "before_gradient_clip",
            "after_gradient_clip",
            "before_optimizer_step",
            "after_optimizer_step",
        }:
            updated = intervention_hook(stage, payload, active_intervention_state)
            if updated is not None:
                active_intervention_state = updated

    for step in range(start_step, end_step):
        order = fixed_batch_order
        batch_x = train_x[order].to(config.device)
        batch_y = train_y[order].to(config.device)
        emit(
            "before_batch",
            {
                "model": model,
                "optimizer": optimizer,
                "order": order,
                "step": step,
                "x": batch_x,
                "y": batch_y,
            },
        )
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(batch_x, batch_y)
        emit(
            "after_forward",
            {
                "logits": logits,
                "loss": loss,
                "model": model,
                "optimizer": optimizer,
                "step": step,
            },
        )
        loss.backward()
        emit(
            "after_backward",
            {
                "loss": loss,
                "model": model,
                "optimizer": optimizer,
                "step": step,
            },
        )
        gradient_clip_context = {
            "max_norm": float(config.gradient_clip),
            "model": model,
            "optimizer": optimizer,
            "step": step,
        }
        emit("before_gradient_clip", gradient_clip_context)
        gradient_clip_max_norm = gradient_clip_context["max_norm"]
        require(
            isinstance(gradient_clip_max_norm, (int, float))
            and not isinstance(gradient_clip_max_norm, bool)
            and math.isfinite(float(gradient_clip_max_norm))
            and float(gradient_clip_max_norm) > 0,
            "INTERVENTION_GRADIENT_CLIP_CONTROL_INVALID",
        )
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(gradient_clip_max_norm),
        )
        emit(
            "after_gradient_clip",
            {
                "model": model,
                "optimizer": optimizer,
                "step": step,
                "total_norm": total_norm,
            },
        )
        emit(
            "before_optimizer_step",
            {"model": model, "optimizer": optimizer, "step": step},
        )
        optimizer.step()
        emit(
            "after_optimizer_step",
            {
                "model": model,
                "optimizer": optimizer,
                "parameter_version": step + 1,
                "step": step,
            },
        )
        losses.append(float(loss.detach().cpu()))
        evaluation_step = step + 1
        actual_end_step = evaluation_step
        if (
            evaluation_step == 1
            or evaluation_step % config.evaluation_interval == 0
            or evaluation_step == end_step
        ):
            emit(
                "before_evaluation",
                {
                    "model": model,
                    "parameter_version": evaluation_step,
                    "step": evaluation_step,
                },
            )
            emit(
                "before_evaluation_split",
                {
                    "model": model,
                    "parameter_version": evaluation_step,
                    "split": "train",
                    "step": evaluation_step,
                },
            )
            train_accuracy, train_predictions, train_logits = evaluation_details(
                model,
                task.train_inputs,
                task.train_targets,
                config.device,
            )
            emit(
                "before_evaluation_split",
                {
                    "model": model,
                    "parameter_version": evaluation_step,
                    "split": "validation",
                    "step": evaluation_step,
                },
            )
            (
                validation_accuracy,
                validation_predictions,
                validation_logits,
            ) = evaluation_details(
                model,
                task.validation_inputs,
                task.validation_targets,
                config.device,
            )
            metrics.append(
                {
                    "loss": losses[-1],
                    "parameter_version": evaluation_step,
                    "step": evaluation_step,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": validation_accuracy,
                }
            )
            emit(
                "evaluation",
                {
                    "metrics": metrics[-1],
                    "model": model,
                    "parameter_version": evaluation_step,
                    "step": evaluation_step,
                    "train_inputs": task.train_inputs,
                    "train_targets": task.train_targets,
                    "train_logits": train_logits,
                    "train_predictions": train_predictions,
                    "validation_inputs": task.validation_inputs,
                    "validation_targets": task.validation_targets,
                    "validation_logits": validation_logits,
                    "validation_predictions": validation_predictions,
                },
            )
            if stop_when is not None and stop_when(metrics[-1], metrics):
                break
    checkpoint = capture_checkpoint(
        step=actual_end_step,
        model=model,
        optimizer=optimizer,
        data_generator=data_generator,
    )
    if config.device == "cuda":
        torch.cuda.synchronize()
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_commitment(checkpoint),
        "config": asdict(config),
        "elapsed_seconds": time.perf_counter() - started,
        "losses": losses,
        "metrics": metrics,
        "intervention_state": active_intervention_state,
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "model_parameter_tensor_count": len(list(model.parameters())),
        "start_step": start_step,
        "stop_step": actual_end_step,
    }


def detect_transition(
    metrics: list[dict[str, Any]],
    *,
    train_threshold: float,
    pre_transition_validation_max: float,
    validation_threshold: float,
    sustained_points: int,
) -> int | None:
    for index in range(0, len(metrics) - sustained_points + 1):
        window = metrics[index : index + sustained_points]
        if not all(row["train_accuracy"] >= train_threshold for row in window):
            continue
        if not any(
            prior["validation_accuracy"] <= pre_transition_validation_max
            for prior in metrics[: index + 1]
        ):
            continue
        if all(row["validation_accuracy"] >= validation_threshold for row in window):
            return int(window[0]["step"])
    return None
