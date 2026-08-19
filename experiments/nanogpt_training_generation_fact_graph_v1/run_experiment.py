from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
import torch

from .capture_runtime import (
    TrainingFactRecorder,
    tensor_content_sha256,
    tensor_descriptor,
)
from .core_snapshot import build_core_snapshot
from .graph_artifacts import validate_graph, write_artifacts


TRAINER_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
RUN_ID = "nanogpt-shakespeare-cuda-training-capture-v1"
SEED = 1337
STEPS = 3
MICRO_STEPS = 2
BATCH_SIZE = 8
BLOCK_SIZE = 128


def _load_model_module(trainer_root: Path) -> Any:
    model_path = trainer_root / "model.py"
    spec = importlib.util.spec_from_file_location(
        "frozen_nanogpt_model",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("NANOGPT_MODEL_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(
    *,
    data: np.memmap,
    generator: torch.Generator,
    recorder: TrainingFactRecorder,
    step: int,
    micro_step: int,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    indices = torch.randint(
        len(data) - BLOCK_SIZE,
        (BATCH_SIZE,),
        generator=generator,
    )
    starts = [int(value) for value in indices.tolist()]
    x_cpu = torch.stack(
        [
            torch.from_numpy(
                data[index : index + BLOCK_SIZE].astype(np.int64)
            )
            for index in starts
        ]
    )
    y_cpu = torch.stack(
        [
            torch.from_numpy(
                data[index + 1 : index + 1 + BLOCK_SIZE].astype(np.int64)
            )
            for index in starts
        ]
    )
    x_ref = f"source:dataset:step:{step:03d}:micro:{micro_step:02d}:x"
    y_ref = f"source:dataset:step:{step:03d}:micro:{micro_step:02d}:y"
    recorder.register_source(
        x_cpu,
        x_ref,
        {
            "batch_role": "input_tokens",
            "content_sha256": tensor_content_sha256(x_cpu),
            "dataset": "shakespeare_char/train.bin",
            "source_kind": "exact_dataset_windows",
            "start_indices": starts,
            "tensor": tensor_descriptor(x_cpu),
        },
    )
    recorder.register_source(
        y_cpu,
        y_ref,
        {
            "batch_role": "target_tokens",
            "content_sha256": tensor_content_sha256(y_cpu),
            "dataset": "shakespeare_char/train.bin",
            "source_kind": "exact_shifted_dataset_windows",
            "start_indices": [value + 1 for value in starts],
            "tensor": tensor_descriptor(y_cpu),
        },
    )
    with recorder.stage(
        step=step,
        micro_step=micro_step,
        phase=phase,
    ):
        x = x_cpu.pin_memory().to("cuda", non_blocking=True)
        y = y_cpu.pin_memory().to("cuda", non_blocking=True)
    return x, y, starts


def _register_initial_sources(
    recorder: TrainingFactRecorder,
    model: torch.nn.Module,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        stable_ref = f"source:parameter:{name}:version:000"
        recorder.register_source(
            parameter,
            stable_ref,
            {
                "content_sha256": tensor_content_sha256(parameter),
                "parameter_name": name,
                "source_kind": "initialized_model_parameter",
                "tensor": tensor_descriptor(parameter),
                "version": 0,
            },
        )
        refs[name] = stable_ref
    recorder.register_literal_source(
        "source:optimizer:adamw:configuration",
        {
            "algorithm": "torch.optim.AdamW",
            "betas": [0.9, 0.99],
            "fused": True,
            "learning_rate": 0.001,
            "source_kind": "frozen_optimizer_configuration",
            "weight_decay": 0.1,
        },
    )
    return refs


def _parameter_gradient_receipts(
    recorder: TrainingFactRecorder,
    model: torch.nn.Module,
    parameter_refs: dict[str, str],
    loss_ref: str,
    prior_gradient_refs: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"PARAMETER_GRADIENT_MISSING:{name}")
        inputs = [
            (loss_ref, "loss_seed"),
            (parameter_refs[name], "differentiated_parameter"),
        ]
        prior = prior_gradient_refs.get(name)
        if prior is not None:
            inputs.append((prior, "prior_accumulated_gradient"))
        refs = recorder.emit_manual(
            transform_reference={
                "framework": "PyTorch Autograd",
                "operation": "parameter_gradient_accumulation",
                "parameter_name": name,
            },
            inputs=inputs,
            output_tensors=[(parameter.grad, "gradient")],
            receipt_payload={
                "gradient_accumulation": prior is not None,
                "parameter_name": name,
            },
        )
        result[name] = refs[0]
    return result


def _gradient_clip_receipt(
    recorder: TrainingFactRecorder,
    model: torch.nn.Module,
    gradient_refs: dict[str, str],
    total_norm: torch.Tensor,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"CLIPPED_PARAMETER_GRADIENT_MISSING:{name}")
        refs = recorder.emit_manual(
            transform_reference={
                "framework": "PyTorch",
                "max_norm": 1.0,
                "operation": "torch.nn.utils.clip_grad_norm_",
                "parameter_name": name,
            },
            inputs=[(gradient_refs[name], "unclipped_accumulated_gradient")],
            output_tensors=[(parameter.grad, "clipped_gradient")],
            receipt_payload={
                "parameter_name": name,
                "total_norm": float(total_norm.detach().cpu()),
            },
        )
        result[name] = refs[0]
    return result


def _optimizer_receipt(
    recorder: TrainingFactRecorder,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parameter_refs_before: dict[str, str],
    gradient_refs: dict[str, str],
    prior_state_refs: dict[str, dict[str, str]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    parameter_refs_after: dict[str, str] = {}
    state_refs_after: dict[str, dict[str, str]] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        inputs = [
            (parameter_refs_before[name], "parameter_before_update"),
            (gradient_refs[name], "clipped_gradient"),
            (
                "source:optimizer:adamw:configuration",
                "optimizer_configuration",
            ),
        ]
        for key in ("exp_avg", "exp_avg_sq"):
            prior = prior_state_refs.get(name, {}).get(key)
            if prior is not None:
                inputs.append((prior, f"{key}_before_update"))
        output_tensors = [
            (parameter, "parameter"),
            (state["exp_avg"], "optimizer_state_exp_avg"),
            (state["exp_avg_sq"], "optimizer_state_exp_avg_sq"),
        ]
        refs = recorder.emit_manual(
            transform_reference={
                "algorithm": "AdamW",
                "fused": True,
                "operation": "torch.optim.AdamW.step",
                "parameter_name": name,
            },
            inputs=inputs,
            output_tensors=output_tensors,
            receipt_payload={
                "parameter_name": name,
                "state_step": float(state["step"].detach().cpu()),
            },
        )
        parameter_refs_after[name] = refs[0]
        state_refs_after[name] = {
            "exp_avg": refs[1],
            "exp_avg_sq": refs[2],
        }
    return parameter_refs_after, state_refs_after


def run(trainer_root: Path, output_dir: Path) -> dict[str, Any]:
    actual_commit = (
        trainer_root / ".git" / "HEAD"
    )
    if not actual_commit.exists():
        raise RuntimeError("NANOGPT_GIT_CHECKOUT_REQUIRED")
    import subprocess

    resolved_commit = subprocess.check_output(
        ["git", "-C", str(trainer_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if resolved_commit != TRAINER_COMMIT:
        raise RuntimeError(
            f"NANOGPT_COMMIT_DRIFT:{resolved_commit}:{TRAINER_COMMIT}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")

    data_dir = trainer_root / "data" / "shakespeare_char"
    train_path = data_dir / "train.bin"
    meta_path = data_dir / "meta.pkl"
    if not train_path.exists() or not meta_path.exists():
        raise RuntimeError("SHAKESPEARE_DATASET_NOT_PREPARED")
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    data = np.memmap(train_path, dtype=np.uint16, mode="r")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_module = _load_model_module(trainer_root)
    config = model_module.GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=int(meta["vocab_size"]),
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
    )
    model = model_module.GPT(config).to("cuda")
    model.train()
    optimizer = model.configure_optimizers(
        0.1,
        0.001,
        (0.9, 0.99),
        "cuda",
    )
    recorder = TrainingFactRecorder(RUN_ID)
    parameter_refs = _register_initial_sources(recorder, model)
    optimizer_state_refs: dict[str, dict[str, str]] = {}
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(SEED + 101)
    losses: list[float] = []
    step_elapsed_ms: list[float] = []
    batch_indices: list[dict[str, Any]] = []

    with recorder:
        x, y, starts = _batch(
            data=data,
            generator=batch_generator,
            recorder=recorder,
            step=0,
            micro_step=0,
            phase="batch_initial",
        )
        batch_indices.append({"micro_step": 0, "starts": starts, "step": 0})
        for step in range(STEPS):
            torch.cuda.synchronize()
            started = time.perf_counter()
            gradient_refs: dict[str, str] = {}
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(MICRO_STEPS):
                with recorder.stage(
                    step=step,
                    micro_step=micro_step,
                    phase="forward",
                ):
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        _logits, loss = model(x, y)
                        scaled_loss = loss / MICRO_STEPS
                    loss_ref = recorder.reference(scaled_loss)

                next_step = step + (1 if micro_step == MICRO_STEPS - 1 else 0)
                next_micro = 0 if micro_step == MICRO_STEPS - 1 else micro_step + 1
                next_x, next_y, starts = _batch(
                    data=data,
                    generator=batch_generator,
                    recorder=recorder,
                    step=next_step,
                    micro_step=next_micro,
                    phase="batch_prefetch",
                )
                batch_indices.append(
                    {
                        "micro_step": next_micro,
                        "starts": starts,
                        "step": next_step,
                    }
                )

                with recorder.stage(
                    step=step,
                    micro_step=micro_step,
                    phase="backward",
                ):
                    scaled_loss.backward()
                with recorder.stage(
                    step=step,
                    micro_step=micro_step,
                    phase="gradient_snapshot",
                ):
                    gradient_refs = _parameter_gradient_receipts(
                        recorder,
                        model,
                        parameter_refs,
                        loss_ref,
                        gradient_refs,
                    )
                losses.append(float(loss.detach().cpu()))
                x, y = next_x, next_y

            with recorder.stage(
                step=step,
                micro_step=None,
                phase="gradient_clip",
            ):
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )
                clipped_gradient_refs = _gradient_clip_receipt(
                    recorder,
                    model,
                    gradient_refs,
                    total_norm,
                )

            parameter_refs_before = dict(parameter_refs)
            with recorder.stage(
                step=step,
                micro_step=None,
                phase="optimizer_update",
            ):
                optimizer.step()
                parameter_refs, optimizer_state_refs = _optimizer_receipt(
                    recorder,
                    model,
                    optimizer,
                    parameter_refs_before,
                    clipped_gradient_refs,
                    optimizer_state_refs,
                )

            with recorder.stage(
                step=step,
                micro_step=None,
                phase="zero_grad",
            ):
                optimizer.zero_grad(set_to_none=True)
                marker = torch.tensor(
                    [step],
                    device="cuda",
                    dtype=torch.int64,
                )
                recorder.emit_manual(
                    transform_reference={
                        "operation": "torch.optim.Optimizer.zero_grad",
                        "set_to_none": True,
                    },
                    inputs=[
                        (ref, "updated_parameter")
                        for ref in parameter_refs.values()
                    ],
                    output_tensors=[(marker, "zero_grad_completion")],
                    receipt_payload={
                        "all_parameter_gradients_none": all(
                            parameter.grad is None
                            for parameter in model.parameters()
                        )
                    },
                )
            torch.cuda.synchronize()
            step_elapsed_ms.append(
                (time.perf_counter() - started) * 1000.0
            )

    capture = recorder.to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "runtime_receipts_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            capture,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "capture_checkpoint": str(checkpoint_path),
                "events": len(capture["events"]),
                "sources": len(capture["sources"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    snapshot, graph = build_core_snapshot(
        capture,
        trainer_commit=TRAINER_COMMIT,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        device_name=torch.cuda.get_device_name(0),
        source_code_path=Path(__file__).resolve(),
    )
    parameter_count = len(list(model.named_parameters()))
    validation = validate_graph(
        capture,
        graph,
        expected_steps=STEPS,
        parameter_count=parameter_count,
    )
    if validation["status"] != "PASS":
        raise RuntimeError(
            "GENERATION_FACT_GRAPH_VALIDATION_FAILED:"
            + json.dumps(validation, sort_keys=True)
        )
    paths = write_artifacts(
        output_dir,
        capture,
        snapshot,
        graph,
        validation,
        expected_steps=STEPS,
    )
    run_manifest = {
        "artifacts": [path.name for path in paths],
        "baseline_profile": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "gradient_accumulation_steps": MICRO_STEPS,
            "model": {
                "n_embd": 128,
                "n_head": 4,
                "n_layer": 4,
                "parameter_tensors": parameter_count,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
            "optimizer": "fused AdamW",
            "steps": STEPS,
        },
        "batch_indices": batch_indices,
        "cuda": str(torch.version.cuda),
        "dataset_sha256": _sha256_file(train_path),
        "device": torch.cuda.get_device_name(0),
        "losses": losses,
        "nanoGPT_commit": TRAINER_COMMIT,
        "run_id": RUN_ID,
        "step_elapsed_ms_instrumented": step_elapsed_ms,
        "torch": torch.__version__,
        "validation_status": validation["status"],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            run_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    args = parser.parse_args()
    result = run(
        args.trainer_root.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
