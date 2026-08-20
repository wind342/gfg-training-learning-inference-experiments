from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .analysis import (
    adam_memory_exchange,
    adamw_delta,
    analyze_update,
    clone_adam_memory,
    clone_state,
    receiving_state_exchange,
    state_delta,
)
from .data import DiffusionSchedule, evaluation_pack, loaders
from .gfg import CompactGFG, add_update_event
from .model import CifarDiffusionUNet
from .numeric import sha256_file, tensor_sha256, write_json


PACKAGE = Path(__file__).resolve().parent
CONTRACT = json.loads((PACKAGE / "MODEL_CONTRACT.json").read_text(encoding="utf-8"))
DEFAULT_DATA = Path(os.environ.get("GFG_CIFAR10_ROOT", "runtime/datasets/cifar10"))
DEFAULT_OUTPUT = Path("runtime") / "gfg_ddpm_cifar_generalization_v1"


def _seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _state_to(
    state: dict[str, Tensor], device: torch.device | str
) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in state.items()}


def _memory_to(
    memory: dict[str, dict[str, Tensor]], device: torch.device | str
) -> dict[str, dict[str, Tensor]]:
    return {
        name: {key: value.to(device) for key, value in row.items()}
        for name, row in memory.items()
    }


def _max_parameter_error(
    left: dict[str, Tensor], right: dict[str, Tensor], model: nn.Module
) -> float:
    names = {name for name, _ in model.named_parameters()}
    return max(float((left[name] - right[name]).abs().max()) for name in names)


def _evaluate(
    model: nn.Module,
    loader,
    schedule: DiffusionSchedule,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    total_squared = 0.0
    total_values = 0
    generator = torch.Generator(device=device).manual_seed(seed + 700_000)
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            timesteps = torch.randint(
                0,
                len(schedule.betas),
                (len(images),),
                generator=generator,
                device=device,
            )
            noise = torch.randn(images.shape, generator=generator, device=device)
            noisy = schedule.q_sample(images, timesteps, noise)
            prediction = model(noisy, timesteps)
            total_squared += float((prediction - noise).square().sum())
            total_values += noise.numel()
    return {"count": total_values, "epsilon_mse": total_squared / total_values}


def _manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema": "gfg-ddpm-cifar-generalization-manifest-v1",
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files.values()),
    }


def _integrity_pass(summary: dict[str, Any]) -> bool:
    integrity = summary["integrity"]
    return bool(
        integrity["gfg_validation"]["status"] == "PASS"
        and integrity["adamw_formula_max_abs_error"] <= 2e-6
        and integrity["alpha0_max_abs_error"] <= 1e-6
        and integrity["alpha1_native_max_abs_error"] <= 1e-6
        and integrity["support_repeat_max_abs_error"] <= 1e-7
    )


def run_seed(
    *,
    seed: int,
    data_root: Path,
    output_root: Path,
    download: bool,
    smoke: bool,
) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"OUTPUT_ROOT_EXISTS:{output_root}")
    output_root.mkdir(parents=True)
    _seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")
    profile = CONTRACT["smoke" if smoke else "formal"]
    optimizer_contract = CONTRACT["optimizer"]
    epochs = int(profile["epochs"])
    batch_size = int(profile["batch_size"])
    registered_epochs = list(profile["registered_epochs"])
    train_loader, test_loader = loaders(
        data_root,
        batch_size=batch_size,
        seed=seed,
        download=download,
        train_subset=profile.get("train_subset"),
        test_subset=profile.get("test_subset"),
        test_batch_size=max(batch_size, int(profile["anchor_count"])),
    )
    system = CONTRACT["system"]
    schedule = DiffusionSchedule.linear(
        int(system["diffusion_steps"]),
        float(system["beta_schedule"][0]),
        float(system["beta_schedule"][1]),
        device,
    )
    evaluation = evaluation_pack(
        test_loader,
        count=int(profile["anchor_count"]),
        seed=seed,
        candidate_count=int(system["candidate_count"]),
        candidate_scale=float(system["candidate_scale"]),
        schedule=schedule,
        device=device,
    )
    model = CifarDiffusionUNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_contract["learning_rate"]),
        betas=tuple(optimizer_contract["betas"]),
        eps=float(optimizer_contract["epsilon"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
        foreach=False,
        fused=False,
    )
    criterion = nn.MSELoss()
    generator = torch.Generator(device=device).manual_seed(seed + 100_000)
    graph = CompactGFG(f"ddpm-cifar10-adamw-seed-{seed}")
    events: list[dict[str, Any]] = []
    exchanges: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    formula_errors: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        squared_loss_sum = 0.0
        value_count = 0
        for batch_index, (images, labels) in enumerate(train_loader):
            del labels
            images = images.to(device, non_blocking=True)
            timesteps = torch.randint(
                0,
                len(schedule.betas),
                (len(images),),
                generator=generator,
                device=device,
            )
            noise = torch.randn(images.shape, generator=generator, device=device)
            noisy = schedule.q_sample(images, timesteps, noise)
            capture = epoch in registered_epochs and batch_index == 0
            pre = clone_state(model) if capture else None
            memory_pre = clone_adam_memory(model, optimizer) if capture else None
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy, timesteps)
            loss = criterion(prediction, noise)
            loss.backward()
            gradients = (
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                if capture
                else None
            )
            predicted_delta = (
                adamw_delta(
                    pre,
                    gradients,
                    memory_pre,
                    learning_rate=float(optimizer_contract["learning_rate"]),
                    betas=tuple(optimizer_contract["betas"]),
                    epsilon=float(optimizer_contract["epsilon"]),
                    weight_decay=float(optimizer_contract["weight_decay"]),
                )
                if capture
                else None
            )
            optimizer.step()
            squared_loss_sum += float(loss.detach()) * noise.numel()
            value_count += noise.numel()
            if not capture:
                continue
            assert pre is not None and memory_pre is not None
            assert gradients is not None and predicted_delta is not None
            post = clone_state(model)
            native_delta = state_delta(pre, post)
            formula_error = _max_parameter_error(
                native_delta, predicted_delta, model
            )
            formula_errors.append(formula_error)
            analysis = analyze_update(
                model=model,
                pre=pre,
                post=post,
                memory_pre=memory_pre,
                evaluation=evaluation,
                alpha_bar=schedule.alpha_bar,
                target_count=int(profile["target_count"]),
            )
            event = {
                "event_index": len(events),
                "epoch": epoch,
                "batch_index": batch_index,
                "training_loss": float(loss.detach()),
                "adamw_formula_max_abs_error": formula_error,
                "analysis": analysis,
            }
            events.append(event)
            batch_ids = list(range(batch_index * batch_size, batch_index * batch_size + len(images)))
            add_update_event(
                graph,
                len(events) - 1,
                epoch,
                batch_ids,
                tensor_sha256(timesteps),
                tensor_sha256(noise),
                float(loss.detach()),
                analysis,
            )
            if previous is not None:
                prior_pre = _state_to(previous["pre"], device)
                prior_memory = _memory_to(previous["memory"], device)
                exchanges.append(
                    {
                        "from_event_index": len(events) - 1,
                        "to_event_index": previous["event_index"],
                        "receiving_state": receiving_state_exchange(
                            model, pre, prior_pre, native_delta, evaluation
                        ),
                        "adam_memory": adam_memory_exchange(
                            model,
                            pre,
                            native_delta,
                            prior_memory,
                            gradients,
                            evaluation,
                            learning_rate=float(optimizer_contract["learning_rate"]),
                            betas=tuple(optimizer_contract["betas"]),
                            epsilon=float(optimizer_contract["epsilon"]),
                            weight_decay=float(optimizer_contract["weight_decay"]),
                        ),
                    }
                )
            previous = {
                "event_index": len(events) - 1,
                "pre": _state_to(pre, "cpu"),
                "memory": _memory_to(memory_pre, "cpu"),
            }
            model.train()
        row = {
            "epoch": epoch,
            "training_epsilon_mse": squared_loss_sum / value_count,
        }
        if epoch in registered_epochs or epoch == epochs:
            row["test"] = _evaluate(model, test_loader, schedule, device, seed + epoch)
        metrics.append(row)
        print(json.dumps({"seed": seed, **row}, sort_keys=True), flush=True)

    graph_document = graph.document()
    integrity_rows = [event["analysis"]["integrity"] for event in events]
    summary = {
        "schema": "gfg-ddpm-cifar-run-summary-v1",
        "seed": seed,
        "smoke": smoke,
        "event_count": len(events),
        "exchange_count": len(exchanges),
        "final_test": metrics[-1]["test"],
        "integrity": {
            "gfg_validation": graph_document["validation"],
            "adamw_formula_max_abs_error": max(formula_errors, default=0.0),
            "alpha0_max_abs_error": max(
                (row["alpha0_max_abs_error"] for row in integrity_rows), default=0.0
            ),
            "alpha1_native_max_abs_error": max(
                (row["alpha1_native_max_abs_error"] for row in integrity_rows), default=0.0
            ),
            "support_repeat_max_abs_error": max(
                (row["support_repeat_max_abs_error"] for row in integrity_rows), default=0.0
            ),
        },
    }
    summary["integrity_pass"] = _integrity_pass(summary)
    write_json(output_root / "EVENTS.json", events)
    write_json(output_root / "EXCHANGES.json", exchanges)
    write_json(output_root / "TRAINING_METRICS.json", metrics)
    write_json(output_root / "COMPACT_GFG.json", graph_document)
    write_json(output_root / "RUN_SUMMARY.json", summary)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        output_root / "FINAL_CHECKPOINT.pt",
    )
    write_json(output_root / "MANIFEST.json", _manifest(output_root))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run_seed(
        seed=args.seed,
        data_root=args.data_root,
        output_root=args.output_root,
        download=args.download,
        smoke=args.smoke,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
