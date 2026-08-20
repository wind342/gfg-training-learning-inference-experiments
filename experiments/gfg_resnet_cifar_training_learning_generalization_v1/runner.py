from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .analysis import (
    analyze_update,
    clone_state,
    momentum_exchange_response,
    receiving_state_exchange,
    state_delta,
)
from .data import loaders
from .gfg import CompactGFG, add_update_event
from .model import CifarResNet18
from .numeric import sha256_file, state_sha256, write_json


PACKAGE = Path(__file__).resolve().parent
CONTRACT = json.loads((PACKAGE / "MODEL_CONTRACT.json").read_text(encoding="utf-8"))
DEFAULT_DATA = Path(os.environ.get("GFG_CIFAR100_ROOT", "runtime/datasets/cifar100"))
DEFAULT_OUTPUT = Path("runtime") / "gfg_resnet_cifar_generalization_v1"


def _seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _momentum_state(
    model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device | str
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
        result[name] = (
            torch.zeros_like(parameter, device=device)
            if buffer is None
            else buffer.detach().clone().to(device)
        )
    return result


def _state_to(state: dict[str, Tensor], device: torch.device | str):
    return {name: value.to(device) for name, value in state.items()}


def _sgd_delta(
    pre: dict[str, Tensor],
    gradients: dict[str, Tensor],
    momentum_pre: dict[str, Tensor],
    native_delta: dict[str, Tensor],
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> dict[str, Tensor]:
    result = {name: value.detach().clone() for name, value in native_delta.items()}
    for name, gradient in gradients.items():
        direction = gradient + weight_decay * pre[name]
        next_buffer = momentum * momentum_pre[name] + direction
        result[name] = -learning_rate * next_buffer
    return result


def _max_parameter_error(
    left: dict[str, Tensor], right: dict[str, Tensor], model: nn.Module
) -> float:
    names = {name for name, _ in model.named_parameters()}
    return max(float((left[name] - right[name]).abs().max()) for name in names)


def _evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    count = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(criterion(logits, labels))
            correct += int(logits.argmax(dim=1).eq(labels).sum())
            count += labels.numel()
    return {
        "count": count,
        "accuracy": correct / count,
        "mean_loss": loss_sum / count,
    }


def _manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema": "gfg-resnet-cifar-generalization-manifest-v1",
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files.values()),
    }


def _integrity_pass(summary: dict[str, Any]) -> bool:
    integrity = summary["integrity"]
    return bool(
        integrity["gfg_validation"]["status"] == "PASS"
        and integrity["sgd_formula_max_abs_error"] <= 1e-6
        and integrity["alpha0_max_abs_error"] <= 1e-5
        and integrity["alpha1_native_max_abs_error"] <= 1e-5
        and integrity["support_repeat_max_abs_error"] <= 1e-6
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
    device = torch.device("cuda")
    formal_training = CONTRACT["training"]
    formal_eval = CONTRACT["evaluation"]
    epochs = 1 if smoke else int(formal_training["epochs"])
    batch_size = 64 if smoke else int(formal_training["batch_size"])
    anchor_count = 100 if smoke else int(formal_eval["anchor_count"])
    target_count = 16 if smoke else int(formal_eval["response_target_count_per_occurrence"])
    registered_epochs = [1] if smoke else list(CONTRACT["registered_epochs"])
    train_limit = 512 if smoke else None
    train_loader, anchor_loader, test_loader = loaders(
        data_root,
        seed,
        batch_size,
        anchor_count,
        download=download,
        train_limit=train_limit,
    )
    anchor_images, anchor_labels, anchor_ids = next(iter(anchor_loader))
    anchor_images = anchor_images.to(device)
    anchor_labels = anchor_labels.to(device)
    anchor_ids = anchor_ids.to(device)
    model = CifarResNet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(formal_training["initial_learning_rate"]),
        momentum=float(formal_training["momentum"]),
        weight_decay=float(formal_training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(formal_training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()
    graph = CompactGFG(f"resnet-cifar100-sgd-seed-{seed}")
    events: list[dict[str, Any]] = []
    exchanges: list[dict[str, Any]] = []
    training_metrics: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0
        for batch_index, (images, labels, sample_ids) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            capture = epoch in registered_epochs and batch_index == 0
            pre = clone_state(model) if capture else None
            momentum_pre = (
                _momentum_state(model, optimizer, device) if capture else None
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = (
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                if capture
                else None
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * labels.numel()
            epoch_correct += int(logits.detach().argmax(dim=1).eq(labels).sum())
            epoch_count += labels.numel()
            if not capture:
                continue
            assert pre is not None and momentum_pre is not None and gradients is not None
            post = clone_state(model)
            delta = state_delta(pre, post)
            formula_delta = _sgd_delta(
                pre,
                gradients,
                momentum_pre,
                delta,
                learning_rate,
                float(formal_training["momentum"]),
                float(formal_training["weight_decay"]),
            )
            formula_error = _max_parameter_error(delta, formula_delta, model)
            analysis = analyze_update(
                model=model,
                pre=pre,
                post=post,
                momentum_pre=momentum_pre,
                anchor_images=anchor_images,
                anchor_labels=anchor_labels,
                anchor_ids=anchor_ids,
                target_count=target_count,
            )
            event = {
                "event_index": len(events),
                "epoch": epoch,
                "batch_index": batch_index,
                "learning_rate": learning_rate,
                "training_loss": float(loss.detach()),
                "training_batch_ids": [int(item) for item in sample_ids.tolist()],
                "sgd_formula_max_abs_error": formula_error,
                "analysis": analysis,
            }
            add_update_event(
                graph,
                event["event_index"],
                epoch,
                event["training_batch_ids"],
                event["training_loss"],
                analysis,
            )
            if previous is not None:
                prior_state = _state_to(previous["pre"], device)
                prior_delta = _state_to(previous["delta"], device)
                exchange = receiving_state_exchange(
                    model=model,
                    state_a=prior_state,
                    delta_a=prior_delta,
                    state_b=pre,
                    delta_b=delta,
                    images=anchor_images[:target_count],
                    labels=anchor_labels[:target_count],
                )
                prior_momentum = _state_to(previous["momentum_pre"], device)
                exchanged_delta = _sgd_delta(
                    pre,
                    gradients,
                    prior_momentum,
                    delta,
                    learning_rate,
                    float(formal_training["momentum"]),
                    float(formal_training["weight_decay"]),
                )
                exchange["momentum_receiving_state_exchange"] = (
                    momentum_exchange_response(
                        model=model,
                        state=pre,
                        native_delta=delta,
                        exchanged_delta=exchanged_delta,
                        images=anchor_images[:target_count],
                        labels=anchor_labels[:target_count],
                    )
                )
                exchange["from_event"] = previous["event_index"]
                exchange["to_event"] = event["event_index"]
                exchanges.append(exchange)
            previous = {
                "event_index": event["event_index"],
                "pre": _state_to(pre, "cpu"),
                "delta": _state_to(delta, "cpu"),
                "momentum_pre": _state_to(momentum_pre, "cpu"),
            }
            events.append(event)
            write_json(output_root / "EVENTS.partial.json", events)
            model.train()
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "capture": event["event_index"],
                        "alpha1_error": analysis["alpha1_native_logit_max_abs_error"],
                        "support_repeat_error": analysis["support"]["repeat_max_abs_error"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_loss / epoch_count,
            "train_accuracy": epoch_correct / epoch_count,
            "learning_rate_after_epoch": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch in registered_epochs or epoch == epochs:
            row["test"] = _evaluate(model, test_loader, device)
        training_metrics.append(row)
        print(json.dumps({"seed": seed, **row}, sort_keys=True), flush=True)

    final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "seed": seed,
            "model_state": final_state,
            "optimizer_state": optimizer.state_dict(),
            "contract_sha256": sha256_file(PACKAGE / "MODEL_CONTRACT.json"),
        },
        output_root / "FINAL_CHECKPOINT.pt",
    )
    graph_document = graph.document()
    integrity = {
        "sgd_formula_max_abs_error": max(
            event["sgd_formula_max_abs_error"] for event in events
        ),
        "alpha0_max_abs_error": max(
            event["analysis"]["alpha0_logit_max_abs_error"] for event in events
        ),
        "alpha1_native_max_abs_error": max(
            event["analysis"]["alpha1_native_logit_max_abs_error"] for event in events
        ),
        "support_repeat_max_abs_error": max(
            event["analysis"]["support"]["repeat_max_abs_error"] for event in events
        ),
        "gfg_validation": graph_document["validation"],
    }
    run_summary = {
        "schema": "gfg-resnet-cifar-generalization-run-v1",
        "seed": seed,
        "smoke": smoke,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "final_test": training_metrics[-1]["test"],
        "event_count": len(events),
        "exchange_count": len(exchanges),
        "integrity": integrity,
        "final_state_sha256": state_sha256(final_state),
    }
    shutil.copy2(PACKAGE / "PROTOCOL_FREEZE.md", output_root / "PROTOCOL_FREEZE.md")
    shutil.copy2(PACKAGE / "MODEL_CONTRACT.json", output_root / "MODEL_CONTRACT.json")
    write_json(output_root / "EVENTS.json", events)
    write_json(output_root / "EXCHANGES.json", exchanges)
    write_json(output_root / "TRAINING_METRICS.json", training_metrics)
    write_json(output_root / "COMPACT_GFG.json", graph_document)
    write_json(output_root / "RUN_SUMMARY.json", run_summary)
    partial = output_root / "EVENTS.partial.json"
    if partial.exists():
        partial.unlink()
    write_json(output_root / "MANIFEST.json", _manifest(output_root))
    return run_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "primary", "formal"), default="smoke")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    seeds = (
        [int(CONTRACT["system"]["primary_seed"])]
        if args.mode in ("smoke", "primary")
        else [
            int(CONTRACT["system"]["primary_seed"]),
            *[int(seed) for seed in CONTRACT["system"]["confirmation_seeds"]],
        ]
    )
    root = args.output_root.resolve()
    if root.exists():
        raise RuntimeError(f"OUTPUT_ROOT_EXISTS:{root}")
    root.mkdir(parents=True)
    summaries = []
    for seed in seeds:
        summary = run_seed(
            seed=seed,
            data_root=args.data_root.resolve(),
            output_root=root / f"seed_{seed}",
            download=args.download,
            smoke=args.mode == "smoke",
        )
        summaries.append(summary)
        if args.mode == "formal" and len(summaries) == 1 and not _integrity_pass(summary):
            write_json(root / "RUNS.json", summaries)
            raise RuntimeError("PRIMARY_INTEGRITY_FAILED_CONFIRMATION_NOT_STARTED")
    write_json(root / "RUNS.json", summaries)
    write_json(root / "MANIFEST.json", _manifest(root))
    print(json.dumps({"mode": args.mode, "runs": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
