from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .analysis import (
    adamw_delta,
    analyze_update,
    clone_adam_memory,
    clone_state,
    inference_projection_tests,
    memory_exchange_response,
    receiving_state_exchange,
    state_delta,
)
from .data import loaders
from .gfg import CompactGFG, add_inference_event, add_update_event
from .model import CifarVisionMoE
from .numeric import sha256_file, state_sha256, write_json


PACKAGE = Path(__file__).resolve().parent
CONTRACT = json.loads((PACKAGE / "MODEL_CONTRACT.json").read_text(encoding="utf-8"))
DEFAULT_DATA = Path(os.environ.get("GFG_CIFAR100_ROOT", "runtime/datasets/cifar100"))
DEFAULT_OUTPUT = Path(os.environ.get("GFG_MOE_OUTPUT", "D:/codex/_runs/gfg_moe_cifar_generalization_v1"))


def _seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _state_to(state: dict[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in state.items()}


def _memory_to(
    memory: dict[str, dict[str, Tensor]], device: torch.device | str
) -> dict[str, dict[str, Tensor]]:
    return {
        name: {key: value.to(device) for key, value in fields.items()}
        for name, fields in memory.items()
    }


def _optimizer_tensors(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        for field, value in optimizer.state.get(parameter, {}).items():
            result[f"{name}:{field}"] = torch.as_tensor(value).detach().clone()
    return result


def _max_parameter_error(
    left: dict[str, Tensor], right: dict[str, Tensor], model: nn.Module
) -> float:
    names = {name for name, _ in model.named_parameters()}
    return max(float((left[name] - right[name]).abs().max()) for name in names)


def _evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    count = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            assert isinstance(logits, Tensor)
            loss_sum += float(criterion(logits, labels))
            correct += int(logits.argmax(dim=1).eq(labels).sum())
            count += labels.numel()
    return {"count": count, "accuracy": correct / count, "mean_loss": loss_sum / count}


def _manifest(output: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema": "gfg-moe-cifar-generalization-manifest-v1",
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files.values()),
    }


def _integrity_pass(summary: dict[str, Any]) -> bool:
    integrity = summary["integrity"]
    return bool(
        integrity["gfg_validation"]["status"] == "PASS"
        and integrity["adamw_formula_max_abs_error"] <= 1e-6
        and integrity["alpha0_max_abs_error"] <= 1e-5
        and integrity["alpha1_native_max_abs_error"] <= 1e-5
        and integrity["support_repeat_max_abs_error"] <= 1e-6
        and integrity["persistent_model_state_frozen"]
        and integrity["persistent_optimizer_state_frozen"]
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
    training = CONTRACT["training"]
    evaluation = CONTRACT["evaluation"]
    epochs = 1 if smoke else int(training["epochs"])
    batch_size = 64 if smoke else int(training["batch_size"])
    anchor_count = 100 if smoke else int(evaluation["anchor_count"])
    target_count = 16 if smoke else int(evaluation["response_target_count_per_occurrence"])
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
    model = CifarVisionMoE(
        dimension=int(CONTRACT["model"]["embedding_dimension"]),
        expert_hidden_dimension=int(CONTRACT["model"]["expert_hidden_dimension"]),
    ).to(device)
    initial_state = clone_state(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]))
    graph = CompactGFG(f"moe-cifar100-adamw-seed-{seed}")
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
            adam_pre = clone_adam_memory(model, optimizer) if capture else None
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                forward = model(images, return_details=True)
                assert isinstance(forward, tuple)
                logits, details = forward
                mean_probability = details["router_probabilities"].mean(dim=(0, 1))
                mean_assignment = details["routing_weights"].gt(0).float().mean(dim=(0, 1))
                balance = 4.0 * torch.sum(mean_probability * mean_assignment)
                loss = criterion(logits, labels) + float(training["load_balance_weight"]) * balance
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
            assert pre is not None and adam_pre is not None and gradients is not None
            post = clone_state(model)
            delta = state_delta(pre, post)
            formula = adamw_delta(
                pre,
                gradients,
                adam_pre,
                learning_rate=learning_rate,
                betas=tuple(float(value) for value in training["betas"]),
                epsilon=float(training["epsilon"]),
                weight_decay=float(training["weight_decay"]),
            )
            formula_error = _max_parameter_error(delta, formula, model)
            analysis = analyze_update(
                model=model,
                pre=pre,
                post=post,
                adam_pre=adam_pre,
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
                "load_balance_loss": float(balance.detach()),
                "training_batch_ids": [int(value) for value in sample_ids.tolist()],
                "adamw_formula_max_abs_error": formula_error,
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
                prior_memory = _memory_to(previous["adam_pre"], device)
                exchanged_delta = adamw_delta(
                    pre,
                    gradients,
                    prior_memory,
                    learning_rate=learning_rate,
                    betas=tuple(float(value) for value in training["betas"]),
                    epsilon=float(training["epsilon"]),
                    weight_decay=float(training["weight_decay"]),
                )
                exchange["adam_receiving_state_exchange"] = memory_exchange_response(
                    model=model,
                    state=pre,
                    native_delta=delta,
                    exchanged_delta=exchanged_delta,
                    images=anchor_images[:target_count],
                    labels=anchor_labels[:target_count],
                )
                exchange["from_event"] = previous["event_index"]
                exchange["to_event"] = event["event_index"]
                exchanges.append(exchange)
            previous = {
                "event_index": event["event_index"],
                "pre": _state_to(pre, "cpu"),
                "delta": _state_to(delta, "cpu"),
                "adam_pre": _memory_to(adam_pre, "cpu"),
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
                        "formula_error": formula_error,
                        "alpha1_error": analysis["alpha1_native_logit_max_abs_error"],
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

    trained_state = clone_state(model)
    model_hash_before = state_sha256(trained_state)
    optimizer_hash_before = state_sha256(_optimizer_tensors(model, optimizer))
    inference = inference_projection_tests(
        model=model,
        trained=trained_state,
        initial=initial_state,
        images=anchor_images[: min(128, anchor_count)],
        labels=anchor_labels[: min(128, anchor_count)],
        thresholds=CONTRACT["frozen_tests"],
    )
    model_hash_after = state_sha256(clone_state(model))
    optimizer_hash_after = state_sha256(_optimizer_tensors(model, optimizer))
    inference["persistent_state"] = {
        "model_before": model_hash_before,
        "model_after": model_hash_after,
        "optimizer_before": optimizer_hash_before,
        "optimizer_after": optimizer_hash_after,
    }
    inference["tests"]["persistent_state_frozen"] = (
        model_hash_before == model_hash_after and optimizer_hash_before == optimizer_hash_after
    )
    inference["pass"] = all(inference["tests"].values())
    add_inference_event(graph, seed, model_hash_before, inference)
    graph_document = graph.document()
    integrity = {
        "adamw_formula_max_abs_error": max(event["adamw_formula_max_abs_error"] for event in events),
        "alpha0_max_abs_error": max(event["analysis"]["alpha0_logit_max_abs_error"] for event in events),
        "alpha1_native_max_abs_error": max(event["analysis"]["alpha1_native_logit_max_abs_error"] for event in events),
        "support_repeat_max_abs_error": max(event["analysis"]["support"]["repeat_max_abs_error"] for event in events),
        "persistent_model_state_frozen": model_hash_before == model_hash_after,
        "persistent_optimizer_state_frozen": optimizer_hash_before == optimizer_hash_after,
        "gfg_validation": graph_document["validation"],
    }
    final_cpu = _state_to(trained_state, "cpu")
    initial_cpu = _state_to(initial_state, "cpu")
    torch.save(
        {
            "seed": seed,
            "initial_model_state": initial_cpu,
            "model_state": final_cpu,
            "optimizer_state": optimizer.state_dict(),
            "contract_sha256": sha256_file(PACKAGE / "MODEL_CONTRACT.json"),
        },
        output_root / "FINAL_CHECKPOINT.pt",
    )
    run_summary = {
        "schema": "gfg-moe-cifar-generalization-run-v1",
        "seed": seed,
        "smoke": smoke,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "final_test": training_metrics[-1]["test"],
        "event_count": len(events),
        "exchange_count": len(exchanges),
        "integrity": integrity,
        "initial_state_sha256": state_sha256(initial_cpu),
        "final_state_sha256": state_sha256(final_cpu),
        "inference_pass": inference["pass"],
    }
    shutil.copy2(PACKAGE / "PROTOCOL_FREEZE.md", output_root / "PROTOCOL_FREEZE.md")
    shutil.copy2(PACKAGE / "MODEL_CONTRACT.json", output_root / "MODEL_CONTRACT.json")
    write_json(output_root / "EVENTS.json", events)
    write_json(output_root / "EXCHANGES.json", exchanges)
    write_json(output_root / "TRAINING_METRICS.json", training_metrics)
    write_json(output_root / "INFERENCE_RESULTS.json", inference)
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
            *[int(value) for value in CONTRACT["system"]["confirmation_seeds"]],
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
