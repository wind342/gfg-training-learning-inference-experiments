from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .common import file_sha256, payload_sha256, write_json
from .nanogpt_adapter import TrainingCheckpoint, checkpoint_commitment


def save_checkpoint(
    path: Path, checkpoint: TrainingCheckpoint
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data_generator_state": checkpoint.data_generator_state,
            "model_state": checkpoint.model_state,
            "numpy_rng_state": checkpoint.numpy_rng_state,
            "optimizer_state": checkpoint.optimizer_state,
            "python_rng_state": checkpoint.python_rng_state,
            "step": checkpoint.step,
            "torch_cpu_rng_state": checkpoint.torch_cpu_rng_state,
            "torch_cuda_rng_state": checkpoint.torch_cuda_rng_state,
        },
        path,
    )
    receipt = {
        "checkpoint_commitment": checkpoint_commitment(checkpoint),
        "checkpoint_file_sha256": file_sha256(path),
        "schema": "exact-training-checkpoint-v1",
        "step": checkpoint.step,
    }
    receipt["receipt_sha256"] = payload_sha256(receipt)
    write_json(path.with_suffix(".receipt.json"), receipt)
    return receipt


def load_checkpoint(path: Path) -> TrainingCheckpoint:
    value = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = TrainingCheckpoint(
        step=int(value["step"]),
        model_state=value["model_state"],
        optimizer_state=value["optimizer_state"],
        torch_cpu_rng_state=value["torch_cpu_rng_state"],
        torch_cuda_rng_state=value["torch_cuda_rng_state"],
        numpy_rng_state=value["numpy_rng_state"],
        python_rng_state=value["python_rng_state"],
        data_generator_state=value["data_generator_state"],
    )
    receipt_path = path.with_suffix(".receipt.json")
    if receipt_path.is_file():
        import json

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["checkpoint_commitment"] != checkpoint_commitment(
            checkpoint
        ):
            raise RuntimeError("CHECKPOINT_COMMITMENT_MISMATCH")
        if receipt["checkpoint_file_sha256"] != file_sha256(path):
            raise RuntimeError("CHECKPOINT_FILE_HASH_MISMATCH")
    return checkpoint


def fork_audit(
    checkpoint: TrainingCheckpoint,
    baseline: TrainingCheckpoint,
    intervention: TrainingCheckpoint,
) -> dict[str, Any]:
    parent = checkpoint_commitment(checkpoint)
    baseline_commitment = checkpoint_commitment(baseline)
    intervention_commitment = checkpoint_commitment(intervention)
    return {
        "baseline_initial_checkpoint": baseline_commitment,
        "identical": (
            parent == baseline_commitment == intervention_commitment
        ),
        "intervention_initial_checkpoint": intervention_commitment,
        "parent_checkpoint": parent,
        "schema": "exact-checkpoint-fork-audit-v1",
    }
