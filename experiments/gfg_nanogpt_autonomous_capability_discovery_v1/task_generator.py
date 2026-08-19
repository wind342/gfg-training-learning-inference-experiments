from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Any

import torch

from .common import payload_sha256


@dataclass(frozen=True)
class TaskInstanceSpec:
    instance_id: str
    modulus: int
    train_fraction: float
    token_permutation_seed: int
    split_seed: int


@dataclass(frozen=True)
class TokenizedTask:
    instance_id: str
    modulus: int
    vocab_size: int
    operator_token: int
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    private_generation_commitment: str
    participant_task_commitment: str


def _validate_prime(value: int) -> None:
    if value < 3:
        raise ValueError("TASK_SCALE_TOO_SMALL")
    for divisor in range(2, int(value**0.5) + 1):
        if value % divisor == 0:
            raise ValueError("TASK_SCALE_NOT_PRIME")


def build_task(spec: TaskInstanceSpec) -> TokenizedTask:
    """Build a randomly relabelled finite cyclic-operation task.

    The returned participant representation contains only relabelled token
    triples, targets, split membership and opaque sample identities. The
    operation rule and permutation remain private generation material.
    """

    _validate_prime(spec.modulus)
    if not 0.1 <= spec.train_fraction <= 0.9:
        raise ValueError("INVALID_TRAIN_FRACTION")

    permutation_rng = random.Random(spec.token_permutation_seed)
    token_map = list(range(spec.modulus))
    permutation_rng.shuffle(token_map)
    operator_token = spec.modulus

    private_rows: list[dict[str, int]] = []
    participant_rows: list[dict[str, Any]] = []
    for left in range(spec.modulus):
        for right in range(spec.modulus):
            result = (left + right) % spec.modulus
            source_id = "sample-" + hashlib.sha256(
                (
                    f"{spec.instance_id}:{spec.token_permutation_seed}:"
                    f"{spec.split_seed}:{left}:{right}"
                ).encode("utf-8")
            ).hexdigest()[:24]
            token_input = [
                token_map[left],
                operator_token,
                token_map[right],
            ]
            token_target = token_map[result]
            private_rows.append(
                {
                    "left": left,
                    "right": right,
                    "result": result,
                    "sample_id": source_id,
                }
            )
            participant_rows.append(
                {
                    "input_tokens": token_input,
                    "sample_id": source_id,
                    "target_token": token_target,
                }
            )

    split_rng = random.Random(spec.split_seed)
    order = list(range(len(participant_rows)))
    split_rng.shuffle(order)
    train_count = max(
        1,
        min(
            len(order) - 1,
            round(len(order) * spec.train_fraction),
        ),
    )
    train_indices = set(order[:train_count])

    train_rows = [
        row for index, row in enumerate(participant_rows) if index in train_indices
    ]
    validation_rows = [
        row for index, row in enumerate(participant_rows) if index not in train_indices
    ]

    def tensors(
        rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        inputs = torch.tensor(
            [row["input_tokens"] for row in rows],
            dtype=torch.long,
        )
        targets = torch.full_like(inputs, -1)
        targets[:, -1] = torch.tensor(
            [row["target_token"] for row in rows],
            dtype=torch.long,
        )
        ids = tuple(str(row["sample_id"]) for row in rows)
        return inputs, targets, ids

    train_inputs, train_targets, train_ids = tensors(train_rows)
    validation_inputs, validation_targets, validation_ids = tensors(
        validation_rows
    )
    private_commitment = payload_sha256(
        {
            "spec": asdict(spec),
            "token_map": token_map,
            "operation_rows": private_rows,
            "train_indices": sorted(train_indices),
        }
    )
    participant_commitment = payload_sha256(
        {
            "instance_id": spec.instance_id,
            "operator_token": operator_token,
            "train": train_rows,
            "validation": validation_rows,
            "vocab_size": spec.modulus + 1,
        }
    )
    return TokenizedTask(
        instance_id=spec.instance_id,
        modulus=spec.modulus,
        vocab_size=spec.modulus + 1,
        operator_token=operator_token,
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        private_generation_commitment=private_commitment,
        participant_task_commitment=participant_commitment,
    )


def participant_description(task: TokenizedTask) -> dict[str, Any]:
    return {
        "input_width": int(task.train_inputs.shape[1]),
        "instance_id": task.instance_id,
        "operator_token": task.operator_token,
        "participant_task_commitment": task.participant_task_commitment,
        "schema": "opaque-token-operation-task-v1",
        "train_sample_count": int(task.train_inputs.shape[0]),
        "validation_sample_count": int(task.validation_inputs.shape[0]),
        "vocab_size": task.vocab_size,
    }
