from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from experiments.gfg_rl_selective_positive_feedback_support_concentration_v1.runtime import (  # noqa: F401
    COMPONENT_COUNT,
    CUE_COUNT,
    SKILL_COUNT,
    STAGE_COUNT,
    MultiSkillGRUPolicy,
    canonical_bytes,
    clone_training_state,
    combined_state_sha256,
    complete_grid,
    evaluate_policy,
    make_optimizer,
    model_sha256,
    object_sha256,
    optimizer_sha256,
    positive_feedback_update,
    restore_state,
    seed_everything,
    state_payload,
    supervised_pretrain_step,
    support_profile,
    targets,
    tensor_dict_sha256,
)


def deterministic_allocated_feedback_batch(
    *,
    seed: int,
    update: int,
    allocation: list[int],
    device: torch.device,
) -> dict[str, Tensor]:
    """Create a condition-matched batch with a common occurrence/randomness ledger."""
    if len(allocation) != SKILL_COUNT or min(allocation) < 0:
        raise ValueError("INVALID_SKILL_ALLOCATION")
    batch_size = sum(int(value) for value in allocation)
    if batch_size <= 0:
        raise ValueError("EMPTY_SKILL_ALLOCATION")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + int(update) * 97 + 43)
    cue_ids = torch.randint(0, CUE_COUNT, (batch_size,), generator=generator)
    uniforms = torch.rand((batch_size, STAGE_COUNT), generator=generator)
    rows = [
        torch.full((int(count),), skill, dtype=torch.long)
        for skill, count in enumerate(allocation)
        if int(count) > 0
    ]
    skill_ids = torch.cat(rows)
    permutation = torch.randperm(batch_size, generator=generator)
    return {
        "skill_ids": skill_ids[permutation].to(device),
        "cue_ids": cue_ids.to(device),
        "uniforms": uniforms.to(device),
    }


def compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_chain_accuracy": evaluation["macro_chain_accuracy"],
        "macro_stage_accuracy": evaluation["macro_stage_accuracy"],
        "per_skill": evaluation["per_skill"],
    }

