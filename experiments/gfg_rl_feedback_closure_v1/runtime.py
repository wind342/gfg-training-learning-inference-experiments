from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import random
from typing import Any

import torch
from torch import nn


CUE_COUNT = 16
CONDITIONS = ("A", "B", "C")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _tensor_digest(hasher: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().to(device="cpu").contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(canonical_bytes(list(value.shape)))
    hasher.update(value.numpy().tobytes())


def model_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _tensor_digest(hasher, name, value)
    return hasher.hexdigest()


def optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    hasher = hashlib.sha256()
    state = optimizer.state_dict()
    hasher.update(canonical_bytes(state["param_groups"]))
    for key, values in sorted(state["state"].items(), key=lambda item: int(item[0])):
        hasher.update(str(key).encode("ascii"))
        for name, value in sorted(values.items()):
            if torch.is_tensor(value):
                _tensor_digest(hasher, name, value)
            else:
                hasher.update(canonical_bytes({name: value}))
    return hasher.hexdigest()


def tensor_dict_sha256(values: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(values.items()):
        _tensor_digest(hasher, name, value)
    return hasher.hexdigest()


def combined_state_sha256(model: nn.Module, optimizer: torch.optim.Optimizer) -> str:
    return object_sha256({"model": model_sha256(model), "optimizer": optimizer_sha256(optimizer)})


def cue_bits(cue_ids: torch.Tensor) -> torch.Tensor:
    return torch.stack([((cue_ids >> shift) & 1).float() for shift in range(4)], dim=1)


def targets(cue_ids: torch.Tensor, phase: str) -> torch.Tensor:
    bits = cue_bits(cue_ids).long()
    result = torch.stack((bits[:, 0] ^ bits[:, 2], bits[:, 1] ^ bits[:, 3]), dim=1)
    if phase == "B":
        result = 1 - result
    elif phase != "A":
        raise ValueError(f"unknown phase {phase}")
    return result


class DelayedGRUPolicy(nn.Module):
    """Two-decision recurrent policy.  It deliberately contains no attention."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(8, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 2)

    def _input(self, cues: torch.Tensor, stage: int, previous_action: torch.Tensor | None) -> torch.Tensor:
        batch = cues.shape[0]
        stage_bits = torch.zeros((batch, 2), dtype=cues.dtype, device=cues.device)
        stage_bits[:, stage] = 1.0
        previous = torch.zeros((batch, 2), dtype=cues.dtype, device=cues.device)
        if previous_action is not None:
            previous.scatter_(1, previous_action[:, None], 1.0)
        return torch.cat((cues, stage_bits, previous), dim=1)

    def forward(self, cues: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros((cues.shape[0], self.hidden_size), dtype=cues.dtype, device=cues.device)
        hidden = self.cell(torch.tanh(self.input_projection(self._input(cues, 0, None))), hidden)
        logits_1 = self.actor(hidden)
        previous = torch.argmax(logits_1, dim=1)
        hidden = self.cell(torch.tanh(self.input_projection(self._input(cues, 1, previous))), hidden)
        logits_2 = self.actor(hidden)
        return logits_1, logits_2

    def sampled_forward(
        self, cues: torch.Tensor, uniforms: torch.Tensor, exploration_epsilon: float,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        hidden = torch.zeros((cues.shape[0], self.hidden_size), dtype=cues.dtype, device=cues.device)
        hidden = self.cell(torch.tanh(self.input_projection(self._input(cues, 0, None))), hidden)
        logits_1 = self.actor(hidden)
        probabilities_1 = torch.softmax(logits_1, dim=1)
        behavior_1 = (1.0 - exploration_epsilon) * probabilities_1 + exploration_epsilon / 2.0
        action_1 = (uniforms[:, 0] >= behavior_1[:, 0].detach()).long()
        hidden = self.cell(torch.tanh(self.input_projection(self._input(cues, 1, action_1))), hidden)
        logits_2 = self.actor(hidden)
        probabilities_2 = torch.softmax(logits_2, dim=1)
        behavior_2 = (1.0 - exploration_epsilon) * probabilities_2 + exploration_epsilon / 2.0
        action_2 = (uniforms[:, 1] >= behavior_2[:, 0].detach()).long()
        return [logits_1, logits_2], torch.stack((action_1, action_2), dim=1)


def make_optimizer(model: nn.Module, contract: dict[str, Any]) -> torch.optim.AdamW:
    cfg = contract["model"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=(float(cfg["beta1"]), float(cfg["beta2"])),
    )


def clone_training_state(
    model: DelayedGRUPolicy,
    optimizer: torch.optim.AdamW,
    contract: dict[str, Any],
    device: torch.device,
) -> tuple[DelayedGRUPolicy, torch.optim.AdamW]:
    clone = DelayedGRUPolicy(int(contract["model"]["hidden_size"])).to(device)
    clone.load_state_dict(deepcopy(model.state_dict()))
    clone_optimizer = make_optimizer(clone, contract)
    clone_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    return clone, clone_optimizer


@torch.no_grad()
def evaluate_policy(model: DelayedGRUPolicy, phase: str, device: torch.device) -> dict[str, Any]:
    cue_ids = torch.arange(CUE_COUNT, device=device)
    logits_1, logits_2 = model(cue_bits(cue_ids))
    actions = torch.stack((logits_1.argmax(dim=1), logits_2.argmax(dim=1)), dim=1)
    expected = targets(cue_ids, phase)
    correct = actions.eq(expected)
    old_correct = actions.eq(targets(cue_ids, "A"))
    return {
        "phase": phase,
        "stage_1_accuracy": float(correct[:, 0].float().mean().item()),
        "stage_2_accuracy": float(correct[:, 1].float().mean().item()),
        "chain_accuracy": float(correct.all(dim=1).float().mean().item()),
        "old_rule_chain_accuracy": float(old_correct.all(dim=1).float().mean().item()),
        "actions": actions.cpu().tolist(),
        "targets": expected.cpu().tolist(),
        "logits": torch.stack((logits_1, logits_2), dim=1).cpu().tolist(),
    }


def deterministic_batch(seed: int, update: int, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1_000_003 + update * 97 + 17)
    cue_ids = torch.randint(0, CUE_COUNT, (batch_size,), generator=generator)
    uniforms = torch.rand((batch_size, 2), generator=generator)
    permutation_1 = torch.randperm(batch_size, generator=generator)
    permutation_2 = torch.randperm(batch_size, generator=generator)
    return {
        "cue_ids": cue_ids.to(device),
        "uniforms": uniforms.to(device),
        "permutation_1": permutation_1.to(device),
        "permutation_2": permutation_2.to(device),
    }


def train_update(
    *,
    model: DelayedGRUPolicy,
    optimizer: torch.optim.AdamW,
    batch: dict[str, torch.Tensor],
    phase: str,
    condition: str,
    entropy_coefficient: float,
    exploration_epsilon: float = 0.0,
    include_episode_ledger: bool = True,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    cue_ids = batch["cue_ids"]
    cues = cue_bits(cue_ids)
    pre_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    pre_state = combined_state_sha256(model, optimizer)
    pre_eval = evaluate_policy(model, "B", cue_ids.device)
    logits, actions = model.sampled_forward(cues, batch["uniforms"], exploration_epsilon)
    expected = targets(cue_ids, phase)
    physical_rewards = torch.where(actions.eq(expected), 1.0, -1.0)
    chosen_log_prob = torch.stack([
        torch.log_softmax(logits[stage], dim=1).gather(1, actions[:, stage, None]).squeeze(1)
        for stage in range(2)
    ], dim=1)
    chosen_probability = torch.exp(chosen_log_prob)
    chosen_behavior_probability = (1.0 - exploration_epsilon) * chosen_probability + exploration_epsilon / 2.0
    importance_weight = (chosen_probability / chosen_behavior_probability).detach()
    entropy = torch.stack([
        -(torch.softmax(value, dim=1) * torch.log_softmax(value, dim=1)).sum(dim=1)
        for value in logits
    ], dim=1)
    batch_size = cue_ids.shape[0]
    identity = torch.arange(batch_size, device=cue_ids.device)
    if condition == "A":
        assigned_rewards = physical_rewards
        source_indices = torch.stack((identity, identity), dim=1)
        credit_targets = torch.tensor([0, 1], device=cue_ids.device).expand(batch_size, 2)
        credited_log_prob = chosen_log_prob
    elif condition == "B":
        assigned_rewards = torch.stack((
            physical_rewards[batch["permutation_1"], 0],
            physical_rewards[batch["permutation_2"], 1],
        ), dim=1)
        source_indices = torch.stack((batch["permutation_1"], batch["permutation_2"]), dim=1)
        credit_targets = torch.tensor([0, 1], device=cue_ids.device).expand(batch_size, 2)
        credited_log_prob = chosen_log_prob
    else:
        assigned_rewards = physical_rewards
        source_indices = torch.stack((identity, identity), dim=1)
        credit_targets = torch.tensor([1, 0], device=cue_ids.device).expand(batch_size, 2)
        credited_log_prob = chosen_log_prob[:, [1, 0]]
    credited_importance = importance_weight if condition != "C" else importance_weight[:, [1, 0]]
    policy_loss = -(assigned_rewards * credited_importance * credited_log_prob).sum(dim=1).mean()
    entropy_bonus = entropy.sum(dim=1).mean()
    loss = policy_loss - entropy_coefficient * entropy_bonus
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    gradient_hash = tensor_dict_sha256(gradients)
    gradient_norm = float(torch.sqrt(sum((value.float() ** 2).sum() for value in gradients.values())).item())
    optimizer.step()
    updates = {
        name: model.state_dict()[name].detach() - pre_model[name]
        for name in pre_model
    }
    update_hash = tensor_dict_sha256(updates)
    update_norm = float(torch.sqrt(sum((value.float() ** 2).sum() for value in updates.values())).item())
    post_state = combined_state_sha256(model, optimizer)
    post_eval = evaluate_policy(model, "B", cue_ids.device)
    result: dict[str, Any] = {
        "condition": condition,
        "phase": phase,
        "pre_state_sha256": pre_state,
        "post_state_sha256": post_state,
        "cue_batch_sha256": object_sha256(cue_ids.cpu().tolist()),
        "uniform_batch_sha256": object_sha256(batch["uniforms"].cpu().tolist()),
        "action_ledger_sha256": object_sha256(actions.cpu().tolist()),
        "physical_consequence_sha256": object_sha256(physical_rewards.cpu().tolist()),
        "assigned_consequence_sha256": object_sha256(assigned_rewards.cpu().tolist()),
        "binding_source_sha256": object_sha256(source_indices.cpu().tolist()),
        "credit_target_sha256": object_sha256(credit_targets.cpu().tolist()),
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "entropy": float(entropy_bonus.detach().item()),
        "behavior_exploration_epsilon": exploration_epsilon,
        "gradient_sha256": gradient_hash,
        "gradient_norm": gradient_norm,
        "actual_update_sha256": update_hash,
        "actual_update_norm": update_norm,
        "pre_chain_accuracy": pre_eval["chain_accuracy"],
        "post_chain_accuracy": post_eval["chain_accuracy"],
        "pre_logits_sha256": object_sha256(pre_eval["logits"]),
        "post_logits_sha256": object_sha256(post_eval["logits"]),
    }
    if include_episode_ledger:
        cue_values = cue_ids.cpu().tolist()
        action_values = actions.cpu().tolist()
        target_values = expected.cpu().tolist()
        reward_values = physical_rewards.cpu().tolist()
        assigned_values = assigned_rewards.cpu().tolist()
        source_values = source_indices.cpu().tolist()
        target_stages = credit_targets.cpu().tolist()
        result["episodes"] = [
            {
                "episode_index": index,
                "cue_id": cue_values[index],
                "actions": action_values[index],
                "targets": target_values[index],
                "physical_consequences": reward_values[index],
                "assigned_consequences": assigned_values[index],
                "assigned_source_episode_indices": source_values[index],
                "credited_to_action_stages": target_stages[index],
            }
            for index in range(batch_size)
        ]
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


__all__ = [
    "CONDITIONS",
    "DelayedGRUPolicy",
    "clone_training_state",
    "combined_state_sha256",
    "deterministic_batch",
    "evaluate_policy",
    "make_optimizer",
    "model_sha256",
    "object_sha256",
    "optimizer_sha256",
    "seed_everything",
    "targets",
    "train_update",
]
