from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import random
from typing import Any, Iterable

import torch
from torch import Tensor, nn


SKILL_COUNT = 4
CUE_COUNT = 16
STAGE_COUNT = 2
COMPONENT_COUNT = 4
CONDITIONS = ("selective", "balanced", "frozen")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _tensor_digest(hasher: Any, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(canonical_bytes(list(value.shape)))
    hasher.update(value.numpy().tobytes())


def tensor_dict_sha256(values: dict[str, Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(values.items()):
        _tensor_digest(hasher, name, value)
    return hasher.hexdigest()


def model_sha256(model: nn.Module) -> str:
    return tensor_dict_sha256({name: value for name, value in model.state_dict().items()})


def optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    hasher = hashlib.sha256(canonical_bytes(state["param_groups"]))
    for key, values in sorted(state["state"].items(), key=lambda row: int(row[0])):
        hasher.update(str(key).encode("ascii"))
        for name, value in sorted(values.items()):
            if torch.is_tensor(value):
                _tensor_digest(hasher, name, value)
            else:
                hasher.update(canonical_bytes({name: value}))
    return hasher.hexdigest()


def combined_state_sha256(model: nn.Module, optimizer: torch.optim.Optimizer) -> str:
    return object_sha256({"model": model_sha256(model), "optimizer": optimizer_sha256(optimizer)})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cue_bits(cue_ids: Tensor) -> Tensor:
    return torch.stack([((cue_ids >> shift) & 1).float() for shift in range(4)], dim=1)


def skill_one_hot(skill_ids: Tensor) -> Tensor:
    return torch.nn.functional.one_hot(skill_ids, num_classes=SKILL_COUNT).float()


def targets(skill_ids: Tensor, cue_ids: Tensor) -> Tensor:
    bits = cue_bits(cue_ids).long()
    result = torch.empty((len(cue_ids), STAGE_COUNT), dtype=torch.long, device=cue_ids.device)
    masks = [
        (bits[:, 0] ^ bits[:, 2], bits[:, 1] ^ bits[:, 3]),
        (bits[:, 0] ^ bits[:, 1], bits[:, 2] ^ bits[:, 3]),
        (bits[:, 0] ^ bits[:, 3], bits[:, 1] ^ bits[:, 2]),
        (bits[:, 0] ^ bits[:, 1] ^ bits[:, 2], bits[:, 1] ^ bits[:, 2] ^ bits[:, 3]),
    ]
    for skill in range(SKILL_COUNT):
        selected = skill_ids.eq(skill)
        result[selected, 0] = masks[skill][0][selected]
        result[selected, 1] = masks[skill][1][selected]
    return result


def component_mask(mask: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if hidden_size % COMPONENT_COUNT:
        raise ValueError("HIDDEN_SIZE_NOT_DIVISIBLE_BY_COMPONENT_COUNT")
    width = hidden_size // COMPONENT_COUNT
    values = torch.zeros(hidden_size, device=device, dtype=dtype)
    for component in range(COMPONENT_COUNT):
        if mask & (1 << component):
            values[component * width : (component + 1) * width] = 1.0
    return values


class MultiSkillGRUPolicy(nn.Module):
    """Attention-free shared policy with four causally gateable hidden components."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size % COMPONENT_COUNT:
            raise ValueError("HIDDEN_SIZE_NOT_DIVISIBLE_BY_COMPONENT_COUNT")
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(12, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 2)

    def _input(
        self,
        skills: Tensor,
        cues: Tensor,
        stage: int,
        previous_action: Tensor | None,
    ) -> Tensor:
        batch = cues.shape[0]
        stage_bits = torch.zeros((batch, 2), dtype=cues.dtype, device=cues.device)
        stage_bits[:, stage] = 1.0
        previous = torch.zeros((batch, 2), dtype=cues.dtype, device=cues.device)
        if previous_action is not None:
            previous.scatter_(1, previous_action[:, None], 1.0)
        return torch.cat((cues, skills, stage_bits, previous), dim=1)

    def forward(
        self,
        skill_ids: Tensor,
        cue_ids: Tensor,
        *,
        active_components: int = 15,
        uniforms: Tensor | None = None,
    ) -> tuple[list[Tensor], Tensor]:
        cues = cue_bits(cue_ids)
        skills = skill_one_hot(skill_ids)
        hidden = torch.zeros((len(cue_ids), self.hidden_size), dtype=cues.dtype, device=cues.device)
        mask = component_mask(active_components, self.hidden_size, cues.device, cues.dtype)
        logits: list[Tensor] = []
        actions: list[Tensor] = []
        previous: Tensor | None = None
        for stage in range(STAGE_COUNT):
            projected = torch.tanh(self.input_projection(self._input(skills, cues, stage, previous)))
            hidden = self.cell(projected, hidden) * mask
            stage_logits = self.actor(hidden)
            if uniforms is None:
                action = stage_logits.argmax(dim=1)
            else:
                probabilities = torch.softmax(stage_logits, dim=1)
                action = (uniforms[:, stage] >= probabilities[:, 0].detach()).long()
            logits.append(stage_logits)
            actions.append(action)
            previous = action
        return logits, torch.stack(actions, dim=1)


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def clone_training_state(
    model: MultiSkillGRUPolicy,
    optimizer: torch.optim.AdamW,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[MultiSkillGRUPolicy, torch.optim.AdamW]:
    clone = MultiSkillGRUPolicy(model.hidden_size).to(device)
    clone.load_state_dict(deepcopy(model.state_dict()))
    clone_optimizer = make_optimizer(clone, learning_rate, weight_decay)
    clone_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    return clone, clone_optimizer


def complete_grid(device: torch.device) -> tuple[Tensor, Tensor]:
    skill_ids = torch.arange(SKILL_COUNT, device=device).repeat_interleave(CUE_COUNT)
    cue_ids = torch.arange(CUE_COUNT, device=device).repeat(SKILL_COUNT)
    return skill_ids, cue_ids


def target_margins(logits: list[Tensor], expected: Tensor) -> Tensor:
    rows = []
    for stage, value in enumerate(logits):
        correct = value.gather(1, expected[:, stage, None]).squeeze(1)
        competitor = value.gather(1, (1 - expected[:, stage])[:, None]).squeeze(1)
        rows.append(correct - competitor)
    return torch.stack(rows, dim=1)


@torch.no_grad()
def evaluate_policy(model: MultiSkillGRUPolicy, device: torch.device) -> dict[str, Any]:
    skill_ids, cue_ids = complete_grid(device)
    logits, actions = model(skill_ids, cue_ids)
    expected = targets(skill_ids, cue_ids)
    correct = actions.eq(expected)
    margins = target_margins(logits, expected)
    per_skill = []
    for skill in range(SKILL_COUNT):
        selected = skill_ids.eq(skill)
        skill_margins = margins[selected]
        skill_correct = correct[selected]
        per_skill.append({
            "skill": skill,
            "stage_1_accuracy": float(skill_correct[:, 0].float().mean()),
            "stage_2_accuracy": float(skill_correct[:, 1].float().mean()),
            "chain_accuracy": float(skill_correct.all(dim=1).float().mean()),
            "mean_margin": float(skill_margins.mean()),
            "minimum_margin": float(skill_margins.min()),
        })
    return {
        "per_skill": per_skill,
        "macro_chain_accuracy": float(correct.all(dim=1).float().mean()),
        "macro_stage_accuracy": float(correct.float().mean()),
        "actions": actions.cpu().tolist(),
        "targets": expected.cpu().tolist(),
        "margins": margins.cpu().tolist(),
    }


def exact_shapley(coalition_values: Tensor) -> Tensor:
    if coalition_values.shape[0] != 2 ** COMPONENT_COUNT:
        raise ValueError("EXPECTED_ALL_COMPONENT_COALITIONS")
    result = torch.zeros(
        (coalition_values.shape[1], COMPONENT_COUNT),
        dtype=coalition_values.dtype,
        device=coalition_values.device,
    )
    factorial = math.factorial
    for component in range(COMPONENT_COUNT):
        for mask in range(2 ** COMPONENT_COUNT):
            if mask & (1 << component):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(COMPONENT_COUNT - size - 1) / factorial(COMPONENT_COUNT)
            result[:, component] += weight * (
                coalition_values[mask | (1 << component)] - coalition_values[mask]
            )
    return result


def _hhi(values: Tensor) -> float:
    total = values.sum()
    if float(total) <= 1e-12:
        return 0.0
    shares = values / total
    return float((shares ** 2).sum())


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


@torch.no_grad()
def support_profile(model: MultiSkillGRUPolicy, device: torch.device) -> dict[str, Any]:
    skill_ids, cue_ids = complete_grid(device)
    expected = targets(skill_ids, cue_ids)
    coalition_rows = []
    for mask in range(2 ** COMPONENT_COUNT):
        logits, _ = model(skill_ids, cue_ids, active_components=mask)
        coalition_rows.append(target_margins(logits, expected).reshape(-1))
    coalition_values = torch.stack(coalition_rows, dim=0)
    shapley_rows = exact_shapley(coalition_values)
    shapley = shapley_rows.reshape(SKILL_COUNT, CUE_COUNT, STAGE_COUNT, COMPONENT_COUNT)
    positive = shapley.clamp_min(0).mean(dim=(1, 2))
    absolute = shapley.abs().mean(dim=(1, 2))
    signed = shapley.mean(dim=(1, 2))
    task_mass = positive.sum(dim=1)
    task_shares = task_mass / task_mass.sum().clamp_min(1e-12)
    within_task_hhi = [_hhi(positive[skill]) for skill in range(SKILL_COUNT)]
    overlaps = {
        f"0:{skill}": _cosine(positive[0], positive[skill])
        for skill in range(1, SKILL_COUNT)
    }
    primary = positive.argmax(dim=1)
    component_primary_task = positive.argmax(dim=0)
    full = coalition_values[-1].reshape(SKILL_COUNT, CUE_COUNT, STAGE_COUNT)
    empty = coalition_values[0].reshape(SKILL_COUNT, CUE_COUNT, STAGE_COUNT)
    return {
        "positive_support": positive.cpu().tolist(),
        "absolute_support": absolute.cpu().tolist(),
        "signed_support": signed.cpu().tolist(),
        "task_support_mass": task_mass.cpu().tolist(),
        "task_support_shares": task_shares.cpu().tolist(),
        "cross_task_hhi": _hhi(task_mass),
        "within_task_hhi": within_task_hhi,
        "primary_component_by_task": primary.cpu().tolist(),
        "primary_task_by_component": component_primary_task.cpu().tolist(),
        "skill_0_overlap": overlaps,
        "full_mean_margin_by_task": full.mean(dim=(1, 2)).cpu().tolist(),
        "full_minimum_margin_by_task": full.amin(dim=(1, 2)).cpu().tolist(),
        "empty_mean_margin_by_task": empty.mean(dim=(1, 2)).cpu().tolist(),
        "coalition_value_sha256": tensor_dict_sha256({"coalitions": coalition_values}),
        "shapley_efficiency_max_error": float(
            (shapley_rows.sum(dim=1) - (coalition_values[-1] - coalition_values[0])).abs().max()
        ),
    }


def supervised_pretrain_step(
    model: MultiSkillGRUPolicy,
    optimizer: torch.optim.AdamW,
    device: torch.device,
) -> dict[str, float]:
    skill_ids, cue_ids = complete_grid(device)
    expected = targets(skill_ids, cue_ids)
    logits, _ = model(skill_ids, cue_ids)
    loss = sum(torch.nn.functional.cross_entropy(logits[stage], expected[:, stage]) for stage in range(STAGE_COUNT))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.sqrt(sum(
        (parameter.grad.detach().float() ** 2).sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    )))
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm": gradient_norm}


def deterministic_feedback_batch(
    *,
    seed: int,
    update: int,
    batch_size: int,
    condition: str,
    device: torch.device,
) -> dict[str, Tensor]:
    if condition not in CONDITIONS:
        raise ValueError(f"UNKNOWN_CONDITION:{condition}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1_000_003 + update * 97 + 43)
    cue_ids = torch.randint(0, CUE_COUNT, (batch_size,), generator=generator)
    uniforms = torch.rand((batch_size, STAGE_COUNT), generator=generator)
    if condition in {"selective", "frozen"}:
        skill_ids = torch.zeros(batch_size, dtype=torch.long)
    else:
        offset = (update * batch_size) % SKILL_COUNT
        skill_ids = (torch.arange(batch_size) + offset) % SKILL_COUNT
        permutation = torch.randperm(batch_size, generator=generator)
        skill_ids = skill_ids[permutation]
    return {
        "skill_ids": skill_ids.to(device),
        "cue_ids": cue_ids.to(device),
        "uniforms": uniforms.to(device),
    }


def positive_feedback_update(
    *,
    model: MultiSkillGRUPolicy,
    optimizer: torch.optim.AdamW,
    batch: dict[str, Tensor],
    apply_update: bool,
) -> dict[str, Any]:
    pre_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    pre_state = combined_state_sha256(model, optimizer)
    logits, actions = model(
        batch["skill_ids"], batch["cue_ids"], uniforms=batch["uniforms"]
    )
    expected = targets(batch["skill_ids"], batch["cue_ids"])
    stage_correct = actions.eq(expected)
    terminal_success = stage_correct.all(dim=1).float()
    chosen_log_probability = torch.stack([
        torch.log_softmax(logits[stage], dim=1).gather(1, actions[:, stage, None]).squeeze(1)
        for stage in range(STAGE_COUNT)
    ], dim=1).sum(dim=1)
    loss = -(terminal_success.detach() * chosen_log_probability).mean()
    optimizer.zero_grad(set_to_none=True)
    if apply_update:
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        gradient_norm = float(torch.sqrt(sum((value.float() ** 2).sum() for value in gradients.values())))
        gradient_hash = tensor_dict_sha256(gradients)
        optimizer.step()
    else:
        gradients = {}
        gradient_norm = 0.0
        gradient_hash = None
    updates = {
        name: model.state_dict()[name].detach() - pre_model[name]
        for name in pre_model
    }
    update_norm = float(torch.sqrt(sum((value.float() ** 2).sum() for value in updates.values())))
    return {
        "pre_state_sha256": pre_state,
        "post_state_sha256": combined_state_sha256(model, optimizer),
        "skill_batch_sha256": tensor_dict_sha256({"skill_ids": batch["skill_ids"]}),
        "cue_batch_sha256": tensor_dict_sha256({"cue_ids": batch["cue_ids"]}),
        "uniform_batch_sha256": tensor_dict_sha256({"uniforms": batch["uniforms"]}),
        "positive_consequence_count": int(terminal_success.sum()),
        "episode_count": len(terminal_success),
        "mean_positive_consequence": float(terminal_success.mean()),
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "gradient_sha256": gradient_hash,
        "update_norm": update_norm,
        "update_sha256": tensor_dict_sha256(updates),
    }


def state_payload(model: MultiSkillGRUPolicy, optimizer: torch.optim.AdamW) -> dict[str, Any]:
    return {
        "model": deepcopy(model.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
        "combined_state_sha256": combined_state_sha256(model, optimizer),
    }


def restore_state(
    payload: dict[str, Any],
    *,
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[MultiSkillGRUPolicy, torch.optim.AdamW]:
    model = MultiSkillGRUPolicy(hidden_size).to(device)
    model.load_state_dict(deepcopy(payload["model"]))
    optimizer = make_optimizer(model, learning_rate, weight_decay)
    optimizer.load_state_dict(deepcopy(payload["optimizer"]))
    if combined_state_sha256(model, optimizer) != payload["combined_state_sha256"]:
        raise RuntimeError("RESTORED_STATE_HASH_MISMATCH")
    return model, optimizer


def iter_component_state_names(model: MultiSkillGRUPolicy, component: int) -> Iterable[tuple[str, Tensor, slice]]:
    width = model.hidden_size // COMPONENT_COUNT
    rows = slice(component * width, (component + 1) * width)
    for name, value in model.state_dict().items():
        if name in {"input_projection.weight", "input_projection.bias"}:
            yield name, value, rows
        elif name in {"cell.weight_ih", "cell.weight_hh", "cell.bias_ih", "cell.bias_hh"}:
            gate_rows = [
                slice(gate * model.hidden_size + rows.start, gate * model.hidden_size + rows.stop)
                for gate in range(3)
            ]
            for gate_slice in gate_rows:
                yield name, value, gate_slice
        elif name == "actor.weight":
            yield name, value, rows


def rollback_component_state(
    trained: MultiSkillGRUPolicy,
    baseline_state: dict[str, Tensor],
    component: int,
) -> MultiSkillGRUPolicy:
    clone = MultiSkillGRUPolicy(trained.hidden_size).to(next(trained.parameters()).device)
    current = deepcopy(trained.state_dict())
    width = trained.hidden_size // COMPONENT_COUNT
    rows = slice(component * width, (component + 1) * width)
    current["input_projection.weight"][rows] = baseline_state["input_projection.weight"][rows]
    current["input_projection.bias"][rows] = baseline_state["input_projection.bias"][rows]
    for name in ("cell.weight_ih", "cell.weight_hh", "cell.bias_ih", "cell.bias_hh"):
        for gate in range(3):
            gate_rows = slice(gate * trained.hidden_size + rows.start, gate * trained.hidden_size + rows.stop)
            current[name][gate_rows] = baseline_state[name][gate_rows]
    current["cell.weight_hh"][:, rows] = baseline_state["cell.weight_hh"][:, rows]
    current["actor.weight"][:, rows] = baseline_state["actor.weight"][:, rows]
    clone.load_state_dict(current)
    return clone


def rollback_components_state(
    trained: MultiSkillGRUPolicy,
    baseline_state: dict[str, Tensor],
    components: Iterable[int],
) -> MultiSkillGRUPolicy:
    clone = trained
    for component in components:
        clone = rollback_component_state(clone, baseline_state, int(component))
    return clone
