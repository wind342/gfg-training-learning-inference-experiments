from __future__ import annotations

import torch
from torch import Tensor


def residual_boundary(
    prediction: Tensor, true_noise: Tensor, candidates: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    true_error = (prediction - true_noise).square().flatten(1).mean(dim=1)
    candidate_error = (
        prediction[:, None] - candidates
    ).square().flatten(2).mean(dim=2)
    pair_margins = candidate_error - true_error[:, None]
    margin, competitor = pair_margins.min(dim=1)
    return margin, competitor, true_error, candidate_error, pair_margins


def transition(before: bool, after: bool) -> str:
    if before and after:
        return "MAINTAIN_CORRECT"
    if before and not after:
        return "CORRECT_TO_WRONG"
    if not before and after:
        return "WRONG_TO_CORRECT"
    return "MAINTAIN_WRONG"


def selected_indices(margins: Tensor, count: int) -> Tensor:
    values = margins.detach().cpu()
    positive = torch.where(values >= 0)[0]
    negative = torch.where(values < 0)[0]
    groups = (
        positive[torch.argsort(values[positive])],
        negative[torch.argsort(values[negative], descending=True)],
    )
    selected: list[int] = []
    used: set[int] = set()
    wanted = max(count // 2, 1)
    for group in groups:
        added = 0
        for item in group.tolist():
            if item not in used:
                selected.append(item)
                used.add(item)
                added += 1
            if added >= wanted:
                break
    for item in torch.argsort(values.abs()).tolist():
        if item not in used:
            selected.append(item)
            used.add(item)
        if len(selected) >= count:
            break
    return torch.tensor(selected[:count], dtype=torch.long, device=margins.device)
