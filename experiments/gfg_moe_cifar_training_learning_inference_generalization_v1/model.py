from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class ExpertMLP(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.input = nn.Linear(dimension, hidden_dimension)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dimension, dimension)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.output(self.activation(self.input(tokens)))


class DenseMixerBlock(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.mlp = ExpertMLP(dimension, hidden_dimension)

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens + self.mlp(self.norm(tokens))


class SparseTop2MoE(nn.Module):
    component_names = ("expert0", "expert1", "expert2", "expert3")

    def __init__(self, dimension: int = 96, hidden_dimension: int = 192) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.router = nn.Linear(dimension, 4)
        self.experts = nn.ModuleList(
            [ExpertMLP(dimension, hidden_dimension) for _ in range(4)]
        )

    def forward(
        self,
        tokens: Tensor,
        gates: tuple[float, float, float, float],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        normalized = self.norm(tokens)
        probabilities = torch.softmax(self.router(normalized), dim=-1)
        top_values, top_indices = probabilities.topk(2, dim=-1)
        routed = torch.zeros_like(probabilities).scatter(-1, top_indices, top_values)
        routed = routed / routed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expert_outputs = torch.stack(
            [expert(normalized) for expert in self.experts], dim=-2
        )
        gate = torch.as_tensor(gates, dtype=tokens.dtype, device=tokens.device)
        weighted = routed * gate
        contributions = expert_outputs * weighted.unsqueeze(-1)
        mixed = contributions.sum(dim=-2)
        return tokens + mixed, {
            "routing_weights": routed,
            "expert_contributions": contributions,
            "router_probabilities": probabilities,
        }


class CifarVisionMoE(nn.Module):
    component_names = SparseTop2MoE.component_names

    def __init__(
        self,
        classes: int = 100,
        dimension: int = 96,
        expert_hidden_dimension: int = 192,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, dimension, kernel_size=4, stride=4)
        self.position = nn.Parameter(torch.zeros(1, 64, dimension))
        self.pre_block = DenseMixerBlock(dimension, dimension * 2)
        self.moe = SparseTop2MoE(dimension, expert_hidden_dimension)
        self.output_norm = nn.LayerNorm(dimension)
        self.readout = nn.Linear(dimension, classes)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(
        self,
        images: Tensor,
        gates: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        tokens = self.stem(images).flatten(2).transpose(1, 2) + self.position
        tokens = self.pre_block(tokens)
        tokens, details = self.moe(tokens, gates)
        pooled = self.output_norm(tokens).mean(dim=1)
        logits = self.readout(pooled)
        if return_details:
            return logits, details
        return logits

    def load_balance_loss(self, images: Tensor) -> Tensor:
        _, details = self.forward(images, return_details=True)
        mean_probability = details["router_probabilities"].mean(dim=(0, 1))
        mean_assignment = details["routing_weights"].gt(0).float().mean(dim=(0, 1))
        return 4.0 * torch.sum(mean_probability * mean_assignment)


def parameter_block(name: str) -> str:
    if name.startswith("stem") or name == "position":
        return "stem"
    if name.startswith("pre_block"):
        return "pre_block"
    if name.startswith("moe.router") or name.startswith("moe.norm"):
        return "router"
    if name.startswith("moe.experts"):
        return "experts"
    if name.startswith("output_norm") or name.startswith("readout"):
        return "readout"
    return "other"


def component_parameter_prefix(component: str) -> str:
    if component.startswith("expert"):
        return f"moe.experts.{int(component.removeprefix('expert'))}."
    if component == "router":
        return "moe.router."
    raise KeyError(component)


def routing_summary(details: dict[str, Tensor]) -> dict[str, Any]:
    routed = details["routing_weights"]
    contributions = details["expert_contributions"]
    nonzero = routed.gt(0).sum(dim=-1)
    return {
        "minimum_nonzero_experts_per_token": int(nonzero.min()),
        "maximum_nonzero_experts_per_token": int(nonzero.max()),
        "maximum_weight_sum_error": float((routed.sum(dim=-1) - 1).abs().max()),
        "mean_routing_profile": routed.mean(dim=(0, 1)).detach().cpu().tolist(),
        "expert_utilization": routed.gt(0).float().mean(dim=(0, 1)).detach().cpu().tolist(),
        "expert_contribution_rms": contributions.square().mean(dim=(0, 1, 3)).sqrt().detach().cpu().tolist(),
    }
