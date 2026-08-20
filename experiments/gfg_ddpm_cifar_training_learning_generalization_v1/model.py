from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    scale = math.log(10_000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dimension: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dimension, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.activation = nn.SiLU()

    def forward(self, value: Tensor, time: Tensor) -> Tensor:
        hidden = self.conv1(self.activation(self.norm1(value)))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return self.skip(value) + hidden


class CifarDiffusionUNet(nn.Module):
    component_names = (
        "high_resolution_skip",
        "low_resolution_skip",
        "bottleneck",
        "decoder_refinement",
    )

    def __init__(self, base_channels: int = 32, time_dimension: int = 128):
        super().__init__()
        self.time_dimension = time_dimension
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dimension, time_dimension),
            nn.SiLU(),
            nn.Linear(time_dimension, time_dimension),
        )
        self.input = nn.Conv2d(3, base_channels, 3, padding=1)
        self.encoder_high = ResidualBlock(base_channels, base_channels, time_dimension)
        self.down_high = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.encoder_low = ResidualBlock(
            base_channels * 2, base_channels * 2, time_dimension
        )
        self.down_low = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
        self.middle = ResidualBlock(
            base_channels * 4, base_channels * 4, time_dimension
        )
        self.up_low = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 4, 2, 1
        )
        self.decoder_low = ResidualBlock(
            base_channels * 4, base_channels * 2, time_dimension
        )
        self.up_high = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 4, 2, 1
        )
        self.decoder_high = ResidualBlock(
            base_channels * 2, base_channels, time_dimension
        )
        self.refinement = ResidualBlock(base_channels, base_channels, time_dimension)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output = nn.Conv2d(base_channels, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        gates: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> Tensor:
        time = self.time_mlp(timestep_embedding(timesteps, self.time_dimension))
        initial = self.input(noisy)
        high = self.encoder_high(initial, time)
        low = self.encoder_low(self.down_high(high), time)
        middle = self.middle(self.down_low(low), time) * gates[2]
        decoded_low = self.up_low(middle)
        decoded_low = self.decoder_low(
            torch.cat((decoded_low, low * gates[1]), dim=1), time
        )
        decoded_high = self.up_high(decoded_low)
        decoded_high = self.decoder_high(
            torch.cat((decoded_high, high * gates[0]), dim=1), time
        )
        refined = self.refinement(decoded_high, time)
        decoded_high = decoded_high + gates[3] * (refined - decoded_high)
        return self.output(torch.nn.functional.silu(self.output_norm(decoded_high)))


def parameter_block(name: str) -> str:
    if name.startswith(("input", "time_mlp")):
        return "input"
    if name.startswith(("encoder_high", "down_high", "encoder_low", "down_low")):
        return "encoder"
    if name.startswith("middle"):
        return "bottleneck"
    if name.startswith(("up_low", "decoder_low", "up_high", "decoder_high", "refinement")):
        return "decoder"
    return "output"
