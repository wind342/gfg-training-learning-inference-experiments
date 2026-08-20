from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DirectionalBatchNorm2d(nn.BatchNorm2d):
    """Standard training BN with an eval formula that supports state JVPs."""

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                True,
                exponential_average_factor,
                self.eps,
            )
        mean = self.running_mean[None, :, None, None]
        variance = self.running_var[None, :, None, None]
        weight = self.weight[None, :, None, None]
        bias = self.bias[None, :, None, None]
        return (x - mean) * torch.rsqrt(variance + self.eps) * weight + bias


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = DirectionalBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = DirectionalBatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                DirectionalBatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor, residual_gate: float | Tensor = 1.0) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return self.relu(identity + residual * residual_gate)


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> None:
        super().__init__()
        members = [BasicBlock(in_channels, out_channels, stride)]
        members.extend(
            BasicBlock(out_channels, out_channels, 1)
            for _ in range(1, blocks)
        )
        self.blocks = nn.ModuleList(members)

    def forward(self, x: Tensor, gate: float | Tensor = 1.0) -> Tensor:
        for block in self.blocks:
            x = block(x, gate)
        return x


class CifarResNet18(nn.Module):
    """ResNet-18 with a CIFAR stem and four reversible residual-stage gates."""

    component_names = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.conv1 = _conv3x3(3, 64)
        self.bn1 = DirectionalBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = ResidualStage(64, 64, 2, 1)
        self.layer2 = ResidualStage(64, 128, 2, 2)
        self.layer3 = ResidualStage(128, 256, 2, 2)
        self.layer4 = ResidualStage(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        stage_gates: Sequence[float | Tensor] | None = None,
    ) -> Tensor:
        gates = stage_gates if stage_gates is not None else (1.0,) * 4
        if len(gates) != 4:
            raise ValueError("EXPECTED_FOUR_STAGE_GATES")
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x, gates[0])
        x = self.layer2(x, gates[1])
        x = self.layer3(x, gates[2])
        x = self.layer4(x, gates[3])
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


def parameter_block(name: str) -> str:
    for block in CifarResNet18.component_names:
        if name.startswith(block + "."):
            return block
    if name.startswith(("conv1.", "bn1.")):
        return "stem"
    if name.startswith("fc."):
        return "readout"
    return "other"
