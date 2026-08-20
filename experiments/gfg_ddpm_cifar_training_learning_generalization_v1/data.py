from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: Tensor
    alphas: Tensor
    alpha_bar: Tensor

    @classmethod
    def linear(
        cls,
        steps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ) -> "DiffusionSchedule":
        betas = torch.linspace(beta_start, beta_end, steps, device=device)
        alphas = 1.0 - betas
        return cls(betas=betas, alphas=alphas, alpha_bar=alphas.cumprod(dim=0))

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        alpha_bar = self.alpha_bar[timesteps][:, None, None, None]
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise


def loaders(
    root: Path,
    *,
    batch_size: int,
    seed: int,
    download: bool,
    train_subset: int | None = None,
    test_subset: int | None = None,
    test_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    train = datasets.CIFAR10(root, train=True, transform=transform, download=download)
    test = datasets.CIFAR10(root, train=False, transform=transform, download=download)
    if train_subset is not None:
        train = Subset(train, range(min(train_subset, len(train))))
    if test_subset is not None:
        test = Subset(test, range(min(test_subset, len(test))))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test,
        batch_size=test_batch_size or batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


def evaluation_pack(
    test_loader: DataLoader,
    *,
    count: int,
    seed: int,
    candidate_count: int,
    candidate_scale: float,
    schedule: DiffusionSchedule,
    device: torch.device,
) -> dict[str, Tensor]:
    images, labels = next(iter(test_loader))
    images = images[:count].to(device)
    labels = labels[:count].to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 90_000)
    timesteps = torch.randint(
        5, len(schedule.betas) - 5, (len(images),), generator=generator, device=device
    )
    true_noise = torch.randn(images.shape, generator=generator, device=device)
    offsets = torch.randn(
        (len(images), candidate_count, *images.shape[1:]),
        generator=generator,
        device=device,
    )
    offsets = offsets / offsets.flatten(2).std(dim=2)[:, :, None, None, None]
    candidates = true_noise[:, None] + candidate_scale * offsets
    noisy = schedule.q_sample(images, timesteps, true_noise)
    identities = torch.arange(len(images), device=device, dtype=torch.long)
    return {
        "clean": images,
        "labels": labels,
        "timesteps": timesteps,
        "true_noise": true_noise,
        "candidates": candidates,
        "noisy": noisy,
        "identities": identities,
    }
