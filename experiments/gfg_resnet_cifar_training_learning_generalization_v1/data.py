from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, label, index


def datasets_for_run(root: Path, download: bool = False):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    train = IndexedDataset(
        datasets.CIFAR100(
            root=str(root), train=True, download=download, transform=train_transform
        )
    )
    test = IndexedDataset(
        datasets.CIFAR100(
            root=str(root), train=False, download=download, transform=test_transform
        )
    )
    return train, test


def class_balanced_anchor_indices(targets: list[int], count: int) -> list[int]:
    if count < 100:
        raise ValueError("ANCHOR_COUNT_MUST_COVER_ALL_CLASSES")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    selected: list[int] = []
    round_index = 0
    while len(selected) < count:
        progressed = False
        for target in range(100):
            if round_index < len(by_class[target]):
                selected.append(by_class[target][round_index])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError("INSUFFICIENT_CLASS_BALANCED_TARGETS")
        round_index += 1
    return selected


def loaders(
    root: Path,
    seed: int,
    batch_size: int,
    anchor_count: int,
    download: bool = False,
    train_limit: int | None = None,
):
    train, test = datasets_for_run(root, download=download)
    if train_limit is not None:
        train = Subset(train, list(range(min(train_limit, len(train)))))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    test_targets = list(test.base.targets)
    anchors = Subset(test, class_balanced_anchor_indices(test_targets, anchor_count))
    anchor_loader = DataLoader(
        anchors,
        batch_size=anchor_count,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, anchor_loader, test_loader
