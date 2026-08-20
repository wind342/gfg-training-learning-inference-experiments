from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode("ascii")
    return sha256_bytes(header + value.numpy().tobytes())


def state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(state[name]).encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def target_margins(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    rows = torch.arange(logits.shape[0], device=logits.device)
    correct = logits[rows, labels]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    competitor, competitor_id = masked.max(dim=1)
    return correct - competitor, competitor_id


def cosine(left: Tensor, right: Tensor) -> float:
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def morphology(alpha: list[float], margins: list[float]) -> str:
    slopes = [
        (margins[index + 1] - margins[index])
        / (alpha[index + 1] - alpha[index])
        for index in range(len(alpha) - 1)
    ]
    scale = max(max(abs(value) for value in slopes), 1e-12)
    active = [value for value in slopes if abs(value) > scale * 0.05]
    signs = [1 if value > 0 else -1 for value in active]
    if len(signs) >= 2 and signs[0] != signs[-1]:
        endpoint = margins[-1] - margins[0]
        if endpoint != 0.0 and math.copysign(1.0, endpoint) != signs[0]:
            return "SIGN_REVERSAL"
        return "TURNBACK"
    if len(active) >= 2:
        ratio = abs(active[-1]) / max(abs(active[0]), 1e-12)
        if ratio <= 0.5:
            return "SATURATING"
        if ratio >= 1.5:
            return "ACCELERATING"
    return "NEAR_LINEAR"
