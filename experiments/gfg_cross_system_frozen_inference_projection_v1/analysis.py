from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor, nn


def tensor_hash(value: Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("ascii")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def object_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, Tensor):
            digest.update(b"tensor:")
            digest.update(tensor_hash(item).encode("ascii"))
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda row: str(row)):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence[")
            for row in item:
                visit(row)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def shapley(values: Tensor) -> Tensor:
    if values.shape[0] != 16:
        raise ValueError("EXPECTED_SIXTEEN_COALITIONS")
    result = torch.zeros(
        (values.shape[1], 4), dtype=values.dtype, device=values.device
    )
    for component in range(4):
        for mask in range(16):
            if mask & (1 << component):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
            result[:, component] += weight * (
                values[mask | (1 << component)] - values[mask]
            )
    return result

def support_diagnostics(values: Tensor) -> dict[str, Any]:
    effects = shapley(values)
    denominator = effects.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
    profiles = effects / denominator
    maximum_profile_l1 = 0.0
    for left in range(len(profiles)):
        for right in range(left + 1, len(profiles)):
            maximum_profile_l1 = max(
                maximum_profile_l1,
                float((profiles[left] - profiles[right]).abs().sum()),
            )
    interactions: dict[str, list[float]] = {}
    maximum_interaction = 0.0
    for left in range(4):
        for right in range(left + 1, 4):
            value = (
                values[15]
                - values[15 ^ (1 << left)]
                - values[15 ^ (1 << right)]
                + values[15 ^ (1 << left) ^ (1 << right)]
            )
            interactions[f"{left}+{right}"] = value.detach().cpu().tolist()
            maximum_interaction = max(maximum_interaction, float(value.abs().max()))
    return {
        "shapley": effects.detach().cpu().tolist(),
        "normalized_profiles": profiles.detach().cpu().tolist(),
        "maximum_query_profile_l1": maximum_profile_l1,
        "pair_interactions": interactions,
        "maximum_absolute_pair_interaction": maximum_interaction,
    }


def component_call_capture(
    modules: list[tuple[str, nn.Module]], execute
) -> tuple[Tensor, dict[str, float]]:
    captured: dict[str, Tensor] = {}
    handles = []
    for name, module in modules:
        def hook(_module, _inputs, output, component_name=name):
            captured[component_name] = output.detach().clone()

        handles.append(module.register_forward_hook(hook))
    try:
        output = execute()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {name for name, _ in modules}:
        raise RuntimeError("MISSING_COMPONENT_CALL")
    return output, {
        name: float(value.double().square().mean().sqrt())
        for name, value in captured.items()
    }


def hybrid_state(
    trained: dict[str, Tensor],
    pre_learning: dict[str, Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, Tensor]:
    result = {name: value.detach().clone() for name, value in trained.items()}
    replaced = 0
    for name in result:
        if name.startswith(prefixes):
            result[name] = pre_learning[name].detach().clone()
            replaced += 1
    if replaced == 0:
        raise RuntimeError("ROLLBACK_REPLACED_NO_STATE")
    return result
