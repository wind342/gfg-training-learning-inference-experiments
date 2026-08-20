from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp

from .model import CifarResNet18, parameter_block
from .numeric import cosine, morphology, state_sha256, target_margins


ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
COMPONENTS = CifarResNet18.component_names


def clone_state(model: nn.Module, device: torch.device | str | None = None):
    result: dict[str, Tensor] = {}
    for name, value in itertools.chain(
        model.named_parameters(), model.named_buffers()
    ):
        copied = value.detach().clone()
        if device is not None:
            copied = copied.to(device)
        result[name] = copied
    return result


def state_delta(pre: dict[str, Tensor], post: dict[str, Tensor]):
    return {name: post[name] - value for name, value in pre.items()}


def add_delta(
    base: dict[str, Tensor], delta: dict[str, Tensor], alpha: float | Tensor
):
    result: dict[str, Tensor] = {}
    for name, value in base.items():
        if value.is_floating_point():
            result[name] = value + delta[name] * alpha
        else:
            result[name] = value
    return result


def _forward(
    model: nn.Module,
    state: dict[str, Tensor],
    images: Tensor,
    gates: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
):
    return functional_call(model, state, (images, gates), strict=True)


def _selected_indices(margins: Tensor, count: int) -> Tensor:
    values = margins.detach().cpu()
    positive = torch.where(values >= 0)[0]
    negative = torch.where(values < 0)[0]
    positive_near = positive[torch.argsort(values[positive])]
    negative_near = negative[torch.argsort(values[negative], descending=True)]
    positive_far = positive[torch.argsort(values[positive], descending=True)]
    negative_far = negative[torch.argsort(values[negative])]
    groups = (positive_near, negative_near, positive_far, negative_far)
    wanted = count // 4
    selected: list[int] = []
    used: set[int] = set()
    for group in groups:
        added = 0
        for item in group.tolist():
            if item not in used:
                selected.append(item)
                used.add(item)
                added += 1
            if added == wanted:
                break
    if len(selected) < count:
        fallback = torch.argsort(torch.abs(values))
        for item in fallback.tolist():
            if item not in used:
                selected.append(item)
                used.add(item)
            if len(selected) == count:
                break
    return torch.tensor(selected[:count], dtype=torch.long, device=margins.device)


def _directional_logits(
    model: nn.Module,
    pre: dict[str, Tensor],
    delta: dict[str, Tensor],
    images: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    alpha = torch.zeros((), device=images.device, dtype=images.dtype)
    tangent = torch.ones_like(alpha)

    def logits_at(value: Tensor) -> Tensor:
        return _forward(model, add_delta(pre, delta, value), images)

    value, first = jvp(logits_at, (alpha,), (tangent,))

    def first_at(value: Tensor) -> Tensor:
        return jvp(logits_at, (value,), (tangent,))[1]

    _, second = jvp(first_at, (alpha,), (tangent,))
    return value, first, second


def _correct(logits: Tensor, labels: Tensor) -> Tensor:
    return logits.argmax(dim=1).eq(labels)


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "MAINTAIN_CORRECT"
    if before and not after:
        return "CORRECT_TO_WRONG"
    if not before and after:
        return "WRONG_TO_CORRECT"
    return "MAINTAIN_WRONG"


def _block_geometry(
    state: dict[str, Tensor],
    delta: dict[str, Tensor],
    momentum: dict[str, Tensor],
) -> tuple[list[str], list[float], list[str], list[float]]:
    blocks = ("stem", "layer1", "layer2", "layer3", "layer4", "readout")
    grouped: dict[str, dict[str, list[Tensor]]] = defaultdict(
        lambda: {"parameter": [], "update": [], "momentum": []}
    )
    for name, value in state.items():
        if name not in delta or name not in momentum or not value.is_floating_point():
            continue
        block = parameter_block(name)
        if block not in blocks:
            continue
        grouped[block]["parameter"].append(value.reshape(-1))
        grouped[block]["update"].append(delta[name].reshape(-1))
        if name in momentum:
            grouped[block]["momentum"].append(momentum[name].reshape(-1))
    f3_names: list[str] = []
    f3_values: list[float] = []
    f5_names: list[str] = []
    f5_values: list[float] = []
    update_norms: dict[str, float] = {}
    for block in blocks:
        parts = grouped[block]
        update = torch.cat(parts["update"]) if parts["update"] else torch.zeros(1)
        update_norms[block] = float(torch.linalg.vector_norm(update.double()))
    total_update = math.sqrt(sum(value * value for value in update_norms.values()))
    for block in blocks:
        parts = grouped[block]
        parameter = (
            torch.cat(parts["parameter"]) if parts["parameter"] else torch.zeros(1)
        )
        update = torch.cat(parts["update"]) if parts["update"] else torch.zeros(1)
        momentum_value = (
            torch.cat(parts["momentum"]) if parts["momentum"] else torch.zeros_like(update)
        )
        parameter_norm = float(torch.linalg.vector_norm(parameter.double()))
        momentum_norm = float(torch.linalg.vector_norm(momentum_value.double()))
        f3_names.append(f"F3_{block}_update_fraction")
        f3_values.append(update_norms[block] / max(total_update, 1e-30))
        f5_names.extend(
            [
                f"F5_{block}_parameter_norm",
                f"F5_{block}_momentum_norm",
                f"F5_{block}_update_to_parameter",
                f"F5_{block}_update_momentum_cosine",
            ]
        )
        f5_values.extend(
            [
                parameter_norm,
                momentum_norm,
                update_norms[block] / max(parameter_norm, 1e-30),
                cosine(update, momentum_value),
            ]
        )
    return f3_names, f3_values, f5_names, f5_values


def _support_values(
    model: nn.Module,
    state: dict[str, Tensor],
    images: Tensor,
    labels: Tensor,
) -> Tensor:
    values = []
    with torch.no_grad():
        for mask in range(16):
            gates = tuple(float((mask >> index) & 1) for index in range(4))
            logits = _forward(model, state, images, gates)
            margin, _ = target_margins(logits, labels)
            values.append(margin)
    return torch.stack(values)


def _shapley(values: Tensor) -> Tensor:
    count = values.shape[1]
    result = torch.zeros((count, 4), dtype=values.dtype, device=values.device)
    factorial = math.factorial
    for component in range(4):
        for mask in range(16):
            if mask & (1 << component):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(3 - size) / factorial(4)
            result[:, component] += weight * (
                values[mask | (1 << component)] - values[mask]
            )
    return result


def _support_record(
    model: nn.Module,
    pre: dict[str, Tensor],
    post: dict[str, Tensor],
    images: Tensor,
    labels: Tensor,
) -> dict[str, Any]:
    pre_values = _support_values(model, pre, images, labels)
    repeated = _support_values(model, pre, images, labels)
    post_values = _support_values(model, post, images, labels)
    pre_shapley = _shapley(pre_values)
    post_shapley = _shapley(post_values)
    denominator = pre_shapley.abs().sum(dim=1).clamp_min(1e-12)
    reallocation = (post_shapley - pre_shapley).abs().sum(dim=1) / denominator
    active = (pre_shapley.abs() >= denominator[:, None] * 0.1).sum(dim=1)
    switches = pre_shapley.abs().argmax(dim=1).ne(post_shapley.abs().argmax(dim=1))
    pair_interactions: dict[str, list[float]] = {}
    full = 15
    for left in range(4):
        for right in range(left + 1, 4):
            without_left = full ^ (1 << left)
            without_right = full ^ (1 << right)
            without_both = full ^ (1 << left) ^ (1 << right)
            pair_interactions[f"{COMPONENTS[left]}+{COMPONENTS[right]}"] = (
                pre_values[full]
                - pre_values[without_left]
                - pre_values[without_right]
                + pre_values[without_both]
            ).detach().cpu().tolist()
    return {
        "repeat_max_abs_error": float((pre_values - repeated).abs().max()),
        "pre_shapley": pre_shapley.detach().cpu().tolist(),
        "post_shapley": post_shapley.detach().cpu().tolist(),
        "reallocation_l1_normalized": reallocation.detach().cpu().tolist(),
        "distributed_active_component_count": active.detach().cpu().tolist(),
        "primary_support_switch": switches.detach().cpu().tolist(),
        "pair_interactions_pre": pair_interactions,
    }


def analyze_update(
    *,
    model: nn.Module,
    pre: dict[str, Tensor],
    post: dict[str, Tensor],
    momentum_pre: dict[str, Tensor],
    anchor_images: Tensor,
    anchor_labels: Tensor,
    anchor_ids: Tensor,
    target_count: int,
) -> dict[str, Any]:
    model.eval()
    delta = state_delta(pre, post)
    with torch.no_grad():
        all_pre_logits = _forward(model, pre, anchor_images)
        all_pre_margins, _ = target_margins(all_pre_logits, anchor_labels)
    selected = _selected_indices(all_pre_margins, target_count)
    images = anchor_images[selected]
    labels = anchor_labels[selected]
    identities = anchor_ids[selected]
    logits0, first, second = _directional_logits(model, pre, delta, images)
    response_logits = []
    with torch.no_grad():
        for alpha in ALPHAS:
            if alpha == 0.0:
                response_state = pre
            elif alpha == 1.0:
                response_state = post
            else:
                response_state = add_delta(pre, delta, alpha)
            response_logits.append(_forward(model, response_state, images))
    logits_path = torch.stack(response_logits)
    margins_path = torch.stack(
        [target_margins(logits, labels)[0] for logits in response_logits]
    )
    true_post = response_logits[-1]
    native_post = _forward(model, post, images)
    linear = logits0 + first
    quadratic = linear + 0.5 * second
    pre_correct = _correct(logits0, labels)
    true_correct = _correct(true_post, labels)
    unchanged_correct = pre_correct
    linear_correct = _correct(linear, labels)
    quadratic_correct = _correct(quadratic, labels)
    pre_margin, pre_competitor = target_margins(logits0, labels)
    first_margin = first.gather(1, labels[:, None]).squeeze(1) - first.gather(
        1, pre_competitor[:, None]
    ).squeeze(1)
    second_margin = second.gather(1, labels[:, None]).squeeze(1) - second.gather(
        1, pre_competitor[:, None]
    ).squeeze(1)
    f3_names, f3_shared, f5_names, f5_shared = _block_geometry(
        pre, delta, momentum_pre
    )
    records = []
    for index in range(images.shape[0]):
        margins = margins_path[:, index].detach().cpu().tolist()
        chord = [margins[0] + alpha * (margins[-1] - margins[0]) for alpha in ALPHAS]
        max_deviation = max(abs(a - b) for a, b in zip(margins, chord))
        endpoint_change = margins[-1] - margins[0]
        probabilities = torch.softmax(logits0[index], dim=0)
        entropy = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
        f1_names = [
            "F1_margin",
            "F1_correct",
            "F1_correct_logit",
            "F1_competitor_logit",
            "F1_logit_std",
            "F1_entropy",
        ]
        f1_values = [
            float(pre_margin[index]),
            float(pre_correct[index]),
            float(logits0[index, labels[index]]),
            float(logits0[index, pre_competitor[index]]),
            float(logits0[index].std()),
            entropy,
        ]
        target_f3_names = [
            "F3_first_margin_direction",
            "F3_second_margin_direction",
            "F3_first_correct_logit_direction",
            "F3_first_competitor_logit_direction",
        ]
        target_f3 = [
            float(first_margin[index]),
            float(second_margin[index]),
            float(first[index, labels[index]]),
            float(first[index, pre_competitor[index]]),
        ]
        records.append(
            {
                "target_id": int(identities[index]),
                "label": int(labels[index]),
                "pre_competitor": int(pre_competitor[index]),
                "pre_margin": margins[0],
                "post_margin": margins[-1],
                "margin_path": margins,
                "morphology": morphology(list(ALPHAS), margins),
                "max_chord_deviation": max_deviation,
                "normalized_chord_deviation": max_deviation
                / max(abs(endpoint_change), 1e-6),
                "pre_correct": bool(pre_correct[index]),
                "post_correct": bool(true_correct[index]),
                "transition": _transition(
                    bool(pre_correct[index]), bool(true_correct[index])
                ),
                "predictions": {
                    "unchanged": bool(unchanged_correct[index]),
                    "linear": bool(linear_correct[index]),
                    "quadratic": bool(quadratic_correct[index]),
                },
                "feature_names": f1_names
                + target_f3_names
                + f3_names
                + f5_names,
                "features": f1_values
                + target_f3
                + f3_shared
                + f5_shared,
                "feature_families": {
                    "F1": [0, len(f1_names)],
                    "F3": [
                        len(f1_names),
                        len(f1_names) + len(target_f3_names) + len(f3_names),
                    ],
                    "F5": [
                        len(f1_names) + len(target_f3_names) + len(f3_names),
                        len(f1_names)
                        + len(target_f3_names)
                        + len(f3_names)
                        + len(f5_names),
                    ],
                },
            }
        )
    support = _support_record(model, pre, post, images, labels)
    return {
        "pre_state_sha256": state_sha256(pre),
        "post_state_sha256": state_sha256(post),
        "delta_state_sha256": state_sha256(delta),
        "selected_target_ids": identities.detach().cpu().tolist(),
        "alpha_grid": list(ALPHAS),
        "alpha0_logit_max_abs_error": float((logits_path[0] - logits0).abs().max()),
        "alpha1_native_logit_max_abs_error": float((true_post - native_post).abs().max()),
        "records": records,
        "support": support,
    }


def receiving_state_exchange(
    *,
    model: nn.Module,
    state_a: dict[str, Tensor],
    delta_a: dict[str, Tensor],
    state_b: dict[str, Tensor],
    delta_b: dict[str, Tensor],
    images: Tensor,
    labels: Tensor,
) -> dict[str, Any]:
    model.eval()

    def response(state: dict[str, Tensor], delta: dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            before = target_margins(_forward(model, state, images), labels)[0]
            after = target_margins(
                _forward(model, add_delta(state, delta, 1.0), images), labels
            )[0]
        return after - before

    aa = response(state_a, delta_a)
    ba = response(state_b, delta_a)
    bb = response(state_b, delta_b)
    ab = response(state_a, delta_b)

    def comparison(native: Tensor, exchanged: Tensor) -> dict[str, float]:
        difference = native - exchanged
        rms = float(torch.sqrt(torch.mean(difference.double() ** 2)))
        denominator = float(torch.sqrt(torch.mean(native.double() ** 2)))
        return {
            "response_rmse": rms,
            "response_nrmse": rms / max(denominator, 1e-12),
            "direction_disagreement_rate": float(
                torch.sign(native).ne(torch.sign(exchanged)).float().mean()
            ),
        }

    return {
        "update_a_state_a_vs_state_b": comparison(aa, ba),
        "update_b_state_b_vs_state_a": comparison(bb, ab),
    }


def momentum_exchange_response(
    *,
    model: nn.Module,
    state: dict[str, Tensor],
    native_delta: dict[str, Tensor],
    exchanged_delta: dict[str, Tensor],
    images: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        baseline = target_margins(_forward(model, state, images), labels)[0]
        native = target_margins(
            _forward(model, add_delta(state, native_delta, 1.0), images), labels
        )[0] - baseline
        exchanged = target_margins(
            _forward(model, add_delta(state, exchanged_delta, 1.0), images), labels
        )[0] - baseline
    difference = native - exchanged
    rms = float(torch.sqrt(torch.mean(difference.double() ** 2)))
    denominator = float(torch.sqrt(torch.mean(native.double() ** 2)))
    return {
        "response_rmse": rms,
        "response_nrmse": rms / max(denominator, 1e-12),
        "direction_disagreement_rate": float(
            torch.sign(native).ne(torch.sign(exchanged)).float().mean()
        ),
    }
