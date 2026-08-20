from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp

from .boundary import residual_boundary, selected_indices, transition
from .model import CifarDiffusionUNet, parameter_block
from .numeric import cosine, morphology, state_sha256


ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
COMPONENTS = CifarDiffusionUNet.component_names


def clone_state(
    model: nn.Module, device: torch.device | str | None = None
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in itertools.chain(model.named_parameters(), model.named_buffers()):
        copied = value.detach().clone()
        result[name] = copied.to(device) if device is not None else copied
    return result


def state_delta(pre: dict[str, Tensor], post: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: post[name] - value for name, value in pre.items()}


def add_delta(
    base: dict[str, Tensor], delta: dict[str, Tensor], alpha: float | Tensor
) -> dict[str, Tensor]:
    return {
        name: value + delta[name] * alpha if value.is_floating_point() else value
        for name, value in base.items()
    }


def forward_state(
    model: nn.Module,
    state: dict[str, Tensor],
    noisy: Tensor,
    timesteps: Tensor,
    gates: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Tensor:
    return functional_call(model, state, (noisy, timesteps, gates), strict=True)


def directional_prediction(
    model: nn.Module,
    pre: dict[str, Tensor],
    delta: dict[str, Tensor],
    noisy: Tensor,
    timesteps: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    alpha = torch.zeros((), device=noisy.device, dtype=noisy.dtype)
    tangent = torch.ones_like(alpha)

    def prediction_at(value: Tensor) -> Tensor:
        return forward_state(model, add_delta(pre, delta, value), noisy, timesteps)

    value, first = jvp(prediction_at, (alpha,), (tangent,))

    def first_at(value: Tensor) -> Tensor:
        return jvp(prediction_at, (value,), (tangent,))[1]

    _, second = jvp(first_at, (alpha,), (tangent,))
    return value, first, second


def clone_adam_memory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, Tensor]]:
    result: dict[str, dict[str, Tensor]] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        target_device = parameter.device if device is None else torch.device(device)
        result[name] = {
            "step": torch.as_tensor(state.get("step", 0)).detach().clone().to(target_device),
            "exp_avg": state.get("exp_avg", torch.zeros_like(parameter)).detach().clone().to(target_device),
            "exp_avg_sq": state.get("exp_avg_sq", torch.zeros_like(parameter)).detach().clone().to(target_device),
        }
    return result


def adamw_delta(
    pre: dict[str, Tensor],
    gradients: dict[str, Tensor],
    memory: dict[str, dict[str, Tensor]],
    *,
    learning_rate: float,
    betas: tuple[float, float],
    epsilon: float,
    weight_decay: float,
) -> dict[str, Tensor]:
    beta1, beta2 = betas
    result = {name: torch.zeros_like(value) for name, value in pre.items()}
    for name, gradient in gradients.items():
        parameter = pre[name]
        old = memory[name]
        step = int(old["step"].item()) + 1
        exp_avg = old["exp_avg"] * beta1 + gradient * (1.0 - beta1)
        exp_avg_sq = old["exp_avg_sq"] * beta2 + gradient.square() * (1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        denominator = exp_avg_sq.sqrt() / math.sqrt(bias2) + epsilon
        updated = parameter * (1.0 - learning_rate * weight_decay)
        updated = updated - (learning_rate / bias1) * exp_avg / denominator
        result[name] = updated - parameter
    return result


def _block_geometry(
    state: dict[str, Tensor],
    delta: dict[str, Tensor],
    memory: dict[str, dict[str, Tensor]],
) -> tuple[list[str], list[float], list[str], list[float]]:
    blocks = ("input", "encoder", "bottleneck", "decoder", "output")
    grouped: dict[str, dict[str, list[Tensor]]] = defaultdict(
        lambda: {"parameter": [], "update": [], "m": [], "v": []}
    )
    for name, old in memory.items():
        block = parameter_block(name)
        grouped[block]["parameter"].append(state[name].reshape(-1))
        grouped[block]["update"].append(delta[name].reshape(-1))
        grouped[block]["m"].append(old["exp_avg"].reshape(-1))
        grouped[block]["v"].append(old["exp_avg_sq"].reshape(-1))
    update_norms: dict[str, float] = {}
    for block in blocks:
        value = torch.cat(grouped[block]["update"])
        update_norms[block] = float(torch.linalg.vector_norm(value.double()))
    total_update = math.sqrt(sum(value * value for value in update_norms.values()))
    f3_names: list[str] = []
    f3_values: list[float] = []
    f5_names: list[str] = []
    f5_values: list[float] = []
    for block in blocks:
        parameter = torch.cat(grouped[block]["parameter"])
        update = torch.cat(grouped[block]["update"])
        exp_avg = torch.cat(grouped[block]["m"])
        exp_avg_sq = torch.cat(grouped[block]["v"])
        parameter_norm = float(torch.linalg.vector_norm(parameter.double()))
        m_norm = float(torch.linalg.vector_norm(exp_avg.double()))
        v_rms = float(exp_avg_sq.double().mean().sqrt())
        f3_names.append(f"F3_{block}_update_fraction")
        f3_values.append(update_norms[block] / max(total_update, 1e-30))
        f5_names.extend(
            (
                f"F5_{block}_parameter_norm",
                f"F5_{block}_first_moment_norm",
                f"F5_{block}_second_moment_rms",
                f"F5_{block}_update_to_parameter",
                f"F5_{block}_update_first_moment_cosine",
            )
        )
        f5_values.extend(
            (
                parameter_norm,
                m_norm,
                v_rms,
                update_norms[block] / max(parameter_norm, 1e-30),
                cosine(update, exp_avg),
            )
        )
    return f3_names, f3_values, f5_names, f5_values


def _support_values(
    model: nn.Module,
    state: dict[str, Tensor],
    noisy: Tensor,
    timesteps: Tensor,
    true_noise: Tensor,
    candidates: Tensor,
) -> Tensor:
    values = []
    with torch.no_grad():
        for mask in range(16):
            gates = tuple(float((mask >> index) & 1) for index in range(4))
            prediction = forward_state(model, state, noisy, timesteps, gates)
            values.append(residual_boundary(prediction, true_noise, candidates)[0])
    return torch.stack(values)


def _shapley(values: Tensor) -> Tensor:
    count = values.shape[1]
    result = torch.zeros((count, 4), dtype=values.dtype, device=values.device)
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


def support_record(
    model: nn.Module,
    pre: dict[str, Tensor],
    post: dict[str, Tensor],
    noisy: Tensor,
    timesteps: Tensor,
    true_noise: Tensor,
    candidates: Tensor,
) -> dict[str, Any]:
    pre_values = _support_values(model, pre, noisy, timesteps, true_noise, candidates)
    repeated = _support_values(model, pre, noisy, timesteps, true_noise, candidates)
    post_values = _support_values(model, post, noisy, timesteps, true_noise, candidates)
    pre_shapley = _shapley(pre_values)
    post_shapley = _shapley(post_values)
    denominator = pre_shapley.abs().sum(dim=1).clamp_min(1e-12)
    reallocation = (post_shapley - pre_shapley).abs().sum(dim=1) / denominator
    active = (pre_shapley.abs() >= denominator[:, None] * 0.1).sum(dim=1)
    switches = pre_shapley.abs().argmax(dim=1).ne(post_shapley.abs().argmax(dim=1))
    interactions: dict[str, list[float]] = {}
    for left in range(4):
        for right in range(left + 1, 4):
            interactions[f"{COMPONENTS[left]}+{COMPONENTS[right]}"] = (
                pre_values[15]
                - pre_values[15 ^ (1 << left)]
                - pre_values[15 ^ (1 << right)]
                + pre_values[15 ^ (1 << left) ^ (1 << right)]
            ).detach().cpu().tolist()
    return {
        "repeat_max_abs_error": float((pre_values - repeated).abs().max()),
        "pre_shapley": pre_shapley.detach().cpu().tolist(),
        "post_shapley": post_shapley.detach().cpu().tolist(),
        "reallocation_l1_normalized": reallocation.detach().cpu().tolist(),
        "distributed_active_component_count": active.detach().cpu().tolist(),
        "primary_support_switch": switches.detach().cpu().tolist(),
        "pair_interactions_pre": interactions,
    }


def analyze_update(
    *,
    model: nn.Module,
    pre: dict[str, Tensor],
    post: dict[str, Tensor],
    memory_pre: dict[str, dict[str, Tensor]],
    evaluation: dict[str, Tensor],
    alpha_bar: Tensor,
    target_count: int,
) -> dict[str, Any]:
    model.eval()
    delta = state_delta(pre, post)
    with torch.no_grad():
        all_pre_prediction = forward_state(
            model, pre, evaluation["noisy"], evaluation["timesteps"]
        )
        all_pre_margin = residual_boundary(
            all_pre_prediction, evaluation["true_noise"], evaluation["candidates"]
        )[0]
    selected = selected_indices(all_pre_margin, target_count)
    noisy = evaluation["noisy"][selected]
    timesteps = evaluation["timesteps"][selected]
    true_noise = evaluation["true_noise"][selected]
    candidates = evaluation["candidates"][selected]
    identities = evaluation["identities"][selected]
    prediction0, first, second = directional_prediction(
        model, pre, delta, noisy, timesteps
    )
    path_predictions = []
    with torch.no_grad():
        for alpha in ALPHAS:
            response_state = pre if alpha == 0.0 else post if alpha == 1.0 else add_delta(pre, delta, alpha)
            path_predictions.append(
                forward_state(model, response_state, noisy, timesteps)
            )
    path_margins = torch.stack(
        [residual_boundary(value, true_noise, candidates)[0] for value in path_predictions]
    )
    pre_boundary = residual_boundary(prediction0, true_noise, candidates)
    post_boundary = residual_boundary(path_predictions[-1], true_noise, candidates)
    linear_prediction = prediction0 + first
    quadratic_prediction = linear_prediction + 0.5 * second
    linear_margin = residual_boundary(linear_prediction, true_noise, candidates)[0]
    quadratic_margin = residual_boundary(quadratic_prediction, true_noise, candidates)[0]
    f3_names, f3_global, f5_names, f5_values = _block_geometry(pre, delta, memory_pre)
    support = support_record(
        model, pre, post, noisy, timesteps, true_noise, candidates
    )
    records = []
    feature_names = [
        "F1_pre_margin",
        "F1_pre_correct",
        "F1_true_error",
        "F1_nearest_candidate_error",
        "F1_timestep_fraction",
        "F1_log_snr",
        "F3_target_first_rms",
        "F3_target_alignment",
        "F3_current_competitor_margin_delta",
        *f3_names,
        *f5_names,
    ]
    f1_stop = 6
    f3_stop = f1_stop + 3 + len(f3_global)
    for index in range(len(selected)):
        pre_margin = float(pre_boundary[0][index])
        post_margin = float(post_boundary[0][index])
        competitor = int(pre_boundary[1][index])
        current_pair_before = pre_boundary[4][index, competitor]
        current_pair_after = residual_boundary(
            prediction0[index : index + 1] + first[index : index + 1],
            true_noise[index : index + 1],
            candidates[index : index + 1],
        )[4][0, competitor]
        target_direction = true_noise[index] - prediction0[index]
        timestep = int(timesteps[index])
        abar = float(alpha_bar[timestep])
        features = [
            pre_margin,
            float(pre_margin >= 0.0),
            float(pre_boundary[2][index]),
            float(pre_boundary[3][index].min()),
            timestep / max(len(alpha_bar) - 1, 1),
            math.log(max(abar, 1e-12) / max(1.0 - abar, 1e-12)),
            float(first[index].square().mean().sqrt()),
            cosine(first[index], target_direction),
            float(current_pair_after - current_pair_before),
            *f3_global,
            *f5_values,
        ]
        pre_correct = pre_margin >= 0.0
        post_correct = post_margin >= 0.0
        curve = path_margins[:, index].detach().cpu().tolist()
        records.append(
            {
                "target_identity": int(identities[index]),
                "timestep": timestep,
                "pre_margin": pre_margin,
                "post_margin": post_margin,
                "pre_correct": pre_correct,
                "post_correct": post_correct,
                "transition": transition(pre_correct, post_correct),
                "morphology": morphology(list(ALPHAS), curve),
                "margin_path": curve,
                "predictions": {
                    "unchanged": pre_correct,
                    "linear": bool(linear_margin[index] >= 0),
                    "quadratic": bool(quadratic_margin[index] >= 0),
                },
                "predicted_margins": {
                    "linear": float(linear_margin[index]),
                    "quadratic": float(quadratic_margin[index]),
                },
                "features": features,
                "feature_names": feature_names,
                "feature_families": {
                    "F1": [0, f1_stop],
                    "F3": [f1_stop, f3_stop],
                    "F5": [f3_stop, len(features)],
                },
                "support": {
                    "pre_shapley": support["pre_shapley"][index],
                    "post_shapley": support["post_shapley"][index],
                    "reallocation_l1_normalized": support["reallocation_l1_normalized"][index],
                    "distributed_active_component_count": support["distributed_active_component_count"][index],
                    "primary_support_switch": support["primary_support_switch"][index],
                },
            }
        )
    native_pre = forward_state(model, pre, noisy, timesteps)
    native_post = forward_state(model, post, noisy, timesteps)
    return {
        "pre_state_sha256": state_sha256(pre),
        "post_state_sha256": state_sha256(post),
        "delta_state_sha256": state_sha256(delta),
        "selected_target_ids": [int(value) for value in identities.detach().cpu()],
        "alpha_grid": list(ALPHAS),
        "response_amplitudes": list(ALPHAS),
        "records": records,
        "support": support,
        "integrity": {
            "alpha0_max_abs_error": float((path_predictions[0] - native_pre).abs().max()),
            "alpha1_native_max_abs_error": float((path_predictions[-1] - native_post).abs().max()),
            "support_repeat_max_abs_error": support["repeat_max_abs_error"],
        },
    }


def response_for_state_and_delta(
    model: nn.Module,
    base: dict[str, Tensor],
    delta: dict[str, Tensor],
    evaluation: dict[str, Tensor],
) -> Tensor:
    with torch.no_grad():
        before = forward_state(
            model, base, evaluation["noisy"], evaluation["timesteps"]
        )
        after = forward_state(
            model, add_delta(base, delta, 1.0), evaluation["noisy"], evaluation["timesteps"]
        )
        before_margin = residual_boundary(
            before, evaluation["true_noise"], evaluation["candidates"]
        )[0]
        after_margin = residual_boundary(
            after, evaluation["true_noise"], evaluation["candidates"]
        )[0]
    return after_margin - before_margin


def response_nrmse(left: Tensor, right: Tensor) -> float:
    scale = torch.sqrt((left.square().mean() + right.square().mean()) / 2).clamp_min(1e-12)
    return float(torch.sqrt((left - right).square().mean()) / scale)


def receiving_state_exchange(
    model: nn.Module,
    current_pre: dict[str, Tensor],
    alternate_pre: dict[str, Tensor],
    delta: dict[str, Tensor],
    evaluation: dict[str, Tensor],
) -> dict[str, float]:
    current = response_for_state_and_delta(model, current_pre, delta, evaluation)
    alternate = response_for_state_and_delta(model, alternate_pre, delta, evaluation)
    return {
        "response_nrmse": response_nrmse(current, alternate),
        "current_response_rms": float(current.square().mean().sqrt()),
        "alternate_response_rms": float(alternate.square().mean().sqrt()),
    }


def adam_memory_exchange(
    model: nn.Module,
    pre: dict[str, Tensor],
    actual_delta: dict[str, Tensor],
    alternate_memory: dict[str, dict[str, Tensor]],
    gradients: dict[str, Tensor],
    evaluation: dict[str, Tensor],
    *,
    learning_rate: float,
    betas: tuple[float, float],
    epsilon: float,
    weight_decay: float,
) -> dict[str, float]:
    alternate_delta = adamw_delta(
        pre,
        gradients,
        alternate_memory,
        learning_rate=learning_rate,
        betas=betas,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )
    actual = response_for_state_and_delta(model, pre, actual_delta, evaluation)
    alternate = response_for_state_and_delta(model, pre, alternate_delta, evaluation)
    return {
        "response_nrmse": response_nrmse(actual, alternate),
        "actual_response_rms": float(actual.square().mean().sqrt()),
        "alternate_response_rms": float(alternate.square().mean().sqrt()),
    }
