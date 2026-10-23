from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp

from .model import CifarVisionMoE, component_parameter_prefix, parameter_block
from .numeric import cosine, morphology, state_sha256, target_margins


ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
COMPONENTS = CifarVisionMoE.component_names


def clone_state(
    model: nn.Module, device: torch.device | str | None = None
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in itertools.chain(model.named_parameters(), model.named_buffers()):
        copied = value.detach().clone()
        result[name] = copied.to(device) if device is not None else copied
    return result


def state_delta(pre: dict[str, Tensor], post: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: post[name] - value if value.is_floating_point() else torch.zeros_like(value)
        for name, value in pre.items()
    }


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
    images: Tensor,
    gates: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    *,
    details: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    return functional_call(model, state, (images, gates, details), strict=True)


def clone_adam_memory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, Tensor]]:
    result: dict[str, dict[str, Tensor]] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        target = parameter.device if device is None else torch.device(device)
        result[name] = {
            "step": torch.as_tensor(state.get("step", 0)).detach().clone().to(target),
            "exp_avg": state.get("exp_avg", torch.zeros_like(parameter)).detach().clone().to(target),
            "exp_avg_sq": state.get("exp_avg_sq", torch.zeros_like(parameter)).detach().clone().to(target),
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


def _selected_indices(margins: Tensor, count: int) -> Tensor:
    if count >= margins.numel():
        return torch.arange(margins.numel(), device=margins.device)
    order = torch.argsort(margins)
    bins = torch.tensor_split(order, 4)
    selected: list[int] = []
    used: set[int] = set()
    quota = max(1, count // 4)
    for group in bins:
        for item in group[:quota].tolist():
            selected.append(int(item))
            used.add(int(item))
    for item in torch.argsort(margins.abs()).tolist():
        if int(item) not in used:
            selected.append(int(item))
            used.add(int(item))
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
        result = forward_state(model, add_delta(pre, delta, value), images)
        assert isinstance(result, Tensor)
        return result

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
    memory: dict[str, dict[str, Tensor]],
) -> tuple[list[str], list[float], list[str], list[float]]:
    blocks = ("stem", "pre_block", "router", "experts", "readout")
    grouped: dict[str, dict[str, list[Tensor]]] = defaultdict(
        lambda: {"parameter": [], "update": [], "m": [], "v": []}
    )
    for name, old in memory.items():
        block = parameter_block(name)
        if block not in blocks:
            continue
        grouped[block]["parameter"].append(state[name].reshape(-1))
        grouped[block]["update"].append(delta[name].reshape(-1))
        grouped[block]["m"].append(old["exp_avg"].reshape(-1))
        grouped[block]["v"].append(old["exp_avg_sq"].reshape(-1))
    update_norms: dict[str, float] = {}
    for block in blocks:
        update = torch.cat(grouped[block]["update"])
        update_norms[block] = float(torch.linalg.vector_norm(update.double()))
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
    images: Tensor,
    labels: Tensor,
) -> Tensor:
    values: list[Tensor] = []
    with torch.no_grad():
        for mask in range(16):
            gates = tuple(float((mask >> index) & 1) for index in range(4))
            logits = forward_state(model, state, images, gates)
            assert isinstance(logits, Tensor)
            values.append(target_margins(logits, labels)[0])
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
    for left in range(4):
        for right in range(left + 1, 4):
            full = 15
            interaction = (
                pre_values[full]
                - pre_values[full ^ (1 << left)]
                - pre_values[full ^ (1 << right)]
                + pre_values[full ^ (1 << left) ^ (1 << right)]
            )
            pair_interactions[f"{COMPONENTS[left]}+{COMPONENTS[right]}"] = (
                interaction.detach().cpu().tolist()
            )
    return {
        "repeat_max_abs_error": float((pre_values - repeated).abs().max()),
        "pre_shapley": pre_shapley.detach().cpu().tolist(),
        "post_shapley": post_shapley.detach().cpu().tolist(),
        "reallocation_l1_normalized": reallocation.detach().cpu().tolist(),
        "distributed_active_component_count": active.detach().cpu().tolist(),
        "primary_support_switch": switches.detach().cpu().tolist(),
        "pair_interactions_pre": pair_interactions,
    }


def _routing_record(
    model: nn.Module,
    pre: dict[str, Tensor],
    post: dict[str, Tensor],
    images: Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        pre_result = forward_state(model, pre, images, details=True)
        post_result = forward_state(model, post, images, details=True)
    assert isinstance(pre_result, tuple) and isinstance(post_result, tuple)
    pre_route = pre_result[1]["routing_weights"]
    post_route = post_result[1]["routing_weights"]
    pre_profile = pre_route.mean(dim=1)
    post_profile = post_route.mean(dim=1)
    query_l1 = torch.cdist(pre_profile.double(), pre_profile.double(), p=1)
    return {
        "minimum_nonzero_experts_per_token": int(pre_route.gt(0).sum(dim=-1).min()),
        "maximum_nonzero_experts_per_token": int(pre_route.gt(0).sum(dim=-1).max()),
        "maximum_weight_sum_error": float((pre_route.sum(dim=-1) - 1).abs().max()),
        "mean_pre_profile": pre_profile.mean(dim=0).cpu().tolist(),
        "mean_post_profile": post_profile.mean(dim=0).cpu().tolist(),
        "maximum_query_profile_l1": float(query_l1.max()),
        "mean_profile_reallocation_l1": float(
            (post_profile - pre_profile).abs().sum(dim=1).mean()
        ),
        "top2_identity_change_fraction": float(
            pre_route.gt(0).ne(post_route.gt(0)).any(dim=-1).float().mean()
        ),
    }


def analyze_update(
    *,
    model: nn.Module,
    pre: dict[str, Tensor],
    post: dict[str, Tensor],
    adam_pre: dict[str, dict[str, Tensor]],
    anchor_images: Tensor,
    anchor_labels: Tensor,
    anchor_ids: Tensor,
    target_count: int,
) -> dict[str, Any]:
    model.eval()
    delta = state_delta(pre, post)
    with torch.no_grad():
        logits = forward_state(model, pre, anchor_images)
        assert isinstance(logits, Tensor)
        margins = target_margins(logits, anchor_labels)[0]
    selected = _selected_indices(margins, target_count)
    images = anchor_images[selected]
    labels = anchor_labels[selected]
    identities = anchor_ids[selected]
    logits0, first, second = _directional_logits(model, pre, delta, images)
    response_logits: list[Tensor] = []
    with torch.no_grad():
        for alpha in ALPHAS:
            state = pre if alpha == 0.0 else post if alpha == 1.0 else add_delta(pre, delta, alpha)
            value = forward_state(model, state, images)
            assert isinstance(value, Tensor)
            response_logits.append(value)
    logits_path = torch.stack(response_logits)
    margins_path = torch.stack([target_margins(value, labels)[0] for value in response_logits])
    native_post = forward_state(model, post, images)
    assert isinstance(native_post, Tensor)
    linear = logits0 + first
    quadratic = linear + 0.5 * second
    pre_correct = _correct(logits0, labels)
    post_correct = _correct(response_logits[-1], labels)
    pre_margin, competitor = target_margins(logits0, labels)
    first_margin = first.gather(1, labels[:, None]).squeeze(1) - first.gather(1, competitor[:, None]).squeeze(1)
    second_margin = second.gather(1, labels[:, None]).squeeze(1) - second.gather(1, competitor[:, None]).squeeze(1)
    f3_names, f3_shared, f5_names, f5_shared = _block_geometry(pre, delta, adam_pre)
    records: list[dict[str, Any]] = []
    for index in range(images.shape[0]):
        path = margins_path[:, index].detach().cpu().tolist()
        chord = [path[0] + alpha * (path[-1] - path[0]) for alpha in ALPHAS]
        maximum = max(abs(left - right) for left, right in zip(path, chord))
        endpoint = path[-1] - path[0]
        probabilities = torch.softmax(logits0[index], dim=0)
        f1_names = ["F1_margin", "F1_correct", "F1_correct_logit", "F1_competitor_logit", "F1_logit_std", "F1_entropy"]
        f1_values = [
            float(pre_margin[index]),
            float(pre_correct[index]),
            float(logits0[index, labels[index]]),
            float(logits0[index, competitor[index]]),
            float(logits0[index].std()),
            float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()),
        ]
        target_f3_names = ["F3_first_margin_direction", "F3_second_margin_direction", "F3_first_correct_logit_direction", "F3_first_competitor_logit_direction"]
        target_f3 = [
            float(first_margin[index]),
            float(second_margin[index]),
            float(first[index, labels[index]]),
            float(first[index, competitor[index]]),
        ]
        names = f1_names + target_f3_names + f3_names + f5_names
        values = f1_values + target_f3 + f3_shared + f5_shared
        records.append(
            {
                "target_id": int(identities[index]),
                "label": int(labels[index]),
                "pre_competitor": int(competitor[index]),
                "pre_margin": path[0],
                "post_margin": path[-1],
                "margin_path": path,
                "morphology": morphology(list(ALPHAS), path),
                "max_chord_deviation": maximum,
                "normalized_chord_deviation": maximum / max(abs(endpoint), 1e-6),
                "pre_correct": bool(pre_correct[index]),
                "post_correct": bool(post_correct[index]),
                "transition": _transition(bool(pre_correct[index]), bool(post_correct[index])),
                "predictions": {
                    "unchanged": bool(pre_correct[index]),
                    "linear": bool(_correct(linear, labels)[index]),
                    "quadratic": bool(_correct(quadratic, labels)[index]),
                },
                "feature_names": names,
                "features": values,
                "feature_families": {
                    "F1": [0, len(f1_names)],
                    "F3": [len(f1_names), len(f1_names) + len(target_f3_names) + len(f3_names)],
                    "F5": [len(f1_names) + len(target_f3_names) + len(f3_names), len(names)],
                },
            }
        )
    return {
        "pre_state_sha256": state_sha256(pre),
        "post_state_sha256": state_sha256(post),
        "delta_state_sha256": state_sha256(delta),
        "selected_target_ids": identities.detach().cpu().tolist(),
        "alpha_grid": list(ALPHAS),
        "alpha0_logit_max_abs_error": float((logits_path[0] - logits0).abs().max()),
        "alpha1_native_logit_max_abs_error": float((response_logits[-1] - native_post).abs().max()),
        "records": records,
        "support": _support_record(model, pre, post, images, labels),
        "routing": _routing_record(model, pre, post, images),
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
    def response(state: dict[str, Tensor], delta: dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            before = forward_state(model, state, images)
            after = forward_state(model, add_delta(state, delta, 1.0), images)
            assert isinstance(before, Tensor) and isinstance(after, Tensor)
            return target_margins(after, labels)[0] - target_margins(before, labels)[0]

    aa, ba = response(state_a, delta_a), response(state_b, delta_a)
    bb, ab = response(state_b, delta_b), response(state_a, delta_b)

    def compare(native: Tensor, exchanged: Tensor) -> dict[str, float]:
        difference = native - exchanged
        rms = float(torch.sqrt(torch.mean(difference.double().square())))
        denominator = float(torch.sqrt(torch.mean(native.double().square())))
        return {
            "response_rmse": rms,
            "response_nrmse": rms / max(denominator, 1e-12),
            "direction_disagreement_rate": float(torch.sign(native).ne(torch.sign(exchanged)).float().mean()),
        }

    return {
        "update_a_state_a_vs_state_b": compare(aa, ba),
        "update_b_state_b_vs_state_a": compare(bb, ab),
    }


def memory_exchange_response(
    *,
    model: nn.Module,
    state: dict[str, Tensor],
    native_delta: dict[str, Tensor],
    exchanged_delta: dict[str, Tensor],
    images: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        baseline = forward_state(model, state, images)
        native = forward_state(model, add_delta(state, native_delta, 1.0), images)
        exchanged = forward_state(model, add_delta(state, exchanged_delta, 1.0), images)
        assert isinstance(baseline, Tensor) and isinstance(native, Tensor) and isinstance(exchanged, Tensor)
        base_margin = target_margins(baseline, labels)[0]
        native_response = target_margins(native, labels)[0] - base_margin
        exchanged_response = target_margins(exchanged, labels)[0] - base_margin
    difference = native_response - exchanged_response
    rms = float(torch.sqrt(torch.mean(difference.double().square())))
    denominator = float(torch.sqrt(torch.mean(native_response.double().square())))
    return {
        "response_rmse": rms,
        "response_nrmse": rms / max(denominator, 1e-12),
        "direction_disagreement_rate": float(torch.sign(native_response).ne(torch.sign(exchanged_response)).float().mean()),
    }


def inference_projection_tests(
    *,
    model: nn.Module,
    trained: dict[str, Tensor],
    initial: dict[str, Tensor],
    images: Tensor,
    labels: Tensor,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        first = forward_state(model, trained, images, details=True)
        repeated = forward_state(model, trained, images, details=True)
    assert isinstance(first, tuple) and isinstance(repeated, tuple)
    logits, details = first
    repeat_logits, repeat_details = repeated
    route = details["routing_weights"]
    profile = route.mean(dim=1)
    query_l1 = torch.cdist(profile.double(), profile.double(), p=1)
    contribution_rms = details["expert_contributions"].square().mean(dim=(0, 1, 3)).sqrt()
    utilization = route.gt(0).float().mean(dim=(0, 1))
    coalition_logits: list[Tensor] = []
    with torch.no_grad():
        for mask in range(16):
            gates = tuple(float((mask >> index) & 1) for index in range(4))
            value = forward_state(model, trained, images, gates)
            assert isinstance(value, Tensor)
            coalition_logits.append(value)
    margins = torch.stack([target_margins(value, labels)[0] for value in coalition_logits])
    full = 15
    gate_effects = {
        component: float((coalition_logits[full] - coalition_logits[full ^ (1 << index)]).abs().max())
        for index, component in enumerate(COMPONENTS)
    }
    interactions: dict[str, float] = {}
    for left in range(4):
        for right in range(left + 1, 4):
            interaction = margins[full] - margins[full ^ (1 << left)] - margins[full ^ (1 << right)] + margins[full ^ (1 << left) ^ (1 << right)]
            interactions[f"{COMPONENTS[left]}+{COMPONENTS[right]}"] = float(interaction.abs().max())
    rollbacks: dict[str, dict[str, Any]] = {}
    for component in (*COMPONENTS, "router"):
        prefix = component_parameter_prefix(component)
        rolled = {name: value for name, value in trained.items()}
        replaced = []
        for name in trained:
            if name.startswith(prefix):
                rolled[name] = initial[name]
                replaced.append(name)
        with torch.no_grad():
            value = forward_state(model, rolled, images)
        assert isinstance(value, Tensor)
        rollbacks[component] = {
            "replaced_parameter_names": replaced,
            "logit_max_abs_change": float((value - logits).abs().max()),
        }
    restored = forward_state(model, trained, images)
    assert isinstance(restored, Tensor)
    tests = {
        "repeat_inference_exact": float((logits - repeat_logits).abs().max()) <= thresholds["repeat_output_tolerance"],
        "repeat_routing_exact": float((route - repeat_details["routing_weights"]).abs().max()) <= thresholds["repeat_output_tolerance"],
        "deterministic_top2_routing": int(route.gt(0).sum(dim=-1).min()) == 2 and int(route.gt(0).sum(dim=-1).max()) == 2 and float((route.sum(dim=-1) - 1).abs().max()) <= thresholds["router_topk_tolerance"],
        "all_experts_recruited": bool(torch.all(utilization >= thresholds["expert_utilization_minimum"])) and bool(torch.all(contribution_rms > 0)),
        "query_conditioned_routing": float(query_l1.max()) > thresholds["query_profile_l1_minimum"],
        "gate_changes_output": min(gate_effects.values()) > thresholds["gate_effect_minimum"],
        "nonadditive_combination": max(interactions.values()) > thresholds["pair_interaction_minimum"],
        "learned_version_dependence": min(row["logit_max_abs_change"] for row in rollbacks.values()) > thresholds["rollback_effect_minimum"],
        "exact_restoration": float((restored - logits).abs().max()) <= thresholds["restoration_tolerance"],
    }
    return {
        "trained_state_sha256": state_sha256(trained),
        "initial_state_sha256": state_sha256(initial),
        "repeat_output_max_abs_error": float((logits - repeat_logits).abs().max()),
        "repeat_routing_max_abs_error": float((route - repeat_details["routing_weights"]).abs().max()),
        "minimum_nonzero_experts_per_token": int(route.gt(0).sum(dim=-1).min()),
        "maximum_nonzero_experts_per_token": int(route.gt(0).sum(dim=-1).max()),
        "maximum_weight_sum_error": float((route.sum(dim=-1) - 1).abs().max()),
        "expert_utilization": dict(zip(COMPONENTS, utilization.cpu().tolist())),
        "expert_contribution_rms": dict(zip(COMPONENTS, contribution_rms.cpu().tolist())),
        "maximum_query_profile_l1": float(query_l1.max()),
        "single_gate_complete_output_effects": gate_effects,
        "pair_interactions": interactions,
        "rollbacks": rollbacks,
        "maximum_restoration_output_error": float((restored - logits).abs().max()),
        "tests": tests,
        "pass": all(tests.values()),
    }
