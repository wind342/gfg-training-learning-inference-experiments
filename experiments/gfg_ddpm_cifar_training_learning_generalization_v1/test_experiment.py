from __future__ import annotations

import copy

import torch

from .analysis import adamw_delta, clone_adam_memory, clone_state, state_delta
from .boundary import residual_boundary, transition
from .model import CifarDiffusionUNet


def test_model_shape_and_gates() -> None:
    model = CifarDiffusionUNet(base_channels=8, time_dimension=32)
    noisy = torch.randn(3, 3, 32, 32)
    timesteps = torch.tensor([1, 10, 50])
    full = model(noisy, timesteps)
    gated = model(noisy, timesteps, (0.0, 1.0, 1.0, 1.0))
    assert full.shape == noisy.shape
    assert gated.shape == noisy.shape


def test_residual_boundary() -> None:
    prediction = torch.zeros(2, 3, 2, 2)
    truth = torch.zeros_like(prediction)
    candidates = torch.ones(2, 3, 3, 2, 2)
    margin, competitor, true_error, candidate_errors, pair = residual_boundary(
        prediction, truth, candidates
    )
    assert margin.shape == (2,)
    assert competitor.shape == (2,)
    assert torch.all(true_error == 0)
    assert torch.all(candidate_errors > 0)
    assert torch.all(pair > 0)
    assert transition(True, False) == "CORRECT_TO_WRONG"


def test_manual_adamw_matches_native() -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        foreach=False,
        fused=False,
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(8, 5)).square().mean().backward()
        optimizer.step()
    pre = clone_state(model)
    memory = clone_adam_memory(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(8, 5)).square().mean().backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    predicted = adamw_delta(
        pre,
        gradients,
        memory,
        learning_rate=2e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=1e-4,
    )
    optimizer.step()
    actual = state_delta(pre, clone_state(model))
    assert max(float((actual[name] - predicted[name]).abs().max()) for name in gradients) < 1e-7


def test_optimizer_memory_changes_update() -> None:
    torch.manual_seed(9)
    first = torch.nn.Linear(3, 2)
    second = copy.deepcopy(first)
    left = torch.optim.AdamW(first.parameters(), lr=1e-3, foreach=False, fused=False)
    right = torch.optim.AdamW(second.parameters(), lr=1e-3, foreach=False, fused=False)
    for sign, model, optimizer in ((1.0, first, left), (-1.0, second, right)):
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            (sign * model(torch.ones(5, 3))).mean().backward()
            optimizer.step()
    left_memory = clone_adam_memory(first, left)
    right_memory = clone_adam_memory(second, right)
    name = next(iter(left_memory))
    assert not torch.equal(left_memory[name]["exp_avg"], right_memory[name]["exp_avg"])
