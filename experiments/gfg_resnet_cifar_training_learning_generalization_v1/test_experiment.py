from __future__ import annotations

import torch
from torch import nn

from .gfg import CompactGFG
from .model import CifarResNet18, DirectionalBatchNorm2d
from .numeric import morphology, target_margins


def test_resnet_gate_contract() -> None:
    model = CifarResNet18().eval()
    images = torch.randn(2, 3, 32, 32)
    assert model(images).shape == (2, 100)
    assert model(images, (0.0, 1.0, 1.0, 1.0)).shape == (2, 100)


def test_directional_batch_norm_matches_standard_training_state() -> None:
    torch.manual_seed(7)
    reference = nn.BatchNorm2d(5).train()
    directional = DirectionalBatchNorm2d(5).train()
    directional.load_state_dict(reference.state_dict())
    images = torch.randn(8, 5, 4, 4)

    expected = reference(images)
    observed = directional(images)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        directional.running_mean, reference.running_mean, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        directional.running_var, reference.running_var, rtol=0.0, atol=0.0
    )
    assert int(directional.num_batches_tracked) == int(reference.num_batches_tracked)


def test_target_margin_excludes_correct_class() -> None:
    logits = torch.tensor([[3.0, 5.0, 4.0], [2.0, -1.0, 0.0]])
    labels = torch.tensor([1, 2])
    margin, competitor = target_margins(logits, labels)
    assert torch.equal(competitor, torch.tensor([2, 0]))
    assert torch.equal(margin, torch.tensor([1.0, -2.0]))


def test_morphology_rules() -> None:
    alpha = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
    assert morphology(alpha, [0.0, 0.4, 0.7, 0.9, 0.98, 1.0]) == "SATURATING"
    assert morphology(alpha, [0.0, 0.01, 0.03, 0.15, 0.45, 1.0]) == "ACCELERATING"
    assert morphology(alpha, [0.0, 0.3, 0.5, 0.4, 0.1, -0.2]) == "SIGN_REVERSAL"


def test_compact_gfg_has_bound_endpoints() -> None:
    graph = CompactGFG("test")
    source = graph.source("source", {"value": 1})
    occurrence = graph.occurrence("execution", {"value": 2})
    outcome = graph.outcome("result", {"value": 3})
    graph.fact(source, "transform", occurrence, outcome, "input")
    assert graph.validate()["status"] == "PASS"
