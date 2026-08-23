from __future__ import annotations

import torch

from .model import CifarVisionMoE
from .numeric import target_margins


def test_moe_gate_and_routing_contract() -> None:
    torch.manual_seed(7)
    model = CifarVisionMoE().eval()
    images = torch.randn(3, 3, 32, 32)
    logits, details = model(images, return_details=True)
    assert logits.shape == (3, 100)
    routed = details["routing_weights"]
    assert routed.shape == (3, 64, 4)
    assert torch.equal(routed.gt(0).sum(dim=-1), torch.full((3, 64), 2))
    torch.testing.assert_close(routed.sum(dim=-1), torch.ones(3, 64))
    gated = model(images, (0.0, 1.0, 1.0, 1.0))
    assert not torch.equal(logits, gated)


def test_target_margin_contract() -> None:
    logits = torch.tensor([[2.0, 5.0, 4.0], [1.0, -1.0, 0.5]])
    labels = torch.tensor([1, 2])
    margin, competitor = target_margins(logits, labels)
    assert torch.equal(competitor, torch.tensor([2, 0]))
    assert torch.equal(margin, torch.tensor([1.0, -0.5]))
