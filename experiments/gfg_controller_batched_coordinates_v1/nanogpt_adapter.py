"""Existing nanoGPT feature extraction; only aggregation uses the new core."""
import numpy as np
import torch
from torch.func import functional_call
from experiments.gfg_nanogpt_joint_rotation_branch_v1.analysis import HiddenStateModel, _hidden_derivatives
from .kernel import boundary_coordinates


def coordinates(model, parameters, delta, inputs, targets):
    names = tuple(n for n, _ in model.named_parameters())
    assert set(names) == set(parameters) == set(delta)
    params = {n: parameters[n].to(inputs.device) for n in names}
    tangent = {'gpt.' + n: delta[n].to(inputs.device) for n in names}
    prior_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            recorded, _ = functional_call(model, params, (inputs, targets), tie_weights=True, strict=True)
            recorded = recorded[:, -1, :].cpu().numpy().astype(np.float64)
            hs = _hidden_derivatives(HiddenStateModel(model),
                {'gpt.' + n: v for n, v in params.items()}, tangent, inputs)
            hs = [v.detach().cpu().numpy().astype(np.float64) for v in hs]
    finally:
        model.train(prior_mode)
    return boundary_coordinates(recorded, targets[:, -1].cpu().numpy(),
        parameters['transformer.wte.weight'].numpy(),
        delta['transformer.wte.weight'].numpy(), *hs)
