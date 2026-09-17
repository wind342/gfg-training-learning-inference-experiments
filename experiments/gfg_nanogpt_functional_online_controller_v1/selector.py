"""Only pre-state tensors, its own formed delta, and fixed target inputs enter."""
import numpy as np
import torch
from torch.func import functional_call
from experiments.gfg_nanogpt_joint_rotation_branch_v1.analysis import HiddenStateModel, _hidden_derivatives
from experiments.gfg_nanogpt_functional_step_size_smoke_v1.selector import endpoints, choose
from experiments.gfg_nanogpt_functional_step_size_smoke_v1.common import ALPHAS

def coordinates(model, parameters, delta, inputs, targets):
    device = inputs.device
    names = tuple(n for n, _ in model.named_parameters())
    assert set(names) == set(parameters) == set(delta)
    params = {n: parameters[n].to(device) for n in names}
    tangent = {'gpt.' + n: delta[n].to(device) for n in names}
    prior_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            recorded, _ = functional_call(model, params, (inputs, targets), tie_weights=True, strict=True)
            recorded = recorded[:, -1, :].cpu().numpy().astype(np.float64)
            h, hp, hpp = _hidden_derivatives(HiddenStateModel(model), {'gpt.' + n: v for n, v in params.items()}, tangent, inputs)
            hs = [v.detach().cpu().numpy().astype(np.float64) for v in (h, hp, hpp)]
    finally:
        model.train(prior_mode)
    w = parameters['transformer.wte.weight'].numpy().astype(np.float64)
    dw = delta['transformer.wte.weight'].numpy().astype(np.float64)
    y = targets[:, -1].cpu().numpy()
    groups = {k: np.empty((len(y), 23), dtype=np.float64) for k in ('g0','first_gap','joint_rotation','hidden_curvature')}
    competitors = np.empty((len(y),23), dtype=np.int64)
    for i, target in enumerate(y):
        order = np.argsort(recorded[i])[::-1]
        cs = order[order != target]
        competitors[i] = cs
        b, bp = w[target][None,:] - w[cs], dw[target][None,:] - dw[cs]
        groups['g0'][i] = recorded[i,target] - recorded[i,cs]
        # Preserve the prior derivative cache's float32 storage -> float64 load.
        groups['first_gap'][i] = (bp @ hs[0][i] + b @ hs[1][i]).astype(np.float32).astype(np.float64)
        groups['joint_rotation'][i] = (bp @ hs[1][i]).astype(np.float32).astype(np.float64)
        groups['hidden_curvature'][i] = (0.5 * (b @ hs[2][i])).astype(np.float32).astype(np.float64)
    for v in groups.values():
        assert np.isfinite(v).all()
    return {**groups, 'hidden': hs[0], 'hidden_first': hs[1], 'hidden_second': hs[2],
            'competitors': competitors, 'pre_logits': recorded}

def select(coords):
    scores = {}
    for a in ALPHAS:
        m = endpoints(*(coords[k] for k in ('g0','first_gap','joint_rotation','hidden_curvature')), a).min(axis=1)
        scores[str(a)] = {'predicted_correct_count': int((m > 0).sum()),
                          'median_predicted_minimum_margin': float(np.median(m))}
    return choose(scores), scores
