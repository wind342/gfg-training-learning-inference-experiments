"""Global clipping delegates DTensor reductions; no independent rank clipping."""
import torch


def clip(registry,model):
    parameters=list(registry.resolve(model).values())
    if any(hasattr(p,'placements') and (len(p.placements)!=1 or not p.placements[0].is_shard(0)) for p in parameters):
        raise ValueError('only pure one-dimensional non-replicated FSDP ownership is certified')
    return torch.nn.utils.clip_grad_norm_(parameters,1.)
