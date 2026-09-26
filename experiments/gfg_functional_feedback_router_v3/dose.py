"""Equal weight for each actual training example; sequential microbatch graphs."""
import numpy as np
import torch

def weights(batches):
    if not batches or any(len(ids)==0 for ids in batches):raise ValueError('empty training microbatch')
    total=sum(len(ids) for ids in batches)
    return [len(ids)/total for ids in batches]

def accumulate(model,opt,x,y,batches,seed):
    coefficients=weights(batches)
    opt.zero_grad(set_to_none=True);model.train();losses=[]
    for j,(ids,w) in enumerate(zip(batches,coefficients)):
        torch.manual_seed(seed+j*10000000)
        if x.device.type=='cuda':torch.cuda.manual_seed_all(seed+j*10000000)
        device=next(model.parameters()).device
        loss=model(x[ids].to(device),y[ids].to(device))[1]
        (loss*w).backward();losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not np.isfinite(losses).all() or not torch.isfinite(norm):raise FloatingPointError('nonfinite training')
    return losses
