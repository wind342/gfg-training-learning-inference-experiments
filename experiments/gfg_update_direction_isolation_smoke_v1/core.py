"""Small candidate generator and execution adapter; no edits to old controllers."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from experiments.gfg_router_v4_ten_task_validation.common import (
    DATA,CHECKPOINT,CKSHA,DATA_SHA,scope,order,summarize_predictions,sha,write,data_audit,
)
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/update-direction-isolation-smoke-v1')
BASE='f35b595c4e4582bd3a52e938b29e2cf253446d25'
ARMS=('native','scalar','projection')
ALPHAS=(.125,.25,.5,1.)
SCHEDULE=tuple(range(1,211,10))
EPS=1e-12

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    files=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments/*.py','experiments/**/*.py'],cwd=REPO,text=True).splitlines()]
    files+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(files))}

def extend(history):
    s=GroupSentinel(GroupMap(np.repeat(np.arange(16),10)))
    for field in ('state','level','anchor','capabilities','q10','last_observed'):
        getattr(s,field)[:8]=copy.deepcopy(getattr(history,field))
    s.last_step=210
    return s

def eligible(s):return [k for k in range(8) if s.state[k]=='PROTECTED']

def threats(margins,s,protected):
    v=s.mapping.values(margins)
    return [k for k in protected if v['capabilities'][k]<s.threshold or
            (s.anchor[k]>=s.absolute_floor and v['q10'][k]<s.erosion_ratio*s.anchor[k])]

def choose(predicted,s,protected):
    scores={str(a):threats(predicted[j],s,protected) for j,a in enumerate(ALPHAS)}
    safe=[a for a in reversed(ALPHAS) if not scores[str(a)]]
    return (safe[0] if safe else .125),not bool(safe),scores

def dot(a,b):
    values=[torch.sum(a[n].double()*b[n].double()) for n in a]
    return float(torch.stack(values).sum())

@torch.no_grad()
def cut(delta,gradient):
    before=dot(gradient,delta);norm2=dot(gradient,gradient)
    coeff=before/(norm2+EPS) if before<0 else 0.
    if coeff:
        for n in delta:delta[n].add_(gradient[n],alpha=-coeff)
    return dict(dot_before=before,gradient_norm_squared=norm2,coefficient=coeff,
                dot_after=dot(gradient,delta),changed=bool(coeff))

@torch.no_grad()
def commit_projected(parameters,delta,alpha):
    """Do not call legacy alpha1 commit: its candidate is the native endpoint."""
    for n,p in parameters.items():p.add_(delta[n],alpha=alpha)

def moment_identity(opt):
    return [(id(v),v.data_ptr(),v._version) for state in opt.state.values() for v in state.values() if torch.is_tensor(v)]

def tensor_hash(values):
    h=hashlib.sha256()
    for n,v in sorted(values.items()):
        h.update(n.encode());h.update(str((tuple(v.shape),v.dtype)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def geometry(native,projected):
    a=b=d=inner=0.
    for n,x in native.items():
        x=x.double();y=projected[n].detach().cpu().double()
        a+=float(x.square().sum());b+=float(y.square().sum());d+=float((y-x).square().sum());inner+=float((x*y).sum())
    return dict(native_norm=a**.5,projected_norm=b**.5,norm_ratio=(b/a)**.5 if a else None,
        cosine=inner/(a*b)**.5 if a and b else None,removed_norm_fraction=(d/a)**.5 if a else None)

def margin_gradient(model,px,py,indices):
    modes=[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.random.fork_rng(devices=[px.device.index] if px.is_cuda else []):
            with torch.no_grad():
                z=model(px[indices])[0][:,-1];target=py[indices,-1]
                other=z.clone().scatter_(1,target[:,None],-torch.inf)
                margins=z.gather(1,target[:,None])[:,0]-other.amax(1)
                at=int(margins.argmin());idx=int(indices[at]);competitor=int(other[at].argmax());label=int(target[at])
            z=model(px[idx:idx+1])[0][0,-1]
            margin=z[label]-z[competitor]
            parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
            grads=torch.autograd.grad(margin,tuple(parameters.values()),allow_unused=True)
            result={n:g if g is not None else torch.zeros_like(parameters[n]) for n,g in zip(parameters,grads)}
            return result,dict(probe_index=idx,target=label,competitor=competitor,receiving_margin=float(margin.detach()))
    finally:
        for m,mode in modes:m.training=mode

def aggregates(margins,mapping,baseline):
    v=mapping.values(margins);b=mapping.values(baseline)
    return dict(old_correct=sum(v['correct_counts'][:8]),new_correct=sum(v['correct_counts'][8:]),
        old_macro=float(np.mean(v['capabilities'][:8])),new_macro=float(np.mean(v['capabilities'][8:])),
        newly_formed_new_groups=sum(x<.9 and y>=.9 for x,y in zip(b['capabilities'][8:],v['capabilities'][8:])),
        groups=v)
