"""Independent head-only runtime; original scientific implementations untouched."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.gfg_governor_fewrel_v1.model import RelationModel
from experiments.gfg_router_v4_ten_task_validation.common import (
    DATA,CHECKPOINT,CKSHA,DATA_SHA,scope,order,summarize_predictions,sha,write,data_audit,
)
from experiments.gfg_update_direction_isolation_smoke_v1.core import extend,tensor_hash

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/head-only-exact-monitor-smoke-v1')
BASE='e84014c2417e631025e9a8dc1b9a69d23997856c'
ARMS=('head_only','head_only_exact_monitor')

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    paths=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(paths))}

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class FrozenRelationModel(RelationModel):
    def train(self,mode=True):
        super().train(mode)
        self.bert.eval()
        return self

def freeze(model):
    for p in model.bert.parameters():p.requires_grad_(False);p.grad=None
    model.bert.eval()

def optimizer(parameters):
    return torch.optim.AdamW(parameters,lr=5e-6,betas=(.9,.999),weight_decay=0,fused=True,foreach=False)

def make_candidate(model,saved_optimizer):
    """Load by the same named-parameter order as original full-model AdamW."""
    shadow=copy.deepcopy(model.head)
    opt=optimizer(shadow.parameters())
    source_ids=[i for g in saved_optimizer['param_groups'] for i in g['params']]
    names=[n for n,p in model.named_parameters()]
    assert len(names)==len(source_ids) and len(saved_optimizer['param_groups'])==1
    source={n:saved_optimizer['state'].get(i,{}) for n,i in zip(names,source_ids)}
    group=copy.deepcopy(saved_optimizer['param_groups'][0]);group['params']=[0,1]
    state={'state':{j:copy.deepcopy(source['head.'+n]) for j,n in enumerate(('weight','bias'))},'param_groups':[group]}
    opt.load_state_dict(state)
    assert equal(opt.state_dict(),state)
    assert opt.param_groups[0]['lr']==5e-6 and opt.param_groups[0]['weight_decay']==0
    return shadow,opt

@torch.no_grad()
def propose(head,shadow,opt):
    for p,q in zip(head.parameters(),shadow.parameters()):
        q.copy_(p);q.grad=None if p.grad is None else p.grad.detach().clone()
    opt.step()

@torch.no_grad()
def commit(head,shadow):
    for p,q in zip(head.parameters(),shadow.parameters()):p.copy_(q)

@torch.no_grad()
def logits(head,h,seen=16):return nn.functional.linear(h,head.weight,head.bias)[:,:seen]

def margins(z,y):
    other=z.clone().scatter_(1,y[:,None],-torch.inf)
    return z.gather(1,y[:,None])[:,0]-other.amax(1)

def numbers(m):
    a=np.asarray(m,dtype=np.float64).reshape(16,10)
    return dict(capability=(a>0).mean(1).tolist(),q10=np.quantile(a,.1,axis=1,method='linear').tolist(),
                correct_counts=(a>0).sum(1).tolist(),old_correct=int((a[:8]>0).sum()),new_correct=int((a[8:]>0).sum()))

def classify(before,after,anchor,protected):
    def unsafe(v):return {g for g in range(8) if v['capability'][g]<.9 or (anchor[g]>=1e-6 and v['q10'][g]<.5*anchor[g])}
    old,new=unsafe(before),unsafe(after)
    return dict(already_unsafe_before_update=sorted(old),newly_threatened_by_this_update=sorted(new-old),
        newly_threatened_protected=sorted((new-old)&set(protected)),candidate_unsafe=sorted(new))

def observe(s,baseline,candidate,step):
    s.last_step=step;s.measured(step,baseline)
    receiving=s.snapshot(step);anchor=s.anchor.copy();protected=[k for k in range(8) if s.state[k]=='PROTECTED']
    b=baseline.detach().cpu().numpy();c=candidate.detach().cpu().numpy();bv=numbers(b);cv=numbers(c)
    report=dict(baseline=b.tolist(),candidate=c.tolist(),baseline_stats=bv,candidate_stats=cv,
                anchor=anchor.tolist(),protected=protected,receiving=receiving,**classify(bv,cv,anchor,protected))
    for k,ids in enumerate(s.mapping.members):s._observe_group(k,c[list(ids)],'endpoint_observation')
    report['after']=s.snapshot(step)
    return report

def rng():return (torch.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()])

def head_hash(head):return tensor_hash(head.state_dict())

def moments_hash(opt):
    return tensor_hash({str(i)+':'+k:v for i,s in opt.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)})
