"""Isolated shared-feature harness; old scientific code imported unchanged."""
from experiments.gfg_final_architecture_v2_ten_task.core import (
    p, old, floor_projection, counts, DATA, CHECKPOINT, CKSHA, DATA_SHA,
    FrozenRelationModel, freeze, make_candidate, propose, commit, head_hash,
    moments_hash, tensor_hash, equal, margins, order, scope, summarize_predictions,
    sha, write, torch, nn, np, copy, json, Path,
)
import gzip
import itertools
import platform
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/frozen-head-baseline-comparison-v1')
BASE='0a62264bcbad0a6dae8bd4e755aae96aec205a07'
METHODS={'native_210':('native',1),**{f'{a}_{b}':(a,b//210) for a in ('er','agem','gem','v2') for b in (210,630)}}

def sync(): torch.cuda.synchronize()

def sources():
    paths=[REPO/v for v in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if v.endswith('.py')]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {v.relative_to(REPO).as_posix():sha(v) for v in sorted(set(paths))}

def guards(pre=None):
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)
    assert subprocess.check_output(['git','rev-parse','paper-experiments-cross-system-feedback-release^{}'],cwd=REPO,text=True).strip()=='36dab5ce347dbbdac157ef23205f556606d18294'
    if pre: assert sources()==pre['sources']
    assert shutil.disk_usage(ROOT.parent).free>5*1024**3

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: torch.save(obj,f)

def flat_grad(head):return torch.cat([q.grad.detach().flatten() for q in head.parameters()])

def assign_grad(head,g):
    off=0
    for q in head.parameters():
        q.grad=g[off:off+q.numel()].reshape_as(q).clone();off+=q.numel()
    assert off==g.numel()

def ce_grad(head,h,y,seen):
    head.zero_grad(set_to_none=True)
    loss=nn.functional.cross_entropy(head(h)[:,:seen],y);loss.backward()
    return flat_grad(head),float(loss.detach())

def dual_qp(gram,q):
    """Exact active-set enumeration for <=9 homogeneous constraints."""
    n=len(q);tol=1e-9*max(1.,float(np.max(np.abs(q))))
    for size in range(n+1):
        for subset in itertools.combinations(range(n),size):
            lam=np.zeros(n)
            if subset:
                ix=list(subset);block=gram[np.ix_(ix,ix)]
                sol=np.linalg.lstsq(block,-q[ix],rcond=1e-12)[0]
                if np.max(np.abs(block@sol+q[ix]))>tol or np.min(sol)<-tol:continue
                lam[ix]=np.maximum(sol,0.)
            residual=gram@lam+q
            if np.min(residual)>=-tol and np.max(np.abs(lam*residual))<1e-7*max(1.,np.max(lam)):
                return lam
    raise RuntimeError('GEM dual KKT solution not found; do not execute an unverified projection')

@torch.no_grad()
def project_gradient(g,references,kind):
    # Normalize rows only: homogeneous halfspaces are unchanged.
    R=torch.stack(references).double();R=R/R.norm(dim=1,keepdim=True).clamp_min(1e-30)
    gd=g.double();q=R@gd
    if bool((q>=0).all()):return g,dict(triggered=False,min_dot_before=float(q.min()),min_dot_after=float(q.min()),dual=[])
    if kind=='agem':lam=(-q).clamp_min(0)
    else:
        lam=torch.from_numpy(dual_qp((R@R.T).cpu().numpy(),q.cpu().numpy())).to(gd)
    result=(gd+lam@R).to(g.dtype)
    after=R@result.double()
    assert float(after.min())>=-2e-6*max(1.,float(gd.norm()))
    return result,dict(triggered=True,min_dot_before=float(q.min()),min_dot_after=float(after.min()),dual=lam.cpu().tolist())

@torch.no_grad()
def monitor(head,h,y,n,seen):
    m=margins(head(h)[:,:seen],y)[:n].cpu().numpy()
    return m,counts(m,y[:n].cpu().numpy(),seen-8)

def bank(d,F,task):
    ids,n=old.task_bank(d,task)
    return F[ids],torch.from_numpy(d['y'][ids]).cuda(),ids,n

def make_state(initial):
    head=nn.Linear(1536,80).cuda();head.load_state_dict(initial['head'])
    shadow=copy.deepcopy(head)
    opt=torch.optim.AdamW(shadow.parameters(),lr=5e-5,fused=True,foreach=False,weight_decay=0)
    opt.load_state_dict(copy.deepcopy(initial['optimizer']))
    return head,shadow,opt

def input_ids(d,task):return order(d,task).reshape(-1)

def newly_released(d,task):
    v=old.clean_validation(d)
    return v[(d['y'][v]>=(task-2)*8)&(d['y'][v]<(task-1)*8)] if task>1 else np.array([],dtype=int)

def output_metrics(d,preds,task):return summarize_predictions(scope(d,task*8)[1],preds,d,task)

def runtime():
    import scipy,transformers
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
                transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads())
