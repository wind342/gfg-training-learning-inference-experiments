"""Exact cached affine checks and analytic head-only directional cuts."""
import time
from experiments.gfg_head_only_lr_smoke_v1.core import *
from experiments.gfg_head_only_lr_smoke_v1.core import ROOT as LR_ROOT
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/head-only-exact-cut-smoke-v1')
BASE='3f7a592a69947d944027c4761f99f7544119ed4a'
ARM='head_only_exact_cut'
LR=5e-5

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    paths=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(paths))}

def guard():
    unchanged();pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_hashes']==sources()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    for p,h in pre['baseline_files'].items():assert sha(p)==h
    return pre

def snapshot(head):return {n:p.detach().clone() for n,p in head.state_dict().items()}
def cpu_state(state):return {n:p.detach().cpu().clone() for n,p in state.items()}
@torch.no_grad()
def affine(state,h,seen=16):return nn.functional.linear(h,state['weight'],state['bias'])[:,:seen]

def violated(m,eligible):return torch.nonzero(eligible&(m[:len(eligible)]<=0),as_tuple=False).flatten().cpu().tolist()

@torch.no_grad()
def projection_pass(delta,h,y,z,eligible):
    pairs=[]
    for i in range(len(eligible)):
        if not bool(eligible[i]):continue
        target=int(y[i]);competitors=torch.nonzero(z[i]>=z[i,target],as_tuple=False).flatten().cpu().tolist()
        for c in competitors:
            if c!=target:pairs.append((i,target,c))
    cert=[]
    for i,t,c in pairs:
        hd=h[i].double();dot=float(torch.dot(hd,delta['weight'][t].double()-delta['weight'][c].double())+delta['bias'][t].double()-delta['bias'][c].double())
        norm2=float(2*(hd.square().sum()+1));coefficient=dot/norm2 if dot<0 else 0.
        if coefficient:
            delta['weight'][t].add_(h[i],alpha=-coefficient);delta['weight'][c].add_(h[i],alpha=coefficient)
            delta['bias'][t].add_(-coefficient);delta['bias'][c].add_(coefficient)
        after=float(torch.dot(hd,delta['weight'][t].double()-delta['weight'][c].double())+delta['bias'][t].double()-delta['bias'][c].double())
        cert.append(dict(control=i,target=t,competitor=c,dot_before=dot,norm2=norm2,coefficient=coefficient,dot_after=after))
    return cert

def sync(h):
    if h.is_cuda:torch.cuda.synchronize()

@torch.no_grad()
def exact_cut(before,native,h,y,old_count=80,seen=16):
    sync(h);start=time.perf_counter()
    bz=affine(before,h,seen);nz=affine(native,h,seen);bm=margins(bz,y);nm=margins(nz,y)
    eligible=bm[:old_count]>0;initial=violated(nm,eligible);passes=[];candidate=native
    cut_seconds=0.;recheck_seconds=0.
    if initial:
        delta={k:native[k]-before[k] for k in before};z=nz
        for j in range(2):
            sync(h);tick=time.perf_counter();cert=projection_pass(delta,h,y,z,eligible);sync(h);cut_seconds+=time.perf_counter()-tick
            tick=time.perf_counter();candidate={k:before[k]+delta[k] for k in before}
            z=affine(candidate,h,seen);cm=margins(z,y);bad=violated(cm,eligible);sync(h);recheck_seconds+=time.perf_counter()-tick
            passes.append(dict(pass_number=j+1,certificates=cert,margins=cm.cpu().tolist(),violating=bad,safe=not bad))
            if not bad:break
    else:cm=nm
    skip=bool(violated(cm,eligible));executed=before if skip else candidate
    sync(h)
    return executed,candidate,dict(baseline=bm.cpu().tolist(),native=nm.cpu().tolist(),cut=cm.cpu().tolist(),
        eligible=torch.nonzero(eligible).flatten().cpu().tolist(),native_violating=initial,passes=passes,skip=skip,
        action='CUT_FAILED_SKIP' if skip else 'PROJECTED' if initial else 'NATIVE',
        first_pass_safe=passes[0]['safe'] if passes else None,second_pass=len(passes)==2,
        projection_seconds=cut_seconds,recheck_seconds=recheck_seconds,filter_seconds=time.perf_counter()-start)

def geometry(before,native,other):
    a=torch.cat([(native[k]-before[k]).double().flatten() for k in before]);b=torch.cat([(other[k]-before[k]).double().flatten() for k in before])
    an=float(a.norm());bn=float(b.norm())
    return dict(native_norm=an,cut_norm=bn,norm_ratio=bn/an if an else None,
        cosine=float(torch.dot(a,b))/(an*bn) if an and bn else None,removed_norm_fraction=float((a-b).norm())/an if an else None)

def counts(m):
    v=np.array(m).reshape(16,10)>0
    return dict(old_correct=int(v[:8].sum()),new_correct=int(v[8:].sum()),per_relation_capability=v.mean(1).tolist())
