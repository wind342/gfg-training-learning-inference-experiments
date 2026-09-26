"""Sparse dual-coordinate projection; no external optimizer/solver."""
from experiments.gfg_head_only_coverage_cut_smoke_v1.core import *
from experiments.gfg_head_only_coverage_cut_smoke_v1.core import ROOT as COVER_ROOT

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/head-only-hildreth-smoke-v1')
BASE='9e0dc71c7692fa16ebeb81dbf73197aad93add4b'
ARM='140_controls_hildreth'
EPS=1e-6
MAX_SWEEPS=100

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    p=[REPO/s for s in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if s.endswith('.py')]
    p+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {v.relative_to(REPO).as_posix():sha(v) for v in sorted(set(p))}

def guard():
    unchanged();p=json.loads((ROOT/'PREFLIGHT.json').read_text())
    assert p['status']=='PASS' and p['source_hashes']==sources()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    for name,h in p['baseline_files'].items():assert sha(name)==h
    return p

def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def augmented(state,seen=16):
    return np.concatenate([state['weight'][:seen].detach().cpu().numpy().astype(np.float64),
                           state['bias'][:seen,None].detach().cpu().numpy().astype(np.float64)],axis=1)

def coordinate(P,H,norm2,i,t,c,lam):
    gap=float(np.dot(H[i],P[t]-P[c]))
    new=max(0.,lam+(EPS-gap)/float(norm2[i]));a=new-lam
    if a:
        P[t]+=a*H[i];P[c]-=a*H[i]
    return new

@torch.no_grad()
def materialize(P,native):
    result={k:v.clone() for k,v in native.items()};seen=len(P)
    result['weight'][:seen].copy_(torch.from_numpy(P[:,:-1]).to(native['weight']))
    result['bias'][:seen].copy_(torch.from_numpy(P[:,-1]).to(native['bias']))
    return result

def failing_pairs(z,y,eligible,eps=EPS):
    # Subtraction is in the actual FP32 forward dtype, not an FP64 surrogate.
    zz=z[:len(eligible)];yy=y[:len(eligible)]
    gaps=zz.gather(1,yy[:,None])-zz
    bad=(gaps.double()<eps)&eligible[:,None]
    bad.scatter_(1,yy[:,None],False)
    return [tuple(v) for v in torch.nonzero(bad,as_tuple=False).cpu().tolist()]

@torch.no_grad()
def hildreth(before,native,h,y,old_count=1120,seen=16):
    sync(h);started=time.perf_counter()
    bz=affine(before,h,seen);nz=affine(native,h,seen)
    bm=margins(bz,y);nm=margins(nz,y);eligible=bm[:old_count]>0
    initial=violated(nm,eligible);candidate=native;cm=nm;sweeps=[];active=[];dual={};linear_bad=0;linear_min=None
    if initial:
        P=augmented(native,seen)
        H=np.concatenate([h[:old_count].cpu().numpy().astype(np.float64),np.ones((old_count,1))],axis=1)
        norm2=2*np.sum(H*H,axis=1);labels=y[:old_count].cpu().tolist()
        active=failing_pairs(nz,y,eligible);dual={pair:0. for pair in active};initial_active=list(active)
        for sweep in range(1,MAX_SWEEPS+1):
            for i,c in active:dual[(i,c)]=coordinate(P,H,norm2,i,int(labels[i]),c,dual[(i,c)])
            candidate=materialize(P,native);cz=affine(candidate,h,seen);cm=margins(cz,y)
            bad=failing_pairs(cz,y,eligible);added=sorted(set(bad)-set(active))
            values=np.array([dual[p] for p in active],dtype=np.float64)
            sweeps.append(dict(sweep=sweep,processed_constraints=len(active),failed_pairs=len(bad),
                minimum_margin=float(cm[:old_count][eligible].min()),added_pairs=[list(p) for p in added],
                primal_fp64_sha256=array_hash(P),dual_fp64_sha256=array_hash(values),safe=not bad))
            if not bad:break
            if sweep<MAX_SWEEPS:
                for p in added:dual[p]=0.
                active=sorted(dual)
        continuous=H@P.T
        gaps=continuous[np.arange(old_count),np.asarray(labels),None]-continuous
        mask=np.broadcast_to(eligible.cpu().numpy()[:,None],gaps.shape).copy();mask[np.arange(old_count),np.asarray(labels)]=False
        linear_bad=int(np.sum(gaps[mask]<EPS));linear_min=float(gaps[mask].min())
        failed=bool(failing_pairs(affine(candidate,h,seen),y,eligible))
    else:initial_active=[];failed=False
    executed=before if failed else candidate
    sync(h)
    return executed,candidate,dict(baseline=bm.cpu().tolist(),native=nm.cpu().tolist(),projected=cm.cpu().tolist(),
        eligible=torch.nonzero(eligible).flatten().cpu().tolist(),native_violating=initial,
        initial_active=[list(p) for p in initial_active],final_active=[list(p) for p in active],
        final_duals=[dual[p] for p in active],sweeps=sweeps,sweep_count=len(sweeps),
        final_safe=not failed,skip=failed,action='HILDRETH_FAILED_SKIP' if failed else 'HILDRETH' if initial else 'NATIVE',
        final_linear_failed_pairs=linear_bad,final_linear_min_gap=linear_min,
        projection_seconds=time.perf_counter()-started)

def summarize(rows):
    chosen=[r for r in rows if r['report']['native_violating']]
    def geometries(name):
        return {k:float(np.mean(v)) if (v:=[r[name][k] for r in chosen if r[name][k] is not None]) else None for k in ('norm_ratio','cosine')}
    return dict(native_safe_steps=len(rows)-len(chosen),hildreth_triggered_steps=len(chosen),
        active_constraints=[len(r['report']['final_active']) for r in rows],sweeps=[r['report']['sweep_count'] for r in rows],
        final_safe_steps=sum(r['report']['final_safe'] for r in rows),failed_steps=sum(r['report']['skip'] for r in rows),
        skips=sum(r['report']['skip'] for r in rows),
        executed_old_correct_to_wrong=sum(sum(a>0 and b<=0 for a,b in zip(r['report']['baseline'][:1120],r['report']['executed'][:1120])) for r in rows),
        old_protected_correct=sum(v>0 for v in rows[-1]['report']['executed'][:1120]),old_protected_total=1120,
        projected_geometry=geometries('projected_geometry'),executed_geometry=geometries('executed_geometry'),
        projection_seconds=sum(r['report']['projection_seconds'] for r in rows))
