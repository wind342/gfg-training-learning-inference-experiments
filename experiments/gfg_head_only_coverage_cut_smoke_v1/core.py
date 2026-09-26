"""Small isolated coverage/cut comparison; parent algorithm remains unchanged."""
from experiments.gfg_head_only_exact_cut_smoke_v1.core import *
from experiments.gfg_head_only_exact_cut_smoke_v1.core import ROOT as PARENT_ROOT
from experiments.gfg_head_only_exact_cut_smoke_v1.core import exact_cut as current_cut

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/head-only-coverage-cut-smoke-v1')
BASE='df98982d476b57976872d9f6fb21b6a62404a4a3'
ARMS={'A_140_current':(140,'current'),'B_10_tight':(10,'tight'),'C_140_tight':(140,'tight')}
EPS=1e-6

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    paths=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(paths))}

def bank_ids(d,n):
    p,_,_=scope(d,16)
    old=np.array([int(i) for k in range(8) for i in np.flatnonzero((d['y']==k)&(d['split']==1))[:n]])
    assert len(old)==8*n
    return np.concatenate([old,p[80:]])

def guard():
    unchanged();p=json.loads((ROOT/'PREFLIGHT.json').read_text())
    assert p['status']=='PASS' and p['source_hashes']==sources()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    for path,h in p['baseline_files'].items():assert sha(path)==h
    return p

@torch.no_grad()
def tight_cut(before,native,h,y,old_count):
    sync(h);started=time.perf_counter()
    bz=affine(before,h);nz=affine(native,h);bm=margins(bz,y);nm=margins(nz,y)
    eligible=bm[:old_count]>0;initial=violated(nm,eligible);passes=[];candidate=native;cm=nm
    projection_seconds=0.;recheck_seconds=0.
    if initial:
        delta={k:native[k]-before[k] for k in before};z=nz
        for passno in range(1,3):
            sync(h);tick=time.perf_counter();pairs=[]
            for i in torch.nonzero(eligible).flatten().cpu().tolist():
                t=int(y[i])
                pairs.extend((i,t,c) for c in torch.nonzero(z[i]>=z[i,t]).flatten().cpu().tolist() if c!=t)
            certificates=[]
            for i,t,c in pairs:
                candidate={k:before[k]+delta[k] for k in before}
                zz=affine(candidate,h[i:i+1]);m=float(zz[0,t]-zz[0,c])
                norm2=float(2*(h[i].double().square().sum()+1))
                a=(EPS-m)/norm2 if m<=0 else 0.
                if a:
                    delta['weight'][t].add_(h[i],alpha=a);delta['weight'][c].add_(h[i],alpha=-a)
                    delta['bias'][t].add_(a);delta['bias'][c].add_(-a)
                certificates.append(dict(control=i,target=t,competitor=c,margin_before=m,norm2=norm2,coefficient=a))
            sync(h);projection_seconds+=time.perf_counter()-tick;tick=time.perf_counter()
            candidate={k:before[k]+delta[k] for k in before};z=affine(candidate,h);cm=margins(z,y);bad=violated(cm,eligible)
            sync(h);recheck_seconds+=time.perf_counter()-tick
            passes.append(dict(pass_number=passno,certificates=certificates,margins=cm.cpu().tolist(),violating=bad,safe=not bad))
            if not bad:break
    skip=bool(violated(cm,eligible));sync(h)
    return before if skip else candidate,candidate,dict(baseline=bm.cpu().tolist(),native=nm.cpu().tolist(),cut=cm.cpu().tolist(),
        eligible=torch.nonzero(eligible).flatten().cpu().tolist(),native_violating=initial,passes=passes,skip=skip,
        action='CUT_FAILED_SKIP' if skip else 'PROJECTED' if initial else 'NATIVE',
        first_pass_safe=passes[0]['safe'] if passes else None,second_pass=len(passes)==2,
        projection_seconds=projection_seconds,recheck_seconds=recheck_seconds,filter_seconds=time.perf_counter()-started)

def cut(before,native,h,y,old_count,kind):
    return current_cut(before,native,h,y,old_count) if kind=='current' else tight_cut(before,native,h,y,old_count)

def summary(rows,old_count):
    selected=[r for r in rows if r['report']['native_violating']]
    def geom(key):
        return {k:float(np.mean([r[key][k] for r in selected if r[key][k] is not None])) if any(r[key][k] is not None for r in selected) else None
                for k in ('norm_ratio','cosine','removed_norm_fraction')}
    return dict(old_protected_correct=sum(v>0 for v in rows[-1]['report']['executed'][:old_count]),old_protected_total=old_count,
        projection_steps=len(selected),violating_occurrences=sum(len(r['report']['native_violating']) for r in rows),
        executed_old_correct_to_wrong=sum(sum(a>0 and b<=0 for a,b in zip(r['report']['baseline'][:old_count],r['report']['executed'][:old_count])) for r in rows),
        skips=sum(r['report']['skip'] for r in rows),attempted_cut_geometry=geom('cut_geometry'),executed_geometry=geom('executed_geometry'))
