"""Relation-floor selection around unchanged Hildreth mathematical primitives."""
from experiments.gfg_final_architecture_v1_ten_task import core as old
from experiments.gfg_head_only_hildreth_roundoff_smoke_v1 import core as p
from experiments.gfg_head_only_exact_monitor_smoke_v1.core import *
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/relation-capability-floor-task8-v1')
BASE='115534a09cd554fcd75fd46dde711fdb98988519'

def counts(m,labels,groups):
    return np.bincount(labels[np.asarray(m)>0],minlength=groups).astype(int)

def choose(m,labels,K,ids):
    C=counts(m,labels,len(K));deficits=np.maximum(K-C,0)
    chosen={}
    for r,d in enumerate(deficits):
        if d:
            wrong=np.flatnonzero((labels==r)&(np.asarray(m)<=0)).tolist()
            chosen[str(r)]=sorted(wrong,key=lambda i:(-float(m[i]),int(ids[i])))[:int(d)]
            assert len(chosen[str(r)])==d
    return C,deficits,chosen

@torch.no_grad()
def floor_projection(before,native,h,y,n,K,ids,seen=64,limit=100):
    p.sync(h);start=time.perf_counter()
    bz=p.affine(before,h,seen);nz=p.affine(native,h,seen)
    bm=p.margins(bz,y).cpu().numpy();nm=p.margins(nz,y).cpu().numpy()
    labels=y[:n].cpu().numpy();K=np.asarray(K,dtype=int)
    assert np.all(counts(bm[:n],labels,len(K))>=K)
    native_counts,deficits,_=choose(nm[:n],labels,K,ids)
    candidate=native;cm=nm;cz=nz;active=[];dual={};selected=set();selections=[];sweeps=[];budget_hash=None
    if deficits.any():
        P=p.augmented(native,seen)
        H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
        lengths=2*np.sum(H*H,axis=1)
        budgets=p.logit_roundoff_budgets(before,native,H,seen);budget_hash=p.array_hash(budgets)

        def add(sweepno):
            nonlocal active
            C,D,chosen=choose(cm[:n],labels,K,ids)
            selected.update(i for values in chosen.values() for i in values)
            mask=torch.zeros(n,dtype=torch.bool,device=h.device)
            if selected:mask[list(sorted(selected))]=True
            bad=p.failing_pairs(cz,y,mask)
            new=sorted(set(bad)-set(dual))
            for pair in new:dual[pair]=0.
            active=sorted(dual)
            selections.append(dict(after_sweep=sweepno,counts=C.tolist(),deficits=D.tolist(),
                chosen=chosen,selected_controls=sorted(selected),added_pairs=[list(v) for v in new]))

        add(0)
        for sweep in range(1,limit+1):
            for i,c in active:
                target=int(labels[i]);padding=float(budgets[i,target]+budgets[i,c])
                dual[(i,c)]=p.coordinate(P,H,lengths,i,target,c,dual[(i,c)],padding)
            candidate=p.materialize(P,native);cz=p.affine(candidate,h,seen)
            cm=p.margins(cz,y).cpu().numpy();C=counts(cm[:n],labels,len(K));safe=bool(np.all(C>=K))
            sweeps.append(dict(sweep=sweep,counts=C.tolist(),safe=safe,constraints=len(active),
                primal_sha256=p.array_hash(P),dual_sha256=p.array_hash(np.array([dual[v] for v in active],dtype=np.float64))))
            if safe:break
            if sweep<limit:add(sweep)
    failed=bool(np.any(counts(cm[:n],labels,len(K))<K))
    executed=before if failed else candidate;em=bm if failed else cm
    C=counts(em[:n],labels,len(K));assert np.all(C>=K)
    lost=(bm[:n]>0)&(em[:n]<=0);gained=(bm[:n]<=0)&(em[:n]>0)
    lost_counts=np.bincount(labels[lost],minlength=len(K));gain_counts=np.bincount(labels[gained],minlength=len(K))
    p.sync(h)
    report=dict(K=K.tolist(),receiving_counts=counts(bm[:n],labels,len(K)).tolist(),native_counts=native_counts.tolist(),
        native_deficits=deficits.tolist(),deficient_relations=np.flatnonzero(deficits).tolist(),
        selections=selections,selected_controls=sorted(selected),final_active=[list(v) for v in active],
        final_duals=[dual[v] for v in active],sweeps=sweeps,sweep_count=len(sweeps),
        skip=failed,projection_triggered=bool(deficits.any()),executed_counts=C.tolist(),floor_violations=int((C<K).sum()),
        correct_to_wrong=int(lost.sum()),wrong_to_correct=int(gained.sum()),lost_per_relation=lost_counts.tolist(),
        gained_per_relation=gain_counts.tolist(),migration_relations=np.flatnonzero((lost_counts>0)&(gain_counts>0)).tolist(),
        count_slack_relations=np.flatnonzero(C<counts(bm[:n],labels,len(K))).tolist(),
        baseline=bm.tolist(),native=nm.tolist(),projected=cm.tolist(),executed=em.tolist(),
        roundoff_budget_sha256=budget_hash,projection_seconds=time.perf_counter()-start)
    return executed,candidate,report

def sources():
    files=[REPO/x for x in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if x.endswith('.py')]
    files+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {x.relative_to(REPO).as_posix():sha(x) for x in sorted(set(files))}

def guard(pre):
    assert sources()==pre['sources']
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)
