"""Separate in-place AdamW replay and diagnostic-only inspection of capped solves.

No diagnostic candidate is ever committed. The frozen experiment is unchanged.
"""
from . import cutter
from .common import *
from .run import history

@torch.no_grad()
def independent_exact(p,bank):
    return torch.cat([functional(p,bank.x[k:k+256]) for k in range(0,bank.padded,256)])[:bank.n,:bank.seen].cpu().numpy()

def independent_counts(z,labels,n):
    correct=[]
    for row,label in zip(z,labels):correct.append(all(float(row[label])>float(row[c]) for c in range(len(row)) if c!=label))
    return np.bincount(labels[np.asarray(correct)],minlength=n)

def main():
    setup();guard()
    for name,digest in read(ROOT/'SMOKE_SOURCE_FREEZE.json')['sources'].items():assert sha(HERE/name)==digest
    dest=ROOT/'audit';dest.mkdir(exist_ok=False)
    write(dest/'SOURCE_FREEZE.json',dict(audit_sha256=sha(Path(__file__))))
    X,y,control=load();yc=y.cpu().numpy();reports=[];snapshots={}
    original_geo=cutter.Geometry;original_solver=cutter.ball_hildreth;captured={}
    class ObservedGeometry(original_geo):
        def __init__(self,bank,p,ordered,z):
            super().__init__(bank,p,ordered,z);captured['geometry']=self;captured['current']=p
    def observed_solver(G,b,Ad0,d0norm2,rho,limit=200):
        result=original_solver(G,b,Ad0,d0norm2,rho,limit)
        captured['qp']=(G,b,Ad0,d0norm2,rho,result)
        return result
    cutter.Geometry=ObservedGeometry;cutter.ball_hildreth=observed_solver
    for budget in (210,630):
        reference=history(ROOT/'smoke'/f'jacobian_{budget}'/'task2/STEPS.jsonl.gz')
        report=read(ROOT/'smoke'/f'jacobian_{budget}'/'task2/CHECKS.json')
        state=torch.load(OLD/'formal'/f'native_{budget}'/'task1/CHECKPOINT.pt',weights_only=True)
        head=model();head.load_state_dict(state['model']);opt=optimizer(head);opt.load_state_dict(state['optimizer'])
        oldids=np.flatnonzero((yc<7)&control);ids=np.flatnonzero((yc//7==1)&~control)
        bank=Bank(X[oldids],y[oldids],oldids,14,Meter());K=independent_counts(independent_exact(params(head),bank),bank.labels,7)
        assert K.tolist()==report['K'];caps=[];headhashes=[]
        for step,batch in enumerate(schedule(ids,2,budget),1):
            before=params(head);opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(head(X[batch])[:,:14],y[batch]);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            native=params(head);mh=opt_hash(opt)
            assert opt_steps(opt)==[budget+step]*4 and mh==reference[step-1]['moments_sha256']
            executed,decision=cutter.project(before,native,bank,K);commit(head,executed)
            assert decision==reference[step-1]['decision'],('decision',budget,step)
            hh=thash(head.state_dict());assert hh==reference[step-1]['head_sha256'],('head',budget,step)
            assert opt_hash(opt)==mh
            actual=independent_counts(independent_exact(params(head),bank),bank.labels,7)
            assert actual.tolist()==reference[step-1]['counts'] and np.all(actual>=K)
            headhashes.append(hh)
            if decision['failure_reason']=='INNER_LIMIT':
                geom=captured['geometry'];current=captured['current'];G,b,ad0,d0n,rho,result=captured['qp'];coef,dual,info=result
                d0=tuple(n-p for n,p in zip(cpu(native),current));corr=geom.reconstruct(dual)
                d=tuple(coef*a+v for a,v in zip(d0,corr));candidate=tuple(torch.from_numpy((p+a).astype(np.float32)).cuda() for p,a in zip(current,d))
                z=independent_exact(candidate,bank);count=independent_counts(z,bank.labels,7)
                norm=np.sqrt(cutter.inner(d,d));linear=geom.apply(d);resid=float(np.maximum(b-linear,0).max())
                diag=np.sqrt(np.diag(G));cos=G/diag[:,None]/diag[None,:];np.fill_diagonal(cos,0.)
                ev=np.linalg.eigvalsh(G);v=dict(step=step,pairs=len(b),candidate_floor_safe=bool(np.all(count>=K)),
                    candidate_counts=count.tolist(),halfspace_violation=resid,displacement_norm=norm,rho=rho,
                    ball_hits=info['ball_hits'],cycle_change=info['cycle_change'],cycle_tolerance=1e-10+1e-8*rho,
                    reserve_min=float(geom.reserve.min()),reserve_max=float(geom.reserve.max()),
                    gram_condition=float(ev[-1]/ev[0]) if ev[0]>0 else None,max_abs_row_cosine=float(np.abs(cos).max()),
                    max_candidate_linearization_error=float(np.max(np.abs((z[geom.rows,geom.target]-z[geom.rows,geom.comp])-geom.g-linear))))
                caps.append(v)
                # Preserve first and worst-residual local systems for independent offline QP inspection.
                key=f'{budget}_first'
                if key not in snapshots:snapshots[key]=(G.copy(),b.copy(),ad0.copy(),d0n,rho,step)
                key=f'{budget}_worst'
                if key not in snapshots or resid>snapshots[key][-1]:snapshots[key]=(G.copy(),b.copy(),ad0.copy(),d0n,rho,step,resid)
            if step%60==0:print('independent audit',budget,step,flush=True)
        write(dest/f'CAP_DIAGNOSTICS_{budget}.json',caps)
        reports.append(dict(budget=budget,steps=budget,heads_identical=True,moments_identical=True,counters_identical=True,
            decisions_identical=True,independent_endpoint_counts_identical=True,floor_violations=0,
            capped=len(caps),capped_candidates_actual_safe=sum(v['candidate_floor_safe'] for v in caps),
            capped_ball_hits=sum(v['ball_hits'] for v in caps),diagnostic_candidates_never_committed=True,
            trajectory_sha256=hashlib.sha256(''.join(headhashes).encode()).hexdigest()))
    for key,v in snapshots.items():
        with (dest/f'LOCAL_QP_{key}.npz').open('xb') as f:np.savez(f,G=v[0],b=v[1],Ad0=v[2],d0norm2=v[3],rho=v[4],step=v[5])
    # Byte reads only; no test deserialization or evaluation. Verify every old authority unchanged.
    old_hashes=read(ROOT/'FREEZE.json')['old_evidence_hashes']
    for name,digest in old_hashes.items():assert sha(OLD/name)==digest,('old_changed',name)
    write(ROOT/'AUDIT.json',dict(status='PASS',reports=reports,old_authority_files_verified=len(old_hashes),
        test_deserialized=False,test_evaluated=False,diagnostics_after_outcomes=True))
    print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
