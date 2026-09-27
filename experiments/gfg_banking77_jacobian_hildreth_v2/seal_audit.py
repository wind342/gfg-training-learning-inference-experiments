"""Read-only all-task evidence audit before the new test reader is allowed."""
from .common import *
from .run import history

def main():
    guard();assert read(ROOT/'FORMAL.json')['status']=='PASS';results=[]
    for budget in (210,630):
        totals=defaultdict(int);cap_residuals=[];previous=OLD/'formal'/f'native_{budget}'/'task1/CHECKPOINT.pt'
        for task in range(2,12):
            folder=ROOT/('smoke' if task==2 else 'formal')/f'jacobian_{budget}'/f'task{task}'
            report=read(folder/'CHECKS.json');rows=history(folder/'STEPS.jsonl.gz');assert len(rows)==budget
            assert report['source_checkpoint_sha256']==sha(previous);K=np.asarray(report['K']);prior_counts=K.tolist()
            for step,row in enumerate(rows,1):
                assert row['step']==step and np.all(np.asarray(row['counts'])>=K) and row['floor_violations']==0
                d=row['decision'];totals['scheduled']+=1;totals['rollback']+=int(d['rollback']);totals['native_safe']+=int(d['native_safe'])
                if d['native_safe']:assert d['jacobian_calls']==0 and not d['rounds'] and not d['rollback']
                elif d['rollback']:assert row['counts']==prior_counts
                else:
                    assert d['rounds'][-1]['safe'] and d['rounds'][-1]['endpoint_counts']==row['counts']
                    if not d['rounds'][-1]['local_converged']:totals['approximate_local_commits']+=1
                for q in d['rounds']:
                    assert q['solver']['ball_violation']<=1e-10 and q['local_displacement_norm']<=q['rho']+1e-8
                    totals['outer_rounds']+=1;totals['inner_sweeps']+=q['solver']['sweeps'];totals['ball_hits']+=q['solver']['ball_hits']
                    if not q['local_converged']:
                        totals['capped_exact_checks']+=1;totals['capped_exact_safe']+=int(q['safe']);cap_residuals.append(q['solver']['halfspace_violation'])
                    if not q['safe']:totals['projected_unsafe_rechecks']+=1
                prior_counts=row['counts']
            state=torch.load(folder/'CHECKPOINT.pt',weights_only=True,map_location='cpu')
            assert thash(state['model'])==rows[-1]['head_sha256'];assert all(int(s['step'])==budget*task for s in state['optimizer']['state'].values())
            mh=thash({f'{i}.{k}':v for i,s in enumerate(state['optimizer']['state'].values()) for k,v in s.items() if isinstance(v,torch.Tensor)})
            assert mh==rows[-1]['moments_sha256'];assert sha(folder/'CHECKPOINT.pt')==report['checkpoint_sha256']
            previous=folder/'CHECKPOINT.pt'
        results.append(dict(budget=budget,totals=dict(totals),floor_violations=0,checkpoint_chains_identical=True,
            local_cap_residual_max=max(cap_residuals,default=0.),local_cap_residual_median=float(np.median(cap_residuals)) if cap_residuals else 0.))
    f=read(ROOT/'FREEZE.json')
    for name,digest in f['old_evidence_hashes'].items():assert sha(OLD/name)==digest
    for name,digest in f['v1_authority_hashes'].items():assert sha(V1/name)==digest
    for name,digest in f['v1_source_hashes'].items():assert sha(HERE.parent/'gfg_banking77_jacobian_hildreth_v1'/name)==digest
    write(ROOT/'SEAL_AUDIT.json',dict(status='PASS',streams=results,old_and_v1_unchanged=True,test_evaluated=False,source_sha256=sha(Path(__file__))))
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
