"""Independent sparse dual replay; does not call production Hildreth/coordinate."""
import gzip
from .core import *

@torch.no_grad()
def independent(before,native,h,y):
    n=1120;seen=16;truth=y.cpu().numpy()
    H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
    lengths=2*np.sum(H*H,axis=1)
    def logits_(state):return nn.functional.linear(h,state['weight'],state['bias'])[:,:seen]
    def gaps_(z):
        zz=z.cpu().numpy();g=zz[np.arange(len(truth)),truth,None]-zz
        return g
    def margin_(z):
        g=gaps_(z);g[np.arange(len(truth)),truth]=np.inf
        return g.min(1).tolist()
    bm=margin_(logits_(before));nz=logits_(native);nm=margin_(nz)
    eligible=np.array(bm[:n])>0
    def failures(z):
        g=gaps_(z)[:n];mask=(g.astype(np.float64)<1e-6)&eligible[:,None]
        mask[np.arange(n),truth[:n]]=False
        return [tuple(map(int,p)) for p in np.argwhere(mask)]
    initial=np.flatnonzero(eligible&(np.array(nm[:n])<=0)).tolist()
    candidate=native;cm=nm;active=[];initial_active=[];dual={};logs=[];linear_bad=0;linear_min=None
    budget_hash=None;final_padding=[]
    if initial:
        P=np.column_stack((native['weight'][:seen].cpu().numpy().astype(np.float64),native['bias'][:seen].cpu().numpy().astype(np.float64)))
        initialP=P.copy()
        receiving=np.column_stack((before['weight'][:seen].cpu().numpy().astype(np.float64),before['bias'][:seen].cpu().numpy().astype(np.float64)))
        u=2.**-24;k=H.shape[1]+2;gamma=k*u/(1-k*u)
        budgets=gamma*(np.abs(H)@np.maximum(np.abs(receiving),np.abs(initialP)).T)
        budget_hash=hashlib.sha256(np.ascontiguousarray(budgets).tobytes()).hexdigest()
        active=failures(nz);initial_active=active.copy();dual={p:0. for p in active}
        for sweep in range(1,101):
            for i,c in active:
                t=int(truth[i]);a=dual[(i,c)]
                padding=float(budgets[i,t]+budgets[i,c])
                residual=1e-6+padding-float(np.dot(H[i],P[t]-P[c]))
                b=max(0.,a+residual/float(lengths[i]));change=b-a
                if change:
                    P[t]+=change*H[i];P[c]-=change*H[i]
                dual[(i,c)]=b
            candidate={k:v.clone() for k,v in native.items()}
            candidate['weight'][:seen]=torch.tensor(P[:,:-1],device=h.device,dtype=native['weight'].dtype)
            candidate['bias'][:seen]=torch.tensor(P[:,-1],device=h.device,dtype=native['bias'].dtype)
            z=logits_(candidate);cm=margin_(z);bad=failures(z);added=sorted(set(bad)-set(active))
            dv=np.array([dual[p] for p in active],dtype=np.float64)
            logs.append(dict(sweep=sweep,processed_constraints=len(active),failed_pairs=len(bad),
                minimum_margin=float(np.asarray(cm[:n])[eligible].min()),added_pairs=[list(p) for p in added],
                primal_fp64_sha256=hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest(),
                dual_fp64_sha256=hashlib.sha256(dv.tobytes()).hexdigest(),safe=not bad))
            if not bad:break
            if sweep<100:
                for p in added:dual[p]=0.
                active=sorted(dual)
        continuous=H@P.T;g=continuous[np.arange(n),truth[:n],None]-continuous
        mask=np.broadcast_to(eligible[:,None],g.shape).copy();mask[np.arange(n),truth[:n]]=False
        linear_bad=int(np.sum(g[mask]<1e-6));linear_min=float(g[mask].min())
        failed=bool(failures(logits_(candidate)))
        final_padding=[float(budgets[i,int(truth[i])]+budgets[i,c]) for i,c in active]
    else:failed=False
    return before if failed else candidate,candidate,dict(baseline=bm,native=nm,projected=cm,eligible=np.flatnonzero(eligible).tolist(),
        native_violating=initial,initial_active=[list(p) for p in initial_active],final_active=[list(p) for p in active],
        final_duals=[dual[p] for p in active],sweeps=logs,sweep_count=len(logs),final_safe=not failed,skip=failed,
        action='HILDRETH_FAILED_SKIP' if failed else 'HILDRETH' if initial else 'NATIVE',
        final_linear_failed_pairs=linear_bad,final_linear_min_gap=linear_min,
        roundoff_budget_sha256=budget_hash,final_pair_padding=final_padding)

def main():
    pre=guard();setup();out=ROOT/ARM;r=json.loads((out/'TRAIN_RESULT.json').read_text());assert r['status']=='COMPLETED'
    for name,key in [('AUDIT.pt','audit_sha256'),('STEPS.jsonl','log_sha256'),('CONFIG.json','config_sha256')]:assert sha(out/name)==r[key]
    ev=json.loads((ROOT/'EVALUATION.json').read_text());barrier=json.loads((ROOT/'TRAINING_COMPLETE.json').read_text())
    assert ev['started_unix']>=r['finished_unix'] and ev['barrier_sha256']==sha(ROOT/'TRAINING_COMPLETE.json')
    assert barrier['train_result_sha256']==sha(out/'TRAIN_RESULT.json') and r['no_test_inference']
    d=np.load(DATA/'data.npz');a=torch.load(out/'AUDIT.pt',map_location='cpu',weights_only=False)
    parent=torch.load(COVER_ROOT/'A_140_current'/'AUDIT.pt',map_location='cpu',weights_only=False)
    assert equal(a['heads'][0],parent['heads'][0]) and equal(a['initial_optimizer'],parent['initial_optimizer'])
    assert a['control_ids']==parent['control_ids']==bank_ids(d,140).tolist()
    cache=torch.load(COVER_ROOT/'CACHE.pt',map_location='cpu',weights_only=False)
    assert torch.equal(cache['features'],parent['control_features']);del parent
    h=cache['features'].cuda();y=torch.from_numpy(d['y'][a['control_ids']]).cuda()
    assert tensor_hash({'H':h})==r['cache_sha256'];rows=[json.loads(v) for v in (out/'STEPS.jsonl').read_text().splitlines()]
    assert len(rows)==210 and len(a['heads'])==211 and len(a['natives'])==len(a['projected'])==210
    assert all(float(s['step'])==420 for s in a['final_optimizer']['state'].values())
    from experiments.gfg_head_only_coverage_cut_smoke_v1.check import independent_geometry
    endpoint_values=0;sweep_total=0
    for j,row in enumerate(rows):
        assert row['step']==j+1 and row['current_ids']==order(d,2)[j].tolist() and row['seed']==2021*100000+211+j
        assert row['lr']==LR and row['old_replay']==0 and row['moments_unmodified_by_projection'] and row['bert_frozen_eval']
        b={k:v.cuda() for k,v in a['heads'][j].items()};native={k:v.cuda() for k,v in a['natives'][j].items()}
        executed,candidate,report=independent(b,native,h,y)
        for key,value in report.items():assert value==row['report'][key],(j+1,key)
        assert equal(cpu_state(executed),a['heads'][j+1]) and equal(cpu_state(candidate),a['projected'][j])
        actual=margins(affine(executed,h),y).cpu().tolist();assert actual==row['report']['executed']
        assert not any(x>0 and z<=0 for x,z in zip(report['baseline'][:1120],actual[:1120]))
        for state,key in [(b,'head_before_sha256'),(native,'native_sha256'),(candidate,'projected_sha256'),(executed,'head_after_sha256')]:assert tensor_hash(state)==row[key]
        for name,state in [('projected_geometry',candidate),('executed_geometry',executed)]:
            gg=independent_geometry(b,native,state)
            for k,v in gg.items():
                q=row[name][k];assert (v is None and q is None) or (v is not None and q is not None and abs(v-q)<=1e-12)
        endpoint_values+=4*len(y);sweep_total+=len(report['sweeps'])
        if (j+1)%10==0:print('CHECKED',j+1,'sweeps',sweep_total,flush=True)
    assert r['summary']==summarize(rows) and tensor_hash(a['heads'][-1])==r['head_sha256']
    assert r['backbone_sha256']==pre['backbone_initial_sha256']
    test=ev['result']['test'];assert test['indices']==scope(d,16)[1].tolist()
    score=summarize_predictions(test['indices'],test['predictions'],d,2);assert score==test['macro']
    train=ev['result']['training'];expected=np.flatnonzero((d['split']==0)&(d['y']>=8)&(d['y']<16)).tolist()
    assert train['indices']==expected and train['accuracy16']==float((np.asarray(train['predictions'])==d['y'][expected]).mean())
    baseline=json.loads((COVER_ROOT/'RESULTS.json').read_text())['arms']['A_140_current']
    previous=json.loads((HILD_ROOT/'RESULTS.json').read_text())['hildreth']
    guard();write(ROOT/'CHECKS.json',dict(status='PASS',steps=210,independent_sweeps=sweep_total,endpoint_values=endpoint_values,
        exact_independent_dual_replay=True,all_head_states_verified=True,test_after_training=True,old_evidence_unchanged=True,source_hashes=sources()))
    write(ROOT/'RESULTS.json',dict(status='COMPLETED',checks='PASS',baseline=baseline,previous_hildreth=previous,
        hildreth=dict(test={k:score['macro_'+k] for k in ('old','new','all')},train16_accuracy=train['accuracy16'],
                      **r['summary'],training_wall=r['training_wall']),
        interpretation='Scale-based numerical padding only; actual FP32 acceptance remains >=1e-6. Finite-budget first-safe projection; not global optimality.'))
    print('CHECKS PASS',score,flush=True)

if __name__=='__main__':main()
