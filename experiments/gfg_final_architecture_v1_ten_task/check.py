"""Independent sparse dual replay; does not call production Hildreth/coordinate."""
import gzip
from .core import *

@torch.no_grad()
def independent(before,native,h,y,n=1120,seen=16):
    truth=y.cpu().numpy()
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
    pre=guard();setup();assert json.loads((ROOT/'COMPLETE.json').read_text())['status']=='COMPLETED'
    d=np.load(DATA/'data.npz');assert isolation(d)==pre['isolation']
    cache=torch.load(ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    cm=json.loads((ROOT/'CACHE_MANIFEST.json').read_text());assert sha(ROOT/'CONTROL_CACHE.pt')==cm['sha256']
    assert tensor_hash({'H':cache['features']})==cm['features_sha256']
    assert cache['indices']==clean_validation(d).tolist();positions={v:j for j,v in enumerate(cache['indices'])}
    previous={};table=[];total_sweeps=0;endpoint_values=0;total_steps=0
    from experiments.gfg_head_only_coverage_cut_smoke_v1.check import independent_geometry
    for task in range(2,11):
        ids,n=task_bank(d,task);h=cache['features'][[positions[int(v)] for v in ids]].cuda();y=torch.from_numpy(d['y'][ids]).cuda()
        metrics={};results={}
        evaldir=ROOT/f'evaluation_task{task}';ev=json.loads((evaldir/'EVALUATION.json').read_text());barrier=json.loads((evaldir/'BARRIER.json').read_text())
        assert ev['status']=='PASS' and ev['readonly'] and ev['barrier_sha256']==sha(evaldir/'BARRIER.json')
        for arm in ARMS:
            folder=ROOT/arm/f'task{task}';r=json.loads((folder/'TRAIN_RESULT.json').read_text());results[arm]=r
            assert r['status']=='COMPLETED' and r['steps']==210 and r['task']==task and r['no_test_in_training']
            assert r['backbone_sha256']==pre['backbone_initial_sha256']
            assert barrier['created_unix']>=r['finished_unix'] and barrier['training_results'][arm]==sha(folder/'TRAIN_RESULT.json')
            for name,key in [('AUDIT.pt','audit_sha256'),('STEPS.jsonl','log_sha256'),('CONFIG.json','config_sha256')]:assert sha(folder/name)==r[key]
            config=json.loads((folder/'CONFIG.json').read_text());assert config['source_hashes']==pre['source_hashes']
            a=torch.load(folder/'AUDIT.pt',map_location='cpu',weights_only=False)
            rows=[json.loads(v) for v in (folder/'STEPS.jsonl').read_text().splitlines()]
            assert len(rows)==len(a['natives'])==len(a['clipped_gradients'])==210 and len(a['heads'])==211
            assert len(a['projected'])==(210 if arm==ARMS[1] else 0)
            if task>2:
                assert equal(a['heads'][0],previous[arm]['head']) and equal(a['initial_optimizer'],previous[arm]['optimizer'])
            else:
                parent=torch.load(PARENT_ROOT/'140_controls_hildreth_roundoff/AUDIT.pt',map_location='cpu',weights_only=False)
                assert equal(a['heads'][0],parent['heads'][0]) and equal(a['initial_optimizer'],parent['initial_optimizer'])
                if arm==ARMS[1]:
                    for key in ('heads','natives','projected'):assert equal(a[key],parent[key])
                del parent
            initial={k:v.cuda() for k,v in a['heads'][0].items()}
            if arm==ARMS[1]:
                probe=nn.Linear(1536,80).cuda();probe.load_state_dict(initial)
                assert transition(probe,h,y,n,task)==r['transition']==config['transition'];del probe
            shadow=nn.Linear(1536,80).cuda();opt=optimizer(shadow.parameters());opt.load_state_dict(a['initial_optimizer'])
            assert opt_counter(opt)==210*(task-1) and moments_hash(opt)==r['moments_start']
            assert tensor_hash(initial)==r['head_start']
            for j,row in enumerate(rows):
                global_step=210*(task-1)+j+1
                assert row['global_step']==row['optimizer_step']==global_step
                assert row['seed']==2021*100000+global_step and row['current_ids']==order(d,task)[j].tolist()
                b={k:v.cuda() for k,v in a['heads'][j].items()};native={k:v.cuda() for k,v in a['natives'][j].items()}
                with torch.no_grad():
                    for name,p in shadow.named_parameters():p.copy_(b[name]);p.grad=a['clipped_gradients'][j][name].cuda()
                    opt.step()
                assert equal(snapshot(shadow),native) and opt_counter(opt)==global_step
                assert moments_hash(opt)==row['optimizer_sha256']
                assert tensor_hash(b)==row['head_before_sha256'] and tensor_hash(native)==row['native_sha256']
                if arm==ARMS[1]:
                    executed,candidate,report=independent(b,native,h,y,n,task*8)
                    for key,value in report.items():assert value==row['report'][key],(task,j+1,key)
                    assert equal(cpu_state(candidate),a['projected'][j]) and tensor_hash(candidate)==row['projected_sha256']
                    actual=margins(affine(executed,h,task*8),y).cpu().tolist();assert actual==row['report']['executed']
                    assert not any(x>0 and z<=0 for x,z in zip(report['baseline'][:n],actual[:n]))
                    for name,state in [('projected_geometry',candidate),('executed_geometry',executed)]:
                        for k,v in independent_geometry(b,native,state).items():
                            q=row[name][k];assert (v is None and q is None) or (v is not None and q is not None and abs(v-q)<=1e-12)
                    total_sweeps+=len(report['sweeps']);endpoint_values+=4*len(ids)
                else:executed=native;assert 'report' not in row
                assert equal(cpu_state(executed),a['heads'][j+1]) and tensor_hash(executed)==row['head_after_sha256']
                assert row['moments_unmodified_by_projection'] and row['bert_frozen_eval'];total_steps+=1
                if (j+1)%70==0:print('CHECK',arm,'task',task,'step',j+1,'total sweeps',total_sweeps,flush=True)
            assert equal(opt.state_dict(),a['final_optimizer']) and moments_hash(opt)==r['moments_end']
            assert tensor_hash(a['heads'][-1])==r['head_end'] and opt_counter(opt)==r['optimizer_step']==210*task
            if arm==ARMS[1]:assert summary(rows,n)==r['summary']
            previous[arm]=dict(head=a['heads'][-1],optimizer=a['final_optimizer'])
            e=ev['arms'][arm];assert e['indices']==scope(d,task*8)[1].tolist() and e['head_sha256']==r['head_end']
            score=summarize_predictions(e['indices'],e['predictions'],d,task)
            score['previous_task_macro']=float(np.mean([score['per_relation'][str(g)] for g in range(8*(task-2),8*(task-1))]))
            assert e['metrics']==score;metrics[arm]=score
            del a,rows,shadow,opt
        table.append(dict(task=task,metrics=metrics,final_diagnostics=results[ARMS[1]]['summary'],
            expansion=results[ARMS[1]]['transition'],training_wall={a:results[a]['training_wall'] for a in ARMS},
            peak_cuda_bytes={a:results[a]['peak_cuda_bytes'] for a in ARMS}))
    means={a:{k:float(np.mean([r['metrics'][a]['macro_'+k] for r in table])) for k in ('old','new','all')} for a in ARMS}
    guard();write(ROOT/'CHECKS.json',dict(status='PASS',steps=total_steps,independent_sweeps=total_sweeps,endpoint_values=endpoint_values,
        native_candidates_reconstructed_from_saved_clipped_gradients=True,task2_parent_trajectory_exact=True,
        independent_projection_replay=True,continuous_heads_and_moments=True,old_evidence_unchanged=True,source_hashes=sources()))
    write(ROOT/'RESULTS.json',dict(status='COMPLETED',checks='PASS',tasks=table,mean_task2_to_10=means,
        task10={a:{k:table[-1]['metrics'][a]['macro_'+k] for k in ('old','new','all')} for a in ARMS}))
    print('CHECKS PASS',total_steps,'steps',total_sweeps,'sweeps',flush=True)

if __name__=='__main__':main()
