"""Independent reconstruction, without production selection/projector calls."""
from .core import *

@torch.no_grad()
def independent(before,native,h,y,n,K,ids):
    labels=y[:n].cpu().numpy();truth=y.cpu().numpy();groups=len(K)
    def margins_(state):
        z=nn.functional.linear(h,state['weight'],state['bias'])[:,:64].cpu().numpy()
        gaps=z[np.arange(len(truth)),truth,None]-z;gaps[np.arange(len(truth)),truth]=np.inf
        return gaps,gaps.min(1)
    def count(m):return np.bincount(labels[m[:n]>0],minlength=groups)
    bg,bm=margins_(before);g,cm=margins_(native);nm=cm.copy();C=count(cm)
    active=[];dual={};selected=set();selections=[];logs=[];candidate=native
    if np.any(C<K):
        H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
        P=np.column_stack((native['weight'][:64].cpu().numpy().astype(np.float64),native['bias'][:64].cpu().numpy().astype(np.float64)))
        R=np.column_stack((before['weight'][:64].cpu().numpy().astype(np.float64),before['bias'][:64].cpu().numpy().astype(np.float64)))
        u=2.**-24;k=H.shape[1]+2;budgets=(k*u/(1-k*u))*(np.abs(H)@np.maximum(np.abs(R),np.abs(P)).T)
        lengths=2*np.sum(H*H,axis=1)
        for sweep in range(1,101):
            deficits=np.maximum(K-count(cm),0);chosen={}
            for r in range(groups):
                if deficits[r]:
                    wrong=[i for i in range(n) if labels[i]==r and cm[i]<=0]
                    wrong.sort(key=lambda i:(-float(cm[i]),int(ids[i])))
                    chosen[str(r)]=wrong[:int(deficits[r])];selected.update(chosen[str(r)])
            mask=np.zeros(n,dtype=bool);mask[list(selected)]=True
            bad=[tuple(map(int,v)) for v in np.argwhere((g[:n].astype(np.float64)<1e-6)&mask[:,None])]
            added=sorted(set(bad)-set(dual))
            for pair in added:dual[pair]=0.
            active=sorted(dual)
            selections.append(dict(after_sweep=sweep-1,counts=count(cm).tolist(),deficits=deficits.tolist(),
                chosen=chosen,selected_controls=sorted(selected),added_pairs=[list(v) for v in added]))
            for i,c in active:
                t=int(labels[i]);previous=dual[(i,c)]
                updated=max(0.,previous+(1e-6+float(budgets[i,t]+budgets[i,c])-float(np.dot(H[i],P[t]-P[c])))/float(lengths[i]))
                change=updated-previous
                if change:P[t]+=change*H[i];P[c]-=change*H[i]
                dual[(i,c)]=updated
            candidate={name:v.clone() for name,v in native.items()}
            candidate['weight'][:64].copy_(torch.from_numpy(P[:,:-1]).to(native['weight']))
            candidate['bias'][:64].copy_(torch.from_numpy(P[:,-1]).to(native['bias']))
            g,cm=margins_(candidate);C=count(cm);safe=bool(np.all(C>=K))
            logs.append(dict(sweep=sweep,counts=C.tolist(),safe=safe,constraints=len(active),
                primal_sha256=hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest(),
                dual_sha256=hashlib.sha256(np.array([dual[v] for v in active],dtype=np.float64).tobytes()).hexdigest()))
            if safe:break
    fail=bool(np.any(count(cm)<K));executed=before if fail else candidate
    return executed,candidate,selections,logs,bm,nm,cm,(bm if fail else cm)

def main():
    p.setup();pre=json.loads((ROOT/'PREFLIGHT.json').read_text());guard(pre)
    assert pre['inputs']=={name:sha(name) for name in pre['inputs']}
    meta=json.loads((ROOT/'TRAIN_RESULT.json').read_text());assert sha(ROOT/'AUDIT.pt')==meta['audit_sha256'] and sha(ROOT/'STEPS.jsonl')==meta['log_sha256']
    cache=torch.load(old.ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    ids=pre['bank_ids'];n=pre['old_count'];pos={i:j for j,i in enumerate(cache['indices'])}
    h=cache['features'][[pos[i] for i in ids]].cuda();d=np.load(DATA/'data.npz');y=torch.from_numpy(d['y'][ids]).cuda();labels=d['y'][ids[:n]]
    audit=torch.load(ROOT/'AUDIT.pt',map_location='cpu',weights_only=False)
    head=nn.Linear(1536,80).cuda();head.load_state_dict(audit['heads'][0])
    opt=optimizer(head.parameters());opt.load_state_dict(audit['initial_optimizer'])
    with torch.no_grad():
        z=head(h)[:,:64];zz=z.cpu().numpy();truth=y.cpu().numpy();g=zz[np.arange(len(truth)),truth,None]-zz;g[np.arange(len(truth)),truth]=np.inf
        entry=g.min(1);K=np.bincount(labels[entry[:n]>0],minlength=56)
    assert K.tolist()==pre['K'];steps=0;total_sweeps=0;flips=0;gains=0;migrated=set();floorchecks=0
    for j,line in enumerate((ROOT/'STEPS.jsonl').open()):
        row=json.loads(line);r=row['report'];before={k:v.cuda() for k,v in audit['heads'][j].items()}
        head.load_state_dict(before)
        for name,q in head.named_parameters():q.grad=audit['clipped_gradients'][j][name].cuda()
        opt.step();native=p.snapshot(head)
        assert equal(native,audit['natives'][j]) and moments_hash(opt)==row['optimizer_sha256']
        e,c,selections,logs,bm,nm,cm,em=independent(before,native,h,y,n,K,ids[:n])
        assert equal(e,audit['heads'][j+1]) and equal(c,audit['projected'][j])
        assert selections==r['selections'] and logs==r['sweeps']
        assert bm.tolist()==r['baseline'] and nm.tolist()==r['native'] and cm.tolist()==r['projected'] and em.tolist()==r['executed']
        C=np.bincount(labels[em[:n]>0],minlength=56);assert np.all(C>=K) and C.tolist()==r['executed_counts'] and r['K']==K.tolist()
        lost=(bm[:n]>0)&(em[:n]<=0);gain=(bm[:n]<=0)&(em[:n]>0)
        lc=np.bincount(labels[lost],minlength=56);gc=np.bincount(labels[gain],minlength=56)
        assert int(lost.sum())==r['correct_to_wrong'] and int(gain.sum())==r['wrong_to_correct']
        assert np.flatnonzero((lc>0)&(gc>0)).tolist()==r['migration_relations']
        migrated.update(r['migration_relations']);flips+=int(lost.sum());gains+=int(gain.sum());floorchecks+=56;steps+=1;total_sweeps+=len(logs)
        if steps==1 or steps%70==0:print('CHECK',steps,flush=True)
    assert equal(opt.state_dict(),audit['final_optimizer'])
    summary=meta['summary'];assert flips==summary['correct_to_wrong'] and gains==summary['wrong_to_correct'] and sorted(migrated)==summary['same_step_migration_relations']
    ev=json.loads((ROOT/'EVALUATION.json').read_text());barrier=json.loads((ROOT/'EVALUATION_BARRIER.json').read_text())
    assert barrier['train_result_sha256']==sha(ROOT/'TRAIN_RESULT.json') and barrier['created_unix']<ev['finished_unix']
    assert summarize_predictions(ev['indices'],ev['predictions'],d,8)==ev['metrics']
    guard(pre)
    write(ROOT/'CHECKS.json',dict(status='PASS',native_candidates=steps,independent_sweeps=total_sweeps,
        relation_step_floor_checks=floorchecks,floor_violations=0,correct_to_wrong=flips,wrong_to_correct=gains,
        optimizer_moments_exact=True,selections_and_ties_exact=True,actual_endpoints_exact=True,
        fixed_entry_floor=True,metrics_recomputed=True,source_and_input_identity=True))
    print('CHECKS PASS',steps,total_sweeps,floorchecks,flush=True)

if __name__=='__main__':main()
