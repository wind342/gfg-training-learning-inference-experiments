"""Independent sparse projection replay and saved-gradient AdamW reconstruction."""
from .core import *

@torch.no_grad()
def independent(before,native,h,y,n):
    # Do not call production projection, coordinate, margin or materialize.
    def z(state):
        return (nn.functional.linear(h[:,:DIM].contiguous(),state['weight'][:,:DIM].contiguous(),state['bias'])+
                nn.functional.linear(h[:,DIM:].contiguous(),state['weight'][:,DIM:].contiguous()))[:,:SEEN]
    truth=y.cpu().numpy()
    def gaps(state):
        zz=z(state).cpu().numpy();g=zz[np.arange(len(truth)),truth,None]-zz
        g[np.arange(len(truth)),truth]=np.inf
        return g
    bg=gaps(before);ng=gaps(native);eligible=bg[:n].min(1)>0
    def pairs(g):return [tuple(map(int,p)) for p in np.argwhere((g[:n].astype(np.float64)<1e-6)&eligible[:,None])]
    candidate=native;logs=[]
    if np.any(eligible&(ng[:n].min(1)<=0)):
        H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
        P=np.column_stack((native['weight'][:SEEN].cpu().numpy().astype(np.float64),native['bias'][:SEEN].cpu().numpy().astype(np.float64)))
        receiving=np.column_stack((before['weight'][:SEEN].cpu().numpy().astype(np.float64),before['bias'][:SEEN].cpu().numpy().astype(np.float64)))
        u=2.**-24;k=H.shape[1]+2;gamma=k*u/(1-k*u)
        budgets=gamma*(np.abs(H)@np.maximum(np.abs(receiving),np.abs(P)).T)
        lengths=2*np.sum(H*H,axis=1)
        active=pairs(ng);dual={p:0. for p in active}
        for sweep in range(1,101):
            for i,c in active:
                t=int(truth[i]);previous=dual[(i,c)]
                desired=1e-6+float(budgets[i,t]+budgets[i,c])
                updated=max(0.,previous+(desired-float(np.dot(H[i],P[t]-P[c])))/float(lengths[i]))
                change=updated-previous
                if change:P[t]+=change*H[i];P[c]-=change*H[i]
                dual[(i,c)]=updated
            candidate={name:v.clone() for name,v in native.items()}
            candidate['weight'][:SEEN].copy_(torch.from_numpy(P[:,:-1]).to(native['weight']))
            candidate['bias'][:SEEN].copy_(torch.from_numpy(P[:,-1]).to(native['bias']))
            g=gaps(candidate);bad=pairs(g)
            dv=np.array([dual[p] for p in active],dtype=np.float64)
            logs.append(dict(primal=hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest(),
                dual=hashlib.sha256(dv.tobytes()).hexdigest(),safe=not bad))
            if not bad:break
            if sweep<100:
                for p in sorted(set(bad)-set(active)):dual[p]=0.
                active=sorted(dual)
        failed=bool(pairs(gaps(candidate)))
    else:failed=False
    executed=before if failed else candidate
    return executed,candidate,logs,bg.min(1).tolist(),ng.min(1).tolist(),gaps(executed).min(1).tolist()

def main():
    pre=json.loads((ROOT/'PREFLIGHT.json').read_text());guard(pre)
    assert pre['inputs']=={p:sha(p) for p in pre['inputs']}
    cache=torch.load(old.ROOT/'CONTROL_CACHE.pt',map_location='cpu',weights_only=False)
    features=torch.load(ROOT/'FEATURES.pt',map_location='cpu',weights_only=False)
    assert sha(ROOT/'FEATURES.pt')==pre['feature_sha256']
    d=np.load(DATA/'data.npz');ids=features['bank_ids'];n=7833
    positions={i:j for j,i in enumerate(cache['indices'])}
    h=cache['features'][[positions[i] for i in ids]].cuda();y=torch.from_numpy(d['y'][ids]).cuda();R=features['R'].cuda()
    setup();assert torch.equal(R.cpu(),random_map())
    total=0;sweeps=0;endpoints=0
    for arm in ARMS:
        folder=ROOT/arm;audit=torch.load(folder/'AUDIT.pt',map_location='cpu',weights_only=False)
        meta=json.loads((folder/'TRAIN_RESULT.json').read_text())
        assert sha(folder/'AUDIT.pt')==meta['audit_sha256'] and sha(folder/'STEPS.jsonl')==meta['log_sha256']
        packed=audit['heads'][0]
        head=JointHead({'weight':packed['weight'][:,:DIM].cuda(),'bias':packed['bias'].cuda()}).cuda()
        unpack(head,{k:v.cuda() for k,v in packed.items()})
        opt=torch.optim.AdamW(head.parameters(),lr=LR,betas=(.9,.999),weight_decay=0.,fused=True,foreach=False)
        opt.load_state_dict(audit['initial_optimizer'])
        q=joint_features(h,R,arm)
        for j,line in enumerate((folder/'STEPS.jsonl').open()):
            row=json.loads(line);before={k:v.cuda() for k,v in audit['heads'][j].items()}
            unpack(head,before)
            for name,p in head.named_parameters():p.grad=audit['clipped_gradients'][j][name].cuda()
            opt.step();native=pack(head)
            assert all(torch.equal(native[k].cpu(),audit['natives'][j][k]) for k in native)
            assert moments_hash(opt)==row['optimizer_sha256']
            executed,candidate,logs,bm,nm,em=independent(before,native,q,y,n)
            assert bm==row['report']['baseline'] and nm==row['report']['native'] and em==row['report']['executed']
            assert all(torch.equal(executed[k].cpu(),audit['heads'][j+1][k]) for k in executed)
            assert all(torch.equal(candidate[k].cpu(),audit['projected'][j][k]) for k in candidate)
            assert len(logs)==row['sweeps']
            for a,b in zip(logs,row['report']['sweeps']):
                assert a['primal']==b['primal_fp64_sha256'] and a['dual']==b['dual_fp64_sha256'] and a['safe']==b['safe']
            assert sum(a>0 and b<=0 for a,b in zip(bm[:n],em[:n]))==0
            total+=1;sweeps+=len(logs);endpoints+=3*len(em)
            if j%70==0:print('CHECK',arm,j+1,flush=True)
        assert equal(opt.state_dict(),audit['final_optimizer'])
    ev=json.loads((ROOT/'EVALUATION.json').read_text());barrier=json.loads((ROOT/'EVALUATION_BARRIER.json').read_text())
    assert barrier['training_hashes']=={a:sha(ROOT/a/'TRAIN_RESULT.json') for a in ARMS}
    for a in ARMS:assert summarize_predictions(ev['indices'],ev['predictions'][a],d,TASK)==ev['metrics'][a]
    assert barrier['time']<ev['finished_unix']
    guard(pre)
    write(ROOT/'CHECKS.json',dict(status='PASS',native_candidates=total,independent_sweeps=sweeps,
        endpoint_margin_values=endpoints,initial_logits_exact=True,optimizer_moments_exact=True,
        executed_protected_flips=0,evaluation_after_both_arms=True,source_and_input_identity=True,
        metrics_recomputed=True))
    print('CHECKS PASS',total,sweeps,endpoints,flush=True)

if __name__=='__main__':main()
