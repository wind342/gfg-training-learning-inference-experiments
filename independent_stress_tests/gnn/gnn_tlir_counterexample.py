"""Independent GNN stress test of specific TLIR experimental relationships.
No original GFG-native archive/checker is claimed. CPU deterministic execution.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, platform
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Experimental controls: no tuning once protocol is frozen.
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
SEEDS=(20261024,20261025,20261026)
CHECKPOINT_STEPS=(25,100,200,350)
STEPS=500
N=12; H=40; TRAIN_N=1024; TEST_N=256; BATCH=64; LR=0.003


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def arrsha(x):
    ar=x.detach().contiguous().cpu().numpy()
    return hashlib.sha256(str((str(ar.dtype), tuple(ar.shape))).encode()+ar.tobytes()).hexdigest()


def state_sha(st):
    h=hashlib.sha256()
    for k in sorted(st):
        h.update(k.encode());h.update(arrsha(st[k]).encode())
    return h.hexdigest()


def clone_state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}


def restore_state(model,state):model.load_state_dict(state,strict=True)


def clone_optimizer(opt):
    return {k:{a:(v.detach().clone() if isinstance(v,torch.Tensor) else v) for a,v in t.items()} for k,t in opt.state_dict()['state'].items()}


def build_data(seed,n_samples):
    rng=np.random.default_rng(seed)
    a=[];x=[];labels=[]; ids=[]
    counts=[0,0]; attempt=0
    while min(counts)<n_samples//2:
        attempt+=1
        if attempt > n_samples*60:raise RuntimeError('Could not balance graph classes')
        p=rng.uniform(0.13,0.37)
        upper=np.triu((rng.random((N,N))<p).astype(np.float32),1)
        adj=upper+upper.T
        marker=int(rng.integers(1,N))
        # Task: marker is within one or two graph hops of node 0.
        twohop=(adj@adj)[0,marker]
        target=int(adj[0,marker]>0 or twohop>0)
        if counts[target]>=n_samples//2:continue
        features=np.zeros((N,3),np.float32)
        features[:,2]=1
        features[0,0]=1;features[marker,1]=1
        # Fixed query: anchor 0, all other node indices randomly permuted.
        perm=np.r_[0,rng.permutation(np.arange(1,N))]
        adj=adj[perm][:,perm];features=features[perm]
        degrees=adj.sum(axis=1,keepdims=True)
        adj=adj/np.maximum(degrees,1)
        a.append(adj);x.append(features);labels.append(target)
        ids.append(f'{seed}:{len(ids)}:{marker}:{target}:{hashlib.sha256(adj.tobytes()+features.tobytes()).hexdigest()[:12]}')
        counts[target]+=1
    order=rng.permutation(len(labels))
    return torch.tensor(np.asarray(a)[order]),torch.tensor(np.asarray(x)[order]),torch.tensor(np.asarray(labels,dtype=np.int64)[order]),[ids[i] for i in order]


class ResidualGNNBlock(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.self_proj=nn.Linear(h,h)
        self.message_proj=nn.Linear(h,h,bias=False)
        self.norm=nn.LayerNorm(h)
    def forward(self,h,adj,gate):
        msg=torch.bmm(adj,h)
        inc=F.gelu(self.norm(self.self_proj(h)+self.message_proj(msg)))
        return h+gate*0.5*inc


class GraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(3,H)
        self.blocks=nn.ModuleList([ResidualGNNBlock(H) for _ in range(4)])
        self.readout=nn.Sequential(nn.Linear(H,H),nn.GELU(),nn.Linear(H,2))
    def forward(self,a,x,mask=15):
        h=self.embed(x)
        for i,blk in enumerate(self.blocks):h=blk(h,a,float((mask>>i)&1))
        return self.readout(h[:,0])


def accuracy(model,ds):
    a,x,y,_=ds
    model.eval()
    with torch.no_grad():out=model(a,x);pred=out.argmax(-1)
    return float((pred==y).float().mean()),float(F.cross_entropy(out,y))


def eval_logits(model,ds,mask=15,state=None):
    if state is not None:restore_state(model,state)
    model.eval()
    with torch.no_grad():return model(ds[0],ds[1],mask).detach().clone()


def task_margin(logits,targets):
    return torch.where(targets==0,logits[:,0]-logits[:,1],logits[:,1]-logits[:,0])


def add_delta(pre,delta,alpha):
    return {k:v+alpha*delta[k] if v.is_floating_point() else v.detach().clone() for k,v in pre.items()}


def tensor_rms(t):return float(torch.mean(t.double()**2).sqrt())


def nrmse(a,b):
    den=math.sqrt((tensor_rms(a)**2+tensor_rms(b)**2)/2)
    return tensor_rms(a-b)/max(den,1e-12)


def coalitions(model,ds,state):
    restore_state(model,state);model.eval()
    with torch.no_grad():return torch.stack([model(ds[0],ds[1],mask).clone() for mask in range(16)],0)


def shapley(values,truth):
    # Characteristic function: true-vs-other logit margin, full 16-mask coalition.
    margins=torch.stack([task_margin(values[m],truth) for m in range(16)])
    out=torch.zeros((len(truth),4),dtype=margins.dtype)
    for j in range(4):
        for mask in range(16):
            if mask&(1<<j):continue
            k=mask.bit_count()
            out[:,j]+=math.factorial(k)*math.factorial(3-k)/math.factorial(4)*(margins[mask|(1<<j)]-margins[mask])
    return out


def relation_tests(model,train,test,initial,records,seed):
    # All tests are done after training, on held-out test query identities only.
    # Do not train to improve a failing test.
    test_small=(test[0][:64],test[1][:64],test[2][:64],test[3][:64])
    checkpoints={k:records[k] for k in CHECKPOINT_STEPS}
    stat=[]
    state_resp=[];opt_resp=[];curve_rows=[];support_rows=[]
    for later,earlier in [(100,25),(200,100),(350,200)]:
        r=records[later];alt=records[earlier]
        # Receiving-state exchange: same exact *realized* parameter update applied to both states.
        def resp(s,d):
            zero=eval_logits(model,test_small,15,s)
            one=eval_logits(model,test_small,15,add_delta(s,d,1.0))
            return task_margin(one,test_small[2])-task_margin(zero,test_small[2])
        d=r['delta'];sr=resp(r['pre'],d);sa=resp(alt['pre'],d)
        state_resp.append(dict(later=later,earlier=earlier,nrmse=nrmse(sr,sa),received_rms=tensor_rms(sr),alternate_rms=tensor_rms(sa),delta_l2=math.sqrt(sum(float(torch.sum(t.double()**2)) for t in d.values()))))
    for k in CHECKPOINT_STEPS:
        r=records[k];p=r['pre'];d=r['delta']
        # Exact finite amplitude (no response above alpha zero enters any prediction).
        amplitudes=(0.,0.125,0.25,0.5,0.75,1.)
        vals=[task_margin(eval_logits(model,test_small,15,add_delta(p,d,a)),test_small[2]) for a in amplitudes]
        endpoint=vals[-1];baseline=vals[0];max_dev=torch.zeros_like(endpoint)
        for a,v in zip(amplitudes,vals):max_dev=torch.maximum(max_dev,(v-baseline-a*(endpoint-baseline)).abs())
        scaled=max_dev/((endpoint-baseline).abs().clamp_min(0.001))
        curve_rows.append(dict(step=k,number=len(endpoint),nonlinear_count=int(((max_dev>0.001)&(scaled>0.1)).sum()),mean_abs_change=float((endpoint-baseline).abs().mean()),max_chord_deviation=float(max_dev.max()),nonlinear_fraction=float(((max_dev>0.001)&(scaled>0.1)).float().mean())))
        # Measure 16-coalition support at the exact same pre/post pair.
        v0=coalitions(model,test_small,p);v1=coalitions(model,test_small,r['post'])
        sh0=shapley(v0,test_small[2]);sh1=shapley(v1,test_small[2])
        den=sh0.abs().sum(-1).clamp_min(1e-12)
        fraction=(sh0.abs()>=den[:,None]*0.1).sum(-1)
        reallocation=(sh1-sh0).abs().sum(-1)/den
        s0=sh0/sh0.abs().sum(-1,keepdim=True).clamp_min(1e-12)
        max_query=0.0
        for i in range(len(s0)):
            max_query=max(max_query,float((s0-s0[i:i+1]).abs().sum(-1).max()))
        interactions=[]
        for i in range(4):
            for j in range(i+1,4):
                full=15
                # Full two-class logits. Not margin-based (rules out competitor-selection artifacts).
                q=v0[full]-v0[full^(1<<i)]-v0[full^(1<<j)]+v0[full^(1<<i)^(1<<j)]
                interactions.append(tensor_rms(q))
        support_rows.append(dict(step=k,active_fraction=float((fraction>=2).float().mean()),reallocation_fraction=float((reallocation>0.01).float().mean()),primary_switch_fraction=float((sh0.abs().argmax(-1)!=sh1.abs().argmax(-1)).float().mean()),max_query_profile_l1=max_query,max_complete_logit_pair_interaction_rms=max(interactions)))
        restore_state(model,r['post'])
    # Frozen-inference integrity, then version rollback into the identical learned checkpoint.
    final=records['final']
    original=eval_logits(model,test_small,15,final)
    frozen_before_sha=state_sha(clone_state(model))
    repeated=eval_logits(model,test_small,15)
    frozen_after_sha=state_sha(clone_state(model))
    frozen_inference_exact=bool(torch.equal(original,repeated) and frozen_before_sha==frozen_after_sha==state_sha(final))
    rollback=[]
    for j in range(4):
        prefix=f'blocks.{j}.'
        hybrid={name:(initial[name].clone() if name.startswith(prefix) else value.clone()) for name,value in final.items()}
        changed=eval_logits(model,test_small,15,hybrid)
        restored=eval_logits(model,test_small,15,final)
        rollback.append(dict(component=j,maximum_full_output_difference=float((changed-original).abs().max()),exact_restoration=bool(torch.equal(restored,original) and state_sha(clone_state(model))==state_sha(final))))
    final_accuracy,final_loss=accuracy(model,test)
    # Strictly separate observation, pass/fail and hypothesis scope.
    g={
        'training_capability':final_accuracy>=0.85,
        'frozen_execution_integrity':frozen_inference_exact,
        'same_update_different_receiver':sum(r['nrmse']>0.05 and max(r['received_rms'],r['alternate_rms'])>1e-4 for r in state_resp)>=2,
        'finite_amplitude_nonlinearity':sum(r['nonlinear_count'] for r in curve_rows)>=0.10*sum(r['number'] for r in curve_rows),
        'distributed_support':sum(r['active_fraction'] for r in support_rows)/len(support_rows)>=0.5,
        'persistent_support_reallocation':sum(r['reallocation_fraction'] for r in support_rows)/len(support_rows)>=0.25,
        'frozen_query_conditioning':all(r['max_query_profile_l1']>0.05 for r in support_rows),
        'nonadditive_complete_logits':all(r['max_complete_logit_pair_interaction_rms']>1e-6 for r in support_rows),
        'exact_learned_version_rollback':all(r['maximum_full_output_difference']>1e-6 and r['exact_restoration'] for r in rollback)
    }
    return dict(seed=seed,training_accuracy=final_accuracy,training_loss=final_loss,receiving_state_exchanges=state_resp,finite_amplitude=curve_rows,support=support_rows,rollbacks=rollback,gates=g,execution_integrity={'frozen_inference_exact':frozen_inference_exact},observation_status='OBSERVED_WITH_FROZEN_THRESHOLDS')


def run(seed,steps=STEPS,formal=False):
    seed_all(seed)
    train=build_data(seed+100,TRAIN_N)
    test=build_data(seed+200,TEST_N)
    model=GraphNet()
    initial=clone_state(model)
    opt=torch.optim.RMSprop(model.parameters(),lr=LR,alpha=0.95,eps=1e-8,momentum=0.0)
    rng=np.random.default_rng(seed+300)
    records={}; eval_log=[]
    for k in range(1,steps+1):
        idx=torch.tensor(rng.integers(0,len(train[2]),size=BATCH),dtype=torch.long)
        a,x,y=train[0][idx],train[1][idx],train[2][idx]
        model.train()
        capture=k in CHECKPOINT_STEPS
        pre=clone_state(model) if capture else None
        opt.zero_grad(set_to_none=True)
        logits=model(a,x)
        loss=F.cross_entropy(logits,y)
        loss.backward()
        opt.step()
        if capture:
            post=clone_state(model)
            delta={n:post[n]-pre[n] for n in pre}
            records[k]=dict(pre=pre,post=post,delta=delta,batch_ids=idx.tolist(),pre_sha=state_sha(pre),post_sha=state_sha(post),update_sha=state_sha(delta),optimizer_memory=clone_optimizer(opt))
        if k in (1,25,100,200,350,steps):
            val_acc,val_loss=accuracy(model,test)
            eval_log.append(dict(step=k,test_accuracy=val_acc,test_loss=val_loss))
    records['final']=clone_state(model)
    if not formal:
        return dict(seed=seed,steps=steps,eval_log=eval_log)
    result=relation_tests(model,train,test,initial,records,seed)
    result['train_log']=eval_log
    result['identities']={'training_rows_sha256':hashlib.sha256(''.join(train[3]).encode()).hexdigest(),'test_rows_sha256':hashlib.sha256(''.join(test[3]).encode()).hexdigest(),'model_initial_sha256':state_sha(initial),'final_sha256':state_sha(records['final']),'updates':[{'step':k,'pre_sha':records[k]['pre_sha'],'post_sha':records[k]['post_sha'],'update_sha':records[k]['update_sha'],'batch_indices_sha':hashlib.sha256(str(records[k]['batch_ids']).encode()).hexdigest()} for k in CHECKPOINT_STEPS]}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--formal',action='store_true');ap.add_argument('--output',default='/mnt/data/gnn_tlir_results.json');ap.add_argument('--seed',type=int,default=None)
    args=ap.parse_args()
    if args.pilot and args.formal:raise RuntimeError('Use either --pilot or --formal')
    if args.pilot:
        results=[run(99999,formal=False)]
    elif args.formal:
        if not Path('/mnt/data/gnn_falsification_frozen.json').exists():raise RuntimeError('Missing frozen protocol')
        frozen=json.loads(Path('/mnt/data/gnn_falsification_frozen.json').read_text())
        source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if source_sha!=frozen['source_sha256']:raise RuntimeError('SOURCE_HASH_MISMATCH')
        results=[run(int(s),formal=True) for s in SEEDS]
    else:raise RuntimeError('Specify --pilot or --formal')
    out=Path(args.output)
    report={'schema':'gnn_tlir_falsification_v1','kind':'PILOT' if args.pilot else 'FORMAL','torch':torch.__version__,'python':platform.python_version(),'seeds':[99999] if args.pilot else list(SEEDS),'runs':results,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    for row in results:
        print(json.dumps({'seed':row['seed'],'eval_log':row.get('eval_log',row.get('train_log')),'gates':row.get('gates')},ensure_ascii=False))
    print(f'SAVED {out} SHA256 {hashlib.sha256(out.read_bytes()).hexdigest()}')

if __name__=='__main__':main()
