"""One paired task1 timing experiment; production policy imported unchanged."""
import copy
import json
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.gfg_head_only_lr_smoke_v1.core import setup
from experiments.gfg_head_only_exact_monitor_smoke_v1.core import (
    FrozenRelationModel, freeze, propose, head_hash, moments_hash, equal, tensor_hash,
)
from experiments.gfg_final_architecture_v1_ten_task.core import (
    clean_validation, isolation, order, scope, DATA, DATA_SHA, sha, write,
)
from experiments.gfg_relation_capability_floor_task8_v1.core import floor_projection
from experiments.gfg_head_only_hildreth_roundoff_smoke_v1.core import snapshot, margins, affine
from experiments.gfg_governor_fewrel_v1.prepare import MODEL

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/task1-conflict-free-timing-v1')
BASE='f24d37cd0d23c4d32c68bebefd3345f662016788'
ARMS=('head_only_native','current_architecture')

def sync(): torch.cuda.synchronize()

def optimizer(head):
    return torch.optim.AdamW(head.parameters(),lr=5e-5,betas=(.9,.999),weight_decay=0,fused=True,foreach=False)

@torch.no_grad()
def features(m,x,ids):
    return torch.cat([m.features(x[ids[j:j+16]].cuda()).cpu() for j in range(0,len(ids),16)])

@torch.no_grad()
def score(head,h,y):
    pred=head(h)[:,:8].argmax(1)
    per=[float((pred[y==g]==g).float().mean()) for g in range(8)]
    return dict(accuracy=float((pred==y).float().mean()),macro=float(np.mean(per)),
                per_relation=per,predictions=pred.cpu().tolist())

def main():
    assert not ROOT.exists(), 'Never overwrite an existing run'
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)
    setup(); ROOT.mkdir(parents=True)
    write(ROOT/'EXECUTION.json',dict(status='STARTED',base=BASE,started_unix=time.time()))
    try:
        assert sha(DATA/'data.npz')==DATA_SHA
        d=np.load(DATA/'data.npz'); iso=isolation(d)
        source_paths=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
        source_paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
        sources={p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(source_paths))}
        pretrained={str(p):sha(p) for p in MODEL.iterdir() if p.is_file()}
        m=FrozenRelationModel().cuda();freeze(m);m.eval();m.seen=8
        backbone=tensor_hash(m.bert.state_dict());versions=[p._version for p in m.bert.parameters()]
        init=copy.deepcopy(m.head);x=torch.from_numpy(d['x']);truth=torch.from_numpy(d['y'])
        batches=order(d,1);assert batches.shape==(210,16)
        cv=clean_validation(d);val_ids=cv[d['y'][cv]<8]
        tick=time.perf_counter();vh=features(m,x,val_ids).cuda();sync();cache_seconds=time.perf_counter()-tick
        vy=truth[val_ids].cuda();positions={int(i):j for j,i in enumerate(val_ids)}
        monitor_ids=scope(d,8)[0];h=vh[[positions[int(i)] for i in monitor_ids]];y=truth[monitor_ids].cuda()
        assert h.shape==(80,1536)
        states={}
        for arm in ARMS:
            head=copy.deepcopy(init);shadow=copy.deepcopy(init) if arm==ARMS[1] else None
            opt=optimizer(shadow if shadow is not None else head)
            states[arm]=(head,shadow,opt)
        torch.save(dict(head=init.state_dict(),bert_sha256=backbone),ROOT/'INITIAL.pt')
        write(ROOT/'PREFLIGHT.json',dict(status='PASS',sources=sources,data_sha256=DATA_SHA,
            pretrained_files=pretrained,backbone_sha256=backbone,initial_head_sha256=head_hash(init),
            initial_sha256=sha(ROOT/'INITIAL.pt'),isolation=iso,batch_order=batches.tolist(),
            validation_ids=val_ids.tolist(),monitor_ids=monitor_ids.tolist(),cache_seconds=cache_seconds,
            task1_checkpoint_loaded=False,optimizer_initial_step=0,seed=2021,lr=5e-5))
        # Warmup is on disposable state, never an experimental head.
        warm=copy.deepcopy(init);wopt=optimizer(warm)
        for _ in range(5):
            with torch.no_grad(): f=m.features(x[batches[0]].cuda())
            warm.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(warm(f)[:,:8],truth[batches[0]].cuda()).backward()
            nn.utils.clip_grad_norm_(warm.parameters(),1.);wopt.step()
        floor_projection(snapshot(init),snapshot(init),h,y,0,[],[],seen=8)
        del warm,wopt,f;sync();torch.cuda.reset_peak_memory_stats()
        curve=[dict(step=0,**{a:score(s[0],vh,vy) for a,s in states.items()})]
        rows=[]
        with (ROOT/'STEPS.jsonl').open('x',encoding='utf8') as ledger:
            for step,batch in enumerate(batches,1):
                row=dict(step=step,ids=batch.tolist(),arms={});native_state=None
                for arm in (ARMS if step%2 else ARMS[::-1]):
                    head,shadow,opt=states[arm]
                    torch.manual_seed(2021*100000+step);torch.cuda.manual_seed_all(2021*100000+step)
                    sync();start=time.perf_counter()
                    with torch.no_grad():f=m.features(x[batch].cuda())
                    target=truth[batch].cuda();sync();t1=time.perf_counter()
                    head.zero_grad(set_to_none=True)
                    loss=nn.functional.cross_entropy(head(f)[:,:8],target);loss.backward()
                    norm=nn.utils.clip_grad_norm_(head.parameters(),1.)
                    assert torch.isfinite(loss) and torch.isfinite(norm)
                    if shadow is None:opt.step()
                    else:propose(head,shadow,opt)
                    sync();t2=time.perf_counter();report=None
                    if shadow is not None:
                        before=snapshot(head);native=snapshot(shadow)
                        executed,_,report=floor_projection(before,native,h,y,0,[],[],seen=8)
                        with torch.no_grad():
                            for name,p in head.named_parameters():p.copy_(executed[name])
                        actual=margins(affine(snapshot(head),h,8),y).cpu().tolist()
                        assert actual==report['executed'] and equal(executed,native)
                        assert not report['projection_triggered'] and not report['skip']
                    sync();end=time.perf_counter()
                    row['arms'][arm]=dict(loss=float(loss),seconds=end-start,bert_seconds=t1-start,
                        head_update_seconds=t2-t1,controller_seconds=end-t2,
                        head_sha256=head_hash(head),moments_sha256=moments_hash(opt),
                        projection=bool(report and report['projection_triggered']),skip=bool(report and report['skip']))
                assert equal(states[ARMS[0]][0].state_dict(),states[ARMS[1]][0].state_dict())
                assert equal(states[ARMS[0]][2].state_dict(),states[ARMS[1]][2].state_dict())
                assert row['arms'][ARMS[0]]['loss']==row['arms'][ARMS[1]]['loss']
                assert versions==[p._version for p in m.bert.parameters()]
                assert all(not p.requires_grad and p.grad is None for p in m.bert.parameters())
                assert all(not sub.training for sub in m.bert.modules())
                rows.append(row);ledger.write(json.dumps(row,allow_nan=False)+'\n');ledger.flush()
                if step%30==0:
                    curve.append(dict(step=step,**{a:score(s[0],vh,vy) for a,s in states.items()}))
                    print('STEP',step,'validation',curve[-1][ARMS[0]]['macro'],flush=True)
        peak=torch.cuda.max_memory_allocated()
        assert tensor_hash(m.bert.state_dict())==backbone
        write(ROOT/'TRAINING_SEALED.json',dict(steps=210,finished_unix=time.time(),ledger_sha256=sha(ROOT/'STEPS.jsonl')))
        # No test tensor/labels have been evaluated before the training barrier.
        ti=scope(d,8)[1];th=features(m,x,ti).cuda();ty=truth[ti].cuda()
        test={a:score(s[0],th,ty) for a,s in states.items()}
        tr=batches.flatten();trh=features(m,x,tr).cuda();try_=truth[tr].cuda()
        training={a:score(s[0],trh,try_) for a,s in states.items()}
        summary={}
        for arm in ARMS:
            times={key:sum(r['arms'][arm][key] for r in rows) for key in ('seconds','bert_seconds','head_update_seconds','controller_seconds')}
            times['median_step_seconds']=float(np.median([r['arms'][arm]['seconds'] for r in rows]))
            crossings={str(threshold):next((r['step'] for r in curve if r[arm]['macro']>=threshold),None) for threshold in (.9,.95)}
            summary[arm]=dict(timing=times,train_accuracy=training[arm]['macro'],test_accuracy=test[arm]['macro'],
                validation_accuracy=curve[-1][arm]['macro'],first_observed_validation_crossing=crossings)
        torch.save({a:dict(head=s[0].state_dict(),optimizer=s[2].state_dict()) for a,s in states.items()},ROOT/'FINAL.pt')
        assert all(sha(REPO/p)==v for p,v in sources.items())
        write(ROOT/'RESULTS.json',dict(summary=summary,curve=curve,test=test,test_ids=ti.tolist(),training=training,
            peak_cuda_bytes=peak,cache_seconds=cache_seconds,final_sha256=sha(ROOT/'FINAL.pt'),
            controller_over_native_wall_ratio=summary[ARMS[1]]['timing']['seconds']/summary[ARMS[0]]['timing']['seconds']))
        write(ROOT/'CHECKS.json',dict(status='PASS',steps=210,head_and_moments_identical_every_step=True,
            projection_steps=0,skips=0,bert_byte_identical=True,source_identity=True,initial_task1_leakage=False,
            data_isolation=iso['status'],test_after_training=True,actual_fp32_endpoint_checks=210))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
