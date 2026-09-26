"""Training and control observations only: no test inference/imported evaluator."""
import argparse
import time
import traceback
from .core import *

def run(arm):
    pre=guard();setup();out=ROOT/arm;out.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
    write(out/'CONFIG.json',dict(base=BASE,arm=arm,lr=ARMS[arm],seed=2021,steps=210,batch=16,old_replay=0,source_hashes=sources(),
        preflight_sha256=sha(ROOT/'PREFLIGHT.json'),data_sha256=DATA_SHA,checkpoint_sha256=CKSHA))
    try:
        d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y']);probes,_,_=scope(d,16)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model);model.seen=16
        shadow,opt=make_candidate(model,saved['optimizer']);s=extend(saved['scheduler']);del saved
        assert head_hash(model.head)==pre['head_initial_sha256'] and moments_hash(opt)==pre['head_optimizer_initial_sha256']
        set_lr(opt,ARMS[arm]);initial_opt=copy.deepcopy(opt.state_dict())
        b_hash=tensor_hash(model.bert.state_dict());assert b_hash==pre['backbone_initial_sha256'];versions=[p._version for p in model.bert.parameters()]
        with torch.no_grad():h=torch.cat([model.features(x[probes[b:b+16]].cuda()) for b in range(0,160,16)])
        labels=y[probes].cuda();h_hash=tensor_hash({'H':h});hv=h._version
        history=[{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()}]
        torch.cuda.synchronize();train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as log:
            for step,batch in enumerate(order(d,2),1):
                tick=time.perf_counter();ids=batch.tolist();seed=2021*100000+210+step
                model.train();assert all(not m.training for m in model.bert.modules())
                assert opt.param_groups[0]['lr']==ARMS[arm]
                model.head.zero_grad(set_to_none=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():features=model.features(x[ids].cuda())
                loss=nn.functional.cross_entropy(model.head(features)[:,:16],y[ids].cuda());loss.backward()
                norm=nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm)
                propose(model.head,shadow,opt)
                rh=rng();mh=moments_hash(opt);hh=head_hash(model.head)
                baseline=margins(logits(model.head,h),labels);candidate=margins(logits(shadow,h),labels)
                obs=observe(s,baseline,candidate,210+step)
                assert equal(rh,rng()) and mh==moments_hash(opt) and hh==head_hash(model.head)
                assert all(p.grad is None and not p.requires_grad for p in model.bert.parameters())
                assert versions==[p._version for p in model.bert.parameters()] and h._version==hv
                commit(model.head,shadow);assert torch.equal(candidate,margins(logits(model.head,h),labels))
                history.append({k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()})
                torch.cuda.synchronize()
                row=dict(step=step,current_ids=ids,seed=seed,lr=opt.param_groups[0]['lr'],loss=float(loss),grad_norm=float(norm),
                    alpha=1,old_replay=0,head_before_sha256=hh,head_after_sha256=head_hash(model.head),optimizer_sha256=mh,
                    observation=obs,monitor_readonly=True,bert_frozen_eval=True,executed_candidate_exact=True,seconds=time.perf_counter()-tick)
                rows.append(row);log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');log.flush()
                if step in (1,50,100,150,210):print('STEP',arm,step,row['loss'],obs['candidate_stats']['old_correct'],obs['candidate_stats']['new_correct'],flush=True)
                if time.perf_counter()-train_start>7200:raise TimeoutError('fixed cap')
        torch.cuda.synchronize();wall=time.perf_counter()-train_start
        assert b_hash==tensor_hash(model.bert.state_dict()) and h_hash==tensor_hash({'H':h})
        with (out/'AUDIT.pt').open('xb') as f:torch.save(dict(heads=history,control_features=h.cpu(),initial_optimizer=initial_opt,
            final_optimizer=opt.state_dict(),base_checkpoint_sha256=CKSHA),f)
        result=dict(status='COMPLETED',arm=arm,lr=ARMS[arm],steps=210,no_test_inference=True,training_wall=wall,total_wall=time.perf_counter()-start,
            finished_unix=time.time(),head_sha256=head_hash(model.head),backbone_sha256=b_hash,cache_sha256=h_hash,
            optimizer_sha256=moments_hash(opt),losses={str(i):rows[i-1]['loss'] for i in (1,50,100,150,210)},
            final_controls=rows[-1]['observation']['candidate_stats'],old_correct_to_wrong=sum(int(np.sum((np.array(r['observation']['baseline'])[:80]>0)&(np.array(r['observation']['candidate'])[:80]<=0))) for r in rows),
            newly_unsafe_steps=sum(bool(r['observation']['newly_threatened_by_this_update']) for r in rows),
            newly_unsafe_group_occurrences=sum(len(r['observation']['newly_threatened_by_this_update']) for r in rows),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            audit_sha256=sha(out/'AUDIT.pt'),log_sha256=sha(out/'STEPS.jsonl'),config_sha256=sha(out/'CONFIG.json'))
    except Exception:
        result=dict(status='FAILED',arm=arm,steps=len(rows),traceback=traceback.format_exc())
    write(out/'TRAIN_RESULT.json',result);print('TRAIN RESULT',arm,result['status'],flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=list(ARMS));run(p.parse_args().arm)
