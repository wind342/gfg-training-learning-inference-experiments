"""One fixed training run; exact-control cuts only, no test inference."""
import traceback
from .core import *

def main():
    pre=guard();setup();out=ROOT/ARM;out.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
    write(out/'CONFIG.json',dict(base=BASE,arm=ARM,lr=LR,seed=2021,steps=210,batch=16,old_replay=0,source_hashes=sources(),
        checkpoint_sha256=CKSHA,data_sha256=DATA_SHA,preflight_sha256=sha(ROOT/'PREFLIGHT.json')))
    try:
        d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y']);probes,_,_=scope(d,16)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model);model.seen=16
        shadow,opt=make_candidate(model,saved['optimizer']);del saved
        assert head_hash(model.head)==pre['head_initial_sha256'] and moments_hash(opt)==pre['head_optimizer_initial_sha256']
        set_lr(opt,LR);initialopt=copy.deepcopy(opt.state_dict());bh=tensor_hash(model.bert.state_dict());assert bh==pre['backbone_initial_sha256']
        versions=[p._version for p in model.bert.parameters()]
        with torch.no_grad():h=torch.cat([model.features(x[probes[b:b+16]].cuda()) for b in range(0,160,16)])
        labels=y[probes].cuda();hhash=tensor_hash({'H':h});hv=h._version
        history=[cpu_state(snapshot(model.head))];natives=[];cuts=[]
        torch.cuda.synchronize();trainstart=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as f:
            for step,batch in enumerate(order(d,2),1):
                tick=time.perf_counter();ids=batch.tolist();seed=2021*100000+210+step
                model.train();assert all(not m.training for m in model.bert.modules()) and opt.param_groups[0]['lr']==LR
                model.head.zero_grad(set_to_none=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():features=model.features(x[ids].cuda())
                loss=nn.functional.cross_entropy(model.head(features)[:,:16],y[ids].cuda());loss.backward()
                norm=nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm)
                propose(model.head,shadow,opt);before=snapshot(model.head);native=snapshot(shadow)
                rh=rng();mh=moments_hash(opt);hh=head_hash(model.head)
                executed,candidate,report=exact_cut(before,native,h,labels)
                assert equal(rh,rng()) and mh==moments_hash(opt) and hh==head_hash(model.head)
                assert versions==[p._version for p in model.bert.parameters()] and all(p.grad is None and not p.requires_grad for p in model.bert.parameters()) and h._version==hv
                with torch.no_grad():
                    for k,p in model.head.named_parameters():p.copy_(executed[k])
                actual=margins(logits(model.head,h),labels).cpu().tolist()
                expected=report['baseline'] if report['skip'] else report['cut'];assert actual==expected
                assert not any(report['baseline'][i]>0 and actual[i]<=0 for i in range(80))
                stats={k:counts(report[k]) for k in ('baseline','native','cut')};stats['executed']=counts(actual)
                report.update(executed=actual,stats=stats,new_gains={k:stats[k]['new_correct']-stats['baseline']['new_correct'] for k in ('native','cut','executed')},
                    cut_geometry=geometry(before,native,candidate),executed_geometry=geometry(before,native,executed))
                history.append(cpu_state(executed));natives.append(cpu_state(native));cuts.append(cpu_state(candidate))
                torch.cuda.synchronize();row=dict(step=step,current_ids=ids,seed=seed,lr=LR,old_replay=0,loss=float(loss.detach()),grad_norm=float(norm),
                    report=report,head_before_sha256=hh,head_after_sha256=head_hash(model.head),native_sha256=tensor_hash(native),cut_sha256=tensor_hash(candidate),
                    optimizer_sha256=mh,moments_unmodified_by_cut=True,bert_frozen_eval=True,seconds=time.perf_counter()-tick)
                rows.append(row);f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()
                if step in (1,50,100,150,210):print('STEP',step,row['loss'],stats['executed'],report['action'],flush=True)
                if time.perf_counter()-trainstart>7200:raise TimeoutError('fixed cap')
        torch.cuda.synchronize();wall=time.perf_counter()-trainstart
        assert bh==tensor_hash(model.bert.state_dict()) and hhash==tensor_hash({'H':h})
        with (out/'AUDIT.pt').open('xb') as f:torch.save(dict(heads=history,natives=natives,cuts=cuts,control_features=h.cpu(),
            initial_optimizer=initialopt,final_optimizer=opt.state_dict(),base_checkpoint_sha256=CKSHA),f)
        result=dict(status='COMPLETED',arm=ARM,lr=LR,steps=210,no_test_inference=True,finished_unix=time.time(),
            training_wall=wall,total_wall=time.perf_counter()-start,projection_seconds=sum(r['report']['projection_seconds'] for r in rows),
            recheck_seconds=sum(r['report']['recheck_seconds'] for r in rows),filter_seconds=sum(r['report']['filter_seconds'] for r in rows),
            projected_steps=sum(bool(r['report']['native_violating']) for r in rows),first_pass_safe_steps=sum(r['report']['first_pass_safe'] is True for r in rows),
            second_pass_steps=sum(r['report']['second_pass'] for r in rows),skipped_steps=sum(r['report']['skip'] for r in rows),
            final_controls=rows[-1]['report']['stats']['executed'],losses={str(k):rows[k-1]['loss'] for k in (1,50,100,150,210)},
            head_sha256=head_hash(model.head),optimizer_sha256=moments_hash(opt),backbone_sha256=bh,cache_sha256=hhash,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            audit_sha256=sha(out/'AUDIT.pt'),log_sha256=sha(out/'STEPS.jsonl'),config_sha256=sha(out/'CONFIG.json'))
    except Exception:result=dict(status='FAILED',steps=len(rows),traceback=traceback.format_exc())
    write(out/'TRAIN_RESULT.json',result);print('TRAIN RESULT',result['status'],flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])

if __name__=='__main__':main()
