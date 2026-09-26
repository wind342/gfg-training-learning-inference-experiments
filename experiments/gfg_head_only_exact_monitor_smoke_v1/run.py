import argparse
import time
import traceback
from contextlib import contextmanager
from .core import *
from experiments.gfg_governor_fewrel_v1.run import evaluate

def run(arm):
    unchanged();pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['status']=='PASS' and pre['source_hashes']==sources()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    out=ROOT/arm;out.mkdir(exist_ok=False);start=time.perf_counter();result={};rows=[]
    write(out/'CONFIG.json',dict(arm=arm,base=BASE,source_hashes=sources(),preflight_sha256=sha(ROOT/'PREFLIGHT.json'),
          seed=2021,lr=5e-6,steps=210,batch=16,seen=16,old_replay=0,checkpoint_sha256=CKSHA,data_sha256=DATA_SHA))
    try:
        torch.set_num_threads(2);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y'][:,None]);probes,primary,full=scope(d,16)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        model=FrozenRelationModel().cuda();model.load_state_dict(saved['model']);freeze(model)
        shadow,opt=make_candidate(model,saved['optimizer']);sentinel=extend(saved['scheduler'])
        initial_anchor=sentinel.anchor.tolist();del saved
        assert head_hash(model.head)==pre['head_initial_sha256'] and moments_hash(opt)==pre['head_optimizer_initial_sha256']
        backbone_start=tensor_hash(model.bert.state_dict());assert backbone_start==pre['backbone_initial_sha256']
        versions=[p._version for p in model.bert.parameters()]
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.no_grad():h=torch.cat([model.features(x[probes[b:b+16]].cuda()) for b in range(0,160,16)])
        torch.cuda.synchronize();cache_seconds=time.perf_counter()-tick
        hy=y[probes,-1].cuda();cache_hash=tensor_hash({'H':h});hversion=h._version
        initial={};oldfull=full[d['y'][full]<8]
        for seen in (8,16):
            model.seen=seen;ev=evaluate(model,x,y,oldfull)
            cm=margins(logits(model.head,h[:80],seen),hy[:80]).cpu().numpy()
            initial[str(seen)]=dict(test=ev,old_test_macro=float(np.mean(list(ev['per_relation'].values()))),
                old_control_capability=(cm.reshape(8,10)>0).mean(1).tolist(),old_control_margins=cm.tolist())
        model.seen=16
        history=[{n:v.detach().cpu().clone() for n,v in model.head.state_dict().items()}]
        checkpoint_optimizer_initial=copy.deepcopy(opt.state_dict())
        torch.cuda.synchronize();setup_seconds=time.perf_counter()-start
        setup_peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats();train_start=time.perf_counter();io_seconds=0.
        @contextmanager
        def timed(key):
            torch.cuda.synchronize();tick=time.perf_counter()
            try:yield
            finally:torch.cuda.synchronize();timing[key]=time.perf_counter()-tick
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as log:
            for offset,batch in enumerate(order(d,2),1):
                timing={};torch.cuda.synchronize();begin=time.perf_counter();ids=batch.tolist();seed=2021*100000+210+offset
                row=dict(offset=offset,current_ids=ids,seed=seed,old_replay=0,alpha=1.,monitor=arm==ARMS[1])
                model.train();assert all(not m.training for m in model.bert.modules())
                model.head.zero_grad(set_to_none=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with timed('current_bert_forward'):
                    with torch.no_grad():features=model.features(x[ids].cuda())
                with timed('head_forward_loss'):
                    loss=nn.functional.cross_entropy(model.head(features)[:,:16],y[ids,-1].cuda())
                with timed('head_backward_clip'):
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
                assert torch.isfinite(loss) and torch.isfinite(norm)
                with timed('proposal'):propose(model.head,shadow,opt)
                with timed('pre_monitor_audit'):
                    before_rng=rng();before_head=head_hash(model.head);before_moment=moments_hash(opt)
                if row['monitor']:
                    with timed('cached_endpoint_calculation'):
                        baseline=margins(logits(model.head,h),hy);candidate=margins(logits(shadow,h),hy)
                    with timed('capability_q10_statistics'):row['observation']=observe(sentinel,baseline,candidate,210+offset)
                with timed('post_monitor_audit'):
                    assert equal(before_rng,rng()) and before_head==head_hash(model.head) and before_moment==moments_hash(opt)
                    assert all(p.grad is None and not p.requires_grad for p in model.bert.parameters())
                    assert versions==[p._version for p in model.bert.parameters()] and h._version==hversion
                    row.update(loss=float(loss),head_grad_norm=float(norm),head_before_sha256=before_head,optimizer_sha256=before_moment,
                               monitor_readonly=True,bert_no_grad=True,bert_eval=True)
                with timed('commit'):commit(model.head,shadow)
                with timed('record_audit'):
                    row['head_after_sha256']=head_hash(model.head);history.append({n:v.detach().cpu().clone() for n,v in model.head.state_dict().items()})
                    if row['monitor']:
                        assert torch.equal(candidate,margins(logits(model.head,h),hy))
                        row['executed_candidate_exact']=True
                torch.cuda.synchronize();row['inclusive_update_seconds']=time.perf_counter()-begin;row['timing']=timing
                row['compute_update_seconds']=sum(v for k,v in timing.items() if k not in ('pre_monitor_audit','post_monitor_audit','record_audit'))
                row['monitor_seconds']=timing.get('cached_endpoint_calculation',0)+timing.get('capability_q10_statistics',0)
                rows.append(row);tick=time.perf_counter();log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');log.flush();io_seconds+=time.perf_counter()-tick
                if offset%30==0:print('STEP',arm,offset,round(row['loss'],5),round(row['monitor_seconds'],6),flush=True)
                if time.perf_counter()-train_start>7200:raise TimeoutError('fixed two-hour cap')
        torch.cuda.synchronize();training_wall=time.perf_counter()-train_start
        peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        tick=time.perf_counter();ev=evaluate(model,x,y,full);pred=np.asarray(ev['predictions']);metrics=dict(
            primary=summarize_predictions(primary,pred[np.isin(full,primary)],d,2),original=summarize_predictions(full,pred,d,2),
            indices=full.tolist(),predictions=pred.tolist());eval_seconds=time.perf_counter()-tick
        assert backbone_start==tensor_hash(model.bert.state_dict()) and cache_hash==tensor_hash({'H':h})
        assert head_hash(model.head)!=pre['head_initial_sha256']
        with (out/'AUDIT.pt').open('xb') as f:torch.save(dict(heads=history,control_features=h.cpu(),initial_optimizer=checkpoint_optimizer_initial,
            final_optimizer=opt.state_dict(),base_checkpoint_sha256=CKSHA),f)
        result=dict(status='COMPLETED',arm=arm,steps=210,initial=initial,initial_anchor=initial_anchor,metrics=metrics,
            backbone_unchanged=True,backbone_sha256=backbone_start,head_sha256=head_hash(model.head),optimizer_sha256=moments_hash(opt),
            cache_sha256=cache_hash,cache_shape=list(h.shape),cache_seconds=cache_seconds,cache_bytes=h.numel()*h.element_size(),
            setup_seconds=setup_seconds,training_wall=training_wall,compute_update_seconds=sum(r['compute_update_seconds'] for r in rows),
            inclusive_update_seconds=sum(r['inclusive_update_seconds'] for r in rows),monitor_seconds=sum(r['monitor_seconds'] for r in rows),
            timing={k:sum(r['timing'].get(k,0) for r in rows) for k in sorted(set(k for r in rows for k in r['timing']))},
            json_io_seconds=io_seconds,evaluation_seconds=eval_seconds,wall=time.perf_counter()-start,
            peak=peak,process_peak={k:max(v,setup_peak[k]) for k,v in peak.items()},old_replay=0,
            log_sha256=sha(out/'STEPS.jsonl'),audit_sha256=sha(out/'AUDIT.pt'),config_sha256=sha(out/'CONFIG.json'))
    except Exception:
        result=dict(status='FAILED',arm=arm,steps=len(rows),traceback=traceback.format_exc(),wall=time.perf_counter()-start)
    write(out/'RESULT.json',result);print('RESULT',arm,result,flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=ARMS);run(p.parse_args().arm)
