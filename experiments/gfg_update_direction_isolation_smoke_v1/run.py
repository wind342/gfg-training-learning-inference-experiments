"""Fixed task2 run; policy inputs never include test results."""
import argparse
import gc
import time
import traceback
from contextlib import contextmanager
from .core import *
from experiments.gfg_governor_fewrel_v1.model import RelationModel,BertInstrument
from experiments.gfg_governor_fewrel_v1.run import evaluate
from experiments.gfg_functional_feedback_router_v3.dose import accumulate
from experiments.gfg_controller_distributed_runtime_v2.identity import CanonicalParameterRegistry
from experiments.gfg_controller_distributed_runtime_v2.ownership import Ownership
from experiments.gfg_functional_update_governor_v2.scheduler import native_margins

def run(arm):
    unchanged();pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['status']=='PASS'
    assert sources()==pre['source_hashes'] and sha(CHECKPOINT)==CKSHA and sha(DATA/'data.npz')==DATA_SHA
    out=ROOT/arm;out.mkdir(parents=True)
    config=dict(arm=arm,base=BASE,source_hashes=sources(),preflight_sha256=sha(ROOT/'PREFLIGHT.json'),
        checkpoint_sha256=CKSHA,data_sha256=DATA_SHA,seed=2021,lr=5e-6,schedule=SCHEDULE,old_replay=0,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    write(out/'CONFIG.json',config);start=time.perf_counter();rows=[];result={}
    try:
        torch.set_num_threads(2);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        d=np.load(DATA/'data.npz');x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y'][:,None])
        probes,primary,full=scope(d,16);batches=order(d,2)
        model=RelationModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(.9,.999),weight_decay=0,fused=True,foreach=False)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);model.seen=16
        sentinel=extend(saved['scheduler']);initial=sentinel.snapshot(210);del saved
        px=x[probes].cuda();py=y[probes].cuda();sentinel.bind(px,py)
        owner=Ownership(model,CanonicalParameterRegistry(model)) if arm!='native' else None
        instrument=BertInstrument(model,chunk=4) if owner else None
        if instrument:instrument.alphas=ALPHAS
        last_actual=None;timing={};torch.cuda.synchronize();load_seconds=time.perf_counter()-start
        torch.cuda.reset_peak_memory_stats();train_start=time.perf_counter()
        @contextmanager
        def timed(key):
            torch.cuda.synchronize();tick=time.perf_counter()
            try:yield
            finally:
                torch.cuda.synchronize();timing[key]=timing.get(key,0.)+time.perf_counter()-tick
        with (out/'STEPS.jsonl').open('x',encoding='utf8') as log:
            for offset,batch in enumerate(batches,1):
                t=210+offset;seed=2021*100000+t;ids=batch.tolist();timing={};torch.cuda.synchronize();tick=time.perf_counter()
                detection=arm!='native' and offset in SCHEDULE
                row=dict(offset=offset,step=t,current_ids=ids,dropout_seed=seed,old_replay=0,detection=detection,alpha=1.,reason='NATIVE')
                with timed('training_gradient'):
                    row['loss']=accumulate(model,opt,x,y,[ids],seed)[0]
                if not detection:
                    with timed('optimizer_commit'):opt.step()
                else:
                    sentinel.begin(t) if offset==1 else None
                    sentinel.last_step=t # skipped sites have no observations or adaptive trigger
                    row['history_before_refresh']=sentinel.snapshot(t)
                    with timed('proposal'):owner.propose(opt)
                    opt.zero_grad(set_to_none=True)
                    state_identity=moment_identity(opt)
                    torch.cuda.empty_cache()
                    with timed('first_prediction'):m=instrument.measure(owner,px,py)
                    baseline=m.baseline_minima.cpu().numpy();native=m.predicted_minima.cpu().numpy()
                    row['baseline']=baseline.tolist();row['native_predicted']=native.tolist()
                    row['since_last_audit_correct_to_wrong']=[] if last_actual is None else np.flatnonzero((last_actual>0)&(baseline<=0)).tolist()
                    row['baseline_scope']='class_expansion_8_to_16' if last_actual is None else 'unguarded_inter_detection_updates'
                    sentinel.measured(t,m.baseline_minima);protected=eligible(sentinel)
                    row['receiving_history']=sentinel.snapshot(t);row['protected']=protected
                    row['lost_protected_since_last_audit']=[g for g in range(8) if row['history_before_refresh']['states'][g]=='PROTECTED' and g not in protected]
                    threatened=threats(native[-1],sentinel,protected);row['threatened']=threatened
                    row['native_delta_sha256']=tensor_hash(owner.proposal.delta)
                    projected=None;direction=native;cuts=[]
                    if arm=='projection' and threatened:
                        original={n:v.detach().cpu().clone() for n,v in owner.proposal.delta.items()}
                        selected=sorted(threatened,key=lambda g:(float(native[-1,g*10:g*10+10].min()),g))
                        for group in selected:
                            with timed('projection_gradient'):
                                gradient,probe=margin_gradient(model,px,py,list(range(group*10,group*10+10)))
                            with timed('projection_pass'):certificate=cut(owner.proposal.delta,gradient)
                            del gradient
                            probe['selected_probe_native_crossing']=bool(baseline[probe['probe_index']]>0 and native[-1,probe['probe_index']]<=0)
                            cuts.append(dict(group=group,probe=probe,**certificate))
                        row['geometry']=geometry(original,owner.proposal.delta);del original;gc.collect()
                        row['projected_delta_sha256']=tensor_hash(owner.proposal.delta)
                        torch.cuda.empty_cache()
                        with timed('second_prediction'):pm=instrument.measure(owner,px,py)
                        assert np.allclose(pm.baseline_minima.cpu().numpy(),baseline,rtol=0,atol=1e-6)
                        projected=pm.predicted_minima.cpu().numpy();direction=projected
                        row['projected_predicted']=projected.tolist()
                        row['second_prediction_delta_sha256']=tensor_hash(owner.proposal.delta)
                        assert row['second_prediction_delta_sha256']==row['projected_delta_sha256']
                    alpha,no_safe,scores=choose(direction,sentinel,protected)
                    if not threatened:assert alpha==1. and not no_safe
                    row.update(alpha=alpha,no_safe=no_safe,safety_scores=scores,cuts=cuts,
                        reason=('NO_SAFE_PROJECTED_ALPHA' if projected is not None else 'NO_SAFE_SCALAR_ALPHA') if no_safe else 'SHRINK' if alpha<1 else 'SAFE_FULL')
                    chosen=direction[ALPHAS.index(alpha)];row['executed_predicted']=chosen.tolist()
                    row['aggregates']={k:aggregates(v,sentinel.mapping,baseline) for k,v in
                        [('receiving',baseline),('native',native[-1]),('projected',direction[-1]),('executed',chosen)]}
                    gains={k:row['aggregates'][k]['new_correct']-row['aggregates']['receiving']['new_correct'] for k in ('native','projected','executed')}
                    row['new_correct_gains']=gains;row['projected_native_gain_ratio']=gains['projected']/gains['native'] if gains['native']>0 else None
                    with timed('optimizer_commit'):
                        if projected is not None:commit_projected(owner.parameters(),owner.proposal.delta,alpha)
                        else:owner.commit(alpha)
                    assert moment_identity(opt)==state_identity
                    row['moments_identity_versions_unchanged']=True
                    with timed('endpoint_audit'):
                        actual=np.concatenate([native_margins(model,px[b:b+16],py[b:b+16]) for b in range(0,160,16)])
                    row['actual']=actual.tolist();last_actual=actual
                    sentinel.measured(t,torch.tensor(actual,dtype=torch.float64));row['history_after']=sentinel.snapshot(t)
                torch.cuda.synchronize();row['update_seconds']=time.perf_counter()-tick;row['timing']=timing
                rows.append(row);log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');log.flush()
                if offset%10==1 or offset==210:print('STEP',arm,offset,row['alpha'],row.get('threatened',[]),round(row['update_seconds'],2),flush=True)
                if time.perf_counter()-train_start>21600:raise TimeoutError('frozen six-hour arm cap')
        torch.cuda.synchronize();training_wall=time.perf_counter()-train_start
        eval_start=time.perf_counter();ev=evaluate(model,x,y,full);pred=np.array(ev['predictions']);cleanpred=pred[np.isin(full,primary)]
        metrics=dict(primary=summarize_predictions(primary,cleanpred,d,2),original=summarize_predictions(full,pred,d,2),indices=full.tolist(),predictions=pred.tolist())
        evaluation_seconds=time.perf_counter()-eval_start
        with (out/'final.pt').open('xb') as stream:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),seen=16,step=420,scheduler=sentinel),stream)
        result=dict(status='COMPLETED',arm=arm,initial_state=initial,metrics=metrics,steps=len(rows),load_seconds=load_seconds,
            training_wall=training_wall,evaluation_seconds=evaluation_seconds,wall=time.perf_counter()-start,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            log_sha256=sha(out/'STEPS.jsonl'),config_sha256=sha(out/'CONFIG.json'),checkpoint_sha256=sha(out/'final.pt'))
    except Exception:
        result=dict(status='FAILED',arm=arm,steps=len(rows),traceback=traceback.format_exc(),wall=time.perf_counter()-start)
    write(out/'RESULT.json',result);print('RESULT',arm,result['status'],result.get('metrics',{}).get('primary',{}),flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=ARMS);run(p.parse_args().arm)
