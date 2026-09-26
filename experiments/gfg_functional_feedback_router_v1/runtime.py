"""Router-before-gradient, measured proposal gate, post-commit observations."""
import time
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v2.scheduler import native_margins
from .policy import emergency_gate

def accumulate(model,opt,x,y,batches,seed):
    opt.zero_grad(set_to_none=True);model.train();losses=[]
    for j,ids in enumerate(batches):
        torch.manual_seed(seed+j*10000000)
        if x.device.type=='cuda':torch.cuda.manual_seed_all(seed+j*10000000)
        # Data may remain in CPU RAM; transfer only the current microbatch.
        device=next(model.parameters()).device
        loss=model(x[ids].to(device),y[ids].to(device))[1]
        (loss/len(batches)).backward();losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not np.isfinite(losses).all() or not torch.isfinite(norm):raise FloatingPointError('nonfinite training')
    return losses

def update(model,opt,engine,router,memory,x,y,current_ids,px,py,t,old_groups,seed):
    if engine.failed:raise RuntimeError('failed runtime cannot continue')
    torch.cuda.synchronize();start=time.perf_counter()
    try:
        s=engine.scheduler;s.bind(px,py)
        # Router reads only history already present, never current endpoint/test.
        decision=router.choose(t,s.mapping.labels,list(s.state),s.level.tolist(),old_groups,memory)
        pending=getattr(engine,'router_pending',[])
        reasons=sorted(set(s.begin(t)+pending));ids=[];pre_events=[]
        if not reasons:
            ids=s.shard(t)
            if ids:pre_events=s.observe(ids,native_margins(model,px[ids],py[ids]))
            reasons=s.reasons(pre_events)
        batches=[current_ids]
        if decision['feedback_example_ids']:batches.append(decision['feedback_example_ids'])
        losses=accumulate(model,opt,x,y,batches,seed)
        info=dict(**decision,current_loss=losses[0],feedback_loss=losses[1] if len(losses)>1 else None,
            alpha=1.,gate_reason='NOT_MEASURED',instrument_called=bool(reasons),instrument_reasons=reasons,
            pre_events=pre_events,pre_observation_phase='receiving-state',post_observation_phase='committed-endpoint')
        post=[]
        if reasons:
            engine.owner.propose(opt)
            torch.cuda.empty_cache() # preserve the resource-only workaround
            m=engine.instrument.measure(engine.owner,px,py)
            gate=emergency_gate(m,s.mapping)
            pre_full=s.measured(t,m.baseline_minima)
            engine.owner.commit(gate['alpha'])
            actual=[]
            for b in range(0,len(px),16):actual.extend(native_margins(model,px[b:b+16],py[b:b+16]).tolist())
            # Register ACTUAL endpoint as next-step history, not as pre-step data.
            post=s.measured(t,torch.tensor(actual,dtype=torch.float64))
            info.update(alpha=gate['alpha'],gate_reason=gate['reason'],gate=gate,
                baseline=m.baseline_minima.cpu().tolist(),predicted=m.predicted_minima.cpu().tolist(),
                actual=actual,pre_full_events=pre_full)
        else:
            opt.step()
            if ids:post=s.observe(ids,native_margins(model,px[ids],py[ids]),source='post_micro')
        engine.router_pending=s.reasons(post)
        info.update(post_events=post,history=s.snapshot(t),next_instrument_reasons=engine.router_pending)
        torch.cuda.synchronize();info['seconds']=time.perf_counter()-start
        return info
    except Exception:
        engine.failed=True
        raise
