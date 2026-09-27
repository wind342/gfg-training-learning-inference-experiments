"""Final actuator: A/B/C, one first-sweep check, no detailed evidence by default."""
from .runtime import *
from .geometry import Geometry,select,inner
from .solver import ball_hildreth

def repair(geom,d,z):
    predicted=geom.g+geom.apply(d)
    actual=(z[geom.rows,geom.target]-z[geom.rows,geom.comp]).astype(np.float64)
    target=EPS+geom.reserve;start=float(np.maximum(target-geom.g,0).sum())
    pr=start-float(np.maximum(target-predicted,0).sum());ar=start-float(np.maximum(target-actual,0).sum())
    return predicted,actual,pr,ar,ar/pr if pr>0 else None

def decision(status,safe,outer):
    if safe:return 'FIRST_SWEEP_ACTUAL_SAFE_ACCEPT' if status=='FIRST_SWEEP_ACTUAL_SAFE' else 'INNER_LIMIT_ACTUAL_SAFE_ACCEPT' if status=='INNER_LIMIT' else 'PASS_ACTUAL_SAFE_ACCEPT'
    return 'OUTER_LIMIT_ROLLBACK' if outer==8 else 'INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE' if status=='INNER_LIMIT' else 'PASS_ACTUAL_UNSAFE_CONTINUE'

@torch.no_grad()
def project(before,native,bank,K,prediction,native_endpoint,before_endpoint,capture_evidence=False):
    meter=bank.meter;z=native_endpoint;nativeC=bank.counts(z);trigger=bool((nativeC<K).any())
    selected=set();pairs=set();rounds=[];brief=[];executed=native;reason=None;native_cpu=None
    if trigger:
        with meter.region('parameter_geometry',True):
            native_cpu=cpu(native);before_cpu=cpu(before);delta=tuple(n-b for n,b in zip(native_cpu,before_cpu))
            base_rho=2*np.sqrt(inner(delta,delta));rho=base_rho;current=native_cpu
        for outer in range(1,9):
            with meter.region('selection'):
                selection_z=prediction if outer==1 and prediction is not None and np.any(bank.counts(prediction)<K) else z
                C,D,chosen=select(bank,selection_z,K,selected,pairs);ordered=sorted(pairs)
            with meter.region('jacobian_gram_guard'):
                geom=Geometry(bank,current,ordered,z)
                if outer==1:d0=None;ad0=np.zeros(len(ordered),dtype=np.float64);d0n=0.
                else:
                    d0=tuple(n-p for n,p in zip(native_cpu,current));ad0=geom.apply(d0);d0n=inner(d0,d0)
                b=EPS+geom.reserve-geom.g
            def materialize(coef,dual):
                with meter.region('reconstruction'):
                    corr=geom.reconstruct(dual);d=corr if d0 is None else tuple(coef*u+v for u,v in zip(d0,corr))
                    next64=tuple(p+v for p,v in zip(current,d));norm=np.sqrt(inner(d,d));assert norm<=rho+1e-8
                meter.flops['reconstruction']+=2*len(ordered)*256*(768+77)+4*216653
                with meter.region('materialization',True):
                    rounded=tuple(p.astype(np.float32) for p in next64);candidate=tuple(torch.from_numpy(p).to(v.device) for p,v in zip(rounded,native))
                    next_current=tuple(p.astype(np.float64) for p in rounded)
                znext=bank.exact(candidate);Cnext=bank.counts(znext)
                return d,norm,rounded,candidate,next_current,znext,Cnext
            first_raw=None;cached=None
            def first_check(coef,dual):
                nonlocal first_raw,cached
                with meter.region('first_check_bookkeeping'):
                    cached=materialize(coef,dual);safe=bool((cached[-1]>=K).all())
                    # Minimal evidence snapshot; its copy cost remains INSIDE control wall.
                    if capture_evidence:first_raw=(coef,dual.copy(),cached[2],cached[-2],cached[-1],safe)
                    return safe
            with meter.region('hildreth_ball'):
                coef,dual,info=ball_hildreth(geom.G,b,ad0,d0n,rho,first_check=first_check)
            meter.flops['hildreth_ball_estimate']+=info['coordinates']*(2*len(ordered)+6)+info['sweeps']*(8*len(ordered)**2+30*len(ordered))
            assert info['status'] in ('PASS','INNER_LIMIT','FIRST_SWEEP_ACTUAL_SAFE')
            assert np.isfinite(coef) and np.isfinite(dual).all()
            if info['sweeps']==1:pack=cached
            else:pack=materialize(coef,dual)
            d,norm,rounded,candidate,next_current,znext,Cnext=pack;safe=bool((Cnext>=K).all());repair_values=None
            if not safe:
                # This calculation controls the next rho and therefore stays timed.
                with meter.region('unsafe_repair_rule'):repair_values=repair(geom,d,znext)
            # Retain existing immutable arrays, no parameter hashing or diagnostics here.
            if capture_evidence:
                rounds.append((outer,selection_z is prediction,C,D,chosen,sorted(selected),ordered,rho,geom,b,
                    coef,dual,info,current,pack,first_raw,repair_values,safe))
            brief.append(dict(outer=outer,status=info['status'],sweeps=info['sweeps'],decision=decision(info['status'],safe,outer)))
            if safe:executed=candidate;z=znext;break
            current=next_current;z=znext;ratio=repair_values[-1]
            if ratio is None or ratio<.25:rho=max(rho*.5,base_rho/16)
            if outer==8:reason='OUTER_LIMIT_ROLLBACK'
        if reason is not None:executed=before;z=before_endpoint
    trace=(nativeC,trigger,selected,pairs,rounds,native_cpu,reason,prediction) if capture_evidence else None
    summary=dict(native_counts=nativeC.tolist(),triggered=trigger,rollback=reason is not None,failure_reason=reason,
        selected_controls=sorted(selected),final_active=[list(v) for v in sorted(pairs)],rounds=brief,
        outer_iterations=len(brief),inner_sweeps=sum(q['sweeps'] for q in brief))
    return executed,z,summary,trace
