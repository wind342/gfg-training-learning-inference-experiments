"""Optional detailed evidence; invoke only outside the training timer."""
import hashlib
from .runtime import *
from .geometry import inner
from .projection import repair,decision

def nhash(p):return hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in p)).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def evidence(trace,bank,K):
    """Called after controller timer ends; reproduce every sealed decision field."""
    nativeC,trigger,selected,pairs,raw,native_cpu,reason,prediction=trace;rounds=[]
    for outer,pred_selection,C,D,chosen,selected_rows,ordered,rho,geom,b,coef,dual,info,current,pack,first_raw,values,safe in raw:
        d,norm,rounded,candidate,next_current,znext,Cnext=pack
        if values is None:values=repair(geom,d,znext)
        predicted,actual,pr,ar,ratio=values
        record=dict(outer=outer,selection_source='prediction' if pred_selection else 'actual',counts=C.tolist(),deficits=D.tolist(),chosen=chosen,
            selected_controls=selected_rows,active_pairs=[list(v) for v in ordered],rho=rho,solver=info,
            reserve_min=float(geom.reserve.min()),reserve_max=float(geom.reserve.max()),gram_sha256=ahash(geom.G),rhs_sha256=ahash(b),dual_sha256=ahash(dual))
        record.update(coef=coef,dual=dual.tolist(),current_sha256=nhash(current),reserve_sha256=ahash(geom.reserve),local_converged=info['status']=='PASS')
        if first_raw is not None:
            fc,fd,fr,fz,fC,fs=first_raw
            record['first_sweep_check']=dict(sweep=1,coef=fc,dual_sha256=ahash(fd),candidate_sha256=nhash(fr),endpoint_sha256=ahash(fz),counts=fC.tolist(),safe=fs)
        record.update(predicted_margin_repair=pr,actual_margin_repair=ar,repair_ratio=ratio,linearization_max_error=float(np.max(np.abs(actual-predicted))),
            decision=decision(info['status'],safe,outer),candidate_sha256=nhash(rounded),endpoint_sha256=ahash(znext),endpoint_counts=Cnext.tolist(),safe=safe,
            local_displacement_norm=norm,cumulative_from_native_norm=np.sqrt(inner(tuple(v-n for v,n in zip(next_current,native_cpu)),tuple(v-n for v,n in zip(next_current,native_cpu)))),
            layer_correction_norms=[float(np.linalg.norm(a-b)) for a,b in zip(next_current,native_cpu)])
        rounds.append(record)
    return dict(predicted_triggered=bool(np.any(bank.counts(prediction)<K)) if prediction is not None else None,native_safe=not trigger,
        native_counts=nativeC.tolist(),triggered=trigger,rollback=reason is not None,failure_reason=reason,selected_controls=sorted(selected),final_active=[list(v) for v in sorted(pairs)],
        rounds=rounds,outer_iterations=len(rounds),inner_sweeps=sum(r['solver']['sweeps'] for r in rounds),jacobian_calls=len(rounds),jacobian_pairs=sum(len(r['active_pairs']) for r in rounds))
