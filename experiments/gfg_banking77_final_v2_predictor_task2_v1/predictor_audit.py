"""Separate functional expression and analytic directional derivative checks."""
from .common import *

def components(before,native,buffer,n):
    tangent=tuple(q-p for p,q in zip(before,native));all_parts=[[],[],[]]
    for i in range(0,len(buffer),256):
        x=buffer[i:i+256]
        def forward(weights):
            hidden=F.linear(x,weights[0],weights[1])
            return F.linear(F.gelu(hidden,approximate='none'),weights[2],weights[3])
        def first(weights):return torch.func.jvp(forward,(weights,),(tangent,))[1]
        value,derivative=torch.func.jvp(forward,(before,),(tangent,))
        curvature=torch.func.jvp(first,(before,),(tangent,))[1]
        for target,part in zip(all_parts,(value,derivative,curvature)):target.append(part.detach())
    return tuple(torch.cat(parts)[:n] for parts in all_parts)

def analytical(before,delta,x):
    w,b,v,c=before;dw,db,dv,dc=delta
    u=x@w.T+b;du=x@dw.T+db;phi=torch.exp(-.5*u*u)/np.sqrt(2*np.pi)
    q=.5*u*(1+torch.erf(u/np.sqrt(2)))
    gp=.5*(1+torch.erf(u/np.sqrt(2)))+u*phi;gpp=(2-u*u)*phi
    return q@v.T+c,(gp*du)@v.T+q@dv.T+dc,(gpp*du*du)@v.T+2*(gp*du)@dv.T

def verify_record(record,before,native,buffer):
    parts=components(before,native,buffer,70);z=(parts[0]+parts[1]+.5*parts[2])[:,:14].cpu().numpy()
    assert [phash((p,)) for p in parts]==record['components_sha256']
    metrics=record.get('predictor',record);assert ahash(z)==metrics['predicted_sha256']
    return z

def replay_native(budget,X,y,control):
    from .audit import records,endpoint,count,margin
    state=torch.load(OLD/f'formal/native_{budget}/task1/CHECKPOINT.pt',weights_only=True)
    head=model();head.load_state_dict(state['model']);shadow=copy.deepcopy(head);opt=optimizer(shadow);opt.load_state_dict(state['optimizer'])
    labels=y.cpu().numpy();ids=np.flatnonzero((labels<7)&control);current=np.flatnonzero((labels//7==1)&~control)
    buf=X.new_zeros((256,768));buf[:70].copy_(X[ids]);K=count(endpoint(params(head),buf,70),labels[ids])
    refs=records(ROOT/f'PREDICTOR_NATIVE_{budget}.jsonl.gz')
    for step,batch in enumerate(schedule(current,2,budget),1):
        r=refs[step-1];assert batch.tolist()==r['batch_ids'] and not control[batch].any()
        before=params(head);assert phash(before)==r['before_sha256'];native_step(head,shadow,opt,X[batch],y[batch],14);native=params(shadow)
        assert phash(native)==r['native_sha256'] and opt_hash(opt)==r['moments_sha256'] and opt_steps(opt)==[budget+step]*4
        predicted=verify_record(r,before,native,buf);actual=endpoint(native,buf,70);m=r['predictor']
        assert ahash(actual)==m['actual_sha256'] and count(actual,labels[ids]).tolist()==m['actual_counts']
        assert count(predicted,labels[ids]).tolist()==m['predicted_counts']
        ps=bool(np.all(count(predicted,labels[ids])>=K));ac=bool(np.all(count(actual,labels[ids])>=K))
        assert (ps and not ac)==m['false_safe'] and (ac and not ps)==m['false_unsafe'];commit(head,native)
    return dict(budget=budget,status='PASS',steps=budget,native_proposals_and_moments_identical=True,jvp_components_byte_identical=True,
        exact_endpoints_identical=True,false_safe_decisions_identical=True)
