"""Independent controller reimplementation; never imports production cutter."""
from .common import *
from .preflight import install,verify_inputs
from .predictor_audit import verify_record,replay_native

def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(line) for line in f]

def product(left,right):return sum(float(np.sum(a*b)) for a,b in zip(left,right))

def margin(z,labels):
    alternatives=z.copy();alternatives[np.arange(len(labels)),labels]=-np.inf
    return z[np.arange(len(labels)),labels]-np.max(alternatives,axis=1)

def count(z,labels):
    good=np.array([all(float(row[y])>float(row[j]) for j in range(14) if j!=y) for row,y in zip(z,labels)])
    return np.bincount(labels[good],minlength=7)

@torch.no_grad()
def endpoint(parameters,buffer,n):
    logits=[]
    for start in range(0,len(buffer),256):
        hidden=F.linear(buffer[start:start+256],parameters[0],parameters[1])
        logits.append(F.linear(F.gelu(hidden,approximate='none'),parameters[2],parameters[3]))
    return torch.cat(logits)[:n,:14].cpu().numpy().copy()

def choose(z,labels,ids,K,chosen,pairs):
    m=margin(z,labels);C=count(z,labels);D=np.maximum(K-C,0);new={}
    for relation in range(7):
        if D[relation]:
            eligible=[i for i in range(len(labels)) if labels[i]==relation and m[i]<=0]
            eligible.sort(key=lambda i:(-float(m[i]),int(ids[i])))
            new[str(relation)]=eligible[:int(D[relation])];chosen.update(new[str(relation)])
    for i in sorted(chosen):
        for competitor in range(14):
            if competitor!=labels[i] and z[i,labels[i]]-z[i,competitor]<EPS:pairs.add((i,competitor))
    return C,D,new

def factors(H,labels,theta,pairs,z):
    row=np.array([i for i,c in pairs]);comp=np.array([c for i,c in pairs]);target=labels[row]
    unique,inverse=np.unique(row,return_inverse=True);w,b,v,c=theta
    u=H[unique]@w.T+b;q=.5*u*(1+erf(u/np.sqrt(2.)))
    derivative=.5*(1+erf(u/np.sqrt(2.)))+u*np.exp(-.5*u*u)/np.sqrt(2*np.pi)
    A=H[row];Q=q[inverse];S=(v[target]-v[comp])*derivative[inverse]
    B=np.zeros((len(row),77));B[np.arange(len(row)),target]=1;B[np.arange(len(row)),comp]=-1
    input_gram=H@H.T+1.
    G=np.asfortranarray((S@S.T)*input_gram[np.ix_(row,row)]+(B@B.T)*(Q@Q.T+1.))
    actual=(z[row,target]-z[row,comp]).astype(np.float64);unit=2.**-24
    gamma=lambda width:width*unit/(1-width*unit)
    first_error=gamma(769)*(np.abs(H[unique])@np.abs(w).T+np.abs(b))
    activation_error=1.13*first_error+8*unit*(1+np.abs(u)+np.abs(q))
    errors=[]
    for classes in (target,comp):
        errors.append(np.einsum('ij,ij->i',np.abs(v[classes]),activation_error[inverse])+gamma(257)*(
            np.einsum('ij,ij->i',np.abs(v[classes]),np.abs(Q))+np.abs(c[classes])))
    fp64_margin=np.einsum('ij,ij->i',Q,v[target]-v[comp])+c[target]-c[comp]
    reserve=np.maximum(errors[0]+errors[1],np.abs(actual-fp64_margin))+8*unit
    return dict(H=A,Q=Q,S=S,B=B,G=G,g=actual,reserve=reserve,row=row,target=target,comp=comp)

def direction(f,d):
    hidden=f['H']@d[0].T+d[1]
    return np.einsum('ij,ij->i',f['S'],hidden)+np.einsum('ij,ij->i',f['Q'],d[2][f['target']]-d[2][f['comp']])+d[3][f['target']]-d[3][f['comp']]

def independent_solver(gram,rhs,linear_native,native_square,radius):
    # Re-execute coordinate and ball corrections, independently of production calls or stored duals.
    size=len(rhs);coefficient=1.;weights=np.zeros(size);halfspace_corrections=np.zeros(size)
    ball_coefficient=0.;ball_weights=np.zeros(size);values=linear_native.copy();temp=np.empty(size)
    def length_squared(a,w):return max(0.,a*a*native_square+2*a*float(w@linear_native)+float(w@(gram@w)))
    for sweep in range(1,201):
        previous_coefficient=coefficient;previous_weights=weights.copy()
        for j in range(size):
            next_correction=max(0.,halfspace_corrections[j]+(rhs[j]-values[j])/gram[j,j])
            increment=next_correction-halfspace_corrections[j];halfspace_corrections[j]=next_correction
            if increment:
                weights[j]+=increment;np.multiply(gram[:,j],increment,out=temp);np.add(values,temp,out=values)
        unprojected_coefficient=coefficient+ball_coefficient;unprojected_weights=weights+ball_weights
        length=np.sqrt(length_squared(unprojected_coefficient,unprojected_weights));scale=min(1.,radius/max(length,1e-300))
        coefficient=unprojected_coefficient*scale;weights=unprojected_weights*scale
        ball_coefficient=unprojected_coefficient*(1-scale);ball_weights=unprojected_weights*(1-scale)
        values=coefficient*linear_native+gram@weights
        residual=float(np.maximum(rhs-values,0).max(initial=0));length=np.sqrt(length_squared(coefficient,weights))
        movement=np.sqrt(length_squared(coefficient-previous_coefficient,weights-previous_weights))
        if residual<=1e-8 and length<=radius+1e-10 and movement<=1e-10+1e-8*radius:return coefficient,weights,'PASS',sweep
    return coefficient,weights,'INNER_LIMIT',200

def rebuild(f,coef,dual,offset,current,explicit=False):
    first=dual[:,None]*f['S'];second=dual[:,None]*f['Q']
    correction=(first.T@f['H'],first.sum(0),f['B'].T@second,f['B'].T@dual)
    delta=tuple(coef*a+b for a,b in zip(offset,correction));next64=tuple(a+b for a,b in zip(current,delta))
    error=0.
    if explicit:
        dense=np.concatenate([(f['S'][:,:,None]*f['H'][:,None,:]).reshape(len(dual),-1),f['S'],
            (f['B'][:,:,None]*f['Q'][:,None,:]).reshape(len(dual),-1),f['B']],axis=1)
        independently_summed=dense.T@dual;expected=np.concatenate([a.ravel() for a in correction])
        error=float(np.max(np.abs(independently_summed-expected)));assert error<=1e-12
        assert np.allclose(dense@dense.T,f['G'],atol=1e-8,rtol=1e-12)
    return tuple(a.astype(np.float32) for a in next64),delta,error

def replay(budget,X,y,control):
    folder=ROOT/f'budget{budget}';reference=records(folder/'STEPS.jsonl.gz');summary=read(folder/'CHECKS.json')
    state=torch.load(OLD/f'formal/native_{budget}/task1/CHECKPOINT.pt',weights_only=True)
    head=model();head.load_state_dict(state['model']);opt=optimizer(head);opt.load_state_dict(state['optimizer'])
    yc=y.cpu().numpy();ids=np.flatnonzero((yc<7)&control);current_ids=np.flatnonzero((yc//7==1)&~control)
    H=X[ids].cpu().numpy().astype(np.float64);labels=yc[ids];buffer=X.new_zeros((256,768));buffer[:70].copy_(X[ids])
    initial=endpoint(params(head),buffer,70);K=count(initial,labels);assert K.tolist()==summary['K']
    audit_rows=[];cap_checks=continuation_checks=0;explicit_error=0.;old_backward=0
    for step,batch in enumerate(schedule(current_ids,2,budget),1):
        row=reference[step-1];assert batch.tolist()==row['batch_ids'];assert not control[batch].any() and np.all(yc[batch]//7==1)
        before=params(head);assert phash(before)==row['before_sha256'];opt.zero_grad(set_to_none=True)
        logits=head(X[batch])[:,:14];F.cross_entropy(logits,y[batch]).backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        native=params(head);assert phash(native)==row['native_sha256'];moments=opt_hash(opt)
        assert moments==row['moments_sha256'] and opt_steps(opt)==row['optimizer_steps']==[budget+step]*4
        predicted=verify_record(row['predictor'],before,native,buffer)
        z=endpoint(native,buffer,70);native_counts=count(z,labels);d=row['decision'];assert native_counts.tolist()==d['native_counts']
        predicted_counts=count(predicted,labels);ps=bool(np.all(predicted_counts>=K));actual_safe=bool(np.all(native_counts>=K))
        assert d['predicted_triggered']==(not ps) and row['predictor']['predicted_counts']==predicted_counts.tolist()
        assert row['predictor']['false_safe']==(ps and not actual_safe) and row['predictor']['false_unsafe']==(actual_safe and not ps)
        selected=set();pairs=set();executed=native;reason=None;observed=[]
        if np.any(native_counts<K):
            n=cpu(native);start=cpu(before);native_delta=tuple(a-b for a,b in zip(n,start));base=2*np.sqrt(product(native_delta,native_delta));rho=base;current=n
            for outer in range(1,9):
                record=d['rounds'][outer-1];assert nhash(current)==record['current_sha256'] and rho==record['rho']
                selection_z=predicted if outer==1 and not ps else z
                assert record['selection_source']==('prediction' if selection_z is predicted else 'actual')
                C,D,new=choose(selection_z,labels,ids,K,selected,pairs);ordered=sorted(pairs)
                assert C.tolist()==record['counts'] and D.tolist()==record['deficits'] and new==record['chosen']
                assert sorted(selected)==record['selected_controls'] and [list(p) for p in ordered]==record['active_pairs']
                f=factors(H,labels,current,ordered,z);offset=tuple(a-b for a,b in zip(n,current));ad=direction(f,offset);rhs=EPS+f['reserve']-f['g']
                assert ahash(f['G'])==record['gram_sha256'] and ahash(rhs)==record['rhs_sha256'] and ahash(f['reserve'])==record['reserve_sha256']
                coef,dual,status,sweeps=independent_solver(f['G'],rhs,ad,product(offset,offset),rho)
                assert coef==record['coef'] and np.array_equal(dual,np.asarray(record['dual']))
                assert ahash(dual)==record['dual_sha256'] and status==record['solver']['status'] and sweeps==record['solver']['sweeps']
                rounded,delta,error=rebuild(f,coef,dual,offset,current,explicit=status=='INNER_LIMIT');explicit_error=max(explicit_error,error)
                assert nhash(rounded)==record['candidate_sha256'];assert np.sqrt(product(delta,delta))<=rho+1e-8
                candidate=tuple(torch.from_numpy(a).cuda() for a in rounded);actual_z=endpoint(candidate,buffer,70);actual_counts=count(actual_z,labels)
                assert ahash(actual_z)==record['endpoint_sha256'] and actual_counts.tolist()==record['endpoint_counts']
                safe=bool(np.all(actual_counts>=K));actual_decision=('INNER_LIMIT_ACTUAL_SAFE_ACCEPT' if status=='INNER_LIMIT' else 'PASS_ACTUAL_SAFE_ACCEPT') if safe else (
                    'OUTER_LIMIT_ROLLBACK' if outer==8 else 'INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE' if status=='INNER_LIMIT' else 'PASS_ACTUAL_UNSAFE_CONTINUE')
                assert safe==record['safe'] and actual_decision==record['decision'];observed.append(actual_decision);cap_checks+=int(status=='INNER_LIMIT')
                assert opt_hash(opt)==moments
                if safe:executed=candidate;break
                continuation_checks+=int(outer<8)
                target=EPS+f['reserve'];predicted=f['g']+direction(f,delta)
                actual=(actual_z[f['row'],f['target']]-actual_z[f['row'],f['comp']]).astype(np.float64)
                initial_deficit=float(np.maximum(target-f['g'],0).sum())
                predicted_repair=initial_deficit-float(np.maximum(target-predicted,0).sum())
                actual_repair=initial_deficit-float(np.maximum(target-actual,0).sum())
                rate=actual_repair/predicted_repair if predicted_repair>0 else None
                assert rate==record['repair_ratio']
                current=tuple(a.astype(np.float64) for a in rounded);z=actual_z
                if rate is None or rate<.25:rho=max(rho*.5,base/16)
                if outer==8:reason='OUTER_LIMIT_ROLLBACK';executed=before
        else:assert d['native_safe'] and not d['rounds'] and d['jacobian_calls']==0
        assert len(observed)==len(d['rounds']) and reason==d['failure_reason']
        commit(head,executed);assert phash(params(head))==row['executed_sha256'] and thash(head.state_dict())==row['head_sha256']
        actual=endpoint(params(head),buffer,70);C=count(actual,labels);assert np.all(C>=K) and C.tolist()==row['counts'] and ahash(actual)==row['endpoint_sha256']
        assert opt_hash(opt)==moments
        audit_rows.append(dict(step=step,native_sha256=phash(native),executed_sha256=phash(params(head)),moments_sha256=moments,
            decisions=observed,counts=C.tolist(),floor_violations=0,old_ce_backward_examples=0))
        if step%60==0:print('independent audit',budget,step,flush=True)
    with gzip.open(ROOT/f'AUDIT_STEPS_{budget}.jsonl.gz','xt',encoding='utf8') as out:
        for r in audit_rows:out.write(json.dumps(r,separators=(',',':'))+'\n')
    return dict(budget=budget,steps=budget,status='PASS',native_proposals_identical=True,moments_and_counters_identical=True,
        predictor_components_byte_identical=True,predicted_trigger_and_selection_identical=True,local_solver_coef_duals_identical=True,candidate_materializations_byte_identical=True,all_endpoint_counts_identical=True,
        all_decisions_identical=True,executed_heads_byte_identical=True,cap_candidate_checks=cap_checks,continuation_checks=continuation_checks,
        explicit_dense_jacobian_reconstruction_max_error=explicit_error,floor_violations=0,old_ce_backward_examples=old_backward)

def main():
    access=install();setup();guard();verify_inputs();freeze=read(ROOT/'FREEZE.json')
    for name,digest in freeze['sources'].items():assert sha(HERE/name)==digest
    X,y,c=load();native_reports=[replay_native(b,X,y,c) for b in (210,630)];reports=[replay(b,X,y,c) for b in (210,630)];verify_inputs();guard()
    write(ROOT/'AUDIT.json',dict(status='PASS',reports=reports,native_predictor_reports=native_reports,access=access.report(),production_cutter_imported=False,test_files_opened=0))
    print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
