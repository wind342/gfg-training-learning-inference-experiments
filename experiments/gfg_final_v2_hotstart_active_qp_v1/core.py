"""New actuator: cached constraint Gram and warm dual block active-set QP."""
from experiments.gfg_final_v2_cached_hildreth_v1.core import *
from .analysis import ROOT,BASE,HERE
import scipy.linalg as la


class GramCache:
    def __init__(self,g):
        self.g=g;self.index={};self.pairs=[];self.capacity=512
        self.matrix=np.full((self.capacity,self.capacity),np.nan)
        self.computed=self.reused=self.zero_entries=self.requests=0

    def block(self,pairs):
        self.requests+=1
        for pair in pairs:
            if pair not in self.index:
                self.index[pair]=len(self.pairs);self.pairs.append(pair)
        if len(self.pairs)>self.capacity:
            size=self.capacity
            while size<len(self.pairs):size*=2
            value=np.full((size,size),np.nan);value[:self.capacity,:self.capacity]=self.matrix
            self.matrix=value;self.capacity=size
        ix=np.array([self.index[p] for p in pairs]);G=self.matrix[np.ix_(ix,ix)]
        n=len(pairs);rows=np.array([i for i,c in pairs]);targets=self.g.labels[rows];comp=np.array([c for i,c in pairs])
        coefficients=(targets[:,None]==targets[None,:]).astype(np.int8)+(comp[:,None]==comp[None,:]).astype(np.int8)
        coefficients-=(targets[:,None]==comp[None,:]);coefficients-=(comp[:,None]==targets[None,:])
        missing=np.isnan(G);self.reused+=int((~missing).sum())
        zeros=missing & (coefficients==0);G[zeros]=0.;self.zero_entries+=int(zeros.sum())
        a,b=np.nonzero(np.triu(missing & (coefficients!=0)))
        for start in range(0,len(a),256):
            aa,bb=a[start:start+256],b[start:start+256]
            v=np.einsum('ij,ij->i',self.g.H[rows[aa]],self.g.H[rows[bb]])*coefficients[aa,bb]
            G[aa,bb]=v;G[bb,aa]=v
        self.computed+=len(a)
        assert np.isfinite(G).all()
        self.matrix[np.ix_(ix,ix)]=G
        return G


def residuals(G,b,x):
    grad=G@x-b
    return dict(dual_negative=float(max(0.,-x.min(initial=0.))),
                primal_negative=float(max(0.,-grad.min(initial=0.))),
                complementarity=float(np.max(np.abs(x*grad),initial=0.)),
                gradient_on_support=float(np.max(np.abs(grad[x>0]),initial=0.)))


def active_qp(G,b,warm):
    """Dual min .5*x'G*x-b'x, x>=0. Deterministic block principal pivots."""
    n=len(b);x=np.maximum(np.asarray(warm,dtype=float),0.).copy();passive=x>0
    tolerance=1e-10*max(1.,float(np.max(np.abs(b),initial=0.)))
    initial=residuals(G,b,x);best=n+1;block_tries=3;linear_solves=0;singular_solves=0
    path=[]
    if initial['primal_negative']<=tolerance and initial['gradient_on_support']<=tolerance:
        return x,dict(status='PASS',iterations=0,linear_solves=0,singular_solves=0,initial=initial,final=initial,path=[])
    for iteration in range(1,201):
        indices=np.flatnonzero(passive);x[:]=0.
        if len(indices):
            block=G[np.ix_(indices,indices)]
            try:solution=la.cho_solve(la.cho_factor(block,lower=True,check_finite=False),b[indices],check_finite=False)
            except la.LinAlgError:
                solution=la.lstsq(block,b[indices],cond=1e-12,lapack_driver='gelsy',check_finite=False)[0]
                singular_solves+=1
            x[indices]=solution;linear_solves+=1
        grad=G@x-b
        wrong_primal=passive & (x<0.)
        wrong_dual=(~passive) & (grad < -tolerance)
        wrong=wrong_primal|wrong_dual;count=int(wrong.sum())
        path.append(dict(iteration=iteration,passive=int(passive.sum()),negative_multipliers=int(wrong_primal.sum()),negative_reduced_gradients=int(wrong_dual.sum())))
        if count==0:
            final=residuals(G,b,x)
            status='PASS' if final['primal_negative']<=tolerance and final['gradient_on_support']<=tolerance else 'KKT_FAIL'
            return x,dict(status=status,iterations=iteration,linear_solves=linear_solves,singular_solves=singular_solves,initial=initial,final=final,path=path)
        if count<best:best=count;block_tries=3;passive[wrong]=~passive[wrong]
        elif block_tries>0:block_tries-=1;passive[wrong]=~passive[wrong]
        else:
            # Deterministic single principal pivot when block progress stalls.
            j=int(np.flatnonzero(wrong)[-1]);passive[j]=~passive[j]
    return np.maximum(x,0.),dict(status='ITERATION_LIMIT',iterations=200,linear_solves=linear_solves,singular_solves=singular_solves,
                                initial=initial,final=residuals(G,b,np.maximum(x,0.)),path=path)


class Actuator:
    def __init__(self,g):
        self.g=g;self.cache=GramCache(g);self.previous={};self.audit=None

    @torch.no_grad()
    def project(self,before,native,K,ids,evaluate):
        g=self.g;receiving=evaluate(before,set(),'receiving');assert np.all(receiving['C']>=K)
        r=evaluate(native,set(),'native');C,D,_=select_exact(r,g,K,ids)
        selected=set();pairs=set();dual={};solves=[];selections=[];candidate=native;failed=False
        previous=self.previous;inherited=set();reserve=None;reason=None
        if D.any():
            reserve=PairReserve(before,native,g);N=reserve.native
            for outer in range(1,101):
                with g.meter.region('selection_and_discovery',True):
                    Cnow,Dnow,chosen=select_exact(r,g,K,ids)
                    selected.update(i for v in chosen.values() for i in v)
                    mask=torch.zeros(g.n,dtype=torch.bool,device=g.h.device)
                    if selected:mask[sorted(selected)]=True
                    bad=set(p.failing_pairs(r['z'],g.y,mask))
                    eligible_previous={pair for pair in previous if pair[0] in selected}
                    new=(bad|eligible_previous)-pairs;pairs.update(new)
                    inherited.update(eligible_previous)
                    reserve.add(sorted(new));ordered=sorted(pairs)
                    selections.append(dict(round=outer,counts=Cnow.tolist(),deficits=Dnow.tolist(),chosen=chosen,
                                           new_pairs=[list(v) for v in sorted(new)]))
                with g.meter.region('gram_and_rhs'):
                    G=self.cache.block(ordered)
                    rhs=np.array([p.EPS+reserve.values[(i,c)]-float(np.dot(g.H[i],N[int(g.labels[i])]-N[c])) for i,c in ordered])
                    warm=np.array([dual.get(pair,previous.get(pair,0.)) for pair in ordered])
                with g.meter.region('qp_solver'):
                    lam,info=active_qp(G,rhs,warm)
                if self.audit is not None:self.audit(G,rhs,warm,lam,info,outer)
                if info['status']!='PASS':failed=True;reason=info['status'];solves.append(info);break
                dual=dict(zip(ordered,lam.tolist()))
                with g.meter.region('primal_reconstruction'):
                    P=N.copy()
                    for (i,c),value in zip(ordered,lam):
                        if value:
                            delta=value*g.H[i];P[int(g.labels[i])]+=delta;P[c]-=delta
                    actual=np.array([float(np.dot(g.H[i],P[int(g.labels[i])]-P[c]))-p.EPS-reserve.values[(i,c)] for i,c in ordered])
                    # Independent primal check against original feature geometry.
                    direct_violation=max(0.,-float(actual.min(initial=0.)))
                    if direct_violation>1e-8:
                        failed=True;reason='DIRECT_PRIMAL_FAIL';info['direct_primal_violation']=direct_violation;solves.append(info);break
                with g.meter.region('materialization',True):candidate=p.materialize(P,native)
                r=evaluate(candidate,selected,'solve'+str(outer));safe=bool(np.all(r['C']>=K))
                info.update(outer=outer,constraints=len(ordered),positive_duals=int((lam>0).sum()),safe=safe,
                            counts=r['C'].tolist(),direct_primal_violation=direct_violation,
                            inherited_pairs=len(set(ordered)&set(previous)),positive_warm=int((warm>0).sum()),
                            objective=float(np.sum((P-N)**2)),primal_sha256=p.array_hash(P),dual_sha256=p.array_hash(lam))
                solves.append(info)
                if safe:break
                if outer==100:failed=True;reason='OUTER_LIMIT'
        executed=before if failed else candidate
        if failed:self.previous={}
        else:self.previous=dual.copy()
        report=dict(native_counts=C.tolist(),native_deficits=D.tolist(),projection_triggered=bool(D.any()),skip=failed,
            failure_reason=reason,selected_controls=sorted(selected),final_active=[list(v) for v in sorted(pairs)],
            final_duals=[dual.get(v,0.) for v in sorted(pairs)],selections=selections,solves=solves,
            solver_iterations=sum(s['iterations'] for s in solves),linear_solves=sum(s['linear_solves'] for s in solves),
            outer_solves=len(solves),previous_pairs=len(previous),inherited_pairs=len(inherited),
            filtered_previous_pairs=len(set(previous)-inherited),executed_counts=(receiving if failed else r)['C'].tolist())
        return executed,report
