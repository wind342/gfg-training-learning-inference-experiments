"""Analytic pair factors, compressed Hildreth/Dykstra ball projection, exact floor."""
from .common import *

def gelu(x):return .5*x*(1+erf(x/np.sqrt(2.)))
def gelu_prime(x):return .5*(1+erf(x/np.sqrt(2.)))+x*np.exp(-.5*x*x)/np.sqrt(2*np.pi)
def inner(a,b):return sum(float(np.sum(x*y)) for x,y in zip(a,b))

class Geometry:
    def __init__(self,bank,p,ordered,z):
        self.bank=bank;self.ordered=ordered;m=len(ordered);clock=time.perf_counter()
        self.rows=np.array([i for i,c in ordered]);self.comp=np.array([c for i,c in ordered]);self.target=bank.labels[self.rows]
        unique,back=np.unique(self.rows,return_inverse=True);H=bank.H[unique];w,b,v,c=p
        u=H@w.T+b;q=gelu(u);prime=gelu_prime(u)
        self.H=bank.H[self.rows];self.Q=q[back];self.S=(v[self.target]-v[self.comp])*prime[back]
        self.B=np.zeros((m,len(v)));self.B[np.arange(m),self.target]=1.;self.B[np.arange(m),self.comp]=-1.
        bank.meter.seconds['geometry_factors_subregion']+=time.perf_counter()-clock;clock=time.perf_counter()
        self.G=np.asfortranarray((self.S@self.S.T)*bank.inputGram[np.ix_(self.rows,self.rows)]+(self.B@self.B.T)*(self.Q@self.Q.T+1.))
        assert np.isfinite(self.G).all() and (self.G.diagonal()>0).all()
        bank.meter.seconds['gram_subregion']+=time.perf_counter()-clock;clock=time.perf_counter()
        self.g=(z[self.rows,self.target]-z[self.rows,self.comp]).astype(np.float64)
        u32=2.**-24;gamma=lambda n:n*u32/(1-n*u32)
        eu=gamma(769)*(bank.absH[unique]@np.abs(w).T+np.abs(b));eq=1.13*eu+8*u32*(1+np.abs(u)+np.abs(q))
        # Only active target/competitor rows; no full controls x classes budget.
        ey=np.einsum('ij,ij->i',np.abs(v[self.target]),eq[back])+gamma(257)*(np.einsum('ij,ij->i',np.abs(v[self.target]),np.abs(self.Q))+np.abs(c[self.target]))
        ec=np.einsum('ij,ij->i',np.abs(v[self.comp]),eq[back])+gamma(257)*(np.einsum('ij,ij->i',np.abs(v[self.comp]),np.abs(self.Q))+np.abs(c[self.comp]))
        ref=np.einsum('ij,ij->i',self.Q,v[self.target]-v[self.comp])+c[self.target]-c[self.comp]
        self.reserve=np.maximum(ey+ec,np.abs(self.g-ref))+8*u32
        bank.meter.seconds['budget_subregion']+=time.perf_counter()-clock
        bank.meter.calls['jacobian_batches']+=1;bank.meter.calls['jacobian_pairs']+=m;bank.meter.calls['jacobian_unique_rows']+=len(unique)
        bank.meter.flops['jacobian_forward_guard']+=4*len(unique)*768*256+8*m*256
        bank.meter.flops['gram']+=2*m*m*(256+256+len(v))+4*m*m
    def apply(self,d):
        # A*d without materializing rows of A.
        self.bank.meter.flops['jacobian_directional_dense_estimate']+=2*len(self.rows)*768*256+6*len(self.rows)*256
        u=self.H@d[0].T+d[1]
        return np.einsum('ij,ij->i',self.S,u)+np.einsum('ij,ij->i',self.Q,d[2][self.target]-d[2][self.comp])+d[3][self.target]-d[3][self.comp]
    def reconstruct(self,t):
        weighted=t[:,None]*self.S;hidden=t[:,None]*self.Q
        return (weighted.T@self.H,weighted.sum(0),self.B.T@hidden,self.B.T@t)
    def explicit(self):
        # Test-only; never called in production cutter.
        return np.concatenate([(self.S[:,:,None]*self.H[:,None,:]).reshape(len(self.rows),-1),self.S,(self.B[:,:,None]*self.Q[:,None,:]).reshape(len(self.rows),-1),self.B],axis=1)

def ball_hildreth(G,b,Ad0,d0norm2,rho,limit=200):
    """Dykstra over halfspaces and ball in span{d0,A.T}; no vector-space clipping."""
    m=len(b);t=np.zeros(m);lam=np.zeros(m);c=1.;tc=np.zeros(m);cc=0.;z=Ad0.copy();work=np.empty(m);hits=0;coordinates=0
    norm2=lambda a,v:max(0.,a*a*d0norm2+2*a*float(v@Ad0)+float(v@(G@v)))
    diag=G.diagonal();change=None
    for sweep in range(1,limit+1):
        oldc=c;oldt=t.copy()
        for i in range(m):
            new=max(0.,lam[i]+(b[i]-z[i])/diag[i]);delta=new-lam[i];lam[i]=new
            if delta:
                t[i]+=delta;np.multiply(G[:,i],delta,out=work);np.add(z,work,out=z)
            coordinates+=1
        yc=c+cc;yt=t+tc;length=np.sqrt(norm2(yc,yt));scale=min(1.,rho/max(length,1e-300))
        hits+=int(scale<1.);c=yc*scale;t=yt*scale;cc=yc*(1-scale);tc=yt*(1-scale);z=c*Ad0+G@t
        violation=float(np.maximum(b-z,0).max(initial=0));length=np.sqrt(norm2(c,t));change=np.sqrt(norm2(c-oldc,t-oldt))
        if violation<=1e-8 and length<=rho+1e-10 and change<=1e-10+1e-8*rho:
            return c,t,dict(status='PASS',sweeps=sweep,coordinates=coordinates,ball_hits=hits,halfspace_violation=violation,displacement_norm=length,ball_violation=max(0.,length-rho),cycle_change=change)
    return c,t,dict(status='INNER_LIMIT',sweeps=limit,coordinates=coordinates,ball_hits=hits,halfspace_violation=violation,displacement_norm=length,ball_violation=max(0.,length-rho),cycle_change=change)

def select(bank,z,K,selected,pairs):
    margins=bank.margins(z);C=bank.counts(z);D=np.maximum(K-C,0);chosen={}
    for r in np.flatnonzero(D):
        wrong=bank.relation_rows[r][margins[bank.relation_rows[r]]<=0]
        ix=sorted(wrong.tolist(),key=lambda i:(-float(margins[i]),int(bank.ids[i])))[:int(D[r])]
        chosen[str(r)]=ix;selected.update(ix)
    for i in sorted(selected):
        gap=z[i,bank.labels[i]]-z[i]
        pairs.update((i,int(c)) for c in np.flatnonzero(gap<EPS) if c!=bank.labels[i])
    return C,D,chosen

@torch.no_grad()
def project(before,native,bank,K):
    meter=bank.meter;z=bank.exact(native);C=bank.counts(z);nativeC=C.copy();selected=set();pairs=set();rounds=[]
    executed=native;reason=None;trigger=bool((C<K).any());native_safe=not trigger;native_cpu=None
    if trigger:
        with meter.region('parameter_geometry',True):
            native_cpu=cpu(native);before_cpu=cpu(before);delta=tuple(n-b for n,b in zip(native_cpu,before_cpu));base_rho=2*np.sqrt(inner(delta,delta));rho=base_rho;current=native_cpu
        for outer in range(1,9):
            with meter.region('selection'):
                C,D,chosen=select(bank,z,K,selected,pairs);ordered=sorted(pairs)
            with meter.region('jacobian_gram_guard'):
                geom=Geometry(bank,current,ordered,z);d0=tuple(n-p for n,p in zip(native_cpu,current));ad0=geom.apply(d0);b=EPS+geom.reserve-geom.g;d0n=inner(d0,d0)
            with meter.region('hildreth_ball'):
                coef,dual,info=ball_hildreth(geom.G,b,ad0,d0n,rho)
            meter.flops['hildreth_ball_estimate']+=info['coordinates']*(2*len(ordered)+6)+info['sweeps']*(8*len(ordered)**2+30*len(ordered))
            record=dict(outer=outer,counts=C.tolist(),deficits=D.tolist(),chosen=chosen,selected_controls=sorted(selected),active_pairs=[list(v) for v in ordered],rho=rho,solver=info,
                reserve_min=float(geom.reserve.min()),reserve_max=float(geom.reserve.max()),gram_sha256=ahash(geom.G),rhs_sha256=ahash(b),dual_sha256=ahash(dual))
            assert info['status'] in ('PASS','INNER_LIMIT')
            assert np.isfinite(coef) and np.isfinite(dual).all(), 'Invalid numerical solver result'
            record.update(coef=coef,dual=dual.tolist(),current_sha256=nhash(current),
                reserve_sha256=ahash(geom.reserve),local_converged=info['status']=='PASS')
            with meter.region('reconstruction'):
                corr=geom.reconstruct(dual);d=tuple(coef*u+v for u,v in zip(d0,corr));next64=tuple(p+v for p,v in zip(current,d))
                predicted=geom.g+geom.apply(d);local_norm=np.sqrt(inner(d,d));assert local_norm<=rho+1e-8
            meter.flops['reconstruction']+=2*len(ordered)*256*(768+77)+4*216653
            with meter.region('materialization',True):
                rounded=tuple(p.astype(np.float32) for p in next64);candidate=tuple(torch.from_numpy(p).to(v.device) for p,v in zip(rounded,native));next_current=tuple(p.astype(np.float64) for p in rounded)
            znext=bank.exact(candidate);Cnext=bank.counts(znext);actual=(znext[geom.rows,geom.target]-znext[geom.rows,geom.comp]).astype(np.float64)
            target=EPS+geom.reserve;start=float(np.maximum(target-geom.g,0).sum());pr=start-float(np.maximum(target-predicted,0).sum());ar=start-float(np.maximum(target-actual,0).sum())
            ratio=ar/pr if pr>0 else None;safe=bool((Cnext>=K).all())
            decision=('INNER_LIMIT_ACTUAL_SAFE_ACCEPT' if info['status']=='INNER_LIMIT' else 'PASS_ACTUAL_SAFE_ACCEPT') if safe else (
                'OUTER_LIMIT_ROLLBACK' if outer==8 else 'INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE' if info['status']=='INNER_LIMIT' else 'PASS_ACTUAL_UNSAFE_CONTINUE')
            record.update(predicted_margin_repair=pr,actual_margin_repair=ar,repair_ratio=ratio,linearization_max_error=float(np.max(np.abs(actual-predicted))),
                decision=decision,candidate_sha256=nhash(rounded),endpoint_sha256=ahash(znext),
                endpoint_counts=Cnext.tolist(),safe=safe,local_displacement_norm=local_norm,cumulative_from_native_norm=np.sqrt(inner(tuple(v-n for v,n in zip(next_current,native_cpu)),tuple(v-n for v,n in zip(next_current,native_cpu)))),
                layer_correction_norms=[float(np.linalg.norm(a-b)) for a,b in zip(next_current,native_cpu)])
            rounds.append(record)
            if safe:executed=candidate;break
            current=next_current;z=znext
            if ratio is None or ratio<.25:rho=max(rho*.5,base_rho/16)
            if outer==8:reason='OUTER_LIMIT_ROLLBACK'
        if reason is not None:executed=before
    return executed,dict(native_safe=native_safe,native_counts=nativeC.tolist(),triggered=trigger,rollback=reason is not None,failure_reason=reason,
        selected_controls=sorted(selected),final_active=[list(v) for v in sorted(pairs)],rounds=rounds,outer_iterations=len(rounds),
        inner_sweeps=sum(r['solver']['sweeps'] for r in rounds),jacobian_calls=len(rounds),jacobian_pairs=sum(len(r['active_pairs']) for r in rounds))
