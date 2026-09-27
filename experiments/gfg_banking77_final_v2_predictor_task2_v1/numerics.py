"""Synthetic branch tests and independent reconstruction/solver checks; no train inputs."""
from .common import *
from . import cutter
from . import audit
from .preflight import install,identity
from .predictor import Predictor
from .predictor_audit import components,analytical
from torch.utils.flop_counter import FlopCounterMode

def main():
    access=install();setup();guard();identity();m=model();p=params(m);x=torch.randn(70,768,device='cuda');y=torch.arange(70,device='cuda')%7
    bank=Bank(x,y,np.arange(70),14,Meter());z=bank.exact(p)
    assert np.array_equal(z,audit.endpoint(p,bank.x,70))
    pairs=[(0,8),(1,5),(0,3)];theta=cpu(p);geo=cutter.Geometry(bank,theta,pairs,z);f=audit.factors(bank.H,bank.labels,theta,pairs,z)
    assert np.array_equal(geo.G,f['G']) and np.array_equal(geo.reserve,f['reserve'])
    dense=geo.explicit();pp=tuple(a.double().requires_grad_() for a in p);rows=[]
    for i,c in pairs:
        zz=functional(pp,x[i:i+1].double())[0];rows.append(torch.cat([g.flatten() for g in torch.autograd.grad(zz[int(y[i])]-zz[c],pp)]).detach().cpu().numpy())
    jacobian_error=float(np.max(np.abs(dense-np.stack(rows))));assert jacobian_error<1e-10
    rng=np.random.default_rng(445);solver_cases=[]
    for case in range(5):
        A=rng.normal(size=(3,5));G=np.asfortranarray(A@A.T);d0=rng.normal(size=5);b=rng.uniform(-.5,.5,size=3);rho=1.
        c,t,info=cutter.ball_hildreth(G,b,A@d0,float(d0@d0),rho)
        cc,tt,status,sweeps=audit.independent_solver(G,b,A@d0,float(d0@d0),rho)
        assert c==cc and np.array_equal(t,tt) and info['status']==status and info['sweeps']==sweeps
        solver_cases.append(dict(case=case,status=status,sweeps=sweeps))
    original=cutter.ball_hildreth
    def force_cap(*args,**kwargs):
        c,t,r=original(*args,**kwargs);r['status']='INNER_LIMIT';return c,t,r
    cutter.ball_hildreth=force_cap;branches={}
    try:
        for kind in ('A','B','C'):
            before=tuple(torch.zeros_like(v) for v in p);before[3][0]=1.;native=tuple(v.clone() for v in before);native[3][1]=2.
            bbank=Bank(torch.zeros_like(x),torch.zeros_like(y),np.arange(70),14,Meter());K=np.array([70,0,0,0,0,0,0])
            actual_exact=bbank.exact;bad=actual_exact(native);calls=[0]
            def synthetic_endpoint(candidate):
                calls[0]+=1
                if kind=='C' or (kind=='B' and calls[0]==2):return bad.copy()
                return actual_exact(candidate)
            bbank.exact=synthetic_endpoint;executed,record=cutter.project(before,native,bbank,K)
            decisions=[r['decision'] for r in record['rounds']];branches[kind]=decisions
            if kind=='A':assert decisions==['INNER_LIMIT_ACTUAL_SAFE_ACCEPT'] and not record['rollback']
            elif kind=='B':assert decisions[0]=='INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE' and decisions[-1]=='INNER_LIMIT_ACTUAL_SAFE_ACCEPT'
            else:assert decisions==['INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE']*7+['OUTER_LIMIT_ROLLBACK'] and all(torch.equal(a,b) for a,b in zip(executed,before))
    finally:cutter.ball_hildreth=original
    out,r=cutter.project(p,p,bank,np.zeros(7,dtype=int));assert r['native_safe'] and r['jacobian_calls']==0
    delta=tuple(torch.randn_like(v)*1e-4 for v in p);native=tuple(v+d for v,d in zip(p,delta));predictor=Predictor(bank,0)
    with FlopCounterMode(display=False) as counter:predicted=predictor.predict(p,native)
    cost=counter.get_total_flops();reference=components(p,native,bank.x,bank.n)
    assert np.array_equal(predicted,(reference[0]+reference[1]+.5*reference[2])[:,:14].cpu().numpy())
    assert predictor.hashes()==[phash((a,)) for a in reference]
    pp=tuple(v.double() for v in p);nn=tuple(v.double() for v in native);xx=bank.x.double()
    analytic=analytical(pp,tuple(b-a for a,b in zip(pp,nn)),xx[:70]);jvp=components(pp,nn,xx,70)
    errors=[float(torch.max(torch.abs(a-b))) for a,b in zip(analytic,jvp)];assert max(errors)<1e-10
    assert cost==read(OLD/'NUMERICS.json')['predictor_dense_flops_per256']
    write(ROOT/'NUMERICS.json',dict(status='PASS',predictor_dense_flops_per256=cost,independent_predictor_byte_identical=True,analytic_directional_max_errors=errors,fixed256_endpoint_byte_identical=True,independent_geometry_identical=True,
        synthetic_autograd_jacobian_max_error=jacobian_error,independent_solver_cases=solver_cases,branches=branches,
        branch_injection_synthetic_only=True,native_safe_no_jacobian=True,access=access.report()))
    print('Synthetic A/B/C state-machine, endpoint and independent solver/geometry checks PASS',flush=True)

if __name__=='__main__':main()
