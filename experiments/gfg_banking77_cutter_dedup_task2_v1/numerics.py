"""No train trajectory: exact candidate reuse and A/B equivalence branch checks."""
from .common import *
from . import reference,optimized
from .preflight import install,identity,verify_inputs

def main():
    access=install();setup();guard();verify_inputs();identity();rng=np.random.default_rng(341)
    solvers=[]
    for k in range(12):
        A=rng.normal(size=(3+k,20));G=np.asfortranarray(A@A.T);d=rng.normal(size=20)
        ad=A@d;dn=float(d@d);b=rng.normal(size=len(A));rho=.5+k/10
        x=reference.ball_hildreth(G,b,ad,dn,rho);y=optimized.ball_hildreth(G,b,ad,dn,rho)
        assert x[0]==y[0] and x[1].tobytes()==y[1].tobytes() and x[2]==y[2]
        solvers.append(dict(case=k,sweeps=x[2]['sweeps'],ball_hits=x[2]['ball_hits']))
    p=params(model());before=tuple(torch.zeros_like(a) for a in p);before[3][0]=1.;native=tuple(a.clone() for a in before);native[3][1]=2.
    x=torch.zeros(70,768,device='cuda');labels=torch.zeros(70,dtype=torch.long,device='cuda');K=np.array([70,0,0,0,0,0,0])
    branches={}
    original=optimized.ball_hildreth
    for kind in ('native_safe','projected_safe','cap_safe','cap_continue','rollback'):
        bank=Bank(x,labels,np.arange(70),14,Meter());good=bank.exact(before);bad=bank.exact(native);actual=bank.exact;calls=[0]
        def capped(*args,**kw):
            c,t,r=original(*args,**kw);r['status']='INNER_LIMIT';return c,t,r
        optimized.ball_hildreth=capped if kind.startswith('cap') or kind=='rollback' else original
        def endpoint(theta):
            calls[0]+=1
            if kind=='rollback' or (kind=='cap_continue' and calls[0]==1):return bad.copy()
            return actual(theta)
        bank.exact=endpoint
        n=before if kind=='native_safe' else native;z=good if kind=='native_safe' else bad
        executed,reused,r=optimized.project(before,n,bank,K,prediction=z,native_endpoint=z,before_endpoint=good)
        assert np.array_equal(reused,actual(executed));assert np.all(bank.counts(reused)>=K)
        if kind=='rollback':assert r['rollback'] and reused is good and phash(executed)==phash(before)
        elif kind=='native_safe':assert reused is good and not r['rounds']
        elif kind=='cap_continue':assert r['rounds'][0]['decision']=='INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE'
        elif kind=='cap_safe':assert r['rounds'][-1]['decision']=='INNER_LIMIT_ACTUAL_SAFE_ACCEPT'
        branches[kind]=[q['decision'] for q in r['rounds']]
    optimized.ball_hildreth=original
    write(ROOT/'NUMERICS.json',dict(status='PASS',synthetic_solver_cases=solvers,endpoint_reuse_branches=branches,access=access.report(),injection_synthetic_only=True))
    print('A/B solver numerics and C endpoint-reuse branches PASS',flush=True)

if __name__=='__main__':main()
