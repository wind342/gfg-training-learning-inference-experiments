from .common import *
from .preflight import install,verify_inputs
from . import cutter,reference,audit

def main():
    access=install();setup();verify_inputs();rng=np.random.default_rng(2021);trials=[]
    for i in range(40):
        m=3+i%13;A=rng.normal(size=(m,m+5));G=np.asfortranarray(A@A.T)
        b=rng.uniform(.001,.3,m);ad=rng.normal(0,.001,m);d0n=.02;rho=.001 if i%2 else .5
        expected=reference.ball_hildreth(G,b,ad,d0n,rho);calls=[]
        def reject(c,t):calls.append((c,t.copy()));return False
        actual=cutter.ball_hildreth(G,b,ad,d0n,rho,first_check=reject)
        assert actual[0]==expected[0] and ahash(actual[1])==ahash(expected[1]) and actual[2]==expected[2]
        first=reference.ball_hildreth(G,b,ad,d0n,rho,limit=1)
        assert len(calls)==1 and calls[0][0]==first[0] and ahash(calls[0][1])==ahash(first[1])
        accepted=cutter.ball_hildreth(G,b,ad,d0n,rho,first_check=lambda c,t:True)
        assert accepted[0]==first[0] and ahash(accepted[1])==ahash(first[1]) and accepted[2]['sweeps']==1
        assert accepted[2]['status']=='FIRST_SWEEP_ACTUAL_SAFE'
        independently=audit.independent_solver(G,b,ad,d0n,rho,first_check=lambda c,t:False)
        assert independently[0]==expected[0] and ahash(independently[1])==ahash(expected[1]) and independently[2:]==(expected[2]['status'],expected[2]['sweeps'])
        trials.append(dict(case=i,status=expected[2]['status'],sweeps=expected[2]['sweeps']))
    head=model();p=params(head);bank=Bank(torch.randn(70,768,device='cuda'),torch.arange(70,device='cuda')%7,np.arange(70),14,Meter())
    z=bank.exact(p);independent_z=audit.endpoint(p,bank.x,70);assert ahash(z)==ahash(independent_z)
    assert np.array_equal(bank.counts(z),audit.count(z,bank.labels))
    theta=cpu(p);pairs=[(i,(int(bank.labels[i])+1)%14) for i in range(20)]
    g=cutter.Geometry(bank,theta,pairs,z);f=audit.factors(bank.H,bank.labels,theta,pairs,z)
    assert ahash(g.G)==ahash(f['G']) and ahash(g.reserve)==ahash(f['reserve'])
    dual=rng.uniform(0,1e-5,len(pairs));corr=g.reconstruct(dual);rounded=tuple((a+b).astype(np.float32) for a,b in zip(theta,corr))
    other,delta,error=audit.rebuild(f,1.,dual,tuple(np.zeros_like(a) for a in theta),theta,explicit=True)
    assert nhash(rounded)==nhash(other)
    verify_inputs();write(ROOT/'NUMERICS.json',dict(status='PASS',rejected_callback_preserves_entire_solve=True,accepted_callback_is_first_completed_sweep=True,
        trials=trials,independent_geometry_endpoint_reconstruction=True,explicit_error=error,access=access.report()))
    print('Numerics PASS: 40 solver cases, false callback state continuity, first candidate and independent geometry/endpoint',flush=True)

if __name__=='__main__':main()
