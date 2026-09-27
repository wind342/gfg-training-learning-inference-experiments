from .run import *

def main():
    access=install();setup();verify();p=params(model());before=tuple(torch.zeros_like(a) for a in p);before[3][0]=1.
    native=tuple(v.clone() for v in before);native[3][1]=2.
    bank=Bank(torch.zeros(70,768,device='cuda'),torch.zeros(70,dtype=torch.long,device='cuda'),np.arange(70),14,Meter())
    receiving=bank.exact(before);K=bank.counts(receiving);actual=bank.exact(native)
    expected,z,decision=cutter.project(before,native,bank,K,prediction=actual,native_endpoint=actual,before_endpoint=receiving)
    original_geometry=cutter.Geometry;original_solver=cutter.ball_hildreth
    with gzip.open(ROOT/'SYNTHETIC_SWEEPS.jsonl.gz','xt',encoding='utf8') as out:
        observer=Observer(bank,K,out);observer.step(1,dict(decision=decision),native)
        class Watching(original_geometry):
            def __init__(self,b,p,ordered,z):
                super().__init__(b,p,ordered,z);observer.geometry(self,p);observer.local_z=z.copy()
        cutter.Geometry=Watching;cutter.ball_hildreth=observer.solve
        try:executed,zz,report=cutter.project(before,native,bank,K,prediction=actual,native_endpoint=actual,before_endpoint=receiving)
        finally:cutter.Geometry=original_geometry;cutter.ball_hildreth=original_solver
    assert report==decision and phash(executed)==phash(expected) and ahash(zz)==ahash(z)
    assert observer.total_probes==decision['inner_sweeps'] and observer.independent_checks>0
    write(ROOT/'NUMERICS.json',dict(status='PASS',original_decisions_and_bytes_identical=True,probes=observer.total_probes,
        independent_candidate_checks=observer.independent_checks,early_observer_did_not_stop_solver=True,access=access.report()))
    print('Read-only sweep observer and independent materialization/endpoint checks PASS',flush=True)

if __name__=='__main__':main()
