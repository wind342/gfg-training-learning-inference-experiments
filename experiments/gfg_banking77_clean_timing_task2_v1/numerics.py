from .common import *
from .preflight import install,verify_inputs
from . import clean,reference,cutter

def main():
    access=install();setup();verify_inputs();head=model();before=params(head)
    x=torch.randn(70,768,device='cuda');y=torch.arange(70,device='cuda')%7;trials=[]
    for amount in (0.,.02,.1):
        native=list(before);native[3]=native[3].clone();native[3][7]+=amount;native=tuple(native)
        for first in (False,True):
            bank=Bank(x,y,np.arange(70),14,Meter());entry=bank.exact(before);K=bank.counts(entry);z=bank.exact(native)
            old=cutter if first else reference
            ep,zp,dp=old.project(before,native,bank,K,prediction=z,native_endpoint=z,before_endpoint=entry)
            other=Bank(x,y,np.arange(70),14,Meter())
            with other.meter.region('control_total',True):
                ec,zc,trace=clean.project(before,native,other,K,z,z,entry,first)
            assert not other.meter.stack
            dc=clean.evidence(trace,other,K)
            assert phash(ec)==phash(ep) and ahash(zc)==ahash(zp) and dc==dp
            ledger={k:v for k,v in other.meter.exclusive.items() if k.startswith('control_total/')}
            assert abs(sum(ledger.values())-other.meter.seconds['control_total'])<1e-7
            trials.append(dict(amount=amount,first=first,trigger=dc['triggered'],rounds=len(dc['rounds']),sweeps=dc['inner_sweeps'],rollback=dc['rollback'],full_record_identical=True))
    assert any(t['trigger'] for t in trials)
    verify_inputs();write(ROOT/'NUMERICS.json',dict(status='PASS',trials=trials,deferred_evidence_identical=True,exclusive_ledger_closes=True,access=access.report()))
    print('Deferred-record numerical and additive-timer checks PASS',flush=True)

if __name__=='__main__':main()
