"""One final training entry point, with optional out-of-timer diagnostics."""
from .runtime import *
from .predictor import Predictor
from .projection import project


class Controller:
    def __init__(self,head,control_features,control_labels,control_ids,seen):
        assert seen>=14 and seen<=77 and seen%7==0
        assert bool(((control_labels>=0)&(control_labels<seen-7)).all())
        self.meter=Meter()
        with self.meter.region('entry_total',True):
            self.bank=Bank(control_features,control_labels,control_ids,seen,self.meter)
            self.previous_z=self.bank.exact(params(head));self.K=self.bank.counts(self.previous_z)
            self.predictor=Predictor(self.bank,1_127_743_488)

    def step(self,head,shadow,opt,x,y,*,capture_evidence=False):
        """Advance native AdamW once, control and commit. Current-task CE only.

        `opt` owns `shadow`; retain both across tasks. Construct a new Controller
        at task entry to freeze its floors. Detailed evidence is off by default.
        """
        assert head is not shadow
        with self.meter.region('native_total',True):
            assert bool(((y>=self.bank.seen-7)&(y<self.bank.seen)).all())
            before=params(head);loss=native_step(head,shadow,opt,x,y,self.bank.seen);native=params(shadow)
        with self.meter.region('control_total',True):
            predicted=self.predictor.predict(before,native);actual=self.bank.exact(native)
            executed,z,summary,trace=project(before,native,self.bank,self.K,predicted,actual,self.previous_z,capture_evidence)
            commit(head,executed);counts=self.bank.counts(z);assert np.all(counts>=self.K)
            self.previous_z=z.copy()
        result=dict(loss=loss,counts=counts.tolist(),summary=summary)
        if capture_evidence:
            from .diagnostics import evidence
            assert not self.meter.stack
            result['evidence']=evidence(trace,self.bank,self.K)
        return result
