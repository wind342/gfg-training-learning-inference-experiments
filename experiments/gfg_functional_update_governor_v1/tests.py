"""Ordinary unit tests and a tiny actual CUDA adapter smoke; no evidence pipeline."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from .governor import ALPHAS,Decision,FunctionalUpdateGovernor
from .instrument import Measurement,measure_boundaries,FunctionalResponseInstrument
from .scheduler import CheapSentinel,ReleaseToNative
from .runtime import GovernedTraining
from experiments.gfg_controller_distributed_runtime_v2.boundary import calculate
from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter


def measured(b,p):return Measurement(ALPHAS,torch.tensor(b,dtype=torch.float64),torch.tensor(p,dtype=torch.float64))


class GovernorTests(unittest.TestCase):
    def test_amplify_formation(self):
        m=measured([1,-1,-2],[[1,-.9,-1.8],[1,-.8,-1.6],[1,.1,-1],[1,.3,-.1],[1,.8,.2]])
        d=FunctionalUpdateGovernor().choose(m)
        self.assertEqual(d.alpha,2.);self.assertEqual(d.scores['2.0']['formation_gain'],2)

    def test_damage_shrinks(self):
        m=measured([1,1,-1],[[1,1,.1],[1,-.1,.2],[-1,-1,.3],[-2,-2,.4],[-3,-3,.5]])
        self.assertEqual(FunctionalUpdateGovernor().choose(m).alpha,.125)

    def test_all_formed_native_safe(self):
        m=measured([1,1],[[1,1],[2,2],[3,3],[4,4],[20,20]])
        self.assertEqual(FunctionalUpdateGovernor().choose(m).alpha,1.)

    def test_all_formed_shrink_not_amplify(self):
        m=measured([1,1],[[.1,.1],[.5,.5],[-1,.5],[-2,.5],[30,30]])
        self.assertEqual(FunctionalUpdateGovernor().choose(m).alpha,.25)

    def test_none_formed_frontier(self):
        m=measured([-2,-2],[[-2,-2],[-1.9,-1.9],[-1.8,-1.8],[-1.7,-1.7],[-1.6,-1.6]])
        d=FunctionalUpdateGovernor(2).choose(m)
        self.assertEqual(d.alpha,2.);self.assertIsNone(d.scores['2.0']['formed_median_margin'])

    def test_nonfinite_and_empty(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(FloatingPointError):FunctionalUpdateGovernor().choose(measured([value],[[1]]*5))
            with self.assertRaises(FloatingPointError):FunctionalUpdateGovernor().choose(measured([1],[[value]]*5))
        with self.assertRaises(ValueError):FunctionalUpdateGovernor().choose(measured([], [[]]*5))


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81);n,v,h=7,11,257
        self.z=torch.randn(n,v);self.y=torch.arange(n)%v
        self.w=torch.randn(v,h)*.1;self.dw=torch.randn(v,h)*.001
        self.hs=[torch.randn(n,h)*s for s in (1,.01,.001)]

    def test_five_actions_three_backends_and_baseline(self):
        for backend in ('exact','scalarized','streaming'):
            m=measure_boundaries(self.z,self.y,self.w,self.dw,self.hs,mode=backend)
            self.assertEqual(tuple(m.predicted_minima.shape),(5,7))
            # Reference coordinate kernel for each backend; choose arithmetic in NumPy.
            _,_,_,c,_=calculate(self.z,self.y,self.w,self.dw,self.hs,mode=backend,audit=True)
            c=c.numpy();mask=np.arange(11)[None,:]==self.y.numpy()[:,None]
            baseline=c[0].copy();baseline[mask]=np.inf
            np.testing.assert_array_equal(m.baseline_minima.numpy(),baseline.min(1))
            for j,a in enumerate(ALPHAS):
                p=((c[0]+a*c[1])+a*a*c[2])+a*a*c[3];p[mask]=np.inf
                np.testing.assert_array_equal(m.predicted_minima[j].numpy(),p.min(1))

    def test_candidate_count_not_fixed(self):
        for mode in ('exact','scalarized','streaming'):
            m=measure_boundaries(self.z,self.y,self.w,self.dw,self.hs,alphas=(.1,1.,3.),mode=mode)
            self.assertEqual(tuple(m.predicted_minima.shape),(3,7))

    def test_bad_values_fail(self):
        for mode in ('exact','scalarized','streaming'):
            bad=self.dw.clone();bad[0,0]=float('nan')
            with self.assertRaises(FloatingPointError):measure_boundaries(self.z,self.y,self.w,bad,self.hs,mode=mode)
        with self.assertRaises(ValueError):measure_boundaries(self.z[:0],self.y[:0],self.w,self.dw,[h[:0] for h in self.hs])


class SchedulerTests(unittest.TestCase):
    def test_first_and_maximum(self):
        s=CheapSentinel(max_interval=3)
        for t in range(1,8):
            r=s.observe(t,1.,1.)
            if t in (1,4,7):
                self.assertIn('FIRST_STEP' if t==1 else 'MAX_INTERVAL',r);s.measured(t)
            else:self.assertEqual(r,['NONE'])

    def test_past_mad_and_both_anomalies(self):
        s=CheapSentinel()
        for t in range(1,6):
            r=s.observe(t,1.,1.)
            if t==1:s.measured(t)
        self.assertEqual(s.observe(6,1.,1.),['NONE'])
        self.assertEqual(s.observe(7,10.,10.),['LOSS_ANOMALY','GRAD_NORM_ANOMALY'])
        with self.assertRaises(FloatingPointError):s.observe(8,float('nan'),1.)

    def test_release(self):
        for a,expected in [(2,[2,1.75,1.5,1.25,1,1]),(.25,[.25,.4375,.625,.8125,1,1])]:
            s=ReleaseToNative(4);s.decided(3,a)
            self.assertEqual([s.at(t) for t in range(3,9)],expected)
        s=ReleaseToNative();s.decided(1,2)
        self.assertEqual(s.at(1),2);self.assertEqual(s.at(2),1)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(5,3);self.head=torch.nn.Linear(3,5,bias=False)
        self.head.weight=self.emb.weight

    def forward(self,x,y):
        z=self.head(self.emb(x).mean(1))
        return z[:,None],torch.nn.functional.cross_entropy(z,y[:,-1])


class StubInstrument:
    def __init__(self,adapter):self.adapter=adapter;self.calls=0
    def measure(self,owner,x,y):
        self.calls+=1;self.adapter.jets(owner,x)
        return measured([-1,1],[[-.5,1]]*5)


class FixedGovernor:
    def __init__(self,alpha):self.alpha=alpha
    def choose(self,m):
        d=FunctionalUpdateGovernor().choose(m)
        return Decision(self.alpha,d.formed_count,d.unformed_count,d.scores)


class RuntimeTests(unittest.TestCase):
    def test_native_fast_path_and_exact_moments(self):
        for a in (.125,.25,1.,2.):
            torch.manual_seed(9);model=TinyModel();oracle=copy.deepcopy(model)
            opt=torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.98),weight_decay=.1,foreach=False)
            ref=torch.optim.AdamW(oracle.parameters(),lr=.003,betas=(.9,.98),weight_decay=.1,foreach=False)
            adapter=Mock();instrument=StubInstrument(adapter)
            engine=GovernedTraining(model,adapter,instrument=instrument,governor=FixedGovernor(a))
            x=torch.tensor([[0,1],[2,3]]);y=torch.tensor([[0,1],[0,2]])
            with patch.object(engine.owner,'propose',wraps=engine.owner.propose) as propose:
                for t in range(1,4):
                    ref.zero_grad(set_to_none=True);oracle(x,y)[1].backward()
                    torch.nn.utils.clip_grad_norm_(oracle.parameters(),1.)
                    pre={n:p.detach().clone() for n,p in oracle.named_parameters()};ref.step()
                    if t==1 and a!=1.:
                        with torch.no_grad():
                            for n,p in oracle.named_parameters():p.copy_(pre[n]+a*(p-pre[n]))
                    info=engine.update(opt,t,x,y,x,y,seed=900+t,lr=.003)
                    for p,q in zip(model.parameters(),oracle.parameters()):
                        self.assertTrue(torch.equal(p,q))
                        for k in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(opt.state[p][k],ref.state[q][k]))
                    self.assertIs(model.emb.weight,model.head.weight)
                    if t>1:self.assertEqual(info['execution_path'],'NATIVE_FAST');self.assertEqual(info['alpha'],1.)
                self.assertEqual(propose.call_count,1);self.assertEqual(adapter.jets.call_count,1)

    def test_nonfinite_signal_no_optimizer_step(self):
        model=TinyModel();opt=torch.optim.AdamW(model.parameters())
        engine=GovernedTraining(model,Mock())
        x=torch.tensor([[0,1]]);y=torch.tensor([[0,1]])
        with patch.object(model,'forward',return_value=(None,model.emb.weight.sum()*float('nan'))),patch.object(opt,'step') as step:
            with self.assertRaises(FloatingPointError):engine.update(opt,1,x,y,x,y,1,.003)
            step.assert_not_called()
        with self.assertRaises(RuntimeError):engine.update(opt,2,x,y,x,y,2,.003)

    def test_failed_measurement_no_commit_or_silent_resume(self):
        model=TinyModel();opt=torch.optim.AdamW(model.parameters())
        pre={n:p.detach().clone() for n,p in model.named_parameters()}
        inst=Mock();inst.measure.side_effect=FloatingPointError('invalid measurement')
        engine=GovernedTraining(model,Mock(),instrument=inst)
        x=torch.tensor([[0,1]]);y=torch.tensor([[0,1]])
        with patch.object(engine.owner,'commit') as commit:
            with self.assertRaises(FloatingPointError):engine.update(opt,1,x,y,x,y,1,.003)
            commit.assert_not_called()
        self.assertTrue(all(torch.equal(p,pre[n]) for n,p in model.named_parameters()))
        with self.assertRaises(RuntimeError):engine.update(opt,2,x,y,x,y,2,.003)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_actual_cuda_adapter(self):
        spec=importlib.util.spec_from_file_location('governor_test_nanogpt','D:/codex/dependencies/nanoGPT-3adf61e/model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.manual_seed(17)
        model=module.GPT(module.GPTConfig(n_layer=2,n_head=2,n_embd=32,vocab_size=24,block_size=3,bias=False,dropout=0)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,fused=True)
        adapter=NanoGPTAdapter(model);engine=GovernedTraining(model,adapter)
        x=torch.tensor([[1,2,23],[2,3,23],[4,5,23]],device='cuda')
        y=torch.tensor([[-1,-1,3],[-1,-1,5],[-1,-1,9]],device='cuda')
        with patch.object(adapter,'jets',wraps=adapter.jets) as jets:
            for t in (1,2):
                info=engine.update(opt,t,x,y,x,y,t,.001)
                self.assertEqual(info['instrument_called'],t==1)
                if t==1:self.assertEqual(len(info['scores']),5)
            self.assertEqual(jets.call_count,1)
        self.assertEqual(engine.owner.events,[])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
