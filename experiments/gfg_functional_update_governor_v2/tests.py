"""Ordinary unit/integration tests, including reused variable-alpha backend tests."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v1.tests import BoundaryTests, measured
from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter
from .governor import FunctionalUpdateGovernor, Decision
from .runtime import GovernedTraining
from .scheduler import FunctionalSentinel, native_margins


class GovernorTests(unittest.TestCase):
    def test_safe_amplification(self):
        d = FunctionalUpdateGovernor().choose(measured([1,-1,-2], [[1,-1,-2],[1,-.5,-1],[1,.1,-1],[1,.1,-.1],[1,.2,.1]]))
        self.assertEqual(d.alpha, 2.)
        self.assertEqual(d.reason, 'AMPLIFY_FORMATION')

    def test_preservation_shrink(self):
        d = FunctionalUpdateGovernor().choose(measured([1,-1], [[1,.1],[1,.2],[-1,.3],[-1,.4],[-1,.5]]))
        self.assertEqual(d.alpha, .25)
        self.assertEqual(d.reason, 'SHRINK_PRESERVATION')

    def test_all_formed_never_amplifies_closest_safe(self):
        d = FunctionalUpdateGovernor().choose(measured([1], [[100],[.1],[-1],[-2],[1000]]))
        self.assertEqual(d.alpha, .25)

    def test_native_safe_and_margin_only_stays_native(self):
        for b, p in [([1], [[2],[3],[4],[5],[100]]), ([-1], [[-1],[-.9],[-.8],[-.7],[-.1]]),
                     ([1,-1], [[1,-1],[1,-.9],[1,-.8],[1,-.7],[1,-.01]])]:
            d = FunctionalUpdateGovernor().choose(measured(b,p))
            self.assertEqual(d.alpha, 1.)
            self.assertTrue(all('utility' not in s for s in d.scores.values()))

    def test_damage_unchanged_gain_improved(self):
        d = FunctionalUpdateGovernor().choose(measured([1,-1], [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,1]]))
        self.assertEqual(d.alpha, 2.)

    def test_preservation_is_explicit_priority(self):
        d = FunctionalUpdateGovernor().choose(measured([1,-1,-1], [[1,-1,-1],[1,-1,-1],[-1,1,1],[-1,1,1],[-1,1,1]]))
        self.assertEqual(d.alpha, .25)
        self.assertEqual(d.objective, 'PRESERVATION')

    def test_no_formed_can_gain(self):
        d = FunctionalUpdateGovernor().choose(measured([-1], [[-1],[-1],[-1],[-1],[1]]))
        self.assertEqual(d.alpha, 2.)

    def test_bad_measurements(self):
        for v in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(FloatingPointError):
                FunctionalUpdateGovernor().choose(measured([v], [[1]]*5))
            with self.assertRaises(FloatingPointError):
                FunctionalUpdateGovernor().choose(measured([1], [[v]]*5))
        with self.assertRaises(ValueError):
            FunctionalUpdateGovernor().choose(measured([], [[]]*5))


class SentinelTests(unittest.TestCase):
    def initialized(self, values=(1.,1.,-1.), **kwargs):
        s = FunctionalSentinel(**kwargs)
        self.assertEqual(s.begin(1), ['FIRST_STEP'])
        s.measured(1, torch.tensor(values))
        return s

    def test_erosion_latch_refresh_escalation_and_rearm(self):
        s = self.initialized()
        self.assertEqual(s.reasons(s.observe([0],[.4])), ['FORMED_MARGIN_EROSION'])
        self.assertEqual(s.reasons(s.observe([0],[.3])), [])
        s.measured(2, torch.tensor([.3,1.,-1.]))
        self.assertEqual(s.anchor[0], 1.)
        self.assertEqual(s.reasons(s.observe([0],[-.1])), ['FORMED_BOUNDARY_CROSSING'])
        s.measured(3, torch.tensor([-.1,1.,-1.]))
        self.assertEqual(s.anchor[0], 1.)
        self.assertEqual(s.reasons(s.observe([0],[-.2])), [])
        s.observe([0],[.6]); self.assertEqual(s.level[0], 2)
        s.observe([0],[.8]); self.assertEqual(s.level[0], 0)
        self.assertEqual(s.reasons(s.observe([0],[-.1])), ['FORMED_BOUNDARY_CROSSING'])

    def test_independent_targets_not_globally_suppressed(self):
        s = self.initialized()
        s.observe([0],[-1])
        self.assertEqual(s.reasons(s.observe([1],[-1])), ['FORMED_BOUNDARY_CROSSING'])

    def test_full_refresh_detects_unobserved_failure(self):
        s = self.initialized()
        events = s.measured(2, torch.tensor([1.,-1.,-1.]))
        self.assertEqual(s.reasons(events), ['FORMED_BOUNDARY_CROSSING'])
        self.assertEqual(s.anchor[1], 1.)
        self.assertEqual(s.level[1], 2)

    def test_new_formation_is_remembered(self):
        s = self.initialized()
        s.observe([2],[.7])
        self.assertEqual(s.reasons(s.observe([2],[-.1])), ['FORMED_BOUNDARY_CROSSING'])

    def test_floor_and_hard_crossing(self):
        s = self.initialized((1e-9,))
        self.assertEqual(s.reasons(s.observe([0],[1e-12])), [])
        self.assertEqual(s.reasons(s.observe([0],[0])), ['FORMED_BOUNDARY_CROSSING'])

    def test_round_robin_tail_and_maximum(self):
        s = self.initialized(tuple([1.]*212), max_interval=50)
        shards = []
        for step in range(2,51):
            self.assertEqual(s.begin(step), [])
            ids = s.shard(step)
            if step < 29: shards.extend(ids)
        self.assertEqual(shards, list(range(212)))
        self.assertEqual(s.begin(51), ['MAX_INTERVAL'])

    def test_cadence_identity_and_bad_values(self):
        s = self.initialized(micro_every=3)
        self.assertEqual(s.shard(2), [])
        self.assertEqual(s.shard(3), [0,1,2])
        x = torch.ones(3,2); y = torch.ones(3,2,dtype=torch.long)
        s.bind(x,y)
        x[0,0] = 2
        with self.assertRaises(ValueError): s.bind(x,y)
        for v in (float('nan'),float('inf')):
            with self.assertRaises(FloatingPointError): s.observe([0],[v])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(5,3)
        self.head = torch.nn.Linear(3,5,bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer('unchanged_buffer', torch.ones(2))

    def forward(self,x,y=None):
        z = self.head(self.emb(x).mean(1))
        loss = None if y is None else torch.nn.functional.cross_entropy(z,y[:,-1])
        return z[:,None], loss


class StubInstrument:
    def __init__(self,adapter): self.adapter = adapter
    def measure(self,owner,x,y):
        self.adapter.jets(owner,x)
        return measured([-1,1], [[-.5,1]]*5)


class FixedGovernor:
    def __init__(self,alpha): self.alpha = alpha
    def choose(self,m):
        d = FunctionalUpdateGovernor().choose(m)
        return Decision(self.alpha,d.formed_count,d.unformed_count,d.scores,'TEST','TEST')


class RuntimeTests(unittest.TestCase):
    def test_smoke_harness_injects_v2_constructor(self):
        from . import smoke
        original = smoke.harness.create
        def check(case):
            self.assertIs(smoke.harness.create,smoke.create)
            self.assertEqual(smoke.harness.CONFIG,smoke.CONFIG)
            self.assertEqual(smoke.harness.OUT,smoke.OUT)
            self.assertEqual(case,'safe')
        with patch.object(smoke.harness,'run',side_effect=check): smoke.run('safe')
        self.assertIs(smoke.harness.create,original)

    def test_world_size_guard_including_default_group(self):
        with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_world_size',return_value=2):
            with self.assertRaises(NotImplementedError): GovernedTraining(TinyModel(),Mock())

    def test_nonfinite_micro_forward(self):
        model = TinyModel(); x = torch.tensor([[0,1]]); y = torch.tensor([[0,1]])
        for value in (float('nan'),float('inf')):
            with patch.object(model,'forward',return_value=(torch.full((1,1,5),value),None)):
                with self.assertRaises(FloatingPointError): native_margins(model,x,y)

    def test_micro_forward_preserves_state_rng_gradients_modes(self):
        model = TinyModel(); model.head.eval()
        opt = torch.optim.AdamW(model.parameters())
        x = torch.tensor([[0,1],[2,3]]); y = torch.tensor([[0,1],[0,2]])
        model(x,y)[1].backward(); opt.step()
        before = copy.deepcopy(model.state_dict()); state = copy.deepcopy(opt.state_dict())
        grads = [p.grad.clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone(); modes = [m.training for m in model.modules()]
        with patch('torch.func.jvp',side_effect=AssertionError('micro called jvp')):
            value = native_margins(model,x,y)
        self.assertEqual(value.shape,(2,))
        for n,p in model.state_dict().items(): self.assertTrue(torch.equal(p,before[n]))
        for key,entry in opt.state_dict()['state'].items():
            for n,p in entry.items(): self.assertTrue(torch.equal(p,state['state'][key][n]))
        for p,g in zip(model.parameters(),grads): self.assertTrue(torch.equal(p.grad,g))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(modes,[m.training for m in model.modules()])

    def test_exact_native_fast_and_micro_paths_after_scaled_update(self):
        for cadence in (1,10):
            for a in (.125,.25,1.,2.):
                torch.manual_seed(9); model = TinyModel(); oracle = copy.deepcopy(model)
                opt = torch.optim.AdamW(model.parameters(),lr=.003,betas=(.9,.98),weight_decay=.1,foreach=False)
                ref = torch.optim.AdamW(oracle.parameters(),lr=.003,betas=(.9,.98),weight_decay=.1,foreach=False)
                adapter = Mock()
                engine = GovernedTraining(model,adapter,instrument=StubInstrument(adapter),governor=FixedGovernor(a),
                                          scheduler=FunctionalSentinel(micro_every=cadence))
                x = torch.tensor([[0,1],[2,3]]); y = torch.tensor([[0,1],[0,2]])
                with patch.object(engine.owner,'propose',wraps=engine.owner.propose) as proposal, \
                     patch('experiments.gfg_functional_update_governor_v2.runtime.native_margins',return_value=np.array([-1.,1.])):
                    for t in range(1,4):
                        ref.zero_grad(set_to_none=True); oracle(x,y)[1].backward()
                        torch.nn.utils.clip_grad_norm_(oracle.parameters(),1.)
                        pre = [p.detach().clone() for p in oracle.parameters()]; ref.step()
                        if t == 1 and a != 1.:
                            with torch.no_grad():
                                for p,q in zip(oracle.parameters(),pre): p.copy_(q+a*(p-q))
                        info = engine.update(opt,t,x,y,x,y,900+t,.003)
                        for p,q in zip(model.parameters(),oracle.parameters()):
                            self.assertTrue(torch.equal(p,q))
                            for key in ('step','exp_avg','exp_avg_sq'): self.assertTrue(torch.equal(opt.state[p][key],ref.state[q][key]))
                        self.assertIs(model.emb.weight,model.head.weight)
                        if t > 1:
                            self.assertEqual(info['execution_path'],'MICRO_PROBE' if cadence==1 else 'NATIVE_FAST')
                            self.assertEqual(info['alpha'],1.)
                    self.assertEqual(proposal.call_count,1)
                    self.assertEqual(adapter.jets.call_count,1)

    def test_failed_measurement_no_commit(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters())
        pre = copy.deepcopy(model.state_dict()); inst = Mock()
        inst.measure.side_effect = FloatingPointError('invalid')
        engine = GovernedTraining(model,Mock(),instrument=inst)
        x = torch.tensor([[0,1]]); y = torch.tensor([[0,1]])
        with patch.object(engine.owner,'commit') as commit:
            with self.assertRaises(FloatingPointError): engine.update(opt,1,x,y,x,y,1,.003)
            commit.assert_not_called()
        for n,p in model.state_dict().items(): self.assertTrue(torch.equal(p,pre[n]))
        with self.assertRaises(RuntimeError): engine.update(opt,2,x,y,x,y,2,.003)

    def test_nan_loss_no_optimizer_step(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters()); engine = GovernedTraining(model,Mock())
        x = torch.tensor([[0,1]]); y = torch.tensor([[0,1]])
        with patch.object(model,'forward',return_value=(None,model.emb.weight.sum()*float('nan'))),patch.object(opt,'step') as step:
            with self.assertRaises(FloatingPointError): engine.update(opt,1,x,y,x,y,1,.003)
            step.assert_not_called()

    def test_nan_micro_no_proposal(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters()); adapter = Mock()
        engine = GovernedTraining(model,adapter,instrument=StubInstrument(adapter))
        x = torch.tensor([[0,1],[2,3]]); y = torch.tensor([[0,1],[0,2]])
        engine.update(opt,1,x,y,x,y,1,.003)
        with patch('experiments.gfg_functional_update_governor_v2.runtime.native_margins',return_value=np.array([np.nan,1.])), \
             patch.object(opt,'step') as step:
            with self.assertRaises(FloatingPointError): engine.update(opt,2,x,y,x,y,2,.003)
            step.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_actual_cuda_adapter_micro_is_derivative_free(self):
        spec = importlib.util.spec_from_file_location('governor_v2_test_nanogpt','D:/codex/dependencies/nanoGPT-3adf61e/model.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        torch.manual_seed(17)
        model = module.GPT(module.GPTConfig(n_layer=2,n_head=2,n_embd=32,vocab_size=24,block_size=3,bias=False,dropout=0)).cuda()
        opt = torch.optim.AdamW(model.parameters(),lr=.001,fused=True)
        adapter = NanoGPTAdapter(model); engine = GovernedTraining(model,adapter)
        x = torch.tensor([[1,2,23],[2,3,23],[4,5,23]],device='cuda')
        y = torch.tensor([[-1,-1,3],[-1,-1,5],[-1,-1,9]],device='cuda')
        engine.update(opt,1,x,y,x,y,1,.001)
        pre = [p.clone() for p in model.parameters()]
        with patch.object(adapter,'jets',side_effect=AssertionError('micro called jets')):
            native_margins(model,x,y)
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(model.parameters(),pre)))
        info = engine.update(opt,2,x,y,x,y,2,.001)
        self.assertTrue(info['micro_probe_called'])
        self.assertIs(model.transformer.wte.weight,model.lm_head.weight)
        self.assertEqual(engine.owner.events,[])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
