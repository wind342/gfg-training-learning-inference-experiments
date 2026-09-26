"""Ordinary tests: grouping, aggregation, selection, latch, reused execution."""
import copy
import importlib.util
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v1.tests import BoundaryTests,measured
from experiments.gfg_functional_update_governor_v2 import tests as old_tests
from experiments.gfg_functional_update_governor_v2.scheduler import native_margins
from experiments.gfg_controller_distributed_runtime_v1.scale import banks
from .groups import GroupMap
from .governor import GroupGovernor
from .scheduler import GroupSentinel
from .runtime import GovernedTraining


class GroupTests(unittest.TestCase):
    def test_task_mapping_counts_and_identity(self):
        x,y,control,_ = banks(); mapping=GroupMap.target_tokens(y[control],expected=range(23))
        self.assertEqual(len(mapping.ids),212)
        self.assertEqual(sum(n<10 for n in mapping.counts),12)
        self.assertEqual((min(mapping.counts),max(mapping.counts)),(4,14))
        self.assertEqual(sorted(i for m in mapping.members for i in m),list(range(212)))
        self.assertEqual(mapping.ids.tolist(),y[control,-1].tolist())
        with self.assertRaises(ValueError):GroupMap.target_tokens(y[control],expected=range(24))
        with self.assertRaises(ValueError):GroupMap([.1,.2])
        with self.assertRaises(ValueError):mapping.ids[0]=7

    def test_q10_matches_tl_e06_definition(self):
        mapping=GroupMap([7,7,7,9,9]); values=np.array([-.2,.3,1.,2.,4.])
        data=mapping.values(values)
        self.assertEqual(data['q10'],[float(np.quantile(values[:3],.1,method='linear')),2.2])
        self.assertEqual(data['capabilities'],[2/3,1.])
        self.assertEqual(data['capability'],4/5)

    def test_initial_target_map_mismatch_rejected(self):
        y=torch.tensor([[0,0],[0,1]])
        mapping=GroupMap.target_tokens(y)
        with self.assertRaises(ValueError):GroupSentinel(mapping).bind(y,y.flip(0))


class GovernorTests(unittest.TestCase):
    def test_safe_step1_net_gain_not_individual_protection(self):
        mapping=GroupMap([0]*11+[1]*11); b=np.full(22,-1.); b[:6]=1.
        small=b.copy(); small[0]=-1; small[6]=1
        native=b.copy(); native[:5]=-1; native[6:13]=1
        g=GroupGovernor(mapping)
        d=g.choose(measured(b.tolist(),[small.tolist()]*3+[native.tolist(),native.tolist()]))
        self.assertEqual(d.alpha,1.)
        self.assertEqual(g.summary['protected_groups'],[])
        self.assertEqual(d.scores['1.0']['transitions']['correct_to_wrong'],5)
        self.assertEqual(d.scores['1.0']['transitions']['wrong_to_correct'],7)
        self.assertEqual(d.scores['1.0']['correct_count'],8)
        self.assertEqual(d.scores['0.125']['correct_count'],6)

    def test_no_protected_group_amplification(self):
        b=[-1.]*20; native=b.copy(); native[:3]=[1.]*3
        bigger=b.copy(); bigger[:5]=[1.]*5
        d=GroupGovernor(GroupMap([0]*10+[1]*10)).choose(measured(b,[b,b,b,native,bigger]))
        self.assertEqual(d.alpha,2.);self.assertEqual(d.reason,'AMPLIFY_FORMATION')

    def test_protected_group_shrink(self):
        b=[1.]*7+[-1.]*7; damaged=b.copy();damaged[0]=-1
        d=GroupGovernor(GroupMap([0]*7+[1]*7)).choose(measured(b,[b,b,b,damaged,damaged]))
        self.assertEqual(d.alpha,.5)
        self.assertEqual(d.reason,'SHRINK_PROTECTED_GROUP')

    def test_one_crossing_allowed_when_group_still_90(self):
        b=[1.]*10+[-1.]*10
        native=b.copy();native[0]=-1;native[10:12]=[1.,1.]
        d=GroupGovernor(GroupMap([0]*10+[1]*10)).choose(measured(b,[b,b,b,native,native]))
        self.assertEqual(d.alpha,1.)
        self.assertEqual(d.scores['1.0']['protected_violation_count'],0)
        self.assertEqual(d.scores['1.0']['capabilities'][0],.9)

    def test_macro_breaks_overall_tie_but_q10_alone_does_not(self):
        mapping=GroupMap([0]*4+[1]*10);b=[-1.]*14
        native=b.copy();native[4]=1
        alternative=b.copy();alternative[0]=1
        d=GroupGovernor(mapping).choose(measured(b,[b,b,b,native,alternative]))
        self.assertEqual(d.alpha,2.)
        high=[v*100 for v in native]
        d=GroupGovernor(mapping).choose(measured(b,[b,b,b,native,high]))
        self.assertEqual(d.alpha,1.)

    def test_all_protected_no_margin_only_amplification(self):
        b=[1.]*20
        d=GroupGovernor(GroupMap([0]*10+[1]*10)).choose(measured(b,[b,b,b,b,[100.]*20]))
        self.assertEqual(d.alpha,1.)

    def test_degraded_history_not_added_to_current_constraints(self):
        b=[1.]*8+[-1.]*2+[-1.]*10
        g=GroupGovernor(GroupMap([0]*10+[1]*10));g.choose(measured(b,[b]*5))
        self.assertEqual(g.summary['protected_groups'],[])

    def test_nonfinite(self):
        g=GroupGovernor(GroupMap([0,1]))
        for v in (float('nan'),float('inf')):
            with self.assertRaises(FloatingPointError):g.choose(measured([1,v],[[1,1]]*5))
            with self.assertRaises(FloatingPointError):g.choose(measured([1,1],[[1,v]]*5))


class SchedulerTests(unittest.TestCase):
    def initial(self,values,mapping=None,**kwargs):
        s=GroupSentinel(mapping or GroupMap([0]*len(values)),**kwargs)
        self.assertEqual(s.begin(1),['FIRST_STEP']);s.measured(1,torch.tensor(values,dtype=torch.float64))
        return s

    def test_forming_fluctuations_do_not_trigger(self):
        s=self.initial([1]*6+[-1]*5)
        events=s.observe(list(range(11)),[.01]*6+[-10]*5)
        self.assertEqual(s.reasons(events),[]);self.assertEqual(s.state,['FORMING'])

    def test_soft_hard_escalation_anchor_refresh_and_rearm(self):
        s=self.initial([1]*10)
        events=s.observe(list(range(10)),[.4]*10)
        self.assertEqual(s.reasons(events),['PROTECTED_GROUP_MARGIN_EROSION'])
        s.measured(1,torch.full((10,),.3,dtype=torch.float64))
        self.assertEqual(s.anchor[0],1.)
        events=s.observe(list(range(10)),[-1]*2+[.3]*8)
        self.assertEqual(s.reasons(events),['PROTECTED_GROUP_DEGRADATION'])
        self.assertEqual(s.state,['DEGRADED'])
        self.assertEqual(s.reasons(s.observe(list(range(10)),[-1]*2+[.2]*8)),[])
        s.measured(1,torch.tensor([-1]*2+[.2]*8,dtype=torch.float64))
        self.assertEqual(s.anchor[0],1.)
        s.observe(list(range(10)),[.6]*10);self.assertEqual(s.level[0],2)
        s.observe(list(range(10)),[.8]*10);self.assertEqual(s.level[0],0)
        self.assertEqual(s.state,['PROTECTED'])
        self.assertEqual(s.reasons(s.observe(list(range(10)),[-1]*2+[1]*8)),['PROTECTED_GROUP_DEGRADATION'])

    def test_q10_crosses_zero_at_still_90_accuracy(self):
        s=self.initial([1]*10)
        events=s.observe(list(range(10)),[-20]+[1]*9)
        self.assertEqual(s.capabilities[0],.9);self.assertLess(s.q10[0],0)
        self.assertEqual(s.reasons(events),['PROTECTED_GROUP_MARGIN_EROSION'])

    def test_one_wrong_without_q10_erosion_no_alarm(self):
        s=self.initial([1]*10)
        self.assertEqual(s.reasons(s.observe(list(range(10)),[-.01]+[1]*9)),[])
        self.assertEqual(s.state,['PROTECTED'])

    def test_small_group_single_wrong_can_degrade(self):
        s=self.initial([1]*7)
        self.assertEqual(s.reasons(s.observe(list(range(7)),[-.01]+[1]*6)),['PROTECTED_GROUP_DEGRADATION'])

    def test_groups_independent_and_full_detects_other_group(self):
        s=self.initial([1]*14,GroupMap([0]*7+[1]*7))
        s.observe(list(range(7)),[-1]*7)
        e=s.measured(1,torch.tensor([-1]*14,dtype=torch.float64))
        self.assertIn('PROTECTED_GROUP_DEGRADATION',s.reasons(e))
        self.assertEqual(s.level.tolist(),[2,2])

    def test_canonical_round_robin_age_interval(self):
        x,y,control,_=banks();mapping=GroupMap.target_tokens(y[control],expected=range(23))
        s=self.initial([-1.]*212,mapping)
        seen=[]
        for step in range(2,25):
            self.assertEqual(s.begin(step),[])
            ids=s.shard(step);seen.extend(ids);s.observe(ids,[-1.]*len(ids))
        self.assertEqual(sorted(seen),list(range(212)))
        self.assertEqual(s.snapshot(24)['steps_since_group_observed'],list(range(22,-1,-1)))
        for step in range(25,51):s.begin(step)
        self.assertEqual(s.begin(51),['MAX_INTERVAL'])

    def test_sparse_identity_partial_group_and_nan(self):
        s=self.initial([-1.]*4,group_micro_every=3)
        self.assertEqual(s.shard(2),[]);self.assertEqual(s.shard(3),list(range(4)))
        x=torch.ones(4,2);y=torch.zeros(4,2,dtype=torch.long);s.bind(x,y)
        with self.assertRaises(ValueError):s.bind(x.clone(),y)
        y[0,0]=1
        with self.assertRaises(ValueError):s.bind(x,y)
        with self.assertRaises(ValueError):s.observe([0,1],[1,1])
        with self.assertRaises(FloatingPointError):s.observe(list(range(4)),[1,1,1,np.nan])


class SharedEngineTests(old_tests.RuntimeTests):
    """Regression tests of untouched v2 execution/sensor actually reused by v3."""


class RuntimeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_actual_v3_cuda_adapter(self):
        from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter
        spec=importlib.util.spec_from_file_location('v3_tiny_nanogpt','D:/codex/dependencies/nanoGPT-3adf61e/model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.manual_seed(23)
        model=module.GPT(module.GPTConfig(n_layer=2,n_head=2,n_embd=32,vocab_size=24,block_size=3,bias=False,dropout=0)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,fused=True)
        x=torch.tensor([[1,2,23],[2,3,23],[4,5,23]],device='cuda')
        y=torch.tensor([[-1,-1,3],[-1,-1,5],[-1,-1,9]],device='cuda')
        adapter=NanoGPTAdapter(model)
        engine=GovernedTraining(model,adapter,mapping=GroupMap.target_tokens(y))
        with patch.object(adapter,'jets',wraps=adapter.jets) as jets:
            for t in (1,2):
                info=engine.update(opt,t,x,y,x,y,t,.001)
                self.assertEqual(info['runtime_version'],'governor_v3')
                if t==2:self.assertEqual(len(info['micro_indices']),1)
            self.assertEqual(jets.call_count,engine.instrument_calls)
        self.assertIs(model.transformer.wte.weight,model.lm_head.weight)

    def test_v3_native_exact_and_no_jvp_on_micro(self):
        torch.manual_seed(19);model=old_tests.TinyModel();oracle=copy.deepcopy(model)
        opt=torch.optim.AdamW(model.parameters(),lr=.003,foreach=False)
        ref=torch.optim.AdamW(oracle.parameters(),lr=.003,foreach=False)
        adapter=Mock();inst=Mock();inst.measure.return_value=measured([-1,-1],[[-1,-1]]*5)
        engine=GovernedTraining(model,adapter,mapping=GroupMap([0,0]),instrument=inst)
        x=torch.tensor([[0,1],[2,3]]);y=torch.tensor([[0,1],[0,2]])
        with patch('experiments.gfg_functional_update_governor_v2.runtime.native_margins',return_value=np.array([-1.,-1.])), \
             patch.object(engine.owner,'propose',wraps=engine.owner.propose) as proposal:
            for t in (1,2,3):
                ref.zero_grad(set_to_none=True);oracle(x,y)[1].backward()
                torch.nn.utils.clip_grad_norm_(oracle.parameters(),1.);ref.step()
                info=engine.update(opt,t,x,y,x,y,t,.003)
                self.assertEqual(info['runtime_version'],'governor_v3')
                self.assertEqual(info['alpha'],1.)
                if t>1:self.assertEqual(info['execution_path'],'MICRO_PROBE')
                for p,q in zip(model.parameters(),oracle.parameters()):
                    self.assertTrue(torch.equal(p,q))
                    for k in opt.state[p]:self.assertTrue(torch.equal(opt.state[p][k],ref.state[q][k]))
            self.assertEqual(proposal.call_count,1);self.assertEqual(inst.measure.call_count,1)
            self.assertIs(model.emb.weight,model.head.weight)

    def test_harness_v3_injection(self):
        from . import smoke
        original=smoke.harness.create
        def check(case):
            self.assertIs(smoke.harness.create,smoke.create)
            self.assertEqual(smoke.harness.CONFIG['group_micro_every'],1)
        with patch.object(smoke.harness,'run',side_effect=check):smoke.run('safe')
        self.assertIs(smoke.harness.create,original)


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
