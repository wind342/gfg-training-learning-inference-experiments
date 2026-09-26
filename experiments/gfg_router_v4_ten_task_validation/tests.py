import unittest
import numpy as np
import torch
from .common import RandomReplay,Memory,exposure_schedule,summarize_predictions,scope,EXCLUDE
from .check import longest
from experiments.gfg_functional_feedback_router_v3.dose import weights

class TenTaskTests(unittest.TestCase):
    def memory(self):
        m=Memory();m.groups={i:tuple(range(10*i,10*i+10)) for i in range(72)}
        return m
    def test_random_exact_dose(self):
        r=RandomReplay(2021);m=self.memory()
        for n in range(0,17,2):
            gs,ids=r.choose(set(range(72)),m,n)
            self.assertEqual(len(ids),n);self.assertEqual(len(set(gs)),n//2);self.assertEqual(len(set(ids)),n)
            self.assertTrue(all(i in m.groups[g] for j,g in enumerate(gs) for i in ids[2*j:2*j+2]))
    def test_random_reproducible(self):
        a=RandomReplay(2021);b=RandomReplay(2021);m=self.memory()
        self.assertEqual([a.choose(range(72),m,8) for _ in range(10)],[b.choose(range(72),m,8) for _ in range(10)])
    def test_random_invalid_dose(self):
        for n in (-1,1,18):
            with self.assertRaises(ValueError):RandomReplay(2021).choose(range(72),self.memory(),n)
    def test_random_no_global_rng_effect(self):
        torch.manual_seed(5);state=torch.get_rng_state().clone();np.random.seed(5);state_np=np.random.get_state()
        RandomReplay(2021).choose(range(72),self.memory(),16)
        self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertTrue(np.array_equal(state_np[1],np.random.get_state()[1]))
    def test_random_no_risk_interface(self):
        import inspect
        self.assertEqual(list(inspect.signature(RandomReplay.choose).parameters),['self','old','memory','n'])
    def test_schedule_strips_identity_and_risk(self):
        row=dict(step=211,task=2,info=dict(feedback_batch_size=4,selected_feedback_groups=[1,2],risk=[1],alpha=.25))
        self.assertEqual(exposure_schedule([row]),[dict(step=211,task=2,old_batch_size=4)])
    def test_dose_gradient_equivalence(self):
        torch.manual_seed(7);x=torch.randn(22,4,dtype=torch.float64);y=torch.randn(22,2,dtype=torch.float64)
        a=torch.nn.Linear(4,2,dtype=torch.float64);b=torch.nn.Linear(4,2,dtype=torch.float64);b.load_state_dict(a.state_dict())
        ((a(x)-y)**2).mean().backward()
        batches=(list(range(16)),list(range(16,22)))
        for ids,w in zip(batches,weights(batches)):
            (w*((b(x[ids])-y[ids])**2).mean()).backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-12,atol=1e-12)
    def test_scope_changes_only_one_control(self):
        y=np.repeat(np.arange(80),700);s=np.tile(np.r_[np.zeros(420),np.ones(140),np.full(140,2)],80)
        # Real split is shuffled: use real data for authorized index identity.
        from .common import DATA
        d=np.load(DATA/'data.npz');p,clean,full=scope(d,80)
        self.assertIn(50130,p);self.assertNotIn(50124,p);self.assertEqual(len(p),800)
        self.assertEqual(set(full)-set(clean),set(EXCLUDE));self.assertEqual(len(clean),11193)
    def test_longest_gap(self):self.assertEqual(longest([0,1,1,0,1,1,1]),3)
    def test_primary_cohort_metrics(self):
        y=np.arange(16);d={'y':y};r=summarize_predictions(list(range(16)),[0]*8+list(range(8,16)),d,2)
        self.assertEqual(r['old'],1/8);self.assertEqual(r['new'],1);self.assertEqual(r['macro_all'],9/16)
    def test_zero_random_does_not_advance_rng(self):
        a=RandomReplay(2021);b=RandomReplay(2021);m=self.memory();a.choose(range(72),m,0)
        self.assertEqual(a.choose(range(72),m,16),b.choose(range(72),m,16))

if __name__=='__main__':unittest.main()
