import unittest
from experiments.gfg_functional_feedback_router_v1.tests import memory
from .policy import Router

class SparseAnchorTests(unittest.TestCase):
    def test_single_risk_8_groups(self):
        d=Router().choose(1,list(range(8)),['DEGRADED']+['PROTECTED']*7,[2]+[0]*7,set(range(8)),memory(8))
        self.assertEqual(d['feedback_batch_size'],4);self.assertEqual(d['selected_feedback_groups'],[0,1])
    def test_single_risk_80_groups(self):
        d=Router().choose(1,list(range(80)),['DEGRADED']+['PROTECTED']*79,[2]+[0]*79,set(range(80)),memory(80))
        self.assertEqual(d['feedback_batch_size'],4)
    def test_two_inherited_debts(self):
        states=['PROTECTED']*8;levels=[0]*8
        for k in (0,4):states[k]='DEGRADED';levels[k]=2
        d=Router().choose(211,list(range(8)),states,levels,set(range(8)),memory(8))
        self.assertEqual(d['feedback_batch_size'],6);self.assertEqual(d['active_degraded_groups'],[0,4])
    def test_rotation_and_persistent_debt_not_erased(self):
        r=Router();m=memory(8);anchors=[]
        for t in range(1,15):
            d=r.choose(t,list(range(8)),['DEGRADED']+['PROTECTED']*7,[2]+[0]*7,set(range(8)),m)
            self.assertEqual(d['router_mode'],'RECOVERY');self.assertFalse(d['release_event']);anchors+=d['selected_anchor_groups']
        self.assertEqual(anchors,list(range(1,8))*2)
    def test_many_real_risks_can_fill_capacity(self):
        d=Router().choose(1,list(range(10)),['DEGRADED']*9+['PROTECTED'],[2]*9+[0],set(range(10)),memory(10))
        self.assertEqual(d['feedback_batch_size'],16);self.assertEqual(d['selected_anchor_groups'],[9])
    def test_no_stable_anchor(self):
        d=Router().choose(1,[0],['DEGRADED'],[2],{0},memory(1));self.assertEqual(d['feedback_batch_size'],2)
    def test_formation_no_memory_and_release(self):
        r=Router();r.choose(1,[0],['DEGRADED'],[2],{0},memory(1))
        class Bad:
            def __getattribute__(self,name):raise AssertionError('memory read')
        d=r.choose(2,[0],['PROTECTED'],[0],{0},Bad());self.assertTrue(d['release_event']);self.assertEqual(d['feedback_batch_size'],0)

if __name__=='__main__':unittest.main(verbosity=2)
