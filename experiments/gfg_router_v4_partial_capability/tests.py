import copy
import unittest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_feedback_router_v2.policy import Router as V3Router
from experiments.gfg_functional_feedback_router_v1.tests import memory
from .policy import PartialSentinel,Router,complete_old

class PartialTests(unittest.TestCase):
    def sentinel(self,n=3):
        s=PartialSentinel(GroupMap(np.repeat(np.arange(n),10)));s.last_step=1;return s
    def test_only_completed_forming_converted(self):
        s=self.sentinel(4);s.state=['FORMING','PROTECTED','DEGRADED','FORMING'];s.level[2]=2
        self.assertEqual(complete_old(s,{0,1,2}),[0]);self.assertEqual(s.state,['PARTIAL','PROTECTED','DEGRADED','FORMING']);self.assertEqual(s.level[2],2)
    def test_idempotent_completion(self):
        s=self.sentinel();complete_old(s,{0});self.assertEqual(complete_old(s,{0}),[])
    def test_numeric_history_unchanged_at_completion(self):
        s=self.sentinel();s.q10[:]=[-1,0,2];s.anchor[:]=[3,4,5];before=(s.q10.copy(),s.anchor.copy(),s.level.copy())
        complete_old(s,{0,1});self.assertTrue(all(np.array_equal(a,b) for a,b in zip(before,(s.q10,s.anchor,s.level))))
    def test_partial_below_threshold_not_degraded(self):
        s=self.sentinel();s.state[0]='PARTIAL'
        self.assertEqual(s._observe_group(0,[1]*8+[-1]*2,'micro'),[]);self.assertEqual(s.state[0],'PARTIAL');self.assertEqual(s.capabilities[0],.8)
        s._observe_group(0,[1]+[-1]*9,'micro');self.assertEqual(s.state[0],'PARTIAL');self.assertEqual(s.level[0],0)
    def test_partial_promotes_at_original_threshold(self):
        s=self.sentinel();s.state[0]='PARTIAL';e=s._observe_group(0,[1]*9+[-1],'micro')
        self.assertEqual(s.threshold,.9);self.assertEqual(s.state[0],'PROTECTED');self.assertEqual(e[0]['kind'],'GROUP_FORMED')
    def test_promoted_partial_uses_original_degradation(self):
        s=self.sentinel();s.state[0]='PARTIAL';s._observe_group(0,[1]*10,'micro');e=s._observe_group(0,[1]*8+[-1]*2,'micro')
        self.assertEqual(s.state[0],'DEGRADED');self.assertEqual(s.level[0],2);self.assertEqual(e[0]['kind'],'PROTECTED_GROUP_DEGRADATION')
    def test_current_forming_unchanged(self):
        s=self.sentinel();s._observe_group(2,[1]*8+[-1]*2,'micro');self.assertEqual(s.state[2],'FORMING')
    def test_invalid_partial_observation_restores_state(self):
        s=self.sentinel();s.state[0]='PARTIAL'
        with self.assertRaises(FloatingPointError):s._observe_group(0,[float('nan')]*10,'micro')
        self.assertEqual(s.state[0],'PARTIAL')
    def test_shared_slot_alternates_two_partials(self):
        r=Router();m=memory(4);got=[]
        for t in range(1,9):
            d=r.choose(t,list(range(4)),['DEGRADED','PARTIAL','PARTIAL','PROTECTED'],[2,0,0,0],set(range(4)),m)
            self.assertEqual(d['feedback_batch_size'],4);self.assertEqual(d['selected_feedback_groups'][0],0)
            self.assertEqual(len(d['selected_partial_groups'])+len(d['selected_anchor_groups']),1)
            got.append(d['selected_feedback_groups'][1])
        self.assertEqual(got,[3,1,3,2,3,1,3,2])
    def test_no_risk_partial_still_served(self):
        r=Router();m=memory(2)
        a=r.choose(1,[0,1],['PARTIAL','PROTECTED'],[0,0],{0,1},m)
        b=r.choose(2,[0,1],['PARTIAL','PROTECTED'],[0,0],{0,1},m)
        self.assertEqual(a['router_mode'],'BALANCE');self.assertEqual(b['selected_partial_groups'],[0])
    def test_no_anchor_partial_only_even(self):
        r=Router();m=memory(1)
        a=r.choose(1,[0],['PARTIAL'],[0],{0},m);b=r.choose(2,[0],['PARTIAL'],[0],{0},m)
        self.assertEqual(a['feedback_batch_size'],0);self.assertEqual(b['feedback_batch_size'],2)
    def test_partial_cannot_displace_eight_risks_without_anchor(self):
        r=Router();d=r.choose(2,list(range(9)),['DEGRADED']*8+['PARTIAL'],[2]*8+[0],set(range(9)),memory(9))
        self.assertEqual(d['selected_feedback_groups'],list(range(8)));self.assertEqual(d['selected_partial_groups'],[])
    def test_many_partials_no_blanket_fill(self):
        d=Router().choose(2,list(range(80)),['PARTIAL']*80,[0]*80,set(range(80)),memory(80))
        self.assertEqual(d['feedback_batch_size'],2)
    def test_current_groups_not_replayed(self):
        d=Router().choose(2,list(range(4)),['PARTIAL','PROTECTED','FORMING','FORMING'],[0]*4,{0,1},memory(2))
        self.assertTrue(set(d['selected_feedback_groups'])<={0,1})
    def test_no_partial_matches_v3(self):
        a=Router();b=V3Router();ma=memory(16);mb=memory(16)
        for t in range(421,451):
            states=['DEGRADED']*3+['PROTECTED']*13;levels=[2]*3+[1]*2+[0]*11
            x=a.choose(t,list(range(16)),states,levels,set(range(16)),ma);y=b.choose(t,list(range(16)),states,levels,set(range(16)),mb)
            for k,v in y.items():
                if k=='rotation_cursors':self.assertEqual({n:x[k][n] for n in v},v)
                else:self.assertEqual(x[k],v)
    def test_task3_two_partials_each_receive_52_or53_turns(self):
        r=Router();m=memory(16);counts={9:0,14:0};states=['PROTECTED']*16
        for g in (0,3,13):states[g]='DEGRADED'
        for g in counts:states[g]='PARTIAL'
        levels=[2 if s=='DEGRADED' else 0 for s in states]
        for t in range(421,631):
            d=r.choose(t,list(range(16)),states,levels,set(range(16)),m)
            for g in d['selected_partial_groups']:counts[g]+=1
            self.assertLessEqual(d['feedback_batch_size'],8)
        self.assertEqual(sorted(counts.values()),[52,53])
    def test_actual_task_boundary_installs_partial_observer(self):
        from experiments.gfg_governor_fewrel_v1 import run as fewrel
        from experiments.gfg_router_v3_three_task_continuation.run import preserving_extend
        s=self.sentinel(16);s.last_step=420;s.state=['PROTECTED']*16
        s.state[9]=s.state[14]='FORMING';s.state[0]='DEGRADED';s.level[0]=2
        x=torch.arange(240*3).reshape(240,3);y=torch.arange(24).repeat_interleave(10)[:,None]
        old_x,old_y=x[:160],y[:160]
        s.bind(old_x,old_y);e=SimpleNamespace(scheduler=s,mapping=s.mapping)
        with patch.object(fewrel,'GroupSentinel',PartialSentinel):
            preserving_extend(fewrel.extend_bank,e,old_x,old_y,x,y)
        self.assertEqual(complete_old(e.scheduler,set(range(16))),[9,14])
        self.assertIsInstance(e.scheduler,PartialSentinel);self.assertEqual(e.scheduler.state[16:],['FORMING']*8)
        self.assertEqual(e.scheduler.state[0],'DEGRADED');self.assertEqual(e.scheduler.last_step,420)

if __name__=='__main__':unittest.main(verbosity=2)
