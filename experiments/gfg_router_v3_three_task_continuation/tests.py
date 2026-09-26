import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel
from experiments.gfg_governor_fewrel_v1.run import extend_bank
from experiments.gfg_functional_feedback_router_v2.policy import Router
from experiments.gfg_functional_feedback_router_v1.tests import memory
from .gate import validate_summary
from .run import preserving_extend
from .check import longest,coverage

class HarnessTests(unittest.TestCase):
    def fixture(self):
        return ({'status':'PASS','router_v3':{'metrics':[{'offset':60}]}},
            {'status':'COMPLETED','steps':60},[{'step':211+i,'task':2} for i in range(60)])
    def test_v3_schema_accepts(self):validate_summary(*self.fixture())
    def test_reject_failed_smoke(self):
        c,r,rows=self.fixture();c['status']='FAIL'
        with self.assertRaises(ValueError):validate_summary(c,r,rows)
    def test_reject_legacy_schema(self):
        c,r,rows=self.fixture();c['router']=c.pop('router_v3')
        with self.assertRaises(ValueError):validate_summary(c,r,rows)
    def test_reject_incomplete_records(self):
        c,r,rows=self.fixture()
        with self.assertRaises(ValueError):validate_summary(c,r,rows[:-1])
    def test_reject_wrong_step_identity(self):
        c,r,rows=self.fixture();rows[-1]['step']=999
        with self.assertRaises(ValueError):validate_summary(c,r,rows)
    def test_real_append_preserves_all_old_states(self):
        x=torch.arange(240*3).reshape(240,3);y=torch.arange(24).repeat_interleave(10)[:,None]
        s=GroupSentinel(GroupMap(y[:160,-1].numpy()));s.state[0]='DEGRADED';s.level[0]=2
        s.state[4]='PROTECTED';s.anchor[:]=np.arange(16)*.1;s.q10[:]=np.arange(16)*.2
        s.capabilities[:]=.7;s.last_observed[:]=419;s.last_step=420;s.bind(x[:160],y[:160])
        e=SimpleNamespace(scheduler=s,mapping=s.mapping)
        audit=preserving_extend(extend_bank,e,x[:160],y[:160],x,y)
        self.assertEqual(audit['old_labels'],list(range(16)));self.assertEqual(e.scheduler.state[0],'DEGRADED')
        self.assertEqual(e.scheduler.state[16:],['FORMING']*8);self.assertEqual(e.scheduler.last_step,420)
    def test_task3_one_risk_does_not_fill_all_old_groups(self):
        r=Router();m=memory(16)
        i=r.choose(421,list(range(24)),['DEGRADED']+['PROTECTED']*15+['FORMING']*8,[2]+[0]*23,set(range(16)),m)
        self.assertEqual(i['feedback_batch_size'],4);self.assertEqual(len(i['selected_anchor_groups']),1)
    def test_risk_rotation_and_priority_are_not_rewritten(self):
        r=Router();m=memory(16);seen=set()
        for t in range(421,431):
            i=r.choose(t,list(range(24)),['DEGRADED']*10+['PROTECTED']*6+['FORMING']*8,[2]*10+[0]*14,set(range(16)),m)
            seen.update(set(i['selected_feedback_groups'])-set(i['selected_anchor_groups']))
            self.assertEqual(i['feedback_batch_size'],16)
        self.assertEqual(seen,set(range(10)))
    def test_longest_counts_consecutive_not_total(self):
        self.assertEqual(longest([True,True,False,True,True,True]),3)
        self.assertEqual(longest([]),0)
    def test_reports_forming_old_omission_without_changing_policy(self):
        r=Router();m=memory(16);rows=[]
        for t in range(421,426):
            states=['DEGRADED']+['PROTECTED']*14+['FORMING']*9;levels=[2]+[0]*23
            i=r.choose(t,list(range(24)),states,levels,set(range(16)),m)
            n=i['feedback_batch_size'];i.update(allocation_receiving_history=dict(labels=list(range(24)),states=states,levels=levels),
                current_loss=1.,feedback_loss=1.,old_loss_weight=n/(16+n),current_loss_weight=16/(16+n),weighted_training_loss=1.,
                instrument_called=False,gate_reason='NOT_MEASURED',alpha=1.,seconds=.1)
            rows.append(dict(task=3,new_ids=list(range(16)),info=i))
        report=coverage(rows,True)
        self.assertEqual(report['per_relation'][15]['forming_steps'],5)
        self.assertEqual(report['per_relation'][15]['replay_occurrences'],0)
        self.assertEqual(report['per_relation'][0]['risk_unserved_steps'],0)

if __name__=='__main__':torch.set_num_threads(2);unittest.main(verbosity=2)
