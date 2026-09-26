"""Policy, data-role, causality and native execution checks."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.gfg_functional_update_governor_v1.instrument import Measurement
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_governor_fewrel_v1.model import RelationModel,BertInstrument
from experiments.gfg_functional_update_governor_v4.runtime import GovernedTraining
from experiments.gfg_governor_fewrel_feedback_router_v1.run import validate_batches
from transformers import BertConfig
from .policy import Memory,Router,emergency_gate
from .runtime import accumulate,update

def memory(n=12):
    m=Memory();y=np.repeat(np.arange(n),10);s=np.zeros(len(y),int)
    for k in range(n):m.append(k,range(k*10,k*10+10),completed=set(range(n)),split=s,labels=y,forbidden=set())
    return m

def choose(states,levels=None,r=None,m=None,step=1,old=None):
    return (r or Router()).choose(step,list(range(len(states))),states,levels or [0]*len(states),set(range(len(states))) if old is None else old,m or memory(len(states)))

class PolicyTests(unittest.TestCase):
    def test_formation_does_not_read_memory(self):
        class Forbidden:
            def __getattribute__(self,name):raise AssertionError('memory accessed')
        r=Router().choose(1,[0],['PROTECTED'],[0],{0},Forbidden());self.assertEqual(r['feedback_batch_size'],0)
    def test_balance_level1(self):self.assertEqual(choose(['PROTECTED'],[1])['router_mode'],'BALANCE')
    def test_recovery(self):self.assertEqual(choose(['DEGRADED'],[2])['router_mode'],'RECOVERY')
    def test_degraded_before_anchor(self):self.assertEqual(choose(['PROTECTED','DEGRADED'],[0,2])['selected_feedback_groups'],[1,0])
    def test_rotation_over8(self):
        r=Router();m=memory();seen=set()
        for t in range(1,4):seen.update(choose(['DEGRADED']*12,[2]*12,r,m,t)['selected_feedback_groups'])
        self.assertEqual(seen,set(range(12)))
    def test_anchor_reserved(self):
        d=choose(['DEGRADED']*9+['PROTECTED'],[2]*9+[0]);self.assertEqual(len(d['selected_feedback_groups']),8);self.assertEqual(d['selected_anchor_groups'],[9])
    def test_underfull_no_duplicate_groups(self):
        d=choose(['DEGRADED','PROTECTED'],[2,0]);self.assertEqual(d['feedback_batch_size'],4)
    def test_no_current_or_future(self):
        d=choose(['PROTECTED','DEGRADED','DEGRADED'],[0,2,2],old={0});self.assertEqual(d['feedback_batch_size'],0)
    def test_cursor_cycle(self):
        m=memory(1);out=[m.take(0) for _ in range(6)];self.assertEqual(out,[[0,1],[2,3],[4,5],[6,7],[8,9],[0,1]])
    def test_release_immediate(self):
        r=Router();m=memory(1);choose(['DEGRADED'],[2],r,m,1)
        d=choose(['PROTECTED'],[0],r,m,2);self.assertTrue(d['release_event']);self.assertEqual(d['feedback_batch_size'],0)
    def test_choose_only_prestate(self):
        states=['DEGRADED'];d=choose(states,[2]);states[0]='PROTECTED';self.assertEqual(d['router_mode'],'RECOVERY')
    def test_append_only_identity(self):
        m=memory(1);m.groups[0]=tuple(range(1,11))
        with self.assertRaises(ValueError):m.take(0)
    def test_future_memory_rejected(self):
        with self.assertRaises(ValueError):Memory().append(1,range(10),completed={0},split=[0]*10,labels=[1]*10,forbidden=set())
    def test_probe_gradient_rejected(self):
        d=dict(split=np.array([0,1,2]),y=np.array([1,1,1]))
        with self.assertRaises(ValueError):validate_batches([1],[],d,{1},Memory(),[1],[2])
    def test_test_gradient_rejected(self):
        d=dict(split=np.array([0,1,2]),y=np.array([1,1,1]))
        with self.assertRaises(ValueError):validate_batches([2],[],d,{1},Memory(),[1],[2])
    def test_current_replay_rejected(self):
        m=memory(1);d=dict(split=np.zeros(10,int),y=np.zeros(10,int))
        with self.assertRaises(ValueError):validate_batches([3],[0],d,{0},m,[],[])
    def test_no_alpha2(self):
        m=Measurement((.125,.25,.5,1.,2.),torch.ones(10),torch.ones(5,10),{})
        with self.assertRaises(ValueError):emergency_gate(m,GroupMap([0]*10))
    def test_native_safe(self):
        m=Measurement((.125,.25,.5,1.),torch.ones(10),torch.ones(4,10),{})
        self.assertEqual(emergency_gate(m,GroupMap([0]*10))['alpha'],1.)
    def test_emergency_shrink_closest(self):
        p=torch.ones(4,10);p[3,:2]=-1
        d=emergency_gate(Measurement((.125,.25,.5,1.),torch.ones(10),p,{}),GroupMap([0]*10))
        self.assertEqual((d['alpha'],d['reason']),(.5,'EMERGENCY_SHRINK'))
    def test_no_safe_alpha(self):
        p=-torch.ones(4,10);d=emergency_gate(Measurement((.125,.25,.5,1.),torch.ones(10),p,{}),GroupMap([0]*10))
        self.assertEqual((d['alpha'],d['reason']),(1.,'NO_SAFE_ALPHA'))
    def test_no_protected_no_formation_optimization(self):
        p=torch.ones(4,10);p[3]=-1
        d=emergency_gate(Measurement((.125,.25,.5,1.),-torch.ones(10),p,{}),GroupMap([0]*10))
        self.assertEqual(d['alpha'],1.)
    def test_nonfinite_failclosed(self):
        p=torch.ones(4,10);p[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):emergency_gate(Measurement((.125,.25,.5,1.),torch.ones(10),p,{}),GroupMap([0]*10))

class ExecutionTests(unittest.TestCase):
    def test_two_backward_mean_and_moments(self):
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Linear(3,2)
            def forward(self,x,y):
                z=self.w(x);return z,torch.nn.functional.cross_entropy(z,y[:,-1])
        torch.manual_seed(7);a=Tiny();b=copy.deepcopy(a);x=torch.randn(8,3);y=torch.randint(0,2,(8,1));oa=torch.optim.AdamW(a.parameters());ob=torch.optim.AdamW(b.parameters())
        for seed in (5,6):
            accumulate(a,oa,x,y,[list(range(4)),list(range(4,8))],seed)
            ob.zero_grad(set_to_none=True);(.5*b(x[:4],y[:4])[1]+.5*b(x[4:],y[4:])[1]).backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
            for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(pa.grad,pb.grad))
            oa.step();ob.step()
            for pa,pb in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(pa,pb))
                for key in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(oa.state[pa][key],ob.state[pb][key]))
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_native_full_then_post_history_and_audit_no_reselect(self):
        torch.manual_seed(4)
        model=RelationModel(BertConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,max_position_embeddings=16)).cuda()
        model.seen=2;x=torch.cat((torch.tensor([[1,2]]*10),torch.randint(1,32,(10,5))),1);y=torch.randint(0,2,(10,1));px=x.cuda();py=y.cuda()
        instrument=BertInstrument(model);instrument.alphas=(.125,.25,.5,1.)
        engine=GovernedTraining(model,None,mapping=GroupMap(y[:,-1].numpy()),instrument=instrument)
        opt=torch.optim.AdamW(model.parameters(),lr=5e-6,fused=True)
        r=Router();m=Memory()
        with patch('experiments.gfg_functional_feedback_router_v1.runtime.emergency_gate',wraps=emergency_gate) as gate:
            info=update(model,opt,engine,r,m,x,y,list(range(10)),px,py,1,set(),22)
            self.assertEqual(gate.call_count,1)
        self.assertTrue(info['instrument_called']);self.assertEqual(info['feedback_batch_size'],0)
        self.assertEqual(engine.scheduler.mapping.values(info['actual'])['capabilities'],info['history']['capabilities'])
        update(model,opt,engine,r,m,x,y,list(range(10)),px,py,2,set(),23)
        self.assertEqual(engine.scheduler.last_step,2)

if __name__=='__main__':torch.set_num_threads(2);unittest.main(verbosity=2)
