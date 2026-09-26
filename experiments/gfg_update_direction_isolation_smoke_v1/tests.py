import copy
import unittest
import numpy as np
import torch
from .core import *
from experiments.gfg_controller_distributed_runtime_v2.ownership import Proposal

class Tests(unittest.TestCase):
    def test_conflict(self):
        d={'p':torch.tensor([-2.,3.],dtype=torch.float64)};g={'p':torch.tensor([1.,0.],dtype=torch.float64)}
        cut(d,g);self.assertLess(abs(dot(d,g)),3e-12)
    def test_nonconflict(self):
        d={'p':torch.tensor([2.,3.])};before=d['p'].clone()
        self.assertFalse(cut(d,{'p':torch.tensor([1.,0.])})['changed']);self.assertTrue(torch.equal(before,d['p']))
    def test_tensorwise_flat(self):
        torch.manual_seed(1);d={'a':torch.randn(2,3,dtype=torch.float64),'b':torch.randn(4,dtype=torch.float64)}
        g={n:-x.clone() for n,x in d.items()};flat=torch.cat([x.flatten() for x in d.values()]);gf=torch.cat([x.flatten() for x in g.values()])
        expected=flat-torch.dot(gf,flat)/(torch.dot(gf,gf)+EPS)*gf
        cut(d,g);torch.testing.assert_close(torch.cat([x.flatten() for x in d.values()]),expected)
    def test_order_reproducible(self):
        def run():
            d={'x':torch.tensor([-1.,-2.],dtype=torch.float64)}
            for g in ([1.,0.],[-1.,1.]):cut(d,{'x':torch.tensor(g,dtype=torch.float64)})
            return d['x']
        self.assertTrue(torch.equal(run(),run()))
        self.assertLess(float(run()[0]),0) # later projection may reintroduce earlier conflict
    def test_zero_gradient(self):
        d={'a':torch.ones(2)};self.assertFalse(cut(d,{'a':torch.zeros(2)})['changed'])
    def test_moments_and_commit(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.1)
        p.grad=torch.tensor([1.,-1.]);proposal=Proposal({'p':p});before=p.detach().clone();proposal.form({'p':p},opt)
        state=copy.deepcopy(opt.state_dict());candidate=proposal.candidate['p'].clone()
        cut(proposal.delta,{'p':torch.tensor([1.,0.])});expected=before+proposal.delta['p']
        commit_projected({'p':p},proposal.delta,1.)
        self.assertTrue(torch.equal(p,expected));self.assertFalse(torch.equal(p,candidate))
        for k,v in state['state'][0].items():self.assertTrue(torch.equal(v,opt.state_dict()['state'][0][k]))
    def test_no_threat_native_bitwise(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.01);p.grad=torch.ones(2)
        prop=Proposal({'p':p});prop.form({'p':p},opt);expected=prop.candidate['p'].clone();prop.commit({'p':p},1.)
        self.assertTrue(torch.equal(p,expected))
    def test_second_prediction_new_delta(self):
        d={'x':torch.tensor([-1.,2.],dtype=torch.float64)}
        predictor=lambda delta:delta['x'].clone()
        old=predictor(d);cut(d,{'x':torch.tensor([1.,0.],dtype=torch.float64)})
        new=predictor(d);self.assertFalse(torch.equal(old,new));self.assertGreater(new[0],-.000001)
    def test_schedule(self):self.assertEqual(SCHEDULE,tuple(1+10*i for i in range(21)))
    def test_all_protected_checked(self):
        s=GroupSentinel(GroupMap(np.repeat(np.arange(16),10)));s.state=['PROTECTED']*16;s.anchor[:]=1
        pred=np.ones((4,160))*2;pred[-1,10:20]=-1
        a,no,scores=choose(pred,s,[0,1]);self.assertEqual(a,.5);self.assertFalse(no)
    def test_erosion(self):
        s=GroupSentinel(GroupMap(np.repeat(np.arange(16),10)));s.anchor[:]=2
        self.assertEqual(threats(np.ones(160)*.9,s,[0]),[0])
    def test_no_safe(self):
        s=GroupSentinel(GroupMap(np.repeat(np.arange(16),10)))
        a,no,_=choose(-np.ones((4,160)),s,[0]);self.assertEqual(a,.125);self.assertTrue(no)
    def test_isolation_batches_no_replay(self):
        d=np.load(DATA/'data.npz');audit=data_audit(d);self.assertEqual(audit['status'],'PASS')
        orders=[order(d,2) for _ in ARMS]
        for ids in orders:self.assertTrue(np.array_equal(ids,orders[0]));self.assertTrue(np.all(d['split'][ids]==0));self.assertTrue(np.all((d['y'][ids]>=8)&(d['y'][ids]<16)))
        self.assertEqual([2021*100000+211+i for i in range(210)],[2021*100000+210+j for j in range(1,211)])
    def test_aggregate_no_new_gain(self):
        m=GroupMap(np.repeat(np.arange(16),10));a=aggregates(np.ones(160),m,np.ones(160))
        self.assertEqual(a['newly_formed_new_groups'],0)
    def test_real_bert_predictor_reads_replaced_delta(self):
        from transformers import BertConfig
        from experiments.gfg_governor_fewrel_v1.model import RelationModel,BertInstrument
        from experiments.gfg_controller_distributed_runtime_v2.identity import CanonicalParameterRegistry
        from experiments.gfg_controller_distributed_runtime_v2.ownership import Ownership
        torch.set_num_threads(2);torch.manual_seed(12)
        model=RelationModel(BertConfig(vocab_size=30,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,
            intermediate_size=16,max_position_embeddings=16,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
        model.seen=16;x=torch.tensor([[1,2,3,4,5,6]]);y=torch.tensor([[1]])
        opt=torch.optim.AdamW(model.parameters(),lr=.001);model(x,y)[1].backward()
        owner=Ownership(model,CanonicalParameterRegistry(model));owner.propose(opt);opt.zero_grad(set_to_none=True)
        instrument=BertInstrument(model,chunk=1);instrument.alphas=ALPHAS
        first=instrument.measure(owner,x,y)
        for v in owner.proposal.delta.values():v.mul_(.5)
        second=instrument.measure(owner,x,y)
        torch.testing.assert_close(first.predicted_minima[2],second.predicted_minima[3],rtol=1e-5,atol=1e-7)
        model.train();rng=torch.get_rng_state().clone();g,p=margin_gradient(model,x,y,[0])
        self.assertTrue(model.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertTrue(all(torch.isfinite(v).all() for v in g.values()))

if __name__=='__main__':unittest.main()
