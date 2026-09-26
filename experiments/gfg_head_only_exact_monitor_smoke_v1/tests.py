import copy
import unittest
import torch
from transformers import BertConfig
from .core import *

class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2021)
        c=BertConfig(vocab_size=40,hidden_size=12,num_hidden_layers=1,num_attention_heads=3,intermediate_size=24,hidden_dropout_prob=.2,attention_probs_dropout_prob=.2)
        self.m=FrozenRelationModel(c).cuda();self.m.seen=16
        self.x=torch.tensor([[0,1,2,4,5,0],[1,2,3,6,7,8]],device='cuda');self.y=torch.tensor([1,9],device='cuda')
        full=optimizer(self.m.parameters());self.m(self.x,self.y[:,None])[1].backward();full.step();full.zero_grad(set_to_none=True)
        self.saved=copy.deepcopy(full.state_dict());freeze(self.m)
        self.shadow,self.opt=make_candidate(self.m,self.saved)
        with torch.no_grad():self.h=self.m.features(self.x)

    def gradient(self):
        self.m.train();self.m.head.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(self.m.head(self.h)[:,:16],self.y).backward()
        torch.nn.utils.clip_grad_norm_(self.m.head.parameters(),1.)

    def test_01_frozen(self):self.assertTrue(all(not p.requires_grad for p in self.m.bert.parameters()))
    def test_02_no_backbone_grad(self):
        self.gradient();self.assertTrue(all(p.grad is None for p in self.m.bert.parameters()))
    def test_03_head_grad(self):
        self.gradient();self.assertTrue(all(p.grad is not None for p in self.m.head.parameters()))
    def test_04_backbone_identity(self):
        before=tensor_hash(self.m.bert.state_dict());self.gradient();propose(self.m.head,self.shadow,self.opt);commit(self.m.head,self.shadow)
        self.assertEqual(before,tensor_hash(self.m.bert.state_dict()))
    def test_05_head_changes(self):
        before=head_hash(self.m.head);self.gradient();propose(self.m.head,self.shadow,self.opt);commit(self.m.head,self.shadow)
        self.assertNotEqual(before,head_hash(self.m.head))
    def test_06_cache_unchanged(self):
        before=self.h.clone();self.gradient();propose(self.m.head,self.shadow,self.opt);logits(self.shadow,self.h)
        self.assertTrue(torch.equal(before,self.h))
    def test_07_exact_candidate(self):
        self.gradient();propose(self.m.head,self.shadow,self.opt)
        actual=logits(self.shadow,self.h);temporary=copy.deepcopy(self.m.head);temporary.load_state_dict(self.shadow.state_dict())
        self.assertTrue(torch.equal(actual,temporary(self.h)[:,:16]))
    def test_08_readonly_parameters(self):
        before=head_hash(self.m.head);logits(self.m.head,self.h);logits(self.shadow,self.h)
        self.assertEqual(before,head_hash(self.m.head))
    def test_09_readonly_moments(self):
        before=moments_hash(self.opt);logits(self.shadow,self.h);self.assertEqual(before,moments_hash(self.opt))
    def test_10_rng(self):
        before=rng();logits(self.m.head,self.h);logits(self.shadow,self.h);self.assertTrue(equal(before,rng()))
    def test_11_migration(self):
        ids=self.saved['param_groups'][0]['params'];names=[n for n,p in self.m.named_parameters()]
        for p,n in zip(self.shadow.parameters(),('head.weight','head.bias')):
            self.assertTrue(equal(self.opt.state[p],self.saved['state'][ids[names.index(n)]]))
    def test_12_native_equivalence(self):
        direct=copy.deepcopy(self.m.head);o=optimizer(direct.parameters());o.load_state_dict(copy.deepcopy(self.opt.state_dict()))
        self.gradient()
        for p,q in zip(self.m.head.parameters(),direct.parameters()):q.grad=p.grad.clone()
        o.step();propose(self.m.head,self.shadow,self.opt);commit(self.m.head,self.shadow)
        self.assertTrue(equal(direct.state_dict(),self.m.head.state_dict()));self.assertTrue(equal(o.state_dict(),self.opt.state_dict()))
    def test_13_eval_persistent(self):
        self.m.train();self.assertTrue(all(not m.training for m in self.m.bert.modules()))
    def test_14_safety(self):
        b=numbers(np.ones(160));c=numbers(np.ones(160));c['q10'][0]=.4;b['q10'][1]=.4;c['q10'][1]=.3
        v=classify(b,c,[1.]*16,[0,1]);self.assertEqual(v['already_unsafe_before_update'],[1]);self.assertEqual(v['newly_threatened_by_this_update'],[0])
    def test_15_two_arms(self):
        other=copy.deepcopy(self.m);shadow=copy.deepcopy(self.shadow);opt=optimizer(shadow.parameters());opt.load_state_dict(copy.deepcopy(self.opt.state_dict()))
        for step in range(3):
            for m,s,o,monitor in ((self.m,self.shadow,self.opt,False),(other,shadow,opt,True)):
                m.train();m.head.zero_grad(set_to_none=True)
                with torch.no_grad():h=m.features(self.x)
                torch.nn.functional.cross_entropy(m.head(h)[:,:16],self.y).backward();torch.nn.utils.clip_grad_norm_(m.head.parameters(),1.)
                propose(m.head,s,o)
                if monitor:logits(s,self.h);logits(m.head,self.h)
                commit(m.head,s)
        self.assertTrue(equal(self.m.state_dict(),other.state_dict()));self.assertTrue(equal(self.opt.state_dict(),opt.state_dict()))

if __name__=='__main__':unittest.main()
