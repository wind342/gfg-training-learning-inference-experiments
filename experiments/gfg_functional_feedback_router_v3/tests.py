import copy
import unittest
import torch
from transformers import BertConfig
from experiments.gfg_governor_fewrel_v1.model import RelationModel
from experiments.gfg_functional_feedback_router_v1.runtime import accumulate as old_accumulate
from .dose import accumulate,weights

class Tiny(torch.nn.Module):
    def __init__(self):super().__init__();self.w=torch.nn.Linear(5,3,dtype=torch.float64)
    def forward(self,x,y):
        z=self.w(x);return z,torch.nn.functional.cross_entropy(z,y[:,-1])

class DoseTests(unittest.TestCase):
    def test_expected_weights(self):
        for n,v in [(0,0),(4,.2),(6,6/22),(16,.5)]:
            b=[list(range(16))]+([list(range(16,16+n))] if n else [])
            w=weights(b);self.assertEqual(w[-1] if n else 0,v);self.assertEqual(sum(w),1.)
    def test_empty_rejected(self):
        for b in ([],[[]],[list(range(16)),[]]):
            with self.assertRaises(ValueError):weights(b)
    def compare(self,n,steps=1):
        torch.manual_seed(4);a=Tiny();b=copy.deepcopy(a);x=torch.randn(16+n,5,dtype=torch.float64);y=torch.randint(0,3,(16+n,1))
        oa=torch.optim.AdamW(a.parameters(),lr=5e-6);ob=torch.optim.AdamW(b.parameters(),lr=5e-6)
        batches=[list(range(16))]+([list(range(16,16+n))] if n else [])
        for t in range(steps):
            accumulate(a,oa,x,y,batches,7+t)
            ob.zero_grad(set_to_none=True);b(x,y)[1].backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
            for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-12,atol=1e-12)
            oa.step();ob.step()
            for pa,pb in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(pa,pb,rtol=1e-12,atol=1e-12)
                for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(oa.state[pa][key],ob.state[pb][key],rtol=1e-12,atol=1e-12)
    def test_native16(self):self.compare(0)
    def test_old4(self):self.compare(4)
    def test_old6(self):self.compare(6)
    def test_old16(self):self.compare(16)
    def test_optimizer_moments_continuous(self):self.compare(4,3)
    def test_equal_batches_exact_v2_regression(self):
        torch.manual_seed(1);a=Tiny();b=copy.deepcopy(a);x=torch.randn(32,5,dtype=torch.float64);y=torch.randint(0,3,(32,1));batches=[list(range(16)),list(range(16,32))]
        accumulate(a,torch.optim.AdamW(a.parameters()),x,y,batches,42)
        old_accumulate(b,torch.optim.AdamW(b.parameters()),x,y,batches,42)
        for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(pa.grad,pb.grad))
    def test_native_exact_v2_regression(self):
        torch.manual_seed(2);a=Tiny();b=copy.deepcopy(a);x=torch.randn(16,5,dtype=torch.float64);y=torch.randint(0,3,(16,1));batches=[list(range(16))]
        accumulate(a,torch.optim.AdamW(a.parameters()),x,y,batches,42)
        old_accumulate(b,torch.optim.AdamW(b.parameters()),x,y,batches,42)
        for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(pa.grad,pb.grad))
    def test_real_bert_concat_equivalence_no_dropout(self):
        torch.manual_seed(3)
        c=BertConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,max_position_embeddings=16,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        a=RelationModel(c);a.seen=3;b=copy.deepcopy(a)
        x=torch.cat((torch.tensor([[1,2]]*20),torch.randint(1,32,(20,5))),1);y=torch.randint(0,3,(20,1))
        accumulate(a,torch.optim.AdamW(a.parameters()),x,y,[list(range(16)),list(range(16,20))],5)
        b.zero_grad(set_to_none=True);b(x,y)[1].backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb);torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-5,atol=2e-6)

if __name__=='__main__':torch.set_num_threads(2);unittest.main(verbosity=2)
