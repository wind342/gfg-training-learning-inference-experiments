import copy
import unittest
from types import SimpleNamespace
import torch
from transformers import BertConfig
from experiments.gfg_controller_distributed_runtime_v2.identity import CanonicalParameterRegistry
from experiments.gfg_controller_distributed_runtime_v2.ownership import Ownership
from experiments.gfg_functional_update_governor_v3.groups import GroupMap
from experiments.gfg_functional_update_governor_v4.runtime import GovernedTraining
from .model import RelationModel,BertInstrument
from .run import extend_bank
from experiments.gfg_functional_update_governor_v3.scheduler import GroupSentinel


class AdapterTests(unittest.TestCase):
    def test_task_extension_preserves_history_and_rejects_changed_probes(self):
        x=torch.arange(24).reshape(4,6);y=torch.tensor([[0],[0],[1],[1]])
        mapping=GroupMap(y[:,-1].numpy());s=GroupSentinel(mapping);s.bind(x,y)
        s.state=['PROTECTED','DEGRADED'];s.anchor[:]=[.7,.8];s.level[:]=[0,2]
        s.capabilities[:]=[1.,.5];s.q10[:]=[.6,-.2];s.last_step=630;s.last_observed[:]=[629,630]
        engine=SimpleNamespace(scheduler=s)
        new_x=torch.cat([x,torch.ones(2,6,dtype=torch.long)])
        new_y=torch.cat([y,torch.full((2,1),2,dtype=torch.long)])
        extend_bank(engine,x,y,new_x,new_y)
        n=engine.scheduler
        self.assertEqual(n.state,['PROTECTED','DEGRADED','FORMING'])
        self.assertEqual(n.anchor.tolist(),[.7,.8,0.])
        self.assertEqual(n.last_step,630);self.assertIsNone(n.last_instrument)
        self.assertEqual(n.last_observed.tolist(),[629,630,-1])
        self.assertIs(engine.governor.scheduler,n)
        changed=new_x.clone();changed[0,0]+=1
        with self.assertRaisesRegex(ValueError,'old probe identities'):
            extend_bank(engine,new_x,new_y,changed,new_y)

    def model(self):
        torch.manual_seed(51)
        model=RelationModel(BertConfig(vocab_size=40,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
            intermediate_size=32,max_position_embeddings=16,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
        x=torch.tensor([[1,3,2,5,6,7,8,0],[2,4,2,3,4,5,6,7]])
        y=torch.tensor([[0],[2]])
        return model,x,y

    def test_affine_head_and_directional_jets_match_native(self):
        model,x,y=self.model();model.eval()
        owner=Ownership(model,CanonicalParameterRegistry(model));instrument=BertInstrument(model,chunk=1)
        torch.manual_seed(52)
        for n,p in model.named_parameters():owner.proposal.delta[n].copy_(torch.randn_like(p)*.0001)
        h,hp,hpp=instrument.jets(owner,x)
        self.assertTrue(torch.allclose(h,model.features(x),atol=2e-6,rtol=2e-5))
        original=copy.deepcopy(model.state_dict())
        output=instrument.measure(owner,x,y)
        for j,a in enumerate(output.alphas):
            with torch.no_grad():
                for n,p in model.named_parameters():p.copy_(original[n]+a*owner.proposal.delta[n])
            z=model(x)[0][:,-1];target=z.gather(1,y).squeeze(1)
            z=z.scatter(1,y,-torch.inf);actual=target-z.max(1).values
            self.assertTrue(torch.allclose(actual.double(),output.predicted_minima[j],atol=2e-5,rtol=2e-4))
        model.load_state_dict(original)

    def test_head_bias_only_direction_exact(self):
        model,x,y=self.model();model.eval();owner=Ownership(model,CanonicalParameterRegistry(model))
        for delta in owner.proposal.delta.values():delta.zero_()
        owner.proposal.delta['head.bias'][:8].copy_(torch.arange(8)*.01)
        instrument=BertInstrument(model);m=instrument.measure(owner,x,y)
        base=model(x)[0][:,-1]
        for j,a in enumerate(m.alphas):
            z=base+a*owner.proposal.delta['head.bias'][:8]
            actual=z.gather(1,y).squeeze(1)-z.scatter(1,y,-torch.inf).max(1).values
            self.assertTrue(torch.allclose(actual.double(),m.predicted_minima[j],atol=2e-6))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA')
    def test_real_cuda_full_then_micro(self):
        model,x,y=self.model();model.cuda();x=x.cuda();y=y.cuda()
        instrument=BertInstrument(model)
        engine=GovernedTraining(model,None,mapping=GroupMap(y[:,-1].cpu().numpy()),instrument=instrument)
        opt=torch.optim.AdamW(model.parameters(),lr=5e-6,fused=True)
        for t in (1,2):
            info=engine.update(opt,t,x,y,x,y,t,5e-6)
            self.assertEqual(info['runtime_version'],'governor_v4')


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
