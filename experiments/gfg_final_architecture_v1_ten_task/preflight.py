import io
import inspect
import unittest
from .core import *
from .check import independent
from experiments.gfg_head_only_hildreth_roundoff_smoke_v1.preflight import HildrethTests,LRChecks,CutTests

class ContinuationTests(unittest.TestCase):
    def test_imported_projection_identity(self):
        from experiments.gfg_head_only_hildreth_roundoff_smoke_v1.core import hildreth as original
        self.assertIs(hildreth,original)
    def test_native_positive_below_epsilon_remains_unchanged(self):
        b={'weight':torch.tensor([[1.],[0.]]),'bias':torch.zeros(2)}
        n={'weight':torch.tensor([[5e-7],[0.]]),'bias':torch.zeros(2)}
        e,p,r=hildreth(b,n,torch.ones(1,1),torch.tensor([0]),1,2)
        self.assertIs(e,n);self.assertEqual(r['action'],'NATIVE')
    def test_dynamic_bank_and_seen_matches_independent(self):
        for seen in (16,24,80):
            b={'weight':torch.zeros(80,3),'bias':torch.full((80,),-2.)};b['bias'][0]=1.
            n=copy.deepcopy(b);n['bias'][1]=2.
            h=torch.tensor([[1.,0.,1.],[0.,1.,1.],[1.,1.,0.]])
            y=torch.zeros(3,dtype=torch.long)
            e,p,r=hildreth(b,n,h,y,2,seen);ee,pp,rr=independent(b,n,h,y,2,seen)
            self.assertTrue(equal(e,ee) and equal(p,pp))
            for k,v in rr.items():self.assertEqual(r[k],v)
            self.assertTrue(torch.equal(p['weight'][seen:],n['weight'][seen:]))
    def test_class_expansion_is_not_an_update(self):
        head=nn.Linear(1,24);head.weight.data.zero_();head.bias.data.fill_(-2.)
        head.bias.data[0]=1.;head.bias.data[16]=2.;before=head_hash(head)
        r=transition(head,torch.ones(2,1),torch.zeros(2,dtype=torch.long),2,3)
        self.assertEqual(r['expansion_correct_to_wrong'],[0,1]);self.assertEqual(head_hash(head),before)
    def test_continuous_optimizer_counter(self):
        head=nn.Linear(2,80);shadow=copy.deepcopy(head);opt=optimizer(shadow.parameters());set_lr(opt,LR)
        for j in range(3):
            for p in head.parameters():p.grad=torch.ones_like(p)
            propose(head,shadow,opt)
            self.assertEqual(opt_counter(opt),j+1)
            if j!=1:commit(head,shadow)
        self.assertEqual(opt_counter(opt),3)
    def test_bank_isolation_and_sizes(self):
        d=np.load(DATA/'data.npz');a=isolation(d)
        self.assertEqual(a['bank_sizes']['3'],2237);self.assertEqual(a['bank_sizes']['10'],10072)
        for t in range(2,11):
            ids,n=task_bank(d,t);self.assertEqual(len(ids),n+80)
            self.assertTrue(np.all(d['y'][ids[:n]]<8*(t-1)))
            self.assertTrue(np.all((d['y'][ids[n:]]>=8*(t-1))&(d['y'][ids[n:]]<8*t)))

def main():
    unchanged();assert not ROOT.exists();setup()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (LRChecks,CutTests,HildrethTests,ContinuationTests))
    log=io.StringIO();r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);print(log.getvalue(),flush=True)
    assert r.wasSuccessful()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    p=json.loads((PARENT_ROOT/'PREFLIGHT.json').read_text());assert json.loads((PARENT_ROOT/'CHECKS.json').read_text())['status']=='PASS'
    deps=[COVER_ROOT/'CACHE.pt',PARENT_ROOT/'PREFLIGHT.json',PARENT_ROOT/'CHECKS.json',PARENT_ROOT/'RESULTS.json',
        PARENT_ROOT/'140_controls_hildreth_roundoff/AUDIT.pt',PARENT_ROOT/'140_controls_hildreth_roundoff/TRAIN_RESULT.json']
    result=json.loads(deps[-1].read_text());assert sha(deps[-2])==result['audit_sha256']
    old=torch.load(COVER_ROOT/'CACHE.pt',map_location='cpu',weights_only=False)
    d=np.load(DATA/'data.npz');assert task_bank(d,2)[0].tolist()==old['indices']
    # Stored-state compatibility check, no training and no test inference.
    audit=torch.load(deps[-2],map_location='cpu',weights_only=False);i=4
    before={k:v.cuda() for k,v in audit['heads'][i].items()};native={k:v.cuda() for k,v in audit['natives'][i].items()}
    h=old['features'].cuda();y=torch.from_numpy(d['y'][old['indices']]).cuda()
    e,q,rp=hildreth(before,native,h,y,1120,16)
    assert equal(cpu_state(e),audit['heads'][i+1]) and equal(cpu_state(q),audit['projected'][i])
    ee,qq,rr=independent(before,native,h,y,1120,16)
    assert equal(e,ee) and equal(q,qq)
    for k,v in rr.items():assert rp[k]==v,k
    ROOT.mkdir()
    write(ROOT/'PREFLIGHT.json',dict(status='PASS',base=BASE,tests=r.testsRun,test_log=log.getvalue(),
        isolation=isolation(d),source_hashes=sources(),dependencies={str(v):sha(v) for v in deps},
        historical_step5_exact_replay='PASS',projection_import_identity=True,
        **{k:p[k] for k in ('head_initial_sha256','head_optimizer_initial_sha256','backbone_initial_sha256')}))
    print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()
