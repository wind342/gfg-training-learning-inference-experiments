import io
import unittest
from unittest.mock import patch
from .core import *
from experiments.gfg_head_only_lr_smoke_v1.preflight import LRChecks

class CutTests(unittest.TestCase):
    def fixture(self):
        b={'weight':torch.tensor([[2.,0.],[1.,0.],[-1.,0.]]),'bias':torch.zeros(3)}
        n={k:v.clone() for k,v in b.items()};n['weight'][2,0]=3.
        return b,n,torch.tensor([[1.,2.]]),torch.tensor([0])
    def test_analytic_gradient(self):
        b,n,h,y=self.fixture();w=b['weight'].clone().requires_grad_();bias=b['bias'].clone().requires_grad_()
        z=nn.functional.linear(h,w,bias);g=torch.autograd.grad(z[0,0]-z[0,2],(w,bias))
        expected=torch.zeros_like(w);expected[0]=h[0];expected[2]=-h[0]
        self.assertTrue(torch.equal(g[0],expected));self.assertTrue(torch.equal(g[1],torch.tensor([1.,0.,-1.])))
    def test_competitor_switch_and_cut(self):
        b,n,h,y=self.fixture();out,c,r=exact_cut(b,n,h,y,1,3)
        self.assertEqual(r['native_violating'],[0]);self.assertFalse(r['skip']);self.assertTrue(r['first_pass_safe'])
        self.assertEqual(r['passes'][0]['certificates'][0]['competitor'],2)
        self.assertGreater(float(margins(affine(out,h,3),y)[0]),0)
    def test_zero_tie(self):
        b,n,h,y=self.fixture();n['weight'][2,0]=2.;out,c,r=exact_cut(b,n,h,y,1,3)
        self.assertEqual(r['native_violating'],[0]);self.assertFalse(r['skip'])
    def test_safe_native_identity(self):
        b,n,h,y=self.fixture();out,c,r=exact_cut(b,b,h,y,1,3)
        self.assertTrue(equal(out,b));self.assertEqual(r['passes'],[]);self.assertEqual(r['action'],'NATIVE')
    def test_two_pass_skip(self):
        b,n,h,y=self.fixture()
        with patch('experiments.gfg_head_only_exact_cut_smoke_v1.core.projection_pass',return_value=[]) as fn:
            out,c,r=exact_cut(b,n,h,y,1,3)
        self.assertEqual(fn.call_count,2);self.assertTrue(r['skip']);self.assertTrue(equal(out,b))
    def test_input_immutable_rng(self):
        b,n,h,y=self.fixture();bb=copy.deepcopy(b);nnn=copy.deepcopy(n);hh=h.clone();rr=rng()
        exact_cut(b,n,h,y,1,3)
        self.assertTrue(equal(b,bb) and equal(n,nnn) and torch.equal(h,hh) and equal(rr,rng()))
    def test_skip_moments_advance_once(self):
        head=nn.Linear(2,3).cuda();shadow=copy.deepcopy(head);opt=optimizer(shadow.parameters());set_lr(opt,LR)
        for p in head.parameters():p.grad=torch.ones_like(p)
        original=snapshot(head);propose(head,shadow,opt);mh=moments_hash(opt)
        commit(head,head)
        self.assertTrue(equal(original,snapshot(head)));self.assertEqual(mh,moments_hash(opt))
        self.assertTrue(all(float(v['step'])==1 for v in opt.state.values()))
        for p in head.parameters():p.grad=torch.ones_like(p)
        propose(head,shadow,opt);self.assertTrue(all(float(v['step'])==2 for v in opt.state.values()))

def main():
    unchanged();ROOT.mkdir(exist_ok=False);setup();assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(LRChecks),unittest.defaultTestLoader.loadTestsFromTestCase(CutTests)])
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    audit=data_audit(np.load(DATA/'data.npz'));old=json.loads((BASE_ROOT/'PREFLIGHT.json').read_text())
    paths=[LR_ROOT/'lr5e-5'/'TRAIN_RESULT.json',LR_ROOT/'EVALUATION.json',LR_ROOT/'lr5e-5'/'AUDIT.pt']
    rr=json.loads(paths[0].read_text());assert rr['status']=='COMPLETED' and sha(paths[2])==rr['audit_sha256']
    out=dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,test_log=stream.getvalue(),source_hashes=sources(),
        baseline_files={str(p):sha(p) for p in paths},data_audit=audit,
        head_initial_sha256=old['head_initial_sha256'],head_optimizer_initial_sha256=old['head_optimizer_initial_sha256'],
        backbone_initial_sha256=old['backbone_initial_sha256'])
    write(ROOT/'PREFLIGHT.json',out);assert r.wasSuccessful();print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()
