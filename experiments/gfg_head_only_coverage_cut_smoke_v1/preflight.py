import io
import unittest
from unittest.mock import patch
from .core import *
from experiments.gfg_head_only_exact_cut_smoke_v1.preflight import CutTests,LRChecks

class CoverageTests(unittest.TestCase):
    def fixture(self):
        b={'weight':torch.tensor([[1.,0.],[0.,0.]]),'bias':torch.zeros(2)}
        n={'weight':torch.tensor([[-1.,0.],[0.,0.]]),'bias':torch.zeros(2)}
        return b,n,torch.tensor([[1.,0.]]),torch.tensor([0])
    def test_single_constraint_formula(self):
        b,n,h,y=self.fixture();o,c,r=tight_cut(b,n,h,y,1)
        cert=r['passes'][0]['certificates'][0]
        self.assertAlmostEqual(cert['coefficient'],(1+EPS)/4)
        self.assertFalse(r['skip']);self.assertGreater(float(margins(affine(c,h),y)[0]),0)
        self.assertLess(abs(float(margins(affine(c,h),y)[0])-EPS),2e-7)
        self.assertTrue(equal(b,{'weight':torch.tensor([[1.,0.],[0.,0.]]),'bias':torch.zeros(2)}))
    def test_rechecks_redundant_pair(self):
        b,n,h,y=self.fixture();o,c,r=tight_cut(b,n,h.repeat(2,1),y.repeat(2),2)
        cert=r['passes'][0]['certificates'];self.assertEqual(len(cert),2)
        self.assertGreater(cert[1]['margin_before'],0);self.assertEqual(cert[1]['coefficient'],0.)
    def test_safe_native_identity(self):
        b,n,h,y=self.fixture();o,c,r=tight_cut(b,b,h,y,1)
        self.assertIs(o,b);self.assertFalse(r['passes']);self.assertFalse(r['skip'])
    def test_tie(self):
        b,n,h,y=self.fixture();n['weight'].zero_();o,c,r=tight_cut(b,n,h,y,1)
        self.assertEqual(r['native_violating'],[0]);self.assertFalse(r['skip'])
    def test_two_pass_skip(self):
        b,n,h,y=self.fixture();f=affine;calls=0
        def stuck(state,features,seen=16):
            nonlocal calls
            calls+=1
            return f(b if calls==1 else n,features,seen)
        with patch(__package__+'.core.affine',side_effect=stuck):o,c,r=tight_cut(b,n,h,y,1)
        self.assertEqual(len(r['passes']),2);self.assertTrue(r['skip']);self.assertTrue(equal(o,b))
    def test_current_delegation(self):
        b,n,h,y=self.fixture()
        o,c,r=cut(b,n,h,y,1,'current');oo,cc,rr=current_cut(b,n,h,y,1)
        self.assertTrue(equal(o,oo) and equal(c,cc));self.assertEqual(r['passes'],rr['passes'])
    def test_gradient_bias(self):
        b,n,h,y=self.fixture();w=b['weight'].clone().requires_grad_();z=b['bias'].clone().requires_grad_()
        f=nn.functional.linear(h,w,z);gw,gb=torch.autograd.grad(f[0,0]-f[0,1],(w,z))
        self.assertTrue(torch.equal(gw,torch.cat([h,-h])));self.assertTrue(torch.equal(gb,torch.tensor([1.,-1.])))

def main():
    unchanged();assert not ROOT.exists();setup()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (LRChecks,CutTests,CoverageTests))
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    assert r.wasSuccessful()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    d=np.load(DATA/'data.npz');a=data_audit(d);ids=bank_ids(d,140)[:1120];p=bank_ids(d,10)
    hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in d['x']]
    tr={hashes[i] for i in np.flatnonzero(d['split']==0)};_,test,_=scope(d,80);te={hashes[i] for i in test};bh={hashes[i] for i in ids}
    assert len(bh)==1120 and not (bh&tr or bh&te)
    assert np.array_equal(p,scope(d,16)[0]);assert set(p[:80]).issubset(set(ids))
    old=json.loads((PARENT_ROOT/'PREFLIGHT.json').read_text())
    paths=[PARENT_ROOT/'PREFLIGHT.json',PARENT_ROOT/'CHECKS.json',PARENT_ROOT/'EVALUATION.json',
        PARENT_ROOT/'head_only_exact_cut'/'TRAIN_RESULT.json',PARENT_ROOT/'head_only_exact_cut'/'AUDIT.pt']
    checks=json.loads(paths[1].read_text());assert checks['status']=='PASS'
    result=json.loads(paths[3].read_text());assert result['status']=='COMPLETED' and sha(paths[4])==result['audit_sha256']
    ROOT.mkdir()
    write(ROOT/'PREFLIGHT.json',dict(status='PASS',tests=r.testsRun,test_log=stream.getvalue(),source_hashes=sources(),
        baseline_files={str(p):sha(p) for p in paths},data_audit=a,coverage_audit=dict(total=1120,unique_encoded=1120,overlap_train=0,overlap_primary_test=0),
        **{k:old[k] for k in ('head_initial_sha256','head_optimizer_initial_sha256','backbone_initial_sha256')}))
    print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()
