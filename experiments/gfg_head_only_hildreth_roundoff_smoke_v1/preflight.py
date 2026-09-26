import io
import unittest
from unittest.mock import patch
from .core import *
from experiments.gfg_head_only_exact_cut_smoke_v1.preflight import LRChecks,CutTests

class HildrethTests(unittest.TestCase):
    def fixture(self):
        b={'weight':torch.tensor([[1.,0.],[0.,0.]]),'bias':torch.zeros(2)}
        n={'weight':torch.tensor([[-1.,0.],[0.,0.]]),'bias':torch.zeros(2)}
        return b,n,torch.tensor([[1.,0.]]),torch.tensor([0])
    def test_analytic_two_constraint_solution_and_release(self):
        H=np.array([[1.,0.],[1.,1.]]);norm=2*np.sum(H*H,axis=1)
        native=np.array([[-.5,-1.],[.5,1.]]);P=native.copy();lam=np.zeros(2);decreased=False
        for _ in range(30):
            for i in range(2):
                prev=lam[i];lam[i]=coordinate(P,H,norm,i,0,1,prev);decreased|=lam[i]<prev
        expected=native.copy();a=(3+EPS)/4;expected[0]+=a*H[1];expected[1]-=a*H[1]
        self.assertTrue(decreased);self.assertEqual(lam[0],0.);self.assertTrue(np.allclose(P,expected,atol=1e-12,rtol=0))
        self.assertTrue(np.allclose(P,native+np.stack([lam@H,-lam@H]),atol=1e-12,rtol=0))
    def test_single_constraint_minimum_correction(self):
        H=np.array([[2.,1.]]);P=np.array([[-1.,0.],[0.,0.]]);original=P.copy();n=2*np.sum(H*H,axis=1)
        lam=coordinate(P,H,n,0,0,1,0.)
        self.assertAlmostEqual(lam,(2+EPS)/10)
        self.assertAlmostEqual(float(H[0]@(P[0]-P[1])),EPS,places=14)
        self.assertTrue(np.allclose(P-original,np.stack([lam*H[0],-lam*H[0]])))
    def test_sparse_gradient_includes_bias(self):
        b,n,h,y=self.fixture();w=b['weight'].clone().requires_grad_();bi=b['bias'].clone().requires_grad_()
        z=nn.functional.linear(h,w,bi);gw,gb=torch.autograd.grad(z[0,0]-z[0,1],(w,bi))
        self.assertTrue(torch.equal(gw,torch.cat([h,-h])));self.assertTrue(torch.equal(gb,torch.tensor([1.,-1.])))
    def test_native_safe_exact_identity(self):
        b,n,h,y=self.fixture();e,p,r=hildreth(b,b,h,y,1,2)
        self.assertIs(e,b);self.assertEqual(r['sweep_count'],0);self.assertFalse(r['skip'])
    def test_full_pipeline_and_immutability(self):
        b,n,h,y=self.fixture();bb=copy.deepcopy(b);nnn=copy.deepcopy(n);rr=rng()
        e,p,r=hildreth(b,n,h,y,1,2)
        self.assertTrue(equal(b,bb) and equal(n,nnn) and equal(rr,rng()));self.assertLessEqual(r['sweep_count'],100)
        if not r['skip']:self.assertGreaterEqual(float(margins(affine(e,h,2),y)[0]),EPS)
        else:self.assertTrue(equal(e,b))
    def test_epsilon_and_tie_pair_discovery(self):
        y=torch.tensor([0]);eligible=torch.tensor([True])
        self.assertEqual(failing_pairs(torch.tensor([[5e-7,0.]]),y,eligible),[(0,1)])
        self.assertEqual(failing_pairs(torch.tensor([[0.,0.]]),y,eligible),[(0,1)])
        self.assertEqual(failing_pairs(torch.tensor([[2e-6,0.]]),y,eligible),[])
        self.assertEqual(failing_pairs(torch.tensor([[-1.,0.]]),y,torch.tensor([False])),[])
    def test_rounding_cannot_pass_via_tolerance(self):
        v=np.float32(EPS)
        self.assertLess(float(v),EPS)
        self.assertEqual(failing_pairs(torch.tensor([[float(v),0.]]),torch.tensor([0]),torch.tensor([True])),[(0,1)])
    def test_limit_and_skip(self):
        b,n,h,y=self.fixture()
        with patch(__package__+'.core.coordinate',side_effect=lambda P,H,n,i,t,c,l,padding=0.:l):e,p,r=hildreth(b,n,h,y,1,2)
        self.assertEqual(r['sweep_count'],100);self.assertTrue(r['skip']);self.assertTrue(equal(e,b))
    def test_dynamic_active_addition(self):
        b={'weight':torch.tensor([[1.],[0.],[0.]]),'bias':torch.zeros(3)}
        n={'weight':torch.tensor([[-1.],[0.],[-2.]]),'bias':torch.zeros(3)}
        h=torch.ones(1,1);y=torch.tensor([0]);call=0;real=affine
        def fake(state,features,seen=16):
            nonlocal call
            call+=1
            if call==3:return torch.tensor([[0.,-1.,1.]])
            return real(state,features,seen)
        with patch(__package__+'.core.affine',side_effect=fake):e,p,r=hildreth(b,n,h,y,1,3)
        self.assertEqual(r['initial_active'],[[0,1]])
        self.assertEqual(r['sweeps'][0]['added_pairs'],[[0,2]])
        self.assertIn([0,2],r['final_active'])

    def test_padding_formula_and_scaling(self):
        b,n,h,y=self.fixture();H=np.array([[1.,0.,1.]])
        v=logit_roundoff_budgets(b,n,H,2);u=2.**-24;gamma=5*u/(1-5*u)
        self.assertTrue(np.array_equal(v,np.array([[gamma,0.]])))
        self.assertTrue(np.array_equal(logit_roundoff_budgets(b,n,2*H,2),2*v))
        doubled={k:2*x for k,x in n.items()}
        self.assertTrue(np.array_equal(logit_roundoff_budgets(b,doubled,H,2),2*v))
    def test_positive_padding_targets_interior_not_looser_check(self):
        H=np.array([[2.,1.]]);P=np.array([[-1.,0.],[0.,0.]]);n=2*np.sum(H*H,axis=1);pad=3e-4
        coordinate(P,H,n,0,0,1,0.,pad)
        self.assertAlmostEqual(float(H[0]@(P[0]-P[1])),EPS+pad,places=14)
        self.assertEqual(failing_pairs(torch.tensor([[5e-7,0.]]),torch.tensor([0]),torch.tensor([True])),[(0,1)])

def numerical_check(d):
    """Three predetermined historical numerical cases, no training or test inputs."""
    original=HILD_ROOT/'140_controls_hildreth'
    records=[json.loads(v) for v in (original/'STEPS.jsonl').read_text().splitlines()]
    failures=[j for j,r in enumerate(records) if r['report']['skip']]
    selected=[failures[0],failures[len(failures)//2],failures[-1]]
    a=torch.load(original/'AUDIT.pt',map_location='cpu',weights_only=False)
    cache=torch.load(COVER_ROOT/'CACHE.pt',map_location='cpu',weights_only=False)
    h=cache['features'].cuda();y=torch.from_numpy(d['y'][cache['indices']]).cuda()
    H=np.column_stack((cache['features'][:1120].numpy().astype(np.float64),np.ones(1120)))
    cases=[]
    for j in selected:
        b={k:v.cuda() for k,v in a['heads'][j].items()};native={k:v.cuda() for k,v in a['natives'][j].items()}
        rr=rng();e,p,r=hildreth(b,native,h,y)
        assert equal(rr,rng()) and equal(cpu_state(b),a['heads'][j]) and equal(cpu_state(native),a['natives'][j])
        # Reconstruct the FP64 iterate from dual coefficients for numerical audit.
        P=augmented(native);truth=y[:1120].cpu().numpy()
        for (i,c),lam in zip(r['final_active'],r['final_duals']):
            P[int(truth[i])]+=lam*H[i];P[c]-=lam*H[i]
        continuous=H@P.T;g=continuous[np.arange(1120),truth,None]-continuous
        z=affine(p,h).cpu().numpy()[:1120];actual=z[np.arange(1120),truth,None]-z
        budget=logit_roundoff_budgets(b,native,H);pair=budget[np.arange(1120),truth,None]+budget
        mask=np.zeros_like(g,dtype=bool);mask[np.array(r['eligible'])]=True;mask[np.arange(1120),truth]=False
        err=np.abs(g-actual.astype(np.float64));covered=bool(np.all(err[mask]<=pair[mask]))
        cases.append(dict(historical_step=j+1,old_hildreth_skip=True,new_sweeps=r['sweep_count'],new_skip=r['skip'],
            new_actual_min_margin=min(r['projected'][i] for i in r['eligible']),
            error_estimate_covers_observed_pairs=covered,max_observed_abs_pair_error=float(err[mask].max()),
            min_pair_padding=float(pair[mask].min()),max_pair_padding=float(pair[mask].max()),
            max_error_to_padding_ratio=float((err[mask]/pair[mask]).max()),
            source_before_sha256=tensor_hash(b),source_native_sha256=tensor_hash(native),
            roundoff_budget_sha256=r['roundoff_budget_sha256']))
    return dict(status='PASS' if all(c['error_estimate_covers_observed_pairs'] for c in cases) else 'FAIL',
        rule='u=2^-24; k=augmented_dimension+2; gamma=k*u/(1-k*u); per-logit gamma*abs(H)@max(abs(receiving),abs(native)).T; pair sums',
        selection='first, middle, last failed historical solver call; no test inputs, no training, no coefficient fitting',
        cases=cases,universal_error_bound_claim=False)

def main():
    unchanged();assert not ROOT.exists();setup()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (LRChecks,CutTests,HildrethTests))
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);print(stream.getvalue(),flush=True)
    assert r.wasSuccessful()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    d=np.load(DATA/'data.npz');audit=data_audit(d);ids=bank_ids(d,140)
    hs=[hashlib.sha256(x.tobytes()).hexdigest() for x in d['x']];bank={hs[i] for i in ids[:1120]}
    assert len(bank)==1120 and not bank&{hs[i] for i in np.flatnonzero(d['split']==0)}
    assert not bank&{hs[i] for i in scope(d,80)[1]}
    names=['PREFLIGHT.json','CHECKS.json','RESULTS.json','EVALUATION.json','TRAINING_COMPLETE.json','CACHE.pt',
           'A_140_current/TRAIN_RESULT.json','A_140_current/AUDIT.pt']
    paths=[COVER_ROOT/n for n in names];previous=json.loads(paths[0].read_text())
    assert json.loads(paths[1].read_text())['status']=='PASS'
    baseline=json.loads((COVER_ROOT/'A_140_current/TRAIN_RESULT.json').read_text())
    assert baseline['status']=='COMPLETED' and baseline['summary']['skips']==112
    assert sha(COVER_ROOT/'A_140_current/AUDIT.pt')==baseline['audit_sha256']
    assert sha(COVER_ROOT/'CACHE.pt')==json.loads((COVER_ROOT/'TRAINING_COMPLETE.json').read_text())['cache_sha256']
    cache=torch.load(COVER_ROOT/'CACHE.pt',map_location='cpu',weights_only=False)
    assert cache['indices']==ids.tolist() and tuple(cache['features'].shape)==(1200,1536)
    paths += [HILD_ROOT/n for n in ('PREFLIGHT.json','CHECKS.json','RESULTS.json',
        '140_controls_hildreth/TRAIN_RESULT.json','140_controls_hildreth/AUDIT.pt','140_controls_hildreth/STEPS.jsonl')]
    assert json.loads((HILD_ROOT/'CHECKS.json').read_text())['status']=='PASS'
    previous_run=json.loads((HILD_ROOT/'140_controls_hildreth/TRAIN_RESULT.json').read_text())
    assert sha(HILD_ROOT/'140_controls_hildreth/AUDIT.pt')==previous_run['audit_sha256']
    assert sha(HILD_ROOT/'140_controls_hildreth/STEPS.jsonl')==previous_run['log_sha256']
    ROOT.mkdir();numerics=numerical_check(d);write(ROOT/'NUMERICAL_CHECK.json',numerics)
    assert numerics['status']=='PASS'
    write(ROOT/'PREFLIGHT.json',dict(status='PASS',tests=r.testsRun,test_log=stream.getvalue(),source_hashes=sources(),
        numerical_check_sha256=sha(ROOT/'NUMERICAL_CHECK.json'),
        baseline_files={str(p):sha(p) for p in paths},data_audit=audit,bank_isolation='1120 unique encoded inputs disjoint from train and primary test',
        **{k:previous[k] for k in ('head_initial_sha256','head_optimizer_initial_sha256','backbone_initial_sha256')}))
    print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()
