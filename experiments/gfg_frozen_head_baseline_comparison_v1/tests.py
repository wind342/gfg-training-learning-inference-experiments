"""Small numerical/unit checks; no FewRel outcome-based parameter selection."""
from .core import *
from .statistics import cv_ridge,ridge_solve

def tests():
    torch.manual_seed(17);np.random.seed(17)
    # GEM <=9-variable solver: feasibility, stationarity and complementarity.
    for n in range(1,10):
        R=np.random.randn(n,15);R/=np.linalg.norm(R,axis=1)[:,None];g=np.random.randn(15)
        P=R@R.T;q=R@g;lam=dual_qp(P,q);out=g+lam@R
        assert np.min(R@out)>-1e-8 and np.min(lam)>=0 and np.max(np.abs(lam*(R@out)))<1e-7
    g=torch.tensor([-1.,2.],device='cuda');ref=[torch.tensor([1.,0.],device='cuda')]
    ag,r=project_gradient(g,ref,'agem');assert torch.equal(ag,torch.tensor([0.,2.],device='cuda'))
    # ER mean is equal-weight concat, not 50/50 when sizes differ.
    h=nn.Linear(5,4).cuda();X=torch.randn(20,5,device='cuda');y=torch.arange(20,device='cuda')%4
    g1,_=ce_grad(h,X,y,4);h.zero_grad()
    (16/20*nn.functional.cross_entropy(h(X[:16]),y[:16])+4/20*nn.functional.cross_entropy(h(X[16:]),y[16:])).backward()
    assert torch.allclose(g1,flat_grad(h),atol=1e-7,rtol=1e-5)
    # Both primal and dual CV paths match explicit regularized solutions.
    for dim in (5,40):
        H=torch.randn(30,dim,device='cuda',dtype=torch.float64);Y=torch.randn(30,3,device='cuda',dtype=torch.float64)
        lam,cv=cv_ridge(H,Y);n=24
        for idx in (8,12,16):
            ll=cv['grid'][idx];W=torch.linalg.solve(H[:n].T@H[:n]+ll*torch.eye(dim,device='cuda'),H[:n].T@Y[:n])
            direct=float((H[n:]@W-Y[n:]).square().mean())
            assert abs(direct-cv['mse'][idx])<1e-7
        W,rr=ridge_solve(H.T@H,H.T@Y,1.)
        assert rr['relative_residual']<1e-12
    # Reuse the existing independent tests of the unchanged V2 mechanism.
    from experiments.gfg_final_architecture_v2_ten_task.tests import tests as v2tests
    inherited=v2tests()
    return dict(status='PASS',gem_kkt_cases=9,agem_analytic=True,er_gradient_equivalence=True,
        ridge_primal_dual_equivalence=True,inherited_v2=inherited)

if __name__=='__main__':p.setup();print(json.dumps(tests(),indent=2))
