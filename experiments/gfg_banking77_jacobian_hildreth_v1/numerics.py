"""Independent derivatives, primal QP oracle, fixed-buffer and AdamW checks."""
from .cutter import *
from scipy.optimize import minimize

def main():
    setup();guard();rng=np.random.default_rng(829);results={}
    h=model();p=params(h);x=torch.randn(70,768,device='cuda');y=torch.arange(70,device='cuda')%7
    bank=Bank(x,y,np.arange(70),14,Meter());z=bank.exact(p)
    with torch.no_grad():ref=functional(p,bank.x)[:70,:14].cpu().numpy()
    assert np.array_equal(z,ref);results['fixed256_byte_identical']=True
    ordered=[(0,8),(1,5),(0,3)];g=Geometry(bank,cpu(p),ordered,z);A=g.explicit()
    pp=tuple(v.detach().double().requires_grad_() for v in p)
    rows=[]
    for i,c in ordered:
        zz=functional(pp,x[i:i+1].double())[0];v=zz[int(y[i])]-zz[c]
        rows.append(torch.cat([a.flatten() for a in torch.autograd.grad(v,pp)]).detach().cpu().numpy())
    auto=np.stack(rows);err=float(np.max(np.abs(A-auto)));assert err<1e-10
    assert np.allclose(g.G,A@A.T,atol=1e-9,rtol=1e-12)
    d=tuple(rng.normal(size=v.shape) for v in cpu(p));flat=np.concatenate([v.ravel() for v in d])
    assert np.allclose(g.apply(d),A@flat,atol=1e-9)
    t=rng.normal(size=3);assert np.allclose(np.concatenate([v.ravel() for v in g.reconstruct(t)]),A.T@t,atol=1e-12)
    results['jacobian_max_abs_error']=err;results['structured_geometry']='PASS'
    cases=[]
    for k in range(12):
        a=np.eye(4)[:2] if k==0 else rng.normal(size=(3,5))
        feasible=rng.normal(size=a.shape[1]);feasible*=.35/np.linalg.norm(feasible)
        b=a@feasible-rng.uniform(.05,.3,len(a));d0=rng.normal(size=a.shape[1])*2;rho=1.
        c,t,info=ball_hildreth(a@a.T,b,a@d0,float(d0@d0),rho)
        out=c*d0+a.T@t
        oracle=minimize(lambda d:.5*np.sum((d-d0)**2),feasible,jac=lambda d:d-d0,
            constraints=[{'type':'ineq','fun':lambda d:a@d-b,'jac':lambda d:a},
                         {'type':'ineq','fun':lambda d:rho*rho-d@d,'jac':lambda d:-2*d}],
            method='SLSQP',options={'ftol':1e-10,'maxiter':1000})
        assert oracle.success,(k,oracle.message)
        assert info['status']=='PASS',(k,info)
        objdiff=abs(.5*np.sum((out-d0)**2)-oracle.fun)
        assert objdiff<1e-6 and np.max(b-a@out)<1e-8 and np.linalg.norm(out)<=rho+1e-10
        cases.append(dict(case=k,objective_difference=objdiff,**info))
    _,_,bad=ball_hildreth(np.ones((1,1)),np.array([2.]),np.array([0.]),0.,1.)
    assert bad['status']=='INNER_LIMIT';results['qp_cases']=cases;results['infeasible_ball']=bad
    # An independent in-place AdamW update must match the shadow path exactly.
    shadow=model();direct=copy.deepcopy(h);os=optimizer(shadow);od=optimizer(direct)
    for step in range(4):
        native_step(h,shadow,os,x[:16],y[:16],14);commit(h,params(shadow))
        direct.zero_grad(set_to_none=True);F.cross_entropy(direct(x[:16])[:,:14],y[:16]).backward()
        nn.utils.clip_grad_norm_(direct.parameters(),1.);od.step()
        assert thash(h.state_dict())==thash(direct.state_dict()) and opt_hash(os)==opt_hash(od)
    results['shadow_adamw_byte_identical']=True
    pp=params(h);bb=Bank(x,y,np.arange(70),14,Meter());out,r=project(pp,pp,bb,np.zeros(7,dtype=int))
    assert r['native_safe'] and r['jacobian_calls']==0 and all(a is b for a,b in zip(out,pp))
    results['native_safe_no_jacobian']=True;results['status']='PASS'
    destination='NUMERICS_CACHED_ABSH.json' if (ROOT/'NUMERICS.json').exists() else 'NUMERICS.json'
    write(ROOT/destination,results);print(json.dumps(results),flush=True)

if __name__=='__main__':main()
