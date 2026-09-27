"""Post-smoke independent local-QP oracle; never changes executed training."""
from .common import *
from scipy.optimize import minimize

def main():
    result=[]
    for path in sorted((ROOT/'audit').glob('LOCAL_QP_*.npz')):
        data=np.load(path);G=data['G'];b=data['b'];ad0=data['Ad0'];rho=float(data['rho'])
        ev,Q=np.linalg.eigh(G);keep=ev>ev.max()*1e-13
        A=Q[:,keep]*np.sqrt(ev[keep]);d0=Q[:,keep].T@ad0/np.sqrt(ev[keep])
        orth=max(0.,float(data['d0norm2'])-float(d0@d0))
        if orth>1e-16:A=np.column_stack((A,np.zeros(len(A))));d0=np.r_[d0,np.sqrt(orth)]
        norm=np.linalg.norm(A,axis=1);B=A/norm[:,None];bound=b/(rho*norm);target=d0/rho
        sol=minimize(lambda w:.5*np.sum((w-target)**2),np.zeros(A.shape[1]),jac=lambda w:w-target,
            constraints=[dict(type='ineq',fun=lambda w:B@w-bound,jac=lambda w:B),
                         dict(type='ineq',fun=lambda w:1.-w@w,jac=lambda w:-2*w)],
            method='SLSQP',options=dict(ftol=1e-11,maxiter=1000))
        physical=rho*sol.x;residual=float(np.maximum(b-A@physical,0).max());length=float(np.linalg.norm(physical))
        result.append(dict(file=path.name,step=int(data['step']),status=bool(sol.success),message=sol.message,
            iterations=sol.nit,halfspace_violation=residual,ball_violation=max(0.,length-rho),
            norm_over_rho=length/rho,gram_condition=float(ev[-1]/ev[0]) if ev[0]>0 else None,
            dimension=len(sol.x),pairs=len(b),rank_retained=int(keep.sum()),
            satisfies_frozen_primal_tolerances=residual<=1e-8 and length<=rho+1e-10))
    write(ROOT/'LOCAL_QP_DIAGNOSIS.json',dict(posthoc=True,no_training_or_protocol_change=True,cases=result))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
