"""Original Stage1 solver arithmetic with a read-only callback after each sweep."""
from .common import *

def observe_solver(G,b,Ad0,d0norm2,rho,limit=200,observe=None):
    """Dykstra over halfspaces and ball in span{d0,A.T}; no vector-space clipping."""
    m=len(b);t=np.zeros(m);lam=np.zeros(m);c=1.;tc=np.zeros(m);cc=0.;z=Ad0.copy();work=np.empty(m);hits=0;coordinates=0
    norm2=lambda a,v:max(0.,a*a*d0norm2+2*a*float(v@Ad0)+float(v@(G@v)))
    diag=G.diagonal();change=None
    for sweep in range(1,limit+1):
        sweep_start=time.perf_counter()
        oldc=c;oldt=t.copy()
        for i in range(m):
            new=max(0.,lam[i]+(b[i]-z[i])/diag[i]);delta=new-lam[i];lam[i]=new
            if delta:
                t[i]+=delta;np.multiply(G[:,i],delta,out=work);np.add(z,work,out=z)
            coordinates+=1
        yc=c+cc;yt=t+tc;length=np.sqrt(norm2(yc,yt));scale=min(1.,rho/max(length,1e-300))
        hits+=int(scale<1.);c=yc*scale;t=yt*scale;cc=yc*(1-scale);tc=yt*(1-scale);Gt=G@t;z=c*Ad0+Gt
        violation=float(np.maximum(b-z,0).max(initial=0));length=np.sqrt(max(0.,c*c*d0norm2+2*c*float(t@Ad0)+float(t@Gt)));change=np.sqrt(norm2(c-oldc,t-oldt))
        elapsed=time.perf_counter()-sweep_start
        if observe is not None:observe(sweep,c,t,violation,length,change,elapsed)
        if violation<=1e-8 and length<=rho+1e-10 and change<=1e-10+1e-8*rho:
            return c,t,dict(status='PASS',sweeps=sweep,coordinates=coordinates,ball_hits=hits,halfspace_violation=violation,displacement_norm=length,ball_violation=max(0.,length-rho),cycle_change=change)
    return c,t,dict(status='INNER_LIMIT',sweeps=limit,coordinates=coordinates,ball_hits=hits,halfspace_violation=violation,displacement_norm=length,ball_violation=max(0.,length-rho),cycle_change=change)
