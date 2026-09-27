"""Cached analytic geometry; active-pair reserves, unchanged numerical rules."""
import time
import numpy as np
from scipy.special import erf
from .runtime import EPS

def gelu(x):return .5*x*(1+erf(x/np.sqrt(2.)))

def gelu_prime(x):return .5*(1+erf(x/np.sqrt(2.)))+x*np.exp(-.5*x*x)/np.sqrt(2*np.pi)

def inner(a,b):return sum(float(np.sum(x*y)) for x,y in zip(a,b))

class Geometry:
    def __init__(self,bank,p,ordered,z):
        self.bank=bank;self.ordered=ordered;m=len(ordered);clock=time.perf_counter()
        self.rows=np.array([i for i,c in ordered]);self.comp=np.array([c for i,c in ordered]);self.target=bank.labels[self.rows]
        unique,back=np.unique(self.rows,return_inverse=True);H=bank.H[unique];w,b,v,c=p
        u=H@w.T+b;q=gelu(u);prime=gelu_prime(u)
        self.H=bank.H[self.rows];self.Q=q[back];self.S=(v[self.target]-v[self.comp])*prime[back]
        self.B=np.zeros((m,len(v)));self.B[np.arange(m),self.target]=1.;self.B[np.arange(m),self.comp]=-1.
        bank.meter.seconds['geometry_factors_subregion']+=time.perf_counter()-clock;clock=time.perf_counter()
        self.G=np.asfortranarray((self.S@self.S.T)*bank.inputGram[np.ix_(self.rows,self.rows)]+(self.B@self.B.T)*(self.Q@self.Q.T+1.))
        assert np.isfinite(self.G).all() and (self.G.diagonal()>0).all()
        bank.meter.seconds['gram_subregion']+=time.perf_counter()-clock;clock=time.perf_counter()
        self.g=(z[self.rows,self.target]-z[self.rows,self.comp]).astype(np.float64)
        u32=2.**-24;gamma=lambda n:n*u32/(1-n*u32)
        eu=gamma(769)*(bank.absH[unique]@np.abs(w).T+np.abs(b));eq=1.13*eu+8*u32*(1+np.abs(u)+np.abs(q))
        # Only active target/competitor rows; no full controls x classes budget.
        ey=np.einsum('ij,ij->i',np.abs(v[self.target]),eq[back])+gamma(257)*(np.einsum('ij,ij->i',np.abs(v[self.target]),np.abs(self.Q))+np.abs(c[self.target]))
        ec=np.einsum('ij,ij->i',np.abs(v[self.comp]),eq[back])+gamma(257)*(np.einsum('ij,ij->i',np.abs(v[self.comp]),np.abs(self.Q))+np.abs(c[self.comp]))
        ref=np.einsum('ij,ij->i',self.Q,v[self.target]-v[self.comp])+c[self.target]-c[self.comp]
        self.reserve=np.maximum(ey+ec,np.abs(self.g-ref))+8*u32
        bank.meter.seconds['budget_subregion']+=time.perf_counter()-clock
        bank.meter.calls['jacobian_batches']+=1;bank.meter.calls['jacobian_pairs']+=m;bank.meter.calls['jacobian_unique_rows']+=len(unique)
        bank.meter.flops['jacobian_forward_guard']+=4*len(unique)*768*256+8*m*256
        bank.meter.flops['gram']+=2*m*m*(256+256+len(v))+4*m*m
    def apply(self,d):
        # A*d without materializing rows of A.
        self.bank.meter.flops['jacobian_directional_dense_estimate']+=2*len(self.rows)*768*256+6*len(self.rows)*256
        u=self.H@d[0].T+d[1]
        return np.einsum('ij,ij->i',self.S,u)+np.einsum('ij,ij->i',self.Q,d[2][self.target]-d[2][self.comp])+d[3][self.target]-d[3][self.comp]
    def reconstruct(self,t):
        weighted=t[:,None]*self.S;hidden=t[:,None]*self.Q
        return (weighted.T@self.H,weighted.sum(0),self.B.T@hidden,self.B.T@t)
    def explicit(self):
        # Test-only; never called in production cutter.
        return np.concatenate([(self.S[:,:,None]*self.H[:,None,:]).reshape(len(self.rows),-1),self.S,(self.B[:,:,None]*self.Q[:,None,:]).reshape(len(self.rows),-1),self.B],axis=1)

def select(bank,z,K,selected,pairs):
    margins=bank.margins(z);C=bank.counts(z);D=np.maximum(K-C,0);chosen={}
    for r in np.flatnonzero(D):
        wrong=bank.relation_rows[r][margins[bank.relation_rows[r]]<=0]
        ix=sorted(wrong.tolist(),key=lambda i:(-float(margins[i]),int(bank.ids[i])))[:int(D[r])]
        chosen[str(r)]=ix;selected.update(ix)
    for i in sorted(selected):
        gap=z[i,bank.labels[i]]-z[i]
        pairs.update((i,int(c)) for c in np.flatnonzero(gap<EPS) if c!=bank.labels[i])
    return C,D,chosen
