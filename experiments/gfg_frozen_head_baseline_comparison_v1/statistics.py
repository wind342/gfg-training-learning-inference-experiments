"""Equation-based cumulative ridge and fixed-RP classifier adaptations."""
from .core import *

@torch.no_grad()
def cv_ridge(H,Y):
    n=int(len(H)*.8);A=H[:n];B=H[n:];T=Y[:n]
    dual=len(A)<A.shape[1]
    G=A@A.T if dual else A.T@A
    eigen,V=torch.linalg.eigh(G);negative=int((eigen<0).sum());eigen=eigen.clamp_min(0.)
    left=B@A.T@V if dual else B@V
    right=V.T@T if dual else V.T@(A.T@T)
    grid=[10.**i for i in range(-8,9)];losses=[]
    for lam in grid:
        pred=(left/(eigen+lam))@right
        losses.append(float((pred-Y[n:]).square().mean()))
    assert all(np.isfinite(losses))
    winner=int(np.argmin(losses))
    return grid[winner],dict(grid=grid,mse=losses,selected_index=winner,dual=dual,negative_eigenvalues=negative)

@torch.no_grad()
def ridge_solve(G,Q,lam):
    A=G.clone();A.diagonal().add_(lam)
    L,info=torch.linalg.cholesky_ex(A)
    if int(info)==0:
        W=torch.cholesky_solve(Q,L);method='cholesky';negative=0
    else:
        eigen,V=torch.linalg.eigh(G);negative=int((eigen<0).sum())
        W=V@((V.T@Q)/(eigen.clamp_min(0)+lam)[:,None]);method='eigh-roundoff-clamp'
    residual=float((A@W-Q).norm()/Q.norm().clamp_min(1e-30))
    assert torch.isfinite(W).all() and np.isfinite(residual)
    return W,dict(solver=method,cholesky_info=int(info),relative_residual=residual,negative_eigenvalues=negative)

class Cumulative:
    def __init__(self,kind):
        self.kind=kind;self.R=None
        if kind=='ranpac_rp':
            self.R=torch.randn(1536,10000,generator=torch.Generator().manual_seed(2021)).cuda()
        dim=10000 if self.R is not None else 1536
        self.G=torch.zeros((dim,dim),dtype=torch.float64,device='cuda')
        self.Q=torch.zeros((dim,80),dtype=torch.float64,device='cuda')
        self.W=None;self.arrived=set()

    @torch.no_grad()
    def transform(self,F):
        return (torch.relu(F@self.R) if self.R is not None else F).double()

    @torch.no_grad()
    def logits(self,F):return self.transform(F)@self.W

    @torch.no_grad()
    def train_task(self,F,d,task):
        sync();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        current=input_ids(d,task);released=newly_released(d,task)
        ids=np.concatenate((current,released));assert not self.arrived.intersection(map(int,ids))
        assert np.all(d['split'][current]==0) and np.all(d['y'][current]//8==task-1)
        assert not len(released) or np.all(d['y'][released]//8==task-2)
        H=self.transform(F[current]);Y=nn.functional.one_hot(torch.from_numpy(d['y'][current]).cuda(),80).double()
        lam,cv=cv_ridge(H,Y)
        self.G.add_(H.T@H);self.Q.add_(H.T@Y)
        del H,Y
        if len(released):
            H=self.transform(F[released]);Y=nn.functional.one_hot(torch.from_numpy(d['y'][released]).cuda(),80).double()
            self.G.add_(H.T@H);self.Q.add_(H.T@Y);del H,Y
        self.W,solver=ridge_solve(self.G,self.Q,lam);self.arrived.update(map(int,ids))
        sync();elapsed=time.perf_counter()-start
        return dict(task=task,training_indices=current.tolist(),released_controls=released.tolist(),
            arrived_count=len(self.arrived),old_feature_reads=len(released),old_backward_examples=0,
            algorithm_wall=elapsed,lambda_value=lam,cv=cv,solver=solver,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())

    def save_task(self,task):
        save(ROOT/self.kind/f'task{task}'/'CLASSIFIER.pt',dict(weight=self.W.cpu(),task=task))

    def save_final(self):
        save(ROOT/self.kind/'STATISTICS.pt',dict(G=self.G.cpu(),Q=self.Q.cpu(),arrived=sorted(self.arrived)))
        if self.R is not None:save(ROOT/self.kind/'RANDOM_PROJECTION.pt',self.R.cpu())
