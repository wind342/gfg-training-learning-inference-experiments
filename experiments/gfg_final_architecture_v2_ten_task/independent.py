"""Existing independent checker generalized ONLY from fixed64 to seen classes."""
from .core import *

@torch.no_grad()
def independent(before,native,h,y,n,K,ids,seen):
    labels=y[:n].cpu().numpy();truth=y.cpu().numpy();groups=len(K)
    def margins_(state):
        z=nn.functional.linear(h,state['weight'],state['bias'])[:,:seen].cpu().numpy()
        gaps=z[np.arange(len(truth)),truth,None]-z;gaps[np.arange(len(truth)),truth]=np.inf
        return gaps,gaps.min(1)
    def count(m):return np.bincount(labels[m[:n]>0],minlength=groups)
    bg,bm=margins_(before);g,cm=margins_(native);nm=cm.copy();C=count(cm)
    active=[];dual={};selected=set();selections=[];logs=[];candidate=native
    if np.any(C<K):
        H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
        P=np.column_stack((native['weight'][:seen].cpu().numpy().astype(np.float64),native['bias'][:seen].cpu().numpy().astype(np.float64)))
        R=np.column_stack((before['weight'][:seen].cpu().numpy().astype(np.float64),before['bias'][:seen].cpu().numpy().astype(np.float64)))
        u=2.**-24;k=H.shape[1]+2;budgets=(k*u/(1-k*u))*(np.abs(H)@np.maximum(np.abs(R),np.abs(P)).T)
        lengths=2*np.sum(H*H,axis=1)
        for sweep in range(1,101):
            deficits=np.maximum(K-count(cm),0);chosen={}
            for r in range(groups):
                if deficits[r]:
                    wrong=[i for i in range(n) if labels[i]==r and cm[i]<=0]
                    wrong.sort(key=lambda i:(-float(cm[i]),int(ids[i])))
                    chosen[str(r)]=wrong[:int(deficits[r])];selected.update(chosen[str(r)])
            mask=np.zeros(n,dtype=bool);mask[list(selected)]=True
            bad=[tuple(map(int,v)) for v in np.argwhere((g[:n].astype(np.float64)<1e-6)&mask[:,None])]
            added=sorted(set(bad)-set(dual))
            for pair in added:dual[pair]=0.
            active=sorted(dual)
            selections.append(dict(after_sweep=sweep-1,counts=count(cm).tolist(),deficits=deficits.tolist(),
                chosen=chosen,selected_controls=sorted(selected),added_pairs=[list(v) for v in added]))
            for i,c in active:
                t=int(labels[i]);previous=dual[(i,c)]
                updated=max(0.,previous+(1e-6+float(budgets[i,t]+budgets[i,c])-float(np.dot(H[i],P[t]-P[c])))/float(lengths[i]))
                change=updated-previous
                if change:P[t]+=change*H[i];P[c]-=change*H[i]
                dual[(i,c)]=updated
            candidate={name:v.clone() for name,v in native.items()}
            candidate['weight'][:seen].copy_(torch.from_numpy(P[:,:-1]).to(native['weight']))
            candidate['bias'][:seen].copy_(torch.from_numpy(P[:,-1]).to(native['bias']))
            g,cm=margins_(candidate);C=count(cm);safe=bool(np.all(C>=K))
            logs.append(dict(sweep=sweep,counts=C.tolist(),safe=safe,constraints=len(active),
                primal_sha256=hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest(),
                dual_sha256=hashlib.sha256(np.array([dual[v] for v in active],dtype=np.float64).tobytes()).hexdigest()))
            if safe:break
    fail=bool(np.any(count(cm)<K));executed=before if fail else candidate
    return executed,candidate,selections,logs,bm,nm,cm,(bm if fail else cm)

