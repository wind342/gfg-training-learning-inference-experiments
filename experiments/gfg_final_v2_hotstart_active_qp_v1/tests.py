"""Independent NNLS/KKT checks before the new actuator touches task10."""
from .core import *
from scipy.optimize import nnls


def main():
    rng=np.random.default_rng(2021);checks=[]
    for n in (1,4,16,32):
        A=rng.normal(size=(n+8,n));G=A.T@A+np.eye(n)*.1
        warm=np.zeros(n)
        for trial in range(8):
            b=rng.normal(size=n) if trial else -np.ones(n)
            L=la.cholesky(G,lower=True);target=la.solve_triangular(L,b,lower=True)
            reference=nnls(L.T,target)[0]
            x,info=active_qp(G,b,warm)
            assert info['status']=='PASS',info
            assert np.allclose(x,reference,rtol=1e-7,atol=1e-9)
            cold,ci=active_qp(G,b,np.zeros(n));assert np.allclose(x,cold,rtol=1e-7,atol=1e-9)
            checks.append(dict(n=n,trial=trial,iterations=info['iterations'],cold_iterations=ci['iterations'],max_error=float(np.max(np.abs(x-reference)))))
            warm=x
    result=dict(status='PASS',independent_nnls_cases=len(checks),cases=checks)
    out=ROOT/'SOLVER_PREFLIGHT.json'
    if out.exists():raise FileExistsError(out)
    write(out,result);print(json.dumps(dict(status='PASS',cases=len(checks),max_error=max(x['max_error'] for x in checks))),flush=True)


def gram_check():
    rng=np.random.default_rng(2022);H=rng.normal(size=(16,7));labels=np.arange(16)%4
    g=SimpleNamespace(H=H,labels=labels);cache=GramCache(g)
    pairs=[(0,1),(1,0),(2,3),(3,1),(4,2),(5,3)]
    cache.block(pairs[:2]);G=cache.block(pairs)
    A=np.zeros((len(pairs),4*7))
    for j,(i,c) in enumerate(pairs):
        A[j,labels[i]*7:(labels[i]+1)*7]=H[i];A[j,c*7:(c+1)*7]=-H[i]
    assert np.allclose(G,A@A.T,rtol=1e-14,atol=1e-14)
    before=cache.computed;reused=cache.block(pairs)
    assert np.array_equal(G,reused) and cache.computed==before
    result=dict(status='PASS',dense_signed_constraint_gram_matches=True,repeat_request_computes_no_entries=True,
                incremental_registry_size=len(cache.pairs),computed_nonzero_entries=cache.computed)
    out=ROOT/'GRAM_CHECKS.json'
    if out.exists():raise FileExistsError(out)
    write(out,result);return result


if __name__=='__main__':main()
