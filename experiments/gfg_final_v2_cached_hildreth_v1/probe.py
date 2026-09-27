"""Read-only reduction-shape probe; writes only a new diagnostic authority."""
from experiments.gfg_final_architecture_v2_safe_screening_v3.run import load
from experiments.gfg_final_architecture_v2_safe_screening_v2.core import *


def main():
    p.setup()
    d,F,initial,h,y,ids,n,K=load()
    head,shadow,opt=make_state(initial)
    current=order(d,10)[0]
    ce_grad(head,F[current],torch.from_numpy(d['y'][current]).cuda(),80)
    nn.utils.clip_grad_norm_(head.parameters(),1.)
    before=p.snapshot(head);propose(head,shadow,opt);native=p.snapshot(shadow)
    H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
    A=np.abs(H);E=np.maximum(np.abs(p.augmented(before,80)),np.abs(p.augmented(native,80)))
    u=float(np.finfo(np.float32).eps)/2;k=H.shape[1]+2;g=k*u/(1-k*u)
    full=g*(A@E.T)
    out={}
    # Known shape arithmetic, not data-dependent reserve/selection tuning.
    for nr in (1,2,4,8,16,32,64):
        ix=np.arange(nr)*17
        for nc in (2,4,8,16,80):
            cs=np.arange(nc)
            value=g*(A[ix]@E[cs].T)
            ref=full[np.ix_(ix,cs)]
            out[f'{nr}x{nc}']=dict(equal=bool(np.array_equal(value,ref)),different=int(np.count_nonzero(value!=ref)),max_error=float(np.max(np.abs(value-ref))))
    root=Path('E:/gfg-evidence/final-v2-cached-hildreth-v1-probe')
    root.mkdir(exist_ok=False)
    write(root/'REDUCTION_SHAPES.json',out)
    print(json.dumps(out,indent=2),flush=True)


def pair_probe():
    p.setup()
    d,F,initial,h,y,ids,n,K=load()
    before=initial['head'];native=initial['head']
    H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
    A=np.abs(H);E=np.abs(p.augmented(before,80))
    full=A@E.T
    rng=np.random.default_rng(2021);out={}
    for nr in (8,16,24,32):
        ix=rng.choice(n,nr,replace=False)
        for nc in (24,32,48,64,80):
            cs=np.tile([3,71],nc//2)
            value=A[ix]@E[cs].T
            ref=full[np.ix_(ix,cs)]
            out[f'{nr}x{nc}']=dict(equal=bool(np.array_equal(value,ref)),different=int(np.count_nonzero(value!=ref)),max_error=float(np.max(np.abs(value-ref))))
    write(Path('E:/gfg-evidence/final-v2-cached-hildreth-v1-probe')/'PAIR_PADDING.json',out)
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
