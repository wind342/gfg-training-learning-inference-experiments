"""Fixed arithmetic and local dependency injection checks."""
from .core import *


@torch.no_grad()
def tests(h, y, n, before, native, K, ids):
    original = p.affine
    assert Screen.evaluate.__wrapped__.__code__ is inspect.unwrap(PreviousScreen.evaluate).__code__
    assert reference_projection.__wrapped__.__code__ is inspect.unwrap(floor_projection).__code__
    full = fixed_affine(before,h,80)
    cases=[]
    for indices in (torch.arange(0,len(h),3,device=h.device),
                    torch.arange(len(h)-1,-1,-5,device=h.device),
                    torch.arange(1,778,device=h.device),
                    torch.tensor([0,1,255,256,541,len(h)-1],device=h.device)):
        z = fixed_affine(before,h[indices],80)
        delta = float((z-full[indices]).abs().max())
        cases.append(dict(rows=len(indices),max_error=delta,byte_identical=torch.equal(z,full[indices])))
        assert torch.equal(z,full[indices]),'Fixed tile arithmetic differs on gathered rows'
    count = counts(margins(full,y)[:n].cpu().numpy(),y[:n].cpu().numpy(),len(K))
    assert count.tolist()==K.tolist(),'ENTRY_COUNT_MISMATCH_DO_NOT_CHANGE_K'
    s=Screen(h,y,n,len(K),80);s.begin(before,1)
    def evaluate(state,selected,label):
        r=s.evaluate(state,selected)
        f=fixed_affine(state,h,80);m=margins(f,y)
        assert torch.equal(r['m'][r['exact_mask']],m[r['exact_mask']])
        assert torch.equal(r['correct'],m>0)
        return r
    executed,r=screened_projection(before,native,s,K,ids[:n],evaluate)
    e,_,ref=reference_projection(before,native,h,y,n,K,ids[:n],seen=80)
    assert equal(executed,e)
    assert all(r[k]==ref[k] for k in r)
    assert p.affine is original,'Imported scientific module must not be mutated'
    return dict(status='PASS',fixed_gathered_logits=cases,task_entry_K_unchanged=True,
                original_projection_bytecode=True,all_one_step_projection_details_equal=True,
                imported_module_unmodified=True)
