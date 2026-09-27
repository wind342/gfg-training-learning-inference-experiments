"""Boundary cases for two-sided sign proofs and executed-only anchor changes."""
from .transition import *
import io


def preflight():
    h=torch.tensor([[1.,0.],[-1.,0.],[1e-8,0.],[-1e-8,0.],[0.,0.],[1.,1.]],device='cuda')
    y=torch.tensor([0,0,0,0,0,1],device='cuda');K=np.array([1])
    g=Geometry(h,y,5,1,2,Meter());s=TransitionEndpoint(g,K)
    stream=io.StringIO();o=Observer(s,K,np.arange(5),stream)
    a=dict(weight=torch.tensor([[1.,0.],[-1.,0.]],device='cuda'),bias=torch.zeros(2,device='cuda'))
    b=dict(weight=-a['weight'],bias=a['bias'])
    zero=dict(weight=torch.zeros_like(a['weight']),bias=a['bias'])
    o.evaluate(a,(),'receiving');assert s.anchor_updates==1
    identical=o.evaluate(a,(),'native')
    assert bool(identical['positive'].any()) and bool(identical['negative'].any())
    assert bool(identical['exact_mask'][4]) # zero margin never falsely positive
    for state in (b,zero,a,b,a):o.evaluate(state,{0,1},'candidate')
    assert s.anchor_updates==1
    o.evaluate(b,(),'commit');assert s.anchor_updates==2
    o.evaluate(b,(),'receiving');assert s.anchor_updates==2
    o.evaluate(a,{0,1},'candidate')
    # A zero displacement still cannot use old margins after a real commit.
    assert torch.equal(s.anchorMargin,margins(fixed_affine(b,h,2),y)[:5].double())
    return dict(status='PASS',observer_calls=o.calls,zero_margin_checked=True,both_signs_certified=True,
                selected_exact=True,return_and_rollback_checked=True,only_commits_refresh_anchor=True)


if __name__=='__main__':
    p.setup();print(json.dumps(preflight(),indent=2))
