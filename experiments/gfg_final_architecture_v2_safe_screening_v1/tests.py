"""Numerical/adversarial preflight; no test set and no parameter selection."""
from .core import *


@torch.no_grad()
def tests():
    p.setup()
    torch.use_deterministic_algorithms(False)
    g = torch.Generator(device='cuda').manual_seed(7001)
    h = torch.randn((120, 12), generator=g, device='cuda')
    y = torch.arange(120, device='cuda') % 3
    before = dict(weight=torch.randn((6, 12), generator=g, device='cuda'), bias=torch.randn(6, generator=g, device='cuda'))
    screen = Screen(h, y, 120, 3, 6)
    screen.begin(before, 1)
    certified = 0
    cases = []
    for amplitude in (0., 1e-7, .001, .1, 1., 10.):
        candidate = {k: v+amplitude*torch.randn(v.shape, generator=g, device=v.device) for k, v in before.items()}
        r = screen.evaluate(candidate, {0, 1})
        fm = margins(p.affine(candidate, h, 6), y)
        assert not bool((r['certified'] & (fm <= 0)).any())
        assert torch.equal(r['correct'], fm > 0)
        assert bool(r['exact_mask'][:2].all())
        V = augmented_gpu(candidate, 6)
        real64 = margins(screen.X@V.T, y)
        assert bool((r['lower'] <= real64).all())
        certified += int(r['certified'].sum())
        cases.append(dict(amplitude=amplitude, certified=int(r['certified'].sum()), exact_counts=True))
    assert certified > 0
    # Exact count aggregation and identity-preserving selection, including ties.
    example = dict(C=np.array([1]), correct=torch.tensor([True, False, False], device='cuda'),
                   m=torch.tensor([float('nan'), -.1, -.1], device='cuda'))
    assert select(example, np.zeros(3, dtype=int), np.array([2]), [10, 8, 3])[2] == {'0': [2]}
    # A mathematically positive certificate cannot exempt already-selected rows.
    positive = torch.nonzero(screen.evaluate(before)['certified']).flatten().tolist()
    assert positive
    assert bool(screen.evaluate(before, {positive[0]})['exact_mask'][positive[0]])
    # The wrapper must reproduce the production algorithm when its evaluator
    # uses the exact original batch. This isolates projection-policy copying.
    labels = y.cpu().numpy()
    K = counts(margins(p.affine(before, h, 6), y).cpu().numpy(), labels, 3)
    differential = []
    for amplitude in (.001, .05, .2):
        native = {k: v+amplitude*torch.randn(v.shape, generator=g, device=v.device) for k, v in before.items()}
        def full(state, selected, label):
            z = p.affine(state, h, 6)
            m = margins(z, y)
            return dict(z=z, m=m, correct=m>0, exact_mask=torch.ones(120, dtype=torch.bool, device='cuda'),
                        C=counts(m.cpu().numpy(), labels, 3))
        e, r = screened_projection(before, native, screen, K, np.arange(120), full)
        re, _, rr = floor_projection(before, native, h, y, 120, K, np.arange(120), seen=6)
        assert equal(e, re)
        for key in r:
            assert r[key] == rr[key], key
        differential.append(dict(amplitude=amplitude, sweeps=r['sweep_count'], byte_identical=True))
    # Zero/tiny gaps certify nothing, and errors remain observed.
    zero = {k: torch.zeros_like(v) for k, v in before.items()}
    zscreen = Screen(h, y, 120, 3, 6)
    zscreen.begin(zero, 1)
    zr = zscreen.evaluate(zero)
    assert not bool(zr['certified'].any()) and bool(zr['exact_mask'].all())
    return dict(status='PASS', certificate_cases=cases, production_differential=differential,
                zero_boundary=True, selected_rows_always_evaluated=True, tie_break=True,
                scope='Numerical implementation checks, not a formal hardware proof')


if __name__ == '__main__':
    print(json.dumps(tests(), indent=2))
