"""Conservative screening around the unchanged V2 Hildreth primitives."""
from experiments.gfg_frozen_head_baseline_comparison_v1.core import *
from experiments.gfg_relation_capability_floor_task8_v1.core import choose
from experiments.gfg_frozen_head_baseline_comparison_v1.core import ROOT as INPUT

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path('E:/gfg-evidence/final-v2-safe-screening-v1')
BASE = '41ca46bef060815d2f106f8235764fb8b03e67b7'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def gamma(k, dtype):
    u = torch.finfo(dtype).eps / 2
    return k*u/(1-k*u)


def upper_norm(a, g):
    v = torch.linalg.vector_norm(a, dim=-1)/(1-g)
    return torch.nextafter(v, torch.full_like(v, torch.inf))


def augmented_gpu(state, seen):
    return torch.cat((state['weight'][:seen].double(), state['bias'][:seen, None].double()), 1)


class Screen:
    """Certificate state independent of observer outcomes and labels beyond bank."""
    @torch.no_grad()
    def __init__(self, h, y, n, groups, seen):
        assert h.dtype == torch.float32 and not torch.is_autocast_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert bool(torch.isfinite(h).all()) and float(h.abs().max()) <= 1e10
        self.h, self.y, self.n, self.groups, self.seen = h, y, n, groups, seen
        self.labels = y[:n].cpu().numpy()
        self.X = torch.cat((h[:n].double(), torch.ones((n, 1), device=h.device, dtype=torch.float64)), 1)
        self.k = 8*self.X.shape[1]+64
        self.g64, self.g32 = gamma(self.k, torch.float64), gamma(self.k, torch.float32)
        self.q = upper_norm(self.X, self.g64)
        self.anchors = torch.empty((groups, seen, self.X.shape[1]), dtype=torch.float64, device=h.device)
        self.anchor_norm = torch.empty(groups, dtype=torch.float64, device=h.device)
        self.lower = torch.empty(n, dtype=torch.float64, device=h.device)
        self.age = np.full(groups, -1000, dtype=int)
        self.last_fraction = np.ones(groups)
        self.refreshes = []
        self.last = None

    @torch.no_grad()
    def begin(self, state, step):
        relations = np.flatnonzero((self.last_fraction > .25) | (step-self.age >= 100))
        if not len(relations):
            return None
        V = augmented_gpu(state, self.seen)
        assert bool(torch.isfinite(V).all()) and float(V.abs().max()) <= 1e10
        rows = np.flatnonzero(np.isin(self.labels, relations))
        ix = torch.as_tensor(rows, device=self.h.device)
        vn = upper_norm(V, self.g64)
        z = self.X[ix] @ V.T
        m = margins(z, self.y[ix])
        err = self.g64*self.q[ix]*(vn[self.y[ix]]+vn.max())
        low = m-err-self.g64*(1+m.abs()+err)
        self.lower[ix] = torch.nextafter(low, torch.full_like(low, -torch.inf))
        self.anchors[relations] = V
        self.anchor_norm[relations] = vn.max()
        self.age[relations] = step
        row = dict(step=step, relations=relations.tolist(), rows=len(rows))
        self.refreshes.append(row)
        return row

    @torch.no_grad()
    def evaluate(self, state, selected=()):
        V = augmented_gpu(state, self.seen)
        assert bool(torch.isfinite(V).all()) and float(V.abs().max()) <= 1e10
        vn = upper_norm(V, self.g64)
        D = V[None] - self.anchors
        target = D[torch.arange(self.groups, device=D.device), torch.arange(self.groups, device=D.device)]
        bound = upper_norm(target[:, None]-D, self.g64).amax(1)
        bound += 2*self.g64*(vn.max()+self.anchor_norm)
        bound = torch.nextafter(bound, torch.full_like(bound, torch.inf))
        movement = self.q*bound[self.y[:self.n]]
        scales = vn[self.y[:self.n]]+vn.max()
        reserve = self.g32*self.q*scales
        reserve += self.k*torch.finfo(torch.float32).tiny*(1+self.q+scales)
        lower = self.lower-movement-reserve
        lower -= self.g64*(1+self.lower.abs()+movement+reserve)
        lower = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
        certified = torch.isfinite(lower) & (lower > 0)
        evaluate = ~certified
        if selected:
            evaluate[list(sorted(selected))] = True
        # Audit-only current controls always evaluated, never counted in floors.
        fullmask = torch.cat((evaluate, torch.ones(len(self.h)-self.n, dtype=torch.bool, device=self.h.device)))
        ix = torch.nonzero(fullmask).flatten()
        zz = p.affine(state, self.h[ix], self.seen)
        mm = margins(zz, self.y[ix])
        z = torch.full((len(self.h), self.seen), torch.nan, device=self.h.device)
        m = torch.full((len(self.h),), torch.nan, device=self.h.device)
        z[ix], m[ix] = zz, mm
        correct = torch.zeros(len(self.h), dtype=torch.bool, device=self.h.device)
        correct[:self.n] = certified
        correct[ix] = mm > 0
        result = dict(z=z, m=m, correct=correct, certified=certified, lower=lower,
                      exact_mask=fullmask, C=counts(correct[:self.n].cpu().numpy().astype(float), self.labels, self.groups))
        self.last = result
        return result

    def finish(self):
        ambiguous = (~self.last['certified']).cpu().numpy()
        self.last_fraction = np.bincount(self.labels, weights=ambiguous, minlength=self.groups)/np.bincount(self.labels, minlength=self.groups)


def select(result, labels, K, ids):
    """Certified positive rows are excluded; do not manufacture their margins."""
    C = result['C']
    deficit = np.maximum(K-C, 0)
    wrong = ~result['correct'][:len(labels)].cpu().numpy()
    m = result['m'][:len(labels)].cpu().numpy()
    chosen = {}
    for r, d in enumerate(deficit):
        if d:
            options = np.flatnonzero((labels == r) & wrong).tolist()
            assert np.isfinite(m[options]).all()
            chosen[str(r)] = sorted(options, key=lambda i: (-float(m[i]), int(ids[i])))[:int(d)]
            assert len(chosen[str(r)]) == d
    return C, deficit, chosen


@torch.no_grad()
def screened_projection(before, native, screen, K, ids, evaluate):
    """Same selection/dual coordinate sequence, with a replaceable evaluator."""
    h, y, n, seen = screen.h, screen.y, screen.n, screen.seen
    labels = screen.labels
    b = evaluate(before, set(), 'receiving')
    assert np.all(b['C'] >= K)
    r = evaluate(native, set(), 'native')
    native_counts, deficits, _ = select(r, labels, K, ids)
    candidate = native
    active, dual, selected, selections, sweeps = [], {}, set(), [], []
    budget_hash = None
    if deficits.any():
        P = p.augmented(native, seen)
        H = np.column_stack((h[:n].cpu().numpy().astype(np.float64), np.ones(n)))
        lengths = 2*np.sum(H*H, axis=1)
        budgets = p.logit_roundoff_budgets(before, native, H, seen)
        budget_hash = p.array_hash(budgets)

        def add(sweepno):
            nonlocal active
            C, D, chosen = select(r, labels, K, ids)
            selected.update(i for values in chosen.values() for i in values)
            mask = torch.zeros(n, dtype=torch.bool, device=h.device)
            if selected:
                mask[sorted(selected)] = True
            # Newly chosen rows were wrong, hence necessarily exactly evaluated.
            assert bool(r['exact_mask'][:n][mask].all())
            bad = p.failing_pairs(r['z'], y, mask)
            new = sorted(set(bad)-set(dual))
            for pair in new:
                dual[pair] = 0.
            active = sorted(dual)
            selections.append(dict(after_sweep=sweepno, counts=C.tolist(), deficits=D.tolist(),
                                   chosen=chosen, selected_controls=sorted(selected), added_pairs=[list(v) for v in new]))

        add(0)
        for sweep in range(1, 101):
            for i, c in active:
                target = int(labels[i])
                padding = float(budgets[i, target]+budgets[i, c])
                dual[(i, c)] = p.coordinate(P, H, lengths, i, target, c, dual[(i, c)], padding)
            candidate = p.materialize(P, native)
            r = evaluate(candidate, selected, 'sweep'+str(sweep))
            safe = bool(np.all(r['C'] >= K))
            sweeps.append(dict(sweep=sweep, counts=r['C'].tolist(), safe=safe, constraints=len(active),
                               primal_sha256=p.array_hash(P), dual_sha256=p.array_hash(np.array([dual[v] for v in active], dtype=np.float64))))
            if safe:
                break
            if sweep < 100:
                add(sweep)
    failed = bool(np.any(r['C'] < K))
    executed = before if failed else candidate
    report = dict(native_counts=native_counts.tolist(), native_deficits=deficits.tolist(),
                  projection_triggered=bool(deficits.any()), skip=failed, selections=selections,
                  sweeps=sweeps, sweep_count=len(sweeps), final_active=[list(v) for v in active],
                  final_duals=[dual[v] for v in active], selected_controls=sorted(selected),
                  executed_counts=(b if failed else r)['C'].tolist(), roundoff_budget_sha256=budget_hash)
    return executed, report
