"""Floor-aware lazy certificates, with the v2 fixed FP32 numerical authority."""
from experiments.gfg_final_architecture_v2_safe_screening_v2.core import *
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
ROOT = Path('E:/gfg-evidence/final-v2-safe-screening-v3')
BASE = '2f5e45beb5bdd2b4031ddb8461d56beb34f0ee3f'
OLD = HERE.parent/'gfg_final_architecture_v2_safe_screening_v2'


class Timings:
    """CUDA timeline intervals, resolved only after the timed computation."""
    def __init__(self):
        self.events = []

    @contextmanager
    def region(self, name):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        try:
            yield
        finally:
            b.record()
            self.events.append((name, a, b))

    def resolve(self):
        sync()
        result = {}
        for name, a, b in self.events:
            result[name] = result.get(name, 0.) + a.elapsed_time(b)/1000
        self.events.clear()
        return result


class Screen:
    """No relation x class x hidden tensors, no periodic/full-relation refresh.

    Cached per-row margins are transported through a cumulative path of class
    displacement norm upper bounds. Exact frontier evaluations also refresh
    these caches, so refreshing entails no additional feature/head matmul.
    The path is the sequence of actual candidate heads, including sweeps and
    rollbacks, not the scheduled optimizer path. No observer output is used.
    """
    @torch.no_grad()
    def __init__(self, h, y, n, groups, seen, K):
        self.h, self.y, self.n, self.groups, self.seen = h, y, n, groups, seen
        self.labels = y[:n].cpu().numpy()
        self.K = torch.as_tensor(K, device=h.device)
        self.k = 8*(h.shape[1]+1)+64
        self.g64, self.g32 = gamma(self.k, torch.float64), gamma(self.k, torch.float32)
        assert h.dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32
        assert bool(torch.isfinite(h).all()) and float(h.abs().max()) <= 1e10
        X = torch.cat((h[:n].double(), h.new_ones((n, 1)).double()), 1)
        self.q = upper_norm(X, self.g64)
        self.path = torch.zeros(seen, dtype=torch.float64, device=h.device)
        self.pathmax = torch.zeros((), dtype=torch.float64, device=h.device)
        self.row_path = torch.zeros(n, dtype=torch.float64, device=h.device)
        self.row_pathmax = torch.zeros(n, dtype=torch.float64, device=h.device)
        self.lower = torch.full((n,), -torch.inf, dtype=torch.float64, device=h.device)
        self.prev, self.prevnorm = None, None
        self.z = h.new_empty((len(h), seen))
        self.m = h.new_empty(len(h))
        self.counts_template = torch.zeros(groups, dtype=torch.long, device=h.device)
        # Group-to-row padded map is constructed once; -1 slots are never read.
        width = max(np.bincount(self.labels))
        grid = np.full((groups, width), n, dtype=np.int64)
        for r in range(groups):
            ids = np.flatnonzero(self.labels == r)
            grid[r, :len(ids)] = ids
        self.grid = torch.as_tensor(grid, device=h.device)
        self.valid = self.grid < n
        self.timings = Timings()
        self.calls = 0
        self.total_exact = self.total_padded = self.total_gemms = self.total_refresh = 0
        self.total_certificate_elements = 0
        self.last = None

    def relation_counts(self, correct):
        return self.counts_template.clone().scatter_add_(0, self.y[:self.n], correct[:self.n].long())

    @torch.no_grad()
    def certificate(self, state):
        V = augmented_gpu(state, self.seen)
        vn = upper_norm(V, self.g64)
        if self.prev is not None:
            # Triangle bound for every target/competitor; only C x (D+1).
            dn = upper_norm(V-self.prev, self.g64)
            dn += 2*self.g64*(vn+self.prevnorm)
            dn = torch.nextafter(dn, torch.full_like(dn, torch.inf))
            self.path = torch.nextafter(self.path+dn, torch.full_like(dn, torch.inf))
            self.pathmax = torch.nextafter(self.pathmax+dn.max(), V.new_tensor(torch.inf))
        self.prev, self.prevnorm = V, vn
        py = self.path[self.y[:self.n]]
        distance = (py-self.row_path)+(self.pathmax-self.row_pathmax)
        distance += 4*self.g64*(1+py.abs()+self.row_path.abs()+self.pathmax.abs()+self.row_pathmax.abs())
        distance = torch.nextafter(distance.clamp_min(0), torch.full_like(distance, torch.inf))
        movement = self.q*distance
        scales = vn[self.y[:self.n]]+vn.max()
        reserve = self.g32*self.q*scales
        reserve += self.k*torch.finfo(torch.float32).tiny*(1+self.q+scales)
        lower = self.lower-movement-reserve
        lower -= self.g64*(1+self.lower.abs()+movement+reserve)
        # Never certify out-of-domain/nonfinite candidate weights.
        domain = torch.isfinite(V).all() & (V.abs().max() <= 1e10)
        certified = domain & torch.isfinite(lower) & (lower > 0)
        self.total_certificate_elements += V.numel()
        return lower, certified, reserve

    @torch.no_grad()
    def evaluate(self, state, selected=()):
        self.calls += 1
        with self.timings.region('certificate'):
            lower, certified, reserve = self.certificate(state)
        self.z.fill_(torch.nan); self.m.fill_(torch.nan)
        exact = torch.zeros(len(self.h), dtype=torch.bool, device=self.h.device)
        correct = torch.zeros_like(exact)
        correct[:self.n] = certified
        call_rows, call_padded, call_gemms = 0, 0, 0

        def execute(mask):
            nonlocal call_rows, call_padded, call_gemms
            with self.timings.region('gather_pack'):
                ix = torch.nonzero(mask & ~exact).flatten()
                gathered = self.h[ix]
            if not len(ix):
                return
            with self.timings.region('fixed_affine'):
                # Preserve v2's 256-row arithmetic, including row order.
                zz = fixed_affine(state, gathered, self.seen)
                mm = margins(zz, self.y[ix])
            with self.timings.region('cache_and_scatter'):
                self.z[ix], self.m[ix], correct[ix], exact[ix] = zz, mm, mm > 0, True
                oldix = ix[ix < self.n]
                val = self.m[oldix].double()
                low = val-reserve[oldix]
                low -= self.g64*(1+val.abs()+reserve[oldix])
                self.lower[oldix] = torch.nextafter(low, torch.full_like(low, -torch.inf))
                self.row_path[oldix] = self.path[self.y[oldix]]
                self.row_pathmax[oldix] = self.pathmax
            call_rows += len(ix)
            call_gemms += (len(ix)+TILE-1)//TILE
            call_padded += TILE*((len(ix)+TILE-1)//TILE)

        with self.timings.region('floor_masks'):
            C = self.relation_counts(correct)
            need = (self.K-C).clamp_min(0)
            # At most K-C initially unproven rows per relation, all relations
            # packed together. Sequential original row order, no GPU sort.
            unknown = torch.cat((~certified, certified.new_zeros(1)))[self.grid] & self.valid
            take = unknown & (unknown.cumsum(1) <= need[:, None])
            mask = torch.zeros_like(exact)
            mask[self.grid[take]] = True
            if selected:
                mask[list(sorted(selected))] = True
            # Existing new controls are diagnostic only; never count in K.
            mask[self.n:] = True
        execute(mask)
        with self.timings.region('floor_masks'):
            C = self.relation_counts(correct)
            deficit_possible = C < self.K
            remaining = (~certified) & (~exact[:self.n]) & deficit_possible[self.y[:self.n]]
            mask = torch.cat((remaining, remaining.new_zeros(len(self.h)-self.n)))
        # A possibly deficient relation is fully resolved to preserve original
        # deficit and closest-wrong selection, with no guessed/NaN wrong margin.
        execute(mask)
        with self.timings.region('floor_masks'):
            C = self.relation_counts(correct)
            count_lower = C.cpu().numpy()
        self.total_exact += call_rows-(len(self.h)-self.n)
        self.total_refresh += call_rows-(len(self.h)-self.n)
        self.total_padded += call_padded
        self.total_gemms += call_gemms
        self.last = dict(z=self.z, m=self.m, correct=correct, certified=certified,
                         lower=lower, exact_mask=exact, C=count_lower,
                         exact_rows=call_rows, padded_rows=call_padded, gemms=call_gemms)
        return self.last


def compare_reports(report, reference, K):
    """Safe groups carry lower bounds; deficits and control decisions are exact."""
    assert np.array_equal(np.maximum(K-np.asarray(report['native_counts']), 0), reference['native_deficits'])
    for key in ('native_deficits','projection_triggered','skip','sweep_count','final_active',
                'final_duals','selected_controls','roundoff_budget_sha256'):
        assert report[key] == reference[key], key
    for key in ('selections', 'sweeps'):
        assert len(report[key]) == len(reference[key]), key
        for a, b in zip(report[key], reference[key]):
            assert np.all(np.asarray(a['counts']) <= b['counts'])
            assert np.array_equal(np.maximum(K-np.asarray(a['counts']), 0), np.maximum(K-np.asarray(b['counts']), 0))
            assert {k:v for k,v in a.items() if k!='counts'} == {k:v for k,v in b.items() if k!='counts'}, key
    assert np.all(np.asarray(report['executed_counts']) >= K)


def source_hashes():
    paths = subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'experiments'],cwd=REPO,text=True).splitlines()
    files = [REPO/q for q in paths if q.endswith('.py')]
    files += list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {q.relative_to(REPO).as_posix():sha(q) for q in sorted(set(files))}
