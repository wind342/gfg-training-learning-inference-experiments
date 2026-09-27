"""Independent controller: immutable geometry and lazy, exact pair reserves."""
from experiments.gfg_final_architecture_v2_safe_screening_v2.core import *
from contextlib import contextmanager
from collections import defaultdict
import ast
import textwrap
import os

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path(os.environ.get('GFG_CACHED_HILDRETH_ROOT','E:/gfg-evidence/final-v2-cached-hildreth-v1'))
BASE='8f00816a6b8c6d61ee87688b242497f95a00d306'
OLD_ROOT=Path('E:/gfg-evidence/final-v2-safe-screening-v3')


class Meter:
    def __init__(self, enabled=False):
        self.enabled=enabled;self.seconds=defaultdict(float)
        self.rows=self.gemms=self.padded=self.calls=0

    @contextmanager
    def region(self,name,gpu=False):
        if not self.enabled:
            yield;return
        if gpu:sync()
        start=time.perf_counter()
        try:yield
        finally:
            if gpu:sync()
            self.seconds[name]+=time.perf_counter()-start

    def affine(self,state,h,seen):
        with self.region('endpoint_forward',True):
            z=fixed_affine(state,h,seen)
        self.rows+=len(h);self.gemms+=(len(h)+TILE-1)//TILE
        self.padded+=TILE*((len(h)+TILE-1)//TILE);self.calls+=1
        return z


class Geometry:
    def __init__(self,h,y,n,groups,seen,meter):
        self.h,self.y,self.n,self.groups,self.seen=h,y,n,groups,seen
        self.meter=meter
        with meter.region('cpu_geometry',True):
            self.H=np.column_stack((h[:n].cpu().numpy().astype(np.float64),np.ones(n)))
            self.absH=np.abs(self.H)
            self.lengths=2*np.sum(self.H*self.H,axis=1)
            self.labels=y[:n].cpu().numpy().copy()
            self.relation_rows=tuple(np.flatnonzero(self.labels==r) for r in range(groups))
            for v in (self.H,self.absH,self.lengths,self.labels,*self.relation_rows):v.flags.writeable=False
        self.builds=1


class PairReserve:
    """Only active-pair values; padded GEMM keeps the reference reduction path.

    Each tile contains <=8 active pairs and their <=16 required class rows.
    Repeat these class rows to width 80 and feature rows to height 8. No
    unrequested class vector is loaded; cross-products/padding are explicitly
    counted. This is an empirically verified backend-specific arithmetic
    contract, not a universal promise about arbitrary BLAS installations.
    """
    def __init__(self,before,native,g):
        self.g=g;self.values={};self.pair_values={};self.tiles=0
        self.computed_entries=0
        with g.meter.region('budget_construction',True):
            self.native=p.augmented(native,g.seen)
            self.envelope=np.maximum(np.abs(p.augmented(before,g.seen)),np.abs(self.native))
            u=float(np.finfo(np.float32).eps)/2;k=g.H.shape[1]+2
            self.gamma=(k*u)/(1-k*u)
            self.a=np.empty((8,g.H.shape[1]),dtype=np.float64)
            self.e=np.empty((80,g.H.shape[1]),dtype=np.float64)

    def add(self,pairs):
        new=[pair for pair in pairs if pair not in self.values]
        with self.g.meter.region('budget_construction'):
            for off in range(0,len(new),8):
                tile=new[off:off+8];ix=np.array([i for i,c in tile])
                classes=np.array([c for i,c in tile])
                targets=self.g.labels[ix]
                self.a[:len(tile)]=self.g.absH[ix]
                self.a[len(tile):]=self.a[0]
                cs=np.concatenate((targets,classes))
                # Repetition is numerical-shape padding, not a changed formula.
                self.e[:]=self.envelope[np.resize(cs,80)]
                values=self.gamma*(self.a@self.e.T)
                for j,pair in enumerate(tile):
                    bt,bc=values[j,j],values[j,j+len(tile)]
                    self.pair_values[pair]=(float(bt),float(bc))
                    self.values[pair]=float(bt+bc)
                self.tiles+=1;self.computed_entries+=8*80


class FullEndpoint:
    def __init__(self,g):
        self.g=g;self.old_rows=0

    def evaluate(self,state,selected=(),label=''):
        g=self.g;z=g.meter.affine(state,g.h,g.seen);m=margins(z,g.y)
        self.old_rows+=g.n
        return dict(z=z,m=m,correct=m>0,exact_mask=torch.ones(len(g.h),dtype=torch.bool,device=g.h.device),
                    C=counts(m[:g.n].cpu().numpy(),g.labels,g.groups))


def select_exact(r,g,K,ids):
    C=r['C'];D=np.maximum(K-C,0);chosen={}
    if D.any():
        m=r['m'][:g.n].cpu().numpy()
        for relation in np.flatnonzero(D):
            ix=g.relation_rows[relation]
            assert np.isfinite(m[ix]).all()
            wrong=ix[m[ix]<=0].tolist()
            chosen[str(relation)]=sorted(wrong,key=lambda i:(-float(m[i]),int(ids[i])))[:int(D[relation])]
            assert len(chosen[str(relation)])==D[relation]
    return C,D,chosen


@torch.no_grad()
def projection(before,native,g,K,ids,evaluate,audit=None):
    b=evaluate(before,set(),'receiving');assert np.all(b['C']>=K)
    r=evaluate(native,set(),'native');native_counts,deficits,_=select_exact(r,g,K,ids)
    candidate=native;active=[];dual={};selected=set();selections=[];sweeps=[];reserve=None
    if deficits.any():
        reserve=PairReserve(before,native,g);P=reserve.native.copy()

        def add(sweepno):
            nonlocal active
            C,D,chosen=select_exact(r,g,K,ids)
            selected.update(i for values in chosen.values() for i in values)
            mask=torch.zeros(g.n,dtype=torch.bool,device=g.h.device)
            if selected:mask[sorted(selected)]=True
            assert bool(r['exact_mask'][:g.n][mask].all())
            bad=p.failing_pairs(r['z'],g.y,mask)
            new=sorted(set(bad)-set(dual))
            for pair in new:dual[pair]=0.
            active=sorted(dual);reserve.add(new)
            if audit is not None:audit(reserve,new)
            selections.append(dict(after_sweep=sweepno,counts=C.tolist(),deficits=D.tolist(),chosen=chosen,
                                   selected_controls=sorted(selected),added_pairs=[list(v) for v in new]))

        add(0)
        for sweep in range(1,101):
            with g.meter.region('hildreth_coordinate'):
                for i,c in active:
                    target=int(g.labels[i])
                    dual[(i,c)]=p.coordinate(P,g.H,g.lengths,i,target,c,dual[(i,c)],reserve.values[(i,c)])
            with g.meter.region('materialization',True):candidate=p.materialize(P,native)
            r=evaluate(candidate,selected,'sweep'+str(sweep));safe=bool(np.all(r['C']>=K))
            sweeps.append(dict(sweep=sweep,counts=r['C'].tolist(),safe=safe,constraints=len(active),
                primal_sha256=p.array_hash(P),dual_sha256=p.array_hash(np.array([dual[v] for v in active],dtype=np.float64))))
            if safe:break
            if sweep<100:add(sweep)
    failed=bool(np.any(r['C']<K));executed=before if failed else candidate
    report=dict(native_counts=native_counts.tolist(),native_deficits=deficits.tolist(),projection_triggered=bool(deficits.any()),
        skip=failed,selections=selections,sweeps=sweeps,sweep_count=len(sweeps),final_active=[list(v) for v in active],
        final_duals=[dual[v] for v in active],selected_controls=sorted(selected),executed_counts=(b if failed else r)['C'].tolist(),
        pair_reserve_sha256=p.array_hash(np.array([reserve.pair_values[v] for v in active],dtype=np.float64)) if reserve else None,
        useful_budget_entries=2*len(active),padded_budget_entries=reserve.computed_entries if reserve else 0,
        budget_tiles=reserve.tiles if reserve else 0)
    return executed,report


def compare(report,reference):
    for key in ('native_counts','native_deficits','projection_triggered','skip','selections','sweeps','sweep_count',
                'final_active','final_duals','selected_controls','executed_counts'):
        assert report[key]==reference[key],key


def instrument_reference(meter):
    """Wrap the untouched reference AST in timers, without changing expressions."""
    original=inspect.unwrap(floor_projection)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree.body[0].decorator_list=[]
    class Wrap(ast.NodeTransformer):
        def visit_Assign(self,node):
            text=ast.unparse(node)
            if text.startswith(('P = p.augmented','H = np.column_stack','lengths = 2 *')):
                name='cpu_geometry'
            elif text.startswith('budgets = p.logit_roundoff_budgets'):name='budget_construction'
            else:return node
            return self.block(node,name)
        def visit_For(self,node):
            if ast.unparse(node.target)=='(i, c)' and ast.unparse(node.iter)=='active':
                return self.block(node,'hildreth_coordinate')
            return self.generic_visit(node)
        def block(self,node,name):
            return ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Attribute(value=ast.Name(id='_meter',ctx=ast.Load()),attr='region',ctx=ast.Load()),
                args=[ast.Constant(name)],keywords=[]))],body=[node])
    tree=ast.fix_missing_locations(Wrap().visit(tree))
    ns=dict(original.__globals__);proxy=dict(vars(ns['p']))
    proxy['affine']=meter.affine
    def budget(before,native,H,seen):
        value=p.logit_roundoff_budgets(before,native,H,seen)
        meter.reference_budget=value
        return value
    proxy['logit_roundoff_budgets']=budget
    def materialize(P,native):
        with meter.region('materialization',True):return p.materialize(P,native)
    proxy['materialize']=materialize;ns['p']=SimpleNamespace(**proxy);ns['_meter']=meter
    exec(compile(tree,'<instrumented unchanged floor_projection>','exec'),ns)
    return torch.no_grad()(ns[original.__name__])
