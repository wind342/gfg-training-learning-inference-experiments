"""Two-sided transition bounds anchored ONLY on real executed endpoints."""
from .core import *


class TransitionEndpoint(FullEndpoint):
    def __init__(self,g,K):
        super().__init__(g);self.K=K;self.anchor=None
        assert g.h.dtype==torch.float32 and not torch.backends.cuda.matmul.allow_tf32
        assert bool(torch.isfinite(g.h).all()) and float(g.h.abs().max())<=1e10
        self.refresh_rows=0;self.cached_receiving_calls=0;self.anchor_updates=0
        self.positive=self.negative=self.ambiguous=0
        self.k=8*(g.h.shape[1]+1)+64
        self.g64,self.g32=gamma(self.k,torch.float64),gamma(self.k,torch.float32)
        with g.meter.region('transition_setup',True):
            X=torch.cat((g.h[:g.n].double(),g.h.new_ones((g.n,1)).double()),1)
            self.q=upper_norm(X,self.g64)
        self.z=g.h.new_empty((len(g.h),g.seen));self.m=g.h.new_empty(len(g.h))
        self.call=0

    def reserve(self,norms):
        scales=norms[self.g.y[:self.g.n]]+norms.max()
        return self.g32*self.q*scales+self.k*torch.finfo(torch.float32).tiny*(1+self.q+scales)

    @torch.no_grad()
    def set_anchor(self,state,r):
        # Called only at initial receiving head or AFTER actual head commit.
        with self.g.meter.region('anchor_refresh',True):
            self.anchorV=augmented_gpu(state,self.g.seen)
            assert bool(torch.isfinite(self.anchorV).all()) and float(self.anchorV.abs().max())<=1e10
            self.anchorNorm=upper_norm(self.anchorV,self.g64)
            self.anchorReserve=self.reserve(self.anchorNorm)
            self.anchorMargin=r['m'][:self.g.n].double().clone()
            self.anchor={k:v.clone() if torch.is_tensor(v) else v.copy() if isinstance(v,np.ndarray) else v for k,v in r.items()}
            self.anchor_updates+=1

    def relation_counts(self,correct):
        return torch.zeros(self.g.groups,dtype=torch.long,device=self.g.h.device).scatter_add_(0,self.g.y[:self.g.n],correct[:self.g.n].long())

    @torch.no_grad()
    def evaluate(self,state,selected=(),label=''):
        g=self.g;self.call+=1
        if label=='receiving' and self.anchor is not None:
            self.cached_receiving_calls+=1
            return self.anchor
        if self.anchor is None or label=='commit':
            r=super().evaluate(state,selected,label)
            r.update(positive=torch.zeros(g.n,dtype=torch.bool,device=g.h.device),negative=torch.zeros(g.n,dtype=torch.bool,device=g.h.device))
            if label=='commit':self.refresh_rows+=g.n
            self.set_anchor(state,r)
            return r
        with g.meter.region('transition_bounds',True):
            V=augmented_gpu(state,g.seen);vn=upper_norm(V,self.g64)
            dn=upper_norm(V-self.anchorV,self.g64)
            dn+=2*self.g64*(vn+self.anchorNorm)
            dn=torch.nextafter(dn,torch.full_like(dn,torch.inf))
            movement=self.q*(dn[g.y[:g.n]]+dn.max())
            reserve=self.anchorReserve+self.reserve(vn)
            B=movement+reserve
            B+=8*self.g64*(1+self.anchorMargin.abs()+movement.abs()+reserve.abs())
            B=torch.nextafter(B,torch.full_like(B,torch.inf))
            lower=torch.nextafter(self.anchorMargin-B,torch.full_like(B,-torch.inf))
            upper=torch.nextafter(self.anchorMargin+B,torch.full_like(B,torch.inf))
            domain=torch.isfinite(V).all() & (V.abs().max()<=1e10)
            positive=domain & torch.isfinite(lower) & (lower>0)
            negative=domain & torch.isfinite(upper) & (upper<=0)
            ambiguous=~(positive|negative)
        self.z.fill_(torch.nan);self.m.fill_(torch.nan)
        exact=torch.zeros(len(g.h),dtype=torch.bool,device=g.h.device)
        correct=torch.zeros_like(exact);correct[:g.n]=positive

        def execute(mask):
            with g.meter.region('transition_gather',True):
                ix=torch.nonzero(mask & ~exact).flatten();gathered=g.h[ix]
            if not len(ix):return
            zz=g.meter.affine(state,gathered,g.seen)
            with g.meter.region('transition_scatter',True):
                mm=margins(zz,g.y[ix]);self.z[ix]=zz;self.m[ix]=mm
                correct[ix]=mm>0;exact[ix]=True
                self.old_rows+=int((ix<g.n).sum())

        with g.meter.region('transition_masks',True):
            mask=torch.cat((ambiguous,ambiguous.new_ones(len(g.h)-g.n)))
            if selected:mask[sorted(selected)]=True
        execute(mask)
        with g.meter.region('transition_masks',True):
            # Both stable signs plus exact transitions give COMPLETE counts.
            C=self.relation_counts(correct)
            deficit=C<torch.as_tensor(self.K,device=g.h.device)
            # Resolve full margins for any deficit; never rank stable-negative
            # rows by stale anchor margins or by their upper/lower bounds.
            mask=torch.cat((deficit[g.y[:g.n]],ambiguous.new_zeros(len(g.h)-g.n)))
        execute(mask)
        with g.meter.region('transition_masks',True):
            C=self.relation_counts(correct).cpu().numpy()
        return dict(z=self.z,m=self.m,correct=correct,exact_mask=exact,C=C,positive=positive,negative=negative,
                    lower=lower,upper=upper)


class Observer:
    """Read-only full observer: values are never passed into screen/cache."""
    def __init__(self,endpoint,K,ids,stream):
        self.endpoint,self.K,self.ids,self.stream=endpoint,K,ids,stream
        self.calls=self.false_safe=self.false_negative=0;self.step=0
        self.rows=self.positive=self.negative=self.ambiguous=0

    def evaluate(self,state,selected,label):
        r=self.endpoint.evaluate(state,selected,label);g=self.endpoint.g
        with torch.no_grad():
            z=fixed_affine(state,g.h,g.seen);m=margins(z,g.y);correct=m>0
            C=counts(m[:g.n].cpu().numpy(),g.labels,g.groups)
            false_safe=int((r['positive'] & ~correct[:g.n]).sum())
            false_negative=int((r['negative'] & correct[:g.n]).sum())
            exact=r['exact_mask'];error=float((r['m'][exact]-m[exact]).abs().max()) if bool(exact.any()) else 0.
            _,D,chosen=select_exact(r,g,self.K,self.ids)
            _,rd,rchosen=choose(m[:g.n].cpu().numpy(),g.labels,self.K,self.ids)
            assert np.array_equal(r['C'],C) and np.array_equal(D,rd),(self.step,label,'counts')
            assert chosen==rchosen and all(bool(exact[i]) for i in selected),(self.step,label,'selection')
            assert false_safe==0 and false_negative==0 and error==0.,(self.step,label,'certificate')
            positive=int(r['positive'].sum());negative=int(r['negative'].sum())
            row=dict(step=self.step,point=label,complete_counts_identical=True,deficits_identical=True,
                choices_identical=True,false_safe=false_safe,false_wrong=false_negative,max_exact_margin_error=error,
                certified_positive=positive,certified_negative=negative,ambiguous=g.n-positive-negative,
                exact_old_rows=int(exact[:g.n].sum()),counts=C.tolist(),candidate_sha256=tensor_hash(state))
            self.stream.write(json.dumps(row,separators=(',',':'))+'\n');self.stream.flush()
            self.calls+=1;self.positive+=positive;self.negative+=negative;self.ambiguous+=row['ambiguous']
        return r
