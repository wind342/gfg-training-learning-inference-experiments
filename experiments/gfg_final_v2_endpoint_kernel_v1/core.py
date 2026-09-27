"""Preallocated fixed-shape endpoints around the unchanged cached controller."""
from experiments.gfg_final_v2_cached_hildreth_v1.core import *
from experiments.gfg_final_v2_cached_hildreth_v1.run import inputs,history
from experiments.gfg_final_architecture_v2_safe_screening_v3.run import load

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE='f357dfdb9d13759dc07a5703d6d11301a3adb946'
ROOT=Path(os.environ.get('GFG_ENDPOINT_ROOT','E:/gfg-evidence/final-v2-endpoint-kernel-v1'))
ARMS={'reference256':('reference',256),'buffer256':('buffer',256),
      'batch256':('batch',256),'batch512':('batch',512),'batch1024':('batch',1024),'batch2048':('batch',2048)}


def history():
    # Use cached full-count reports, not v3's safe-relation count lower bounds.
    with gzip.open('E:/gfg-evidence/final-v2-cached-hildreth-v1/cached_validation/STEPS.jsonl.gz','rt') as f:
        return [json.loads(s) for s in f]


def guard():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',
        ':(exclude)experiments/gfg_final_v2_endpoint_kernel_v1',
        ':(exclude)experiments/gfg_final_v2_hotstart_active_qp_v1'],cwd=REPO,check=True,capture_output=True)


class EndpointKernel:
    def __init__(self,h,kind,tile,classes=80):
        self.h=h;self.kind=kind;self.tile=tile;self.classes=classes
        self.tiles=(len(h)+tile-1)//tile;self.padded=self.tiles*tile
        if kind!='reference':
            self.bank=h.new_zeros((self.padded,h.shape[1]));self.bank[:len(h)].copy_(h)
            self.out=h.new_empty((self.padded,classes))
            self.blocks=self.bank.view(self.tiles,tile,h.shape[1])
            self.output_blocks=self.out.view(self.tiles,tile,classes)

    @torch.no_grad()
    def __call__(self,state):
        if self.kind=='reference':return fixed_affine(state,self.h,self.classes)
        weight=state['weight'][:self.classes].T
        bias=state['bias'][:self.classes]
        if self.kind=='buffer':
            for j in range(self.tiles):
                torch.addmm(bias,self.blocks[j],weight,out=self.output_blocks[j])
        else:
            # Only views are constructed per endpoint; data buffers are fixed.
            torch.baddbmm(bias.view(1,1,-1).expand(self.tiles,self.tile,self.classes),
                self.blocks,weight.unsqueeze(0).expand(self.tiles,-1,-1),out=self.output_blocks)
        return self.out[:len(self.h)]


class EndpointMeter(Meter):
    def __init__(self,h,kind,tile,profile=False):
        super().__init__(profile);self.kernel=EndpointKernel(h,kind,tile);self.api_calls=0
    def affine(self,state,h,seen):
        assert h.data_ptr()==self.kernel.h.data_ptr() and len(h)==len(self.kernel.h)
        with self.region('endpoint_forward',True):z=self.kernel(state)
        k=self.kernel;self.rows+=len(h);self.padded+=k.padded;self.gemms+=k.tiles;self.calls+=1
        self.api_calls+=1 if k.kind=='batch' else k.tiles
        return z


def same_report(a,b):
    return {k:a[k]==b[k] for k in ('native_deficits','selected_controls','final_active','final_duals','sweep_count','skip','sweeps','selections')}


class Observer:
    def __init__(self,endpoint,K,ids,stream):
        self.endpoint=endpoint;self.K=K;self.ids=ids;self.stream=stream;self.step=0
        self.calls=self.signs=self.old_signs=self.pred_diffs=self.count_diff_calls=self.choice_diff_calls=0
        self.executed_signs=self.reference_floor=0;self.max_error=0.
    def evaluate(self,state,selected,label):
        r=self.endpoint.evaluate(state,selected,label);g=self.endpoint.g
        with torch.no_grad():
            z=fixed_affine(state,g.h,g.seen);m=margins(z,g.y)
            diff=(r['m']>0)!=(m>0);C=counts(m[:g.n].cpu().numpy(),g.labels,g.groups)
            _,_,chosen=select_exact(r,g,self.K,self.ids)
            _,_,truth=choose(m[:g.n].cpu().numpy(),g.labels,self.K,self.ids)
            err=float((r['z']-z).abs().max());sign=int(diff.sum());oldsign=int(diff[:g.n].sum())
            pred=int((r['z'].argmax(1)!=z.argmax(1)).sum())
            floor=int((C<self.K).sum()) if label=='commit' else None
            self.calls+=1;self.signs+=sign;self.old_signs+=oldsign;self.pred_diffs+=pred
            self.max_error=max(self.max_error,err);self.count_diff_calls+=int(not np.array_equal(r['C'],C))
            self.choice_diff_calls+=int(chosen!=truth)
            if label=='commit':self.executed_signs+=oldsign;self.reference_floor+=floor
            row=dict(step=self.step,point=label,sign_differences=sign,old_sign_differences=oldsign,
                     argmax_differences=pred,max_logit_difference=err,counts_identical=bool(np.array_equal(r['C'],C)),
                     choices_identical=chosen==truth,reference_floor_violations=floor)
            self.stream.write(json.dumps(row,separators=(',',':'))+'\n')
        return r


def numerical_preflight(data,arm):
    d,F,initial,h,y,ids,n,K=data;kind,tile=ARMS[arm]
    head,shadow,opt=make_state(initial);before=p.snapshot(head)
    first=order(d,10)[0];ce_grad(head,F[first],torch.from_numpy(d['y'][first]).cuda(),80)
    nn.utils.clip_grad_norm_(head.parameters(),1.);propose(head,shadow,opt)
    native=p.snapshot(shadow)
    final=torch.load(Path('E:/gfg-evidence/final-v2-cached-hildreth-v1/cached_validation/CHECKPOINT.pt'),weights_only=False,map_location='cuda')['head']
    kernel=EndpointKernel(h,kind,tile);checks=[]
    for name,state in (('receiving',before),('first_native',native),('historical_final',final)):
        z=kernel(state).clone();again=kernel(state).clone();ref=fixed_affine(state,h,80)
        assert torch.equal(z,again) and bool(torch.isfinite(z).all())
        m=margins(z,y);rm=margins(ref,y)
        C=counts(m[:n].cpu().numpy(),y[:n].cpu().numpy(),72)
        checks.append(dict(state=name,repeatable=True,logits_identical=bool(torch.equal(z,ref)),
            max_logit_difference=float((z-ref).abs().max()),max_margin_difference=float((m-rm).abs().max()),
            sign_differences=int(((m>0)!=(rm>0)).sum()),floor_violations=int((C<K).sum())))
    assert checks[0]['floor_violations']==0,'initial receiving floor fails; K must not be changed'
    # FP64 error sanity check on fixed deterministic sample, not a new controller.
    ix=torch.arange(0,len(h),max(1,len(h)//64),device=h.device)
    V=augmented_gpu(native,80);X=torch.cat((h[ix].double(),h.new_ones((len(ix),1)).double()),1)
    ref64=X@V.T;z=kernel(native)[ix].double()
    k=h.shape[1]+3;u=torch.finfo(torch.float32).eps/2;gm=k*u/(1-k*u)
    bound=gm*(X.abs()@V.abs().T)+1e-12
    assert bool(((z-ref64).abs()<=bound).all())
    return dict(status='PASS',arm=arm,tile=tile,checks=checks,fp64_error_bound_sample_rows=len(ix),
                input_padding_once=kind!='reference',persistent_output=kind!='reference')
