"""Configured decoder interface and tested nanoGPT binding; no generic-model claim."""
from dataclasses import dataclass
import torch
from torch.func import functional_call,jvp
from torch.nn.attention import sdpa_kernel,SDPBackend
from experiments.gfg_controller_resident_reuse_v1.engine import shared_derivatives


@dataclass
class Stage:
    label:str
    names:tuple
    forward:object


class ConfiguredDecoderAdapter:
    def __init__(self,embedding,stages,readout_name,native_hidden):
        self.embedding=embedding;self.stages=stages;self.readout_name=readout_name;self.native_hidden=native_hidden

    @torch.no_grad()
    def directional(self,owner,inputs):
        with sdpa_kernel(SDPBackend.MATH):
            with owner.materialize(self.embedding.label,self.embedding.names) as (p,d):
                h,hp,hpp=shared_derivatives(lambda values:self.embedding.forward(values,inputs),p,d)
            for stage in self.stages:
                with owner.materialize(stage.label,stage.names) as (p,d):
                    def first(x,pars,tangent):
                        y,yp=jvp(lambda xx,pp:stage.forward(pp,xx),(x,pars),(tangent,d))
                        return yp,y
                    hp,hpp,h=jvp(first,(h,p,hp),(hp,d,hpp),has_aux=True)
        return tuple(x[:,-1,:] for x in (h,hp,hpp))


class NanoGPTAdapter(ConfiguredDecoderAdapter):
    def __init__(self,model):
        if model.config.dropout!=0:raise ValueError('this certified adapter requires dropout=0')
        names=tuple(dict(model.named_parameters()))
        en=('transformer.wte.weight','transformer.wpe.weight')
        def embedding(p,x):
            pos=torch.arange(x.shape[1],device=x.device)
            return torch.nn.functional.embedding(x,p[en[0]])+torch.nn.functional.embedding(pos,p[en[1]])
        stages=[]
        for prefix,module in [(f'transformer.h.{i}.',m) for i,m in enumerate(model.transformer.h)]+[('transformer.ln_f.',model.transformer.ln_f)]:
            ns=tuple(n for n in names if n.startswith(prefix))
            def forward(p,x,module=module,prefix=prefix):return functional_call(module,{n[len(prefix):]:v for n,v in p.items()},(x,),strict=True)
            stages.append(Stage(prefix,ns,forward))
        def native_hidden(x):
            p=dict(model.named_parameters());h=embedding(p,x)
            for block in model.transformer.h:h=block(h)
            return model.transformer.ln_f(h)
        super().__init__(Stage('embedding',en,embedding),stages,en[0],native_hidden)
