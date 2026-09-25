"""Decoder contract and nanoGPT-specific binding; runtime never names model internals."""
import copy
from dataclasses import dataclass
import torch
from torch.func import functional_call,jvp
from torch.nn.attention import sdpa_kernel,SDPBackend


@dataclass
class Stage:
    label:str
    names:tuple
    forward:object


class GenericDecoderAdapter:
    def __init__(self,embedding,stages,readout_name):
        self.embedding=embedding;self.stages=stages;self.readout_name=readout_name

    @torch.no_grad()
    def jets(self,owner,inputs):
        with owner.stage(self.embedding.label,self.embedding.names) as (p,d):
            def first(values):
                out,tangent=jvp(lambda q:self.embedding.forward(q,inputs),(values,),(d,))
                return tangent,out
            hp,hpp,h=jvp(first,(p,),(d,),has_aux=True);native=h
        for stage in self.stages:
            with owner.stage(stage.label,stage.names) as (p,d):
                native=stage.forward(p,native)
                with sdpa_kernel(SDPBackend.MATH):
                    def first(x,pars,tangent):
                        out,yp=jvp(lambda xx,pp:stage.forward(pp,xx),(x,pars),(tangent,d))
                        return yp,out
                    hp,hpp,h=jvp(first,(h,p,hp),(hp,d,hpp),has_aux=True)
        return native,tuple(x[:,-1,:] for x in (h,hp,hpp))


class NanoGPTAdapter(GenericDecoderAdapter):
    def __init__(self,model):
        if model.config.dropout!=0:raise ValueError('certified contract requires dropout=0')
        names=tuple(dict(model.named_parameters()));en=('transformer.wte.weight','transformer.wpe.weight')
        def embed(p,x):return torch.nn.functional.embedding(x,p[en[0]])+torch.nn.functional.embedding(torch.arange(x.shape[1],device=x.device),p[en[1]])
        stages=[]
        # Stateless META templates have no FSDP hooks. Calling an FSDP module from
        # inside functional_call/JVP would otherwise re-enter its lifecycle hooks.
        for prefix,module in [(f'transformer.h.{i}.',b) for i,b in enumerate(model.transformer.h)]+[('transformer.ln_f.',model.transformer.ln_f)]:
            pure=copy.deepcopy(module).to('meta');ns=tuple(n for n in names if n.startswith(prefix))
            def fn(p,x,pure=pure,prefix=prefix):return functional_call(pure,{n[len(prefix):]:v for n,v in p.items()},(x,),strict=True)
            stages.append(Stage(prefix,ns,fn))
        super().__init__(Stage('embedding',en,embed),stages,en[0])

    def fsdp_units(self,model):
        return {'embedding':model,'transformer.ln_f.':model,**{f'transformer.h.{i}.':b for i,b in enumerate(model.transformer.h)}}

    def wrap_fsdp(self,model,mesh):
        from torch.distributed.fsdp import fully_shard
        for block in model.transformer.h:fully_shard(block,mesh=mesh,reshard_after_forward=True)
        fully_shard(model,mesh=mesh,reshard_after_forward=True)
        return self.fsdp_units(model)
