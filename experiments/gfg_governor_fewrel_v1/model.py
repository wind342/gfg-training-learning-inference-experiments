"""Entity-marker BERT classifier and affine-head functional boundary adapter."""
import copy
import torch
from torch import nn
from torch.func import functional_call,jvp
from transformers import BertModel,BertConfig
from experiments.gfg_functional_update_governor_v1.instrument import Measurement,measure_boundaries
from .prepare import MODEL


class RelationModel(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            self.bert=BertModel.from_pretrained(MODEL,local_files_only=True,add_pooling_layer=False,attn_implementation='eager')
            self.bert.resize_token_embeddings(30526,mean_resizing=False)
        else:
            config._attn_implementation='eager'
            self.bert=BertModel(config,add_pooling_layer=False)
        self.head=nn.Linear(2*self.bert.config.hidden_size,80,bias=True)
        self.seen=8

    def features(self,x):
        tokens=x[:,2:];hidden=self.bert(input_ids=tokens,attention_mask=tokens.ne(0))[0]
        return select_entities(hidden,x)

    def forward(self,x,y=None):
        z=self.head(self.features(x))[:,:self.seen]
        loss=nn.functional.cross_entropy(z,y[:,-1]) if y is not None else None
        return z[:,None,:],loss


def select_entities(h,x):
    rows=torch.arange(len(x),device=x.device)
    return torch.cat((h[rows,x[:,0]],h[rows,x[:,1]]),dim=-1)


class BertInstrument:
    def __init__(self,model,chunk=4):
        self.model=model;self.chunk=chunk;self.alphas=(.125,.25,.5,1.,2.)
        self.templates=[]
        for prefix,module in [('bert.embeddings.',model.bert.embeddings)]+[(f'bert.encoder.layer.{i}.',m) for i,m in enumerate(model.bert.encoder.layer)]:
            pure=copy.deepcopy(module).eval().to('meta')
            names=tuple(n for n,_ in model.named_parameters() if n.startswith(prefix))
            self.templates.append((prefix,pure,names))

    @torch.no_grad()
    def jets(self,owner,x):
        tokens=x[:,2:];mask=(1.-tokens.ne(0).float())[:,None,None,:]*torch.finfo(torch.float32).min
        for index,(prefix,module,names) in enumerate(self.templates):
            with owner.stage(prefix,names) as (p,d):
                pars={n[len(prefix):]:v for n,v in p.items()};delta={n[len(prefix):]:v for n,v in d.items()}
                if index==0:
                    def f(q):return functional_call(module,q,(),{'input_ids':tokens,'token_type_ids':torch.zeros_like(tokens),
                        'position_ids':torch.arange(tokens.shape[1],device=tokens.device)[None,:]},strict=False)
                    def first(q):
                        out,der=jvp(f,(q,),(delta,));return der,out
                    hp,hpp,h=jvp(first,(pars,),(delta,),has_aux=True)
                else:
                    def f(hh,q):return functional_call(module,q,(hh,),{'attention_mask':mask},strict=False)[0]
                    def first(hh,q,dh):
                        out,der=jvp(f,(hh,q),(dh,delta));return der,out
                    hp,hpp,h=jvp(first,(h,pars,hp),(hp,delta,hpp),has_aux=True)
        return tuple(select_entities(value,x) for value in (h,hp,hpp))

    @torch.no_grad()
    def measure(self,owner,probes,targets):
        modes=[(m,m.training) for m in self.model.modules()];self.model.eval()
        baseline=[];predicted=[]
        try:
            # Affine head -> homogeneous coordinates. Bias derivative is retained.
            w=self.model.head.weight.detach()[:self.model.seen];b=self.model.head.bias.detach()[:self.model.seen]
            dw=owner.proposal.delta['head.weight'][:self.model.seen];db=owner.proposal.delta['head.bias'][:self.model.seen]
            wa=torch.cat((w,b[:,None]),dim=1);da=torch.cat((dw,db[:,None]),dim=1)
            for begin in range(0,len(probes),self.chunk):
                x=probes[begin:begin+self.chunk];y=targets[begin:begin+self.chunk,-1]
                h,hp,hpp=self.jets(owner,x)
                ones=torch.ones(len(x),1,device=x.device);zero=torch.zeros_like(ones)
                hs=(torch.cat((h,ones),1),torch.cat((hp,zero),1),torch.cat((hpp,zero),1))
                z=nn.functional.linear(h,w,b)
                m=measure_boundaries(z,y,wa,da,hs,alphas=self.alphas,mode='scalarized')
                baseline.append(m.baseline_minima);predicted.append(m.predicted_minima)
            return Measurement(self.alphas,torch.cat(baseline),torch.cat(predicted,dim=1),
                {'adapter':'BERT entity markers, biased affine head via homogeneous coordinates','chunk':self.chunk})
        finally:
            owner.events.clear()
            for module,training in modes:module.training=training
