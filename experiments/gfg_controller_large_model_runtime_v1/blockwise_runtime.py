"""Explicit blockwise backend, certified separately from global-JVP runtime."""
import numpy as np
import torch
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel, SDPBackend
from .runtime import RuntimeController, blockwise_hidden, streamed_boundary, shared_derivatives, HiddenStateModel


class BlockwiseController(RuntimeController):
    def response(self,model,inputs,targets):
        parameters={n:p.detach() for n,p in model.named_parameters()};delta=self.buffers.delta
        model.eval()
        with torch.no_grad():
            recorded=functional_call(model,parameters,(inputs,targets),tie_weights=True,strict=True)[0][:,-1,:].cpu().numpy().astype(np.float64)
            with sdpa_kernel(SDPBackend.MATH):
                hs=blockwise_hidden(model,parameters,delta,inputs)
                reference=None
                if self.audit:
                    wrapper=HiddenStateModel(model)
                    prefix={'gpt.'+n:z for n,z in parameters.items()};direction={'gpt.'+n:z for n,z in delta.items()}
                    reference=shared_derivatives(lambda p:functional_call(wrapper,p,(inputs,),tie_weights=True),prefix,direction)
            arrays=torch.stack(hs).cpu().numpy().astype(np.float64)
            w=parameters['transformer.wte.weight'].cpu().numpy();dw=delta['transformer.wte.weight'].cpu().numpy()
            result=streamed_boundary(recorded,targets[:,-1].cpu().numpy(),w,dw,*arrays,retain_coordinates=self.audit)
            if self.audit:
                self.last_debug={'hidden':arrays,'pre_logits':recorded,'minima':result[3],
                    'blockwise_exact':[torch.equal(a,z) for a,z in zip(hs,reference)],
                    'blockwise_max_abs':[float((a-z).abs().max()) for a,z in zip(hs,reference)]}
        return result
