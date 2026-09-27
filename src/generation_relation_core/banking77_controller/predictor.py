from .runtime import torch,functional

class Predictor:
    def __init__(self,bank,dense_flops):
        self.bank=bank;self.dense_flops=dense_flops;self.calls=0
        # Persistent result storage; forward AD owns its temporary intermediates.
        self.buffers=tuple(bank.x.new_empty((bank.n,77)) for _ in range(3))
    def predict(self,before,native):
        b=self.bank
        with b.meter.region('predictor',True),torch.no_grad():
            delta=tuple(n-p for n,p in zip(native,before))
            for k in range(0,b.padded,256):
                x=b.x[k:k+256];fn=lambda p:functional(p,x)
                f0,f1=torch.func.jvp(fn,(before,),(delta,))
                _,f2=torch.func.jvp(lambda p:torch.func.jvp(fn,(p,),(delta,))[1],(before,),(delta,))
                size=min(256,b.n-k)
                for dst,src in zip(self.buffers,(f0,f1,f2)):dst[k:k+size].copy_(src[:size])
            z=self.buffers[0]+self.buffers[1]+.5*self.buffers[2]
            result=z[:,:b.seen].cpu().numpy().copy()
        self.calls+=1;b.meter.calls['predictor']+=1
        b.meter.flops['predictor_dense']+=self.dense_flops*(b.padded//256)
        return result
