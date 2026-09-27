"""Frozen Banking77 MLP/AdamW conventions and buffer256 endpoint runtime."""
import time,random
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
EPS=1e-6

def setup():
    random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def sync():torch.cuda.synchronize()

def model():return nn.Sequential(nn.Linear(768,256),nn.GELU(approximate='none'),nn.Linear(256,77)).cuda()

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True,foreach=False)

def params(m):return tuple(v.detach().clone() for v in m.parameters())

def cpu(p):return tuple(v.detach().cpu().numpy().astype(np.float64) for v in p)

def native_step(head,shadow,opt,x,y,seen):
    head.zero_grad(set_to_none=True);loss=F.cross_entropy(head(x)[:,:seen],y);loss.backward();norm=nn.utils.clip_grad_norm_(head.parameters(),1.)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    with torch.no_grad():
        for a,b in zip(head.parameters(),shadow.parameters()):b.copy_(a);b.grad=a.grad.detach().clone()
    opt.step();return float(loss.detach())

def commit(head,p):
    with torch.no_grad():
        for a,b in zip(head.parameters(),p):a.copy_(b)

def functional(p,h):return torch.addmm(p[3],F.gelu(torch.addmm(p[1],h,p[0].T),approximate='none'),p[2].T)

class Meter:
    def __init__(self):
        self.seconds=defaultdict(float);self.flops=defaultdict(int);self.calls=defaultdict(int)
        self.exclusive=defaultdict(float);self.stack=[]
    @contextmanager
    def region(self,name,gpu=False):
        start=time.perf_counter()
        if self.stack:self.exclusive[self.stack[0][0]+'/'+self.stack[-1][0]]+=start-self.stack[-1][1]
        frame=[name,start];self.stack.append(frame)
        try:
            if gpu:sync()
            yield
        finally:
            if gpu:sync()
            end=time.perf_counter();self.exclusive[self.stack[0][0]+'/'+name]+=end-frame[1]
            self.seconds[name]+=end-start;assert self.stack.pop() is frame
            if self.stack:self.stack[-1][1]=end

class Bank:
    def __init__(self,x,y,ids,seen,meter):
        self.n=len(x);self.padded=((self.n+255)//256)*256;self.y=y;self.ids=np.asarray(ids);self.seen=seen;self.meter=meter
        self.x=x.new_zeros((self.padded,768));self.x[:self.n].copy_(x)
        self.hidden=x.new_empty((256,256));self.out=x.new_empty((self.padded,77))
        self.H=x.cpu().numpy().astype(np.float64);self.absH=np.abs(self.H);self.labels=y.cpu().numpy().copy()
        self.inputGram=self.H@self.H.T+1.;self.relation_rows=[np.flatnonzero(self.labels==c) for c in range(seen-7)]
        self.calls=0;self.rows=0;self.pairs=0
    def exact(self,p):
        with self.meter.region('endpoint',True),torch.no_grad():
            for k in range(0,self.padded,256):
                torch.addmm(p[1],self.x[k:k+256],p[0].T,out=self.hidden)
                a=F.gelu(self.hidden,approximate='none')
                torch.addmm(p[3],a,p[2].T,out=self.out[k:k+256])
            z=self.out[:self.n,:self.seen].cpu().numpy().copy()
        self.calls+=1;self.rows+=self.n;self.meter.flops['endpoint']+=2*self.padded*(768*256+256*77)
        return z
    def margins(self,z):
        other=z.copy();other[np.arange(self.n),self.labels]=-np.inf
        return z[np.arange(self.n),self.labels]-other.max(1)
    def counts(self,z):return np.bincount(self.labels[self.margins(z)>0],minlength=self.seen-7)
