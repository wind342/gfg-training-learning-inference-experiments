"""Standalone shared conventions, reading only verified original authorities."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,hashlib,time,copy,random,subprocess,gzip,platform
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
from scipy.special import erf
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
ROOT=Path(os.environ.get('GFG_PREDICTOR_TASK2_ROOT','E:/gfg-evidence/banking77-final-v2-predictor-task2-v1'))
OLD=Path('E:/gfg-evidence/banking77-nonlinear-v1');BASE='bd7d29815b78c7b048b863a337d67a246d18b5fe';EPS=1e-6
V1=Path('E:/gfg-evidence/banking77-jacobian-hildreth-v1')

def phash(p):return thash({str(i):v for i,v in enumerate(p)})
def nhash(p):return hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in p)).hexdigest()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:torch.save(v,f)
def thash(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def setup():
    random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def sync():torch.cuda.synchronize()
def guard():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)
    if (ROOT/'FREEZE.json').exists():assert sha(HERE/'PROTOCOL.md')==read(ROOT/'FREEZE.json')['protocol_sha256']
def model():return nn.Sequential(nn.Linear(768,256),nn.GELU(approximate='none'),nn.Linear(256,77)).cuda()
def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True,foreach=False)
def params(m):return tuple(v.detach().clone() for v in m.parameters())
def cpu(p):return tuple(v.detach().cpu().numpy().astype(np.float64) for v in p)
def opt_hash(o):return thash({f'{i}.{k}':v for i,s in enumerate(o.state.values()) for k,v in s.items() if isinstance(v,torch.Tensor)})
def opt_steps(o):return [int(s['step']) for s in o.state.values()]
def schedule(ids,task,budget):
    rng=np.random.default_rng(2021+2000+task);v=[]
    while len(v)<3360:v.extend(rng.permutation(ids).tolist())
    return np.tile(np.array(v[:3360]).reshape(210,16),(budget//210,1))
def load():
    c=read(OLD/'CACHE.json');assert sha(OLD/'TRAIN_FEATURES.pt')==c['feature_file_sha256']
    d=torch.load(OLD/'TRAIN_FEATURES.pt',weights_only=True);assert thash({'features':d['features']})==c['feature_tensor_sha256']
    return d['features'].cuda(),d['labels'].cuda(),d['control'].numpy()
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
    def __init__(self):self.seconds=defaultdict(float);self.flops=defaultdict(int);self.calls=defaultdict(int)
    @contextmanager
    def region(self,name,gpu=False):
        if gpu:sync()
        t=time.perf_counter()
        try:yield
        finally:
            if gpu:sync()
            self.seconds[name]+=time.perf_counter()-t

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

def metrics(head,x,y,seen):
    with torch.no_grad():
        z=head(x)[:,:seen];pred=z.argmax(1);yy=y.cpu().numpy();acc={str(c):float((pred[y==int(c)]==int(c)).float().mean()) for c in np.unique(yy)}
        return dict(ce=float(F.cross_entropy(z,y)),macro=float(np.mean(list(acc.values()))),per_intent=acc)
