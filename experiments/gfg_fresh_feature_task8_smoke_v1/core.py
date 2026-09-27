"""Joint-coordinate adapter; historical production functions remain untouched."""
import types
from experiments.gfg_final_architecture_v1_ten_task import core as old
from experiments.gfg_head_only_hildreth_roundoff_smoke_v1 import core as proj
from experiments.gfg_head_only_exact_monitor_smoke_v1.core import *
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/fresh-feature-task8-smoke-v1')
BASE='115534a09cd554fcd75fd46dde711fdb98988519'
ARMS=('linear_fresh','relu_fresh')
DIM=1536
EXTRA=128
LR=5e-5
TASK=8
SEEN=64

def setup():
    proj.setup()

def random_map():
    return torch.randn(EXTRA,DIM,generator=torch.Generator().manual_seed(2021))/DIM**.5

def joint_features(h,R,arm):
    extra=nn.functional.linear(h,R)
    if arm=='relu_fresh':extra=extra.relu()
    return torch.cat((h,extra),dim=1)

def joint_affine(state,h,seen=SEEN):
    a=nn.functional.linear(h[:,:DIM].contiguous(),state['weight'][:,:DIM].contiguous(),state['bias'])
    b=nn.functional.linear(h[:,DIM:].contiguous(),state['weight'][:,DIM:].contiguous())
    return (a+b)[:,:seen]

# Preserve the entire frozen projection algorithm; only actual forward is adapted.
_raw_projection=proj.hildreth.__wrapped__
_globals=dict(_raw_projection.__globals__)
_globals['affine']=joint_affine
hildreth=torch.no_grad()(types.FunctionType(_raw_projection.__code__,_globals,'joint_hildreth',_raw_projection.__defaults__))
assert hildreth.__wrapped__.__code__ is _raw_projection.__code__

class JointHead(nn.Module):
    def __init__(self,state):
        super().__init__()
        self.weight=nn.Parameter(state['weight'].clone())
        self.bias=nn.Parameter(state['bias'].clone())
        self.residual=nn.Parameter(torch.zeros(80,EXTRA,device=self.weight.device))

    def forward(self,h):
        return nn.functional.linear(h[:,:DIM].contiguous(),self.weight,self.bias)+nn.functional.linear(h[:,DIM:].contiguous(),self.residual)

def pack(head):
    return {'weight':torch.cat((head.weight.detach(),head.residual.detach()),dim=1).clone(),
            'bias':head.bias.detach().clone()}

@torch.no_grad()
def unpack(head,state):
    head.weight.copy_(state['weight'][:,:DIM]);head.residual.copy_(state['weight'][:,DIM:]);head.bias.copy_(state['bias'])

def create(state,moments):
    head=JointHead(state).cuda();shadow=copy.deepcopy(head)
    opt=torch.optim.AdamW(shadow.parameters(),lr=LR,betas=(.9,.999),weight_decay=0.,fused=True,foreach=False)
    ids=moments['param_groups'][0]['params'];assert len(ids)==2
    for p,k in zip(list(shadow.parameters())[:2],ids):
        opt.state[p]={name:v.cuda().clone() if torch.is_tensor(v) else copy.deepcopy(v) for name,v in moments['state'][k].items()}
    opt.state[shadow.residual]={'step':torch.tensor(0.,device='cuda'),
         'exp_avg':torch.zeros_like(shadow.residual),'exp_avg_sq':torch.zeros_like(shadow.residual)}
    assert [int(opt.state[p]['step']) for p in shadow.parameters()]==[1470,1470,0]
    for p,k in zip(list(shadow.parameters())[:2],ids):assert equal(opt.state[p],moments['state'][k])
    return head,shadow,opt

def snapshots(opt):
    return {str(i)+':'+k:v.detach().cpu().clone() for i,s in opt.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)}

def hashes():
    files=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    files+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(files))}

def guard(pre):
    assert hashes()==pre['sources']
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def save_tensor(p,x):
    with p.open('xb') as f:torch.save(x,f)

def check_safety(before,after,h,y,n):
    b=margins(joint_affine(before,h),y);a=margins(joint_affine(after,h),y)
    return int(((b[:n]>0)&(a[:n]<=0)).sum())
