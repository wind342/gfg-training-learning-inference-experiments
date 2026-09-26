"""Reuse frozen head-only semantics; only LR and evaluation schedule differ."""
from experiments.gfg_head_only_exact_monitor_smoke_v1.core import *
from experiments.gfg_head_only_exact_monitor_smoke_v1.core import ROOT as BASE_ROOT
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('E:/gfg-evidence/head-only-lr-smoke-v1')
BASE='c6490aa26f83d046e0101dc2661ab8dae750cb6a'
ARMS={'lr5e-5':5e-5,'lr5e-4':5e-4,'lr1e-3':1e-3}
BASELINE='lr5e-6'

def unchanged():
    subprocess.run(['git','diff','--exit-code',BASE,'--','.',':(exclude)experiments/'+HERE.name],cwd=REPO,check=True,capture_output=True)

def sources():
    paths=[REPO/p for p in subprocess.check_output(['git','ls-files','experiments'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    paths+=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(set(paths))}

def set_lr(opt,lr):
    before=moments_hash(opt)
    for group in opt.param_groups:group['lr']=lr
    assert before==moments_hash(opt)

def setup():
    torch.set_num_threads(2);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def guard():
    unchanged();pre=json.loads((ROOT/'PREFLIGHT.json').read_text())
    assert pre['status']=='PASS' and pre['source_hashes']==sources()
    assert sha(DATA/'data.npz')==DATA_SHA and sha(CHECKPOINT)==CKSHA
    for path,h in pre['baseline_files'].items():assert sha(path)==h
    return pre
