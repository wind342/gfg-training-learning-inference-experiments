"""Audit I/O only; not imported by standalone runtime."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import torch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
OUT=Path(os.environ.get('GFG_V2_OUT','E:/gfg-evidence/controller-distributed-runtime-v2/attempt-1'))
BASE='ec891393178e52ab5f0e5e2571e0f3880dfa7e13'
TAG='paper-experiments-cross-system-feedback-release'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def setup():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def seal():
    assert git('rev-parse','HEAD')==BASE and not OUT.exists()
    hardware={'os':platform.platform(),'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu_count':torch.cuda.device_count(),'gpus':[{'name':torch.cuda.get_device_name(i),'vram':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
        'nccl':torch.distributed.is_nccl_available(),'gloo':torch.distributed.is_gloo_available(),
        'dtensor':importlib.util.find_spec('torch.distributed.tensor') is not None,'fsdp2':hasattr(__import__('torch.distributed.fsdp',fromlist=['fully_shard']),'fully_shard'),
        'triton':importlib.util.find_spec('triton') is not None,'torch_compile_api':hasattr(torch,'compile')}
    value={'base':BASE,'time_ns':time.time_ns(),'hardware':hardware,'sources':{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.suffix in ('.py','.md')},
        'old_files':{n:sha(REPO/n) for n in git('ls-files').splitlines()},'tag_object':git('rev-parse',TAG),'tag_commit':git('rev-parse',TAG+'^{}')}
    write(OUT/'ARCHITECTURE_SEAL.json',value);return value
