"""Run the pinned official minimal forgetting demo with observation-only logging."""
import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
import urllib.request
import numpy as np
import torch

ROOT=Path('E:/gfg-evidence/governor-sfp-v1')
SOURCE=Path('E:/gfg-evidence/sfp-source-audit-v1')
MODEL=Path('E:/gfg-models/SmolLM2-135M-Instruct-sfp-v1')
REVISION='e4ef486c169dcc95af52dc504a8fcebcb26f9959'
MODEL_REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'
EXPECTED={
    'forget.py':'ff5a0035450a288a3f038dfbec540d15bebb2c345be294334f1ef7fd75595ebd',
    'model.py':'cd2c2a6b54f9ef299fb126126a29ceb4ed37095b65c92188f7905303bf4da082',
    'data.py':'c2a910ad16eb5bf7794e52be1cf2e3f4372279d87c9a9ebbae92059c15255f31',
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)


def tensor_hash(tensors):
    h=hashlib.sha256()
    for name,t in sorted(tensors.items()):
        t=t.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((tuple(t.shape),t.dtype)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Tee:
    def __init__(self,console,file):self.console=console;self.file=file
    def write(self,value):self.console.write(value);self.file.write(value);return len(value)
    def flush(self):self.console.flush();self.file.flush()
    def __getattr__(self,name):return getattr(self.console,name)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);attempt=ROOT/'attempt-1';attempt.mkdir()
    SOURCE.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}))
    for name in (*EXPECTED,'LICENSE','AGENTS.md','README.md','tasks/math.py','tasks/code.py',
                 'tasks/ifeval.py','tasks/safety.py','tasks/domain.py','configs/demo.yaml'):
        path=SOURCE/name.replace('/','-')
        if not path.exists():
            with opener.open(f'https://raw.githubusercontent.com/paradigmxyz/sfp/{REVISION}/{name}',timeout=120) as response,path.open('xb') as f:
                f.write(response.read())
    for name,expected in EXPECTED.items():assert sha(SOURCE/name)==expected,name
    assert sha(MODEL/'model.safetensors')=='5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c'
    config=dict(official_commit=REVISION,model_revision=MODEL_REVISION,seed=42,
        official_sources={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()},
        model_files={p.name:sha(p) for p in MODEL.iterdir() if p.is_file()},
        wrapper_sha256=sha(Path(__file__)),torch=torch.__version__,python=sys.version,
        policy='unchanged forget.main; only pin local model, seed42, observe outputs/batches/losses, save adapter',
        scope='official 10-question substring-score demo; NOT full SFP benchmark',
        expected_training=dict(steps=200,batch=4,lr=1e-4,lora_rank=16,lora_alpha=128,
            target_modules=['q_proj','v_proj'],max_length=512),
        code_plasticity='NOT_MEASURED_BY_OFFICIAL_DEMO')
    write(attempt/'CONFIG.json',config)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(2)
    sys.path.insert(0,str(SOURCE))
    spec=importlib.util.spec_from_file_location('official_sfp_forget',SOURCE/'forget.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    assert all(Path(sys.modules[n].__file__).resolve()==(SOURCE/f'{n}.py').resolve() for n in ('data','model'))
    saved={};observations=[];batches=[];losses=[];start=time.perf_counter()
    old_load,old_eval,old_task,old_batch=official.load_model,official.eval_math,official.load_task,official.get_batch
    def load(*a,**kw):
        kw['name']=str(MODEL);model,tok=old_load(*a,**kw);saved['model']=model
        saved['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        saved['parameters']=sum(p.numel() for p in model.parameters())
        saved['before_adapter_sha256']=tensor_hash({n:p for n,p in model.named_parameters() if p.requires_grad})
        def observe(module,inputs,output):
            if output.loss is not None:
                value=float(output.loss.detach());losses.append(value)
                if not math.isfinite(value):raise FloatingPointError('official training produced nonfinite loss')
        saved['hook']=model.register_forward_hook(observe)
        return model,tok
    def evaluate(*a,**kw):
        flags,texts=old_eval(*a,**kw)
        assert len(flags)==len(texts)==10
        observations.append(dict(phase='before' if not observations else 'after',
            flags=flags,texts=texts,golds=[gold for _,gold in official.QUESTIONS],
            accuracy=sum(flags)/10,elapsed=time.perf_counter()-start))
        write(attempt/f'{observations[-1]["phase"]}.json',observations[-1])
        return flags,texts
    def task(*a,**kw):
        dataset=old_task(*a,**kw)
        saved['data_count']=len(dataset)
        saved['data_sha256']=hashlib.sha256(json.dumps(dataset,sort_keys=True).encode()).hexdigest()
        return dataset
    def batch(*a,**kw):
        value=old_batch(*a,**kw)
        batches.append(dict(step=len(batches)+1,sha256=tensor_hash(value)))
        return value
    official.load_model=load;official.eval_math=evaluate;official.load_task=task;official.get_batch=batch
    original_argv=sys.argv;sys.argv=['forget.py','--device','cuda']
    try:
        with (attempt/'RUN.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            torch.cuda.reset_peak_memory_stats();official.main()
        assert len(batches)==len(losses)==200 and len(observations)==2
        model=saved['model'];saved['hook'].remove()
        model.save_pretrained(attempt/'adapter')
        after_hash=tensor_hash({n:p for n,p in model.named_parameters() if p.requires_grad})
        result=dict(status='COMPLETED',observations=observations,batches=batches,losses=losses,
            parameters=saved['parameters'],trainable_parameters=saved['trainable_parameters'],
            code_examples=saved['data_count'],code_data_sha256=saved['data_sha256'],
            before_adapter_sha256=saved['before_adapter_sha256'],after_adapter_sha256=after_hash,
            peak_allocated=torch.cuda.max_memory_allocated(),wall=time.perf_counter()-start,
            code_plasticity='NOT_MEASURED: training loss is not capability',
            adapter_files={str(p.relative_to(attempt)):sha(p) for p in (attempt/'adapter').iterdir() if p.is_file()})
    except Exception:
        result=dict(status='FAILED',traceback=traceback.format_exc(),observations=observations,batches=batches,losses=losses,wall=time.perf_counter()-start)
    finally:sys.argv=original_argv
    for name,expected in EXPECTED.items():assert sha(SOURCE/name)==expected
    write(attempt/'RESULT.json',result);print('SFP_RESULT',result['status'],flush=True)
    if result['status']!='COMPLETED':raise RuntimeError(result['traceback'])


if __name__=='__main__':main()
