"""Independent three-task entry point; no Router V3 policy changes."""
import argparse
import sys
import time
from pathlib import Path
from experiments.gfg_functional_feedback_router_v3 import run as v3
from .gate import HERE,ROOT,BASE,admit

def plain(value):return value.tolist() if hasattr(value,'tolist') else list(value)

def preserving_extend(previous_extend,engine,*args):
    old=engine.scheduler
    before={f:plain(getattr(old,f)) for f in ('state','level','anchor','capabilities','q10','last_observed')}
    labels=list(old.mapping.labels);step=old.last_step
    previous_extend(engine,*args)
    new=engine.scheduler
    for f,values in before.items():
        if plain(getattr(new,f)[:len(labels)])!=values:raise AssertionError('state changed at boundary: '+f)
    if new.last_step!=step:raise AssertionError('step changed at boundary')
    return dict(old_labels=labels,new_labels=list(new.mapping.labels),preserved=before,last_step=step)

def imported_sources(sha):
    result={}
    for module in tuple(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if not f:continue
        p=Path(f).resolve()
        if p.suffix=='.py' and p.is_relative_to(HERE.parent):
            result[str(p.relative_to(HERE.parent))]=sha(p)
    return result

def setup(gate):
    base=v3.setup();base.ROOT=ROOT;base.HERE=HERE;base.CORE=HERE
    previous_write=base.write;boundaries=[];task_evaluations=[]
    def write(path,obj):
        if path.name=='CONFIG.json':
            obj=dict(obj,continuation_base=BASE,admission=gate,
                imported_sources=imported_sources(base.sha),
                python=sys.version,torch_version=base.torch.__version__,gpu=base.torch.cuda.get_device_name(0))
        if path.name=='RESULT.json':
            obj=dict(obj,task_boundary_audit=boundaries,evaluation_seconds=task_evaluations)
        return previous_write(path,obj)
    base.write=write
    previous_extend=base.extend_bank
    def extend(engine,*args):
        boundaries.append(preserving_extend(previous_extend,engine,*args))
    base.extend_bank=extend
    previous_update=base.update
    def update(*args,**kwargs):
        s=args[2].scheduler
        before=dict(labels=list(s.mapping.labels),states=list(s.state),levels=s.level.tolist())
        info=previous_update(*args,**kwargs)
        info['allocation_receiving_history']=before
        return info
    base.update=update
    previous_evaluate=base.evaluate
    def evaluate(*args,**kwargs):
        tick=time.perf_counter();value=previous_evaluate(*args,**kwargs)
        task_evaluations.append(dict(seen=args[0].seen,seconds=time.perf_counter()-tick))
        return value
    base.evaluate=evaluate
    return base

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('native','replay','router'));a=p.parse_args()
    gate=admit()
    print('V3_SMOKE_ADMISSION_PASS',gate['authority_sha256'],flush=True)
    setup(gate).run(a.arm,'three')
