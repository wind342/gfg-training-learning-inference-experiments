"""Reuse V1 harness, changing only router type and output authority."""
from pathlib import Path
from experiments.gfg_governor_fewrel_feedback_router_v1 import run as base
from .policy import Router

HERE=Path(__file__).resolve().parent
ROOT=Path('E:/gfg-evidence/functional-feedback-router-v2')

def setup():
    original_here=base.HERE;original_core=base.CORE
    base.Router=Router;base.ROOT=ROOT;base.HERE=HERE;base.CORE=HERE
    original_write=base.write
    def observed_write(path,obj):
        if path.name=='CONFIG.json':
            obj=dict(obj,inherited_sources={str(p.relative_to(HERE.parent)):base.sha(p)
                for folder in (original_here,original_core) for p in folder.glob('*.py')},
                base_commit='41a9724e1d0f32a7ffbdf07fd360779823c26cac',
                change='stable anchors: at most one rotating group, no fill')
        return original_write(path,obj)
    base.write=observed_write
    native_update=base.update
    def checked(*args,**kwargs):
        info=native_update(*args,**kwargs)
        n=len(info['active_degraded_groups'])+len(info['active_at_risk_groups'])
        assert len(info['selected_anchor_groups'])<=1
        assert info['feedback_batch_size']<=2*min(8,n+1)
        if n==1:assert info['feedback_batch_size']<=4
        step=args[10]
        if step==220:print('TEN_STEP_ALLOCATION_GATE_PASS: no anchor overfill; data roles checked before backward',flush=True)
        return info
    base.update=checked

if __name__=='__main__':setup();base.run('router','smoke')
