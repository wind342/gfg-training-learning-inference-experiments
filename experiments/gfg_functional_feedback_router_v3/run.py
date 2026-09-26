"""Inject only sample-count weighting into the unchanged V2 training route."""
from pathlib import Path
from experiments.gfg_functional_feedback_router_v2 import run as v2
from experiments.gfg_functional_feedback_router_v1 import runtime
from .dose import accumulate

HERE=Path(__file__).resolve().parent
ROOT=Path('E:/gfg-evidence/functional-feedback-router-v3')

def setup():
    v2.setup();base=v2.base
    base.HERE=HERE;base.CORE=HERE;base.ROOT=ROOT
    runtime.accumulate=accumulate
    previous_write=base.write
    def annotate(path,obj):
        if path.name=='CONFIG.json':
            obj=dict(obj,v3_base_commit='cf6c0f30d0e35173771ab2fc52fa2474b1d940ee',
                dose='actual sample-count normalized means; sequential backwards',
                router_v2_sources={str(p.relative_to(HERE.parent)):base.sha(p) for p in (HERE.parent/'gfg_functional_feedback_router_v2').glob('*.py')})
        return previous_write(path,obj)
    base.write=annotate
    inherited_update=base.update
    def observe_dose(*args,**kwargs):
        info=inherited_update(*args,**kwargs)
        nc=len(args[7]);no=info['feedback_batch_size'];total=nc+no
        info.update(current_batch_size=nc,current_loss_weight=nc/total,old_loss_weight=no/total,
            weighted_training_loss=(nc*info['current_loss']+no*(info['feedback_loss'] or 0.))/total)
        return info
    base.update=observe_dose
    return base

if __name__=='__main__':setup().run('router','smoke')
