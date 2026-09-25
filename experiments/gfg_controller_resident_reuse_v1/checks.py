"""Post-run defensive checks: snapshot independence and deliberate mutation."""
from . import run
from .engine import ResidentBuffers, NanoGPTAdapter, Controller
import torch

b=run.b


def main():
    b.old.setup()
    model=torch.nn.Linear(5,3).cuda()
    buffers=ResidentBuffers(model)
    first=buffers.snapshot_model(); digest=b.phash(first)
    with torch.no_grad():model.weight.add_(1.)
    second=buffers.snapshot_model()
    assert b.phash(first)==digest,'old snapshot was overwritten'
    assert b.phash(second)!=digest,'snapshot did not capture mutation'
    assert first['weight'].untyped_storage().data_ptr()!=second['weight'].untyped_storage().data_ptr()

    class CorruptingAdapter(NanoGPTAdapter):
        def coordinates(self,model,*args):
            result=super().coordinates(model,*args)
            with torch.no_grad():next(model.parameters()).reshape(-1)[0].add_(.125)
            return result

    old=b.read(run.previous.HERE/'verification_v2/RESULTS.json')['histories'][0]
    row=next(r for r in b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'] if r['entry_id']==old['entry'])
    loaded=b.load_row(row)
    try:
        rt,start,batches,vx,vy=loaded
        engine=Controller(rt.model,CorruptingAdapter(rt.model),b.select,b.phash,b.selection_boundary)
        opt=b.make_optimizer(rt.model,'controller',b.RATES['controller'][old['config']])
        rng_hash=b.read(b.HERE/'FREEZE_SEAL.json')['source_rng_protocol_sha256']
        seed=b.old.seed_for(row,0,rng_hash); x,y=batches[0]
        caught=False
        try:engine.update(rt.model,opt,1,x,y,vx,vy,seed,1.)
        except AssertionError as exc:
            assert str(exc)=='SELECTOR_MUTATION',repr(exc)
            caught=True
        assert caught,'mutation check disabled'
    finally:loaded[0].close()
    report={'status':'PASS','packed_snapshot_independence':True,
        'deliberate_parameter_mutation_detected':True,
        'negative_control_scope':'one in-memory historical step; no old artifacts written'}
    b.write(run.HERE/'EXTRA_CHECKS.json',report)
    print(report)


if __name__=='__main__':main()
