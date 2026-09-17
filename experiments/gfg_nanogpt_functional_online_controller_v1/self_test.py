"""Synthetic initialized model only. No formal checkpoints/outcomes are read."""
import copy
import time
from .common import *
from .selector import coordinates, select
from .independent_checker import independently_choose
from experiments.gfg_nanogpt_full_network_receptive_state_v1.analysis import _model

def synthetic_runtime(model,x,y):
    rt=CausalTrainingRuntime.__new__(CausalTrainingRuntime)
    rt.model=model
    rt.device='cuda'
    rt.named_parameters=dict(model.named_parameters())
    rt.optimizer=model.configure_optimizers(1.,.003,(.9,.98),'cuda')
    rt.training_inputs=x
    rt.training_targets=y
    return rt

def main():
    setup()
    torch.manual_seed(719)
    model=_model(TRAINER).cuda()
    x=torch.randint(0,23,(212,3),device='cuda')
    y=torch.randint(0,23,(212,3),device='cuda')
    rt=synthetic_runtime(model,x,y)
    rt.train_step(execute_optimizer=True,seed=719)
    initial=rt.snapshot()
    replica=synthetic_runtime(copy.deepcopy(model),x,y)
    replica.restore(initial)
    derivative_seconds=[]
    for t in range(4):
        before=rt.snapshot()
        step=rt.train_step(execute_optimizer=True,seed=720+t)
        proposal=rt.snapshot()
        other=replica.train_step(execute_optimizer=True,seed=720+t)
        require(state_hash(proposal)==state_hash(replica.snapshot()),'SYNTHETIC_NATIVE_REPLAY')
        torch.cuda.synchronize()
        start=time.perf_counter()
        c=coordinates(model,before.parameters,step.parameter_updates,x,y)
        torch.cuda.synchronize()
        derivative_seconds.append(time.perf_counter()-start)
        require(state_hash(rt.snapshot())==state_hash(proposal),'SYNTHETIC_FUNCTIONAL_CALL_MUTATION')
        require(select(c)==independently_choose(c),'SYNTHETIC_SELECTOR_ARITHMETIC')
        for a in (.25,.5,.75,1.):
            scaled=scaled_state(before,proposal,step.parameter_updates,a)
            require(scaled.commitment()['optimizer']==proposal.commitment()['optimizer'],'SYNTHETIC_MOMENT_SCALING')
            if a==1.: require(state_hash(scaled)==state_hash(proposal),'SYNTHETIC_ALPHA_ONE')
        # Change parameters to prove functional_call replaces ALL current model
        # parameters rather than accidentally reading the native post endpoint.
        with torch.no_grad():
            for p in model.parameters(): p.add_(.125)
        reconstructed=coordinates(model,before.parameters,step.parameter_updates,x,y)
        require(all(np.array_equal(v,reconstructed[k]) for k,v in c.items()),'PARAMETER_SUBSTITUTION_LEAK')
        rt.restore(proposal)
    report={'status':'PASS','formal_checkpoint_or_outcome_access':False,
        'synthetic_native_replay_steps':4,'selector_arithmetic':'same imported smoke rule, independent recomputation agrees',
        'all_current_model_parameters_substituted_by_pre_state':True,
        'no_moment_scaling':True,'alpha_one_native_bytes_exact':True,
        'coordinate_seconds':derivative_seconds,'estimated_2400_coordinate_seconds':float(np.median(derivative_seconds[1:])*2400)}
    write_new(HERE / 'SELF_TEST.json',report)
    print(report,flush=True)

if __name__=='__main__':main()
