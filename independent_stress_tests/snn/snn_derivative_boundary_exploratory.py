"""Post-hoc adversarial diagnostic, NOT part of the preregistered formal test.
Compares custom backward/surrogate JVP with physical native finite differences.
"""
from __future__ import annotations
import json,sys,math
from pathlib import Path
import torch
from torch.func import functional_call
from snn_tlir_adversarial import HardSpikingNetwork,digits_dataset,SNAPSHOTS,SEEDS

torch.set_num_threads(2)

def rms(x): return float(x.double().square().mean().sqrt())

def four_way_macro(before,actual,pred):
    labels={ (True,True):'stay_correct', (True,False):'lost', (False,False):'stay_wrong', (False,True):'gained'}
    counted={n:[0,0] for n in labels.values()}
    for a,b,c in zip(before.tolist(),actual.tolist(),pred.tolist()):
        bucket=labels[(a,b)]; counted[bucket][1]+=1
        if c==b:counted[bucket][0]+=1
    recalls={n:(a/b if b else None) for n,(a,b) in counted.items()}
    observed=[z for z in recalls.values() if z is not None]
    return sum(observed)/len(observed),recalls


def run():
    results=[]
    for seed in SEEDS:
        r=torch.load(Path('/mnt/data/SNN_TLIR_formal')/f'seed_{seed}'/'NATIVE_STATES_AND_OUTCOMES.pt',weights_only=False,map_location='cpu')
        _,test,_=digits_dataset(seed)
        X,y,_=test;X=X[:96];y=y[:96]
        model=HardSpikingNetwork().eval()
        for step in SNAPSHOTS:
            ss=r['snapshots'][step]
            def f(alpha):
                st={n:t+alpha*ss['delta'][n] for n,t in ss['pre'].items()}
                return functional_call(model,st,(X,15),strict=True)
            zero=torch.tensor(0.,requires_grad=True)
            v,j=torch.autograd.functional.jvp(f,(zero,),(torch.tensor(1.),),create_graph=False)
            with torch.no_grad():
                actual=f(torch.tensor(1.))
                fd=(f(torch.tensor(0.001))-v)/.001
                pred=v+j
                c0=v.argmax(-1).eq(y)
                ct=actual.argmax(-1).eq(y)
                cp=pred.argmax(-1).eq(y)
                cbase=c0
            macro,recs=four_way_macro(c0,ct,cp)
            macro_base,_=four_way_macro(c0,ct,cbase)
            results.append({'seed':seed,'step':step,'true_forward_finite_diff_epsilon':.001,
              'surrogate_JVP_rms':rms(j),'physical_forward_FD_rms':rms(fd),
              'surrogate_vs_physical_FD_rms':rms(j-fd),
              'surrogate_vs_physical_FD_nrmse':rms(j-fd)/max((rms(j)**2+rms(fd)**2)/2,1e-24)**.5,
              'surrogate_linear_endpoint_logit_rmse':rms(pred-actual),
              'unchanged_endpoint_logit_rmse':rms(v-actual),
              'surrogate_linear_endpoint_class_accuracy':float(cp.float().mean()),
              'unchanged_endpoint_class_accuracy':float(cbase.float().mean()),
              'true_endpoint_class_accuracy':float(ct.float().mean()),
              'surrogate_transition_macro_recall':macro,'unchanged_transition_macro_recall':macro_base,
              'transition_recalls':recs})
    aggregate={'run_count':len(SEEDS),'snapshot_count':len(results),
              'mean_physical_jvp_nrmse':sum(z['surrogate_vs_physical_FD_nrmse'] for z in results)/len(results),
              'mean_surrogate_linear_macro':sum(z['surrogate_transition_macro_recall'] for z in results)/len(results),
              'mean_unchanged_macro':sum(z['unchanged_transition_macro_recall'] for z in results)/len(results),
              'mean_predicted_logit_rmse':sum(z['surrogate_linear_endpoint_logit_rmse'] for z in results)/len(results),
              'mean_unchanged_logit_rmse':sum(z['unchanged_endpoint_logit_rmse'] for z in results)/len(results),
              'logit_rmse_worse_count':sum(z['surrogate_linear_endpoint_logit_rmse']>z['unchanged_endpoint_logit_rmse'] for z in results)}
    report={'schema':'hard_spike_local_autodiff_boundary_v1','status':'EXPLORATORY_POST_HOC_NOT_A_FROZEN_GATE',
           'meaning':'A surrogate-gradient JVP is not the physical forward-map derivative of discontinuous hard spikes. This does not test the TL-P01 predictor, which was not claimed universal to hard spiking.',
           'aggregate':aggregate,'snapshots':results}
    Path('/mnt/data/SNN_TLIR_local_derivative_exploratory.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))
    for z in results:
        print(z['seed'],z['step'],'JVP-vs-physical-NRMSE',round(z['surrogate_vs_physical_FD_nrmse'],2),'linear_macro',round(z['surrogate_transition_macro_recall'],3),'unchanged_macro',round(z['unchanged_transition_macro_recall'],3))
if __name__=='__main__':run()
