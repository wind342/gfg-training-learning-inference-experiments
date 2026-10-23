"""Independent numerical audit of hard-spiking TLI/TLIR stress-test native records.
Does not reuse the producer's metric_report() or evaluation_gates().
Model forward and dataset generation must be shared to replay the exact system.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from snn_tlir_adversarial import (HardSpikingNetwork, digits_dataset, state_hash,
                                 raw_digest, SNAPSHOTS, STATE_PAIRS, ALPHAS,
                                 QUERY_N, LR, SEEDS)

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(a,b,eps=1e-5):
    return abs(float(a)-float(b)) <= eps


def forward(model,inputs,state,mask=15):
    model.load_state_dict(state,strict=True)
    model.eval()
    with torch.no_grad():
        return model(inputs,mask).detach().clone()


def margin(logits,truth):
    target=logits.gather(-1,truth[:,None]).flatten()
    distractor=logits.detach().clone()
    distractor.scatter_(1,truth[:,None],-float('inf'))
    return target-distractor.max(-1).values


def relative(x,y):
    xd=x.double().reshape(-1);yd=y.double().reshape(-1)
    den=torch.sqrt((xd.square().mean()+yd.square().mean())/2).clamp_min(1e-12)
    return float(((xd-yd).square().mean().sqrt())/den)


def response(model,x,y,s,d):
    original=margin(forward(model,x,s),y)
    future=margin(forward(model,x,{k:v+d[k] for k,v in s.items()}),y)
    return future-original


def shapley(margins):
    # Source values are 16 coalitions. Recalculate without producer function.
    out=torch.zeros((margins.shape[1],4),dtype=torch.float64)
    for coal in range(16):
        for comp in range(4):
            if (coal & (1<<comp))==0:
                size=coal.bit_count()
                weight=math.factorial(size)*math.factorial(3-size)/24
                out[:,comp]+=weight*(margins[coal|(1<<comp)].double()-margins[coal].double())
    return out


def check_seed(root,seed,thresholds):
    raw=torch.load(root/'NATIVE_STATES_AND_OUTCOMES.pt',weights_only=False,map_location='cpu')
    measured=json.loads((root/'MEASUREMENTS.json').read_text())
    frozen=json.loads((root/'FROZEN_GATES.json').read_text())
    train,test,digest=digits_dataset(seed)
    x,y,ids=test
    x=x[:QUERY_N];y=y[:QUERY_N]; ids=ids[:QUERY_N]
    require(raw['seed']==seed==measured['seed'],'IDENTITY_SEED')
    require(raw['dataset_digest']==digest==measured['dataset_digest'],'DATASET_DIGEST')
    require(np.array_equal(raw['heldout_indices'],test[2]),'FULL_TEST_IDENTITIES')
    require(torch.equal(raw['heldout_labels'],test[1]),'FULL_TEST_LABELS')
    require(np.array_equal(raw['fixed_query_ids'],ids),'QUERY_IDENTITIES')
    require(hashlib.sha256(ids.tobytes()).hexdigest()==measured['query_ids_sha256'],'QUERY_HASH')
    require(state_hash(raw['initial'])==measured['initial_version_sha256'],'INITIAL_HASH')
    require(state_hash(raw['final'])==measured['parameter_version_sha256'],'FINAL_HASH')
    model=HardSpikingNetwork()
    final_logits=forward(model,test[0],raw['final'])
    accuracy=float((final_logits.argmax(-1)==test[1]).float().mean())
    require(close(accuracy,measured['heldout_accuracy'],1e-8),'FINAL_ACCURACY')
    require(bool(frozen['formed_capability'])==(accuracy>=thresholds['minimum_heldout_accuracy']),'CAPABILITY_GATE')
    before=state_hash(raw['final'])
    a=forward(model,x,raw['final'])
    b=forward(model,x,raw['final'])
    require(torch.equal(a,b),'FINAL_FROZEN_OUTPUT_REPEAT')
    require(state_hash({n:p.detach().clone() for n,p in model.state_dict().items()})==before,'FINAL_FROZEN_STATE')
    for j, rr in enumerate(measured['rollback']):
        hybrid={k:(raw['initial'][k] if k.startswith(f'blocks.{j}.') else v)
                for k,v in raw['final'].items()}
        c=forward(model,x,hybrid)
        d=forward(model,x,raw['final'])
        require(torch.equal(a,d),'ROLLBACK_RESTORATION')
        require(close((a-c).abs().max(),rr['max_full_logits_difference'],1e-7),'ROLLBACK_EFFECT')
        require(rr['restoration_exact'],'ROLLBACK_RESTORATION_GATE')
    for i,step in enumerate(SNAPSHOTS):
        snap=raw['snapshots'][step]
        record=measured['snapshots'][i]
        require(snap['optimizer_step']==step==record['step'],'SNAPSHOT_STEP')
        for n,sfield,mfield in [('pre','pre_sha256','pre_state_sha256'),
                                ('post','post_sha256','post_state_sha256'),
                                ('delta','delta_sha256','delta_sha256'),
                                ('gradients','grad_sha256','gradient_sha256')]:
            require(state_hash(snap[n])==snap[sfield]==record[mfield],f'{n.upper()}_HASH')
        for name in snap['delta']:
            require(torch.equal(snap['post'][name]-snap['pre'][name],snap['delta'][name]),'EXACT_NATIVE_UPDATE')
        expected_batch=hashlib.sha256(np.asarray(snap['train_batch_ids'],dtype=np.int64).tobytes()).hexdigest()
        require(expected_batch==record['batch_ids_sha256'],'BATCH_ID_HASH')
        stored=raw['raw_measurement_arrays'][str(step)]
        curves=measured['finite_amplitude'][i]
        observed=[]
        for scale in ALPHAS:
            state={n:p+scale*snap['delta'][n] for n,p in snap['pre'].items()}
            observed.append(margin(forward(model,x,state),y))
        actual_path=torch.stack(observed)
        require(torch.equal(actual_path,stored['pre_margin_path']),'NATIVE_PATH_BYTES')
        require(raw_digest(actual_path)==curves['finite_path_sha256'],'NATIVE_PATH_HASH')
        dev=torch.zeros_like(observed[0])
        for scale,part in zip(ALPHAS,observed):
            chord=observed[0]+scale*(observed[-1]-observed[0])
            dev=torch.maximum(dev,(part-chord).abs())
        count=int(((dev>0.002)&(dev/(observed[-1]-observed[0]).abs().clamp_min(0.002)>0.10)).sum())
        require(count==curves['nonlinear_count'],'CURVE_COUNT')
        require(curves['number']==QUERY_N,'CURVE_LENGTH')
        support=measured['functional_support'][i]
        margins=[]
        for phase,key in [('pre','precoalition_logits'),('post','postcoalition_logits')]:
            logits=torch.stack([forward(model,x,snap[phase],m) for m in range(16)])
            require(torch.equal(logits,stored[key]),f'{phase.upper()}_COALITION_BYTES')
            require(raw_digest(logits)==support[f'{phase}coalition_logits_sha256'],'COALITION_HASH')
            margins.append(torch.stack([margin(logits[m],y) for m in range(16)]))
        pre_s=shapley(margins[0]);post_s=shapley(margins[1])
        require(torch.allclose(pre_s.float(),stored['pre_shapley'],atol=2e-5,rtol=1e-6),'SHAPLEY_PRE_INDEPENDENT')
        require(torch.allclose(post_s.float(),stored['post_shapley'],atol=2e-5,rtol=1e-6),'SHAPLEY_POST_INDEPENDENT')
        abs_mass=pre_s.abs().sum(-1).clamp_min(1e-12)
        active=(pre_s.abs()>=.1*abs_mass[:,None]).sum(-1)
        reallocated=(pre_s-post_s).abs().sum(-1)/abs_mass
        require(close((active>=2).double().mean(),support['distributed_fraction'],1e-7),'DISTRIBUTED_FRACTION')
        require(close((reallocated>0.01).double().mean(),support['support_reallocation_fraction'],1e-7),'REALLOCATION_FRACTION')
        full=stored['precoalition_logits'].double()
        interactions=[]
        for lhs in range(4):
            for rhs in range(lhs+1,4):
                d=full[15]-full[15^(1<<lhs)]-full[15^(1<<rhs)]+full[15^(1<<lhs)^(1<<rhs)]
                interactions.append(float(d.square().mean().sqrt()))
        require(all(close(left,right,3e-6) for left,right in zip(interactions,support['pair_full_logit_interaction_rms'])),'LOGIT_INTERACTIONS')
    for i,(later,earlier) in enumerate(STATE_PAIRS):
        snap=raw['snapshots'][later];old=raw['snapshots'][earlier]
        native=response(model,x,y,snap['pre'],snap['delta'])
        transplant=response(model,x,y,old['pre'],snap['delta'])
        require(close(relative(native,transplant),measured['receiver_state_exchanges'][i]['response_nrmse'],1e-7),'RECEIVER_RESPONSE')
        sums=snap['accumulator_pre'];changed_sums=old['accumulator_pre']
        reference={n:-LR*g/((sums[n]+g.square()).sqrt()+1e-10) for n,g in snap['gradients'].items()}
        transplanted={n:-LR*g/((changed_sums[n]+g.square()).sqrt()+1e-10) for n,g in snap['gradients'].items()}
        rec_err=max(float((reference[n]-snap['delta'][n]).abs().max()) for n in reference)
        require(close(rec_err,measured['native_optimizer_memory_exchanges'][i]['native_formula_max_abs_error'],1e-7),'ADAGRAD_FORMULA')
        update_nrmse=relative(torch.cat([d.flatten() for d in snap['delta'].values()]),torch.cat([d.flatten() for d in transplanted.values()]))
        alt_response=response(model,x,y,snap['pre'],transplanted)
        response_nrmse=relative(native,alt_response)
        require(close(update_nrmse,measured['native_optimizer_memory_exchanges'][i]['same_gradient_update_nrmse'],1e-7),'MEMORY_UPDATE_DIFF')
        require(close(response_nrmse,measured['native_optimizer_memory_exchanges'][i]['same_gradient_response_nrmse'],1e-7),'MEMORY_RESPONSE_DIFF')
    # Gate decisions are independently compared, without counting these as new predictions.
    def csum(name):
        return sum(float(a[name]) for a in measured['functional_support'])/len(SNAPSHOTS)
    receiver=sum(a['response_nrmse']>.05 and max(a['receiver_rms'],a['alternate_rms'])>.0005 for a in measured['receiver_state_exchanges'])>=2
    memory=sum(a['same_gradient_update_nrmse']>.05 and a['same_gradient_response_nrmse']>.05 and max(a['native_response_rms'],a['altered_response_rms'])>.0005 for a in measured['native_optimizer_memory_exchanges'])>=2
    independent_gates={
      'formed_capability':accuracy>=thresholds['minimum_heldout_accuracy'],
      'actual_adagrad_formula_integrity':all(a['native_formula_max_abs_error']<=thresholds['maximum_update_formula_error'] for a in measured['native_optimizer_memory_exchanges']),
      'frozen_inference_integrity':bool(measured['inference_frozen_exact']) and all(a['repeated_pre_exact'] for a in measured['functional_support']),
      'same_update_different_receiver':receiver,
      'optimizer_memory_changes_action_and_response':memory,
      'finite_amplitude_nonlinearity':sum(a['nonlinear_count'] for a in measured['finite_amplitude'])>=.10*len(SNAPSHOTS)*QUERY_N,
      'distributed_functional_support':csum('distributed_fraction')>=thresholds['minimum_distributed_fraction'],
      'persistent_support_reorganization':csum('support_reallocation_fraction')>=thresholds['minimum_reallocation_fraction'],
      'query_conditioned_support':all(a['max_query_profile_distance']>thresholds['minimum_query_distance'] for a in measured['functional_support']),
      'nonadditive_full_logit_combination':all(max(a['pair_full_logit_interaction_rms'])>thresholds['minimum_full_logit_interaction_rms'] for a in measured['functional_support']),
      'exact_learning_version_rollback':all(a['max_full_logits_difference']>thresholds['minimum_rollback_effect'] and a['restoration_exact'] for a in measured['rollback']),
      'shapley_efficiency_identity':all(a['complete_shapley_sum_error']<=thresholds['maximum_shapley_efficiency_error'] for a in measured['functional_support']),
    }
    require(independent_gates==frozen,'GATE_MISMATCH')
    return {'seed':seed,'heldout_accuracy':accuracy,'gate_count_passed':sum(frozen.values()),
            'gate_count_total':len(frozen),'failed_gates':[k for k,v in frozen.items() if not v]}


def check(root:Path,protocol:Path):
    freeze=json.loads(protocol.read_text())
    formal=json.loads((root/'FORMAL_SUMMARY.json').read_text())
    require(hashlib.sha256(protocol.read_bytes()).hexdigest()==formal['protocol_sha256'],'PROTOCOL_SHA')
    model_source=Path(__file__).with_name('snn_tlir_adversarial.py')
    require(hashlib.sha256(model_source.read_bytes()).hexdigest()==freeze['source_sha256']==formal['source_sha256'],'MODEL_SHA')
    require(tuple(freeze['formal_seeds'])==SEEDS,'SEED_FREEZE')
    for name,item in formal['file_manifest'].items():
        p=root/name
        require(p.is_file(),'MISSING:'+name)
        require(p.stat().st_size==item['size'],'SIZE:'+name)
        require(hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'SHA:'+name)
    checks=[]
    for seed in SEEDS:
        checks.append(check_seed(root/f'seed_{seed}',seed,freeze['thresholds']))
    print(json.dumps({'verifier_status':'PASS','independent_native_replay':'PASS',
                      'verified_seeds':checks,'sha256_summary':hashlib.sha256((root/'FORMAL_SUMMARY.json').read_bytes()).hexdigest()},
                     indent=2,sort_keys=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/mnt/data/SNN_TLIR_formal')
    ap.add_argument('--protocol',default='/mnt/data/snn_tlir_frozen_protocol.json')
    ar=ap.parse_args()
    check(Path(ar.root),Path(ar.protocol))
