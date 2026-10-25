"""Prospective stress test of bounded TLI/TLIR relations in a hard-spiking RNN.

Frozen testing is performed from original native PyTorch states and update
occurrences. This is not a reproduction of GFG-Core-v3 or the original studies.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Instrument and training constants frozen before formal collection.
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
SEEDS = (71001, 71002, 71003, 71004)
PILOT_SEED = 42
SNAPSHOTS = (50, 200, 400, 700)
STATE_PAIRS = ((200, 50), (400, 200), (700, 400))
STEPS = 800
N_COMPONENTS = 4
N_CLASSES = 10
HIDDEN = 64
TIMESTEPS = 6
BATCH_SIZE = 64
LR = 0.05
QUERY_N = 96
ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


class HardSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (voltage,) = ctx.saved_tensors
        # A surrogate gradient is used for training only. Forward is hard-spiking.
        return gradient * (1.0 / (1.0 + 2.0 * voltage.abs()) ** 2)


class SpikingBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.feed = nn.Linear(h, h)
        self.recurrent = nn.Linear(h, h, bias=False)

    def forward(self, signal: torch.Tensor, membrane: torch.Tensor,
                old_spike: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = 0.65 * membrane + self.feed(signal) + 0.2 * self.recurrent(old_spike)
        spike = HardSpike.apply(membrane - 0.6)
        membrane = membrane - 0.6 * spike
        return spike, membrane


class HardSpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(64, HIDDEN)
        self.blocks = nn.ModuleList(SpikingBlock(HIDDEN) for _ in range(N_COMPONENTS))
        self.head = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, image: torch.Tensor, mask: int = 15) -> torch.Tensor:
        drive_base = F.relu(self.stem(image))
        membrane = [torch.zeros((image.shape[0], HIDDEN), dtype=image.dtype, device=image.device)
                    for _ in range(N_COMPONENTS)]
        prior = [torch.zeros_like(membrane[0]) for _ in range(N_COMPONENTS)]
        pooled = torch.zeros_like(membrane[0])
        for _ in range(TIMESTEPS):
            drive = drive_base
            for block_id, block in enumerate(self.blocks):
                spike, membrane[block_id] = block(drive, membrane[block_id], prior[block_id])
                prior[block_id] = spike
                drive = spike if (mask & (1 << block_id)) else torch.zeros_like(spike)
            pooled = pooled + drive
        return self.head(pooled / TIMESTEPS)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def raw_digest(value: torch.Tensor) -> str:
    c = value.detach().contiguous().cpu()
    return hashlib.sha256(f'{c.dtype}|{tuple(c.shape)}|'.encode()+c.numpy().tobytes()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(raw_digest(value).encode())
    return h.hexdigest()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: v.detach().clone() for n, v in model.state_dict().items()}


def restore(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def plus(state: dict[str, torch.Tensor], delta: dict[str, torch.Tensor], scale: float):
    return {n: t + scale*delta[n] for n,t in state.items()}


def memory_state(model: nn.Module, opt: torch.optim.Optimizer):
    return {n: opt.state[p]['sum'].detach().clone() for n,p in model.named_parameters()}


def digits_dataset(seed: int):
    images, labels = load_digits(return_X_y=True)
    images = torch.tensor(images / 16.0, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    all_ids = np.arange(len(labels), dtype=np.int64)
    tr_idx, te_idx = train_test_split(all_ids, test_size=0.25,
                                      stratify=labels.numpy(), random_state=seed)
    dataset_digest = hashlib.sha256(images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    train = (images[tr_idx], labels[tr_idx], tr_idx)
    test = (images[te_idx], labels[te_idx], te_idx)
    return train, test, dataset_digest


def predict(model: nn.Module, images: torch.Tensor,
            state: dict[str, torch.Tensor] | None = None, mask: int = 15):
    if state is not None:
        restore(model, state)
    model.eval()
    with torch.no_grad():
        return model(images, mask).detach().clone()


def classification(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


def margins(logits: torch.Tensor, labels: torch.Tensor):
    correct = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    alternative = logits.clone()
    alternative.scatter_(-1, labels.unsqueeze(-1), -float('inf'))
    return correct - alternative.max(-1).values


def rms(t: torch.Tensor):
    return float(torch.square(t.double()).mean().sqrt())


def relative_error(a: torch.Tensor, b: torch.Tensor):
    den = math.sqrt((rms(a)**2 + rms(b)**2) / 2)
    return rms(a-b) / max(den, 1e-12)


def native_adagrad_formula(gradients: dict[str, torch.Tensor],
                           accumulator: dict[str, torch.Tensor]) -> dict[str,torch.Tensor]:
    return {n: -LR * g / ((accumulator[n] + g.square()).sqrt() + 1e-10)
            for n,g in gradients.items()}


def support_values(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                   state: dict[str, torch.Tensor]):
    restore(model, state)
    model.eval()
    with torch.no_grad():
        logits = torch.stack([model(images, m).clone() for m in range(16)])
    return logits, torch.stack([margins(logits[m], labels) for m in range(16)])


def shapley_values(value: torch.Tensor):
    # value shape = (coalition, query). Independently recomputable from all 16 masks.
    out = torch.zeros((value.shape[1], 4), dtype=value.dtype)
    for j in range(4):
        for m in range(16):
            if m & (1 << j):
                continue
            k = m.bit_count()
            weight = math.factorial(k) * math.factorial(3-k) / math.factorial(4)
            out[:, j] += weight * (value[m | (1 << j)] - value[m])
    return out


def pairwise_logits_interactions(values: torch.Tensor):
    interactions = []
    for i in range(4):
        for j in range(i+1,4):
            d = (values[15] - values[15^(1<<i)] - values[15^(1<<j)]
                 + values[15^(1<<i)^(1<<j)])
            interactions.append(rms(d))
    return interactions


def measured_response(model, images, labels, state, delta):
    baseline = margins(predict(model, images, state),labels)
    changed = margins(predict(model, images, plus(state,delta,1.0)), labels)
    return changed-baseline


def metric_report(model, seed, train, test, initial, snaps, final):
    images,labels,identities = test
    selected_images,selected_labels,selected_ids = images[:QUERY_N],labels[:QUERY_N],identities[:QUERY_N]
    effects=[];curves=[];supports=[];memory=[]
    raw_payload={}
    for later, earlier in STATE_PAIRS:
        record, other=snaps[later],snaps[earlier]
        delta = record['delta']
        a = measured_response(model,selected_images,selected_labels,record['pre'],delta)
        b = measured_response(model,selected_images,selected_labels,other['pre'],delta)
        effects.append({'from_step':earlier,'update_step':later,
                        'response_nrmse':relative_error(a,b),
                        'receiver_rms':rms(a),'alternate_rms':rms(b),
                        'response_delta_sha256':raw_digest(a-b)})
        predicted = native_adagrad_formula(record['gradients'],record['accumulator_pre'])
        alternate = native_adagrad_formula(record['gradients'],other['accumulator_pre'])
        native_resp = measured_response(model,selected_images,selected_labels,record['pre'],record['delta'])
        alternate_resp = measured_response(model,selected_images,selected_labels,record['pre'],alternate)
        v_native = torch.cat([t.flatten() for t in record['delta'].values()])
        v_alt = torch.cat([t.flatten() for t in alternate.values()])
        v_rebuilt = torch.cat([t.flatten() for t in predicted.values()])
        memory.append({'from_step':earlier,'update_step':later,
                       'native_formula_max_abs_error':float((v_native-v_rebuilt).abs().max()),
                       'same_gradient_update_nrmse':relative_error(v_native,v_alt),
                       'same_gradient_response_nrmse':relative_error(native_resp,alternate_resp),
                       'native_response_rms':rms(native_resp),'altered_response_rms':rms(alternate_resp),
                       'adagrad_pre_accumulator_sha256':state_hash(record['accumulator_pre']),
                       'exchanged_accumulator_sha256':state_hash(other['accumulator_pre'])})

    for step in SNAPSHOTS:
        r=snaps[step]
        bases=[margins(predict(model,selected_images,plus(r['pre'],r['delta'],a)),selected_labels)
               for a in ALPHAS]
        maximum=torch.zeros_like(bases[0])
        for a,observed in zip(ALPHAS,bases):
            chord=bases[0] + a*(bases[-1]-bases[0])
            maximum=torch.maximum(maximum,(observed-chord).abs())
        rescaled=maximum/(bases[-1]-bases[0]).abs().clamp_min(0.002)
        positive=(maximum>0.002)&(rescaled>0.10)
        curves.append({'step':step,'number':QUERY_N,'nonlinear_count':int(positive.sum()),
                       'nonlinear_fraction':float(positive.float().mean()),
                       'max_deviation':float(maximum.max()),
                       'average_abs_update_response':float((bases[-1]-bases[0]).abs().mean()),
                       'finite_path_sha256':raw_digest(torch.stack(bases)),
                       'pre_post_transition_counts':{
                          'correct_to_correct':int(((bases[0]>=0)&(bases[-1]>=0)).sum()),
                          'correct_to_wrong':int(((bases[0]>=0)&(bases[-1]<0)).sum()),
                          'wrong_to_wrong':int(((bases[0]<0)&(bases[-1]<0)).sum()),
                          'wrong_to_correct':int(((bases[0]<0)&(bases[-1]>=0)).sum())}})
        logits0, margin0=support_values(model,selected_images,selected_labels,r['pre'])
        logits1, margin1=support_values(model,selected_images,selected_labels,r['post'])
        again=predict(model,selected_images,r['pre'])
        shap0,shap1=shapley_values(margin0),shapley_values(margin1)
        d0=shap0.abs().sum(-1).clamp_min(1e-12)
        d1=shap1.abs().sum(-1).clamp_min(1e-12)
        active=(shap0.abs()>=d0[:,None]*.1).sum(-1)
        realloc=(shap0-shap1).abs().sum(-1)/d0
        a0=shap0/d0[:,None]
        max_dist=float((a0[:,None,:]-a0[None,:,:]).abs().sum(-1).max())
        interactions=pairwise_logits_interactions(logits0)
        supports.append({'step':step,'distributed_fraction':float((active>=2).float().mean()),
                         'support_reallocation_fraction':float((realloc>0.01).float().mean()),
                         'primary_component_changed_fraction':float((shap0.abs().argmax(-1)!=shap1.abs().argmax(-1)).float().mean()),
                         'max_query_profile_distance':max_dist,
                         'pair_full_logit_interaction_rms':interactions,
                         'precoalition_logits_sha256':raw_digest(logits0),
                         'postcoalition_logits_sha256':raw_digest(logits1),
                         'pre_shapley_sha256':raw_digest(shap0),
                         'post_shapley_sha256':raw_digest(shap1),
                         'repeated_pre_exact':bool(torch.equal(logits0[15],again)),
                         'complete_shapley_sum_error':float((shap0.sum(-1)-(margin0[15]-margin0[0])).abs().max())})
        raw_payload[str(step)]={'precoalition_logits':logits0,
                                'postcoalition_logits':logits1,
                                'pre_margin_path':torch.stack(bases),
                                'pre_shapley':shap0,'post_shapley':shap1}
    raw_before_sha=state_hash(final)
    pred=predict(model,selected_images,final)
    frozen_state_before=state_hash(clone_state(model))
    pred2=predict(model,selected_images)
    frozen_state_after=state_hash(clone_state(model))
    frozen_exact=bool(torch.equal(pred,pred2) and raw_before_sha==frozen_state_before==frozen_state_after)
    rollback=[]
    for j in range(4):
        modified={n:(initial[n] if n.startswith(f'blocks.{j}.') else t) for n,t in final.items()}
        hybrid=predict(model,selected_images,modified)
        restoration=predict(model,selected_images,final)
        rollback.append({'component':j,'max_full_logits_difference':float((hybrid-pred).abs().max()),
                         'restoration_exact':bool(torch.equal(restoration,pred) and state_hash(clone_state(model))==raw_before_sha)})
    full_accuracy=classification(predict(model,images,final),labels)
    observed = {
      'seed':seed,'heldout_accuracy':full_accuracy,'heldout_count':len(labels),
      'query_count':QUERY_N,'query_ids_sha256':hashlib.sha256(selected_ids.tobytes()).hexdigest(),
      'native_optimizer_memory_exchanges':memory,
      'receiver_state_exchanges':effects,
      'finite_amplitude':curves,
      'functional_support':supports,
      'inference_frozen_exact':frozen_exact,
      'rollback':rollback,
      'parameter_version_sha256':state_hash(final),
      'initial_version_sha256':state_hash(initial),
    }
    return observed,raw_payload


def evaluation_gates(observed, thresholds):
    effects = observed['receiver_state_exchanges']
    memory = observed['native_optimizer_memory_exchanges']
    curves = observed['finite_amplitude']
    support = observed['functional_support']
    rollback = observed['rollback']
    return {
       'formed_capability': observed['heldout_accuracy']>=thresholds['minimum_heldout_accuracy'],
       'actual_adagrad_formula_integrity':all(x['native_formula_max_abs_error']<=thresholds['maximum_update_formula_error'] for x in memory),
       'frozen_inference_integrity': observed['inference_frozen_exact'] and all(x['repeated_pre_exact'] for x in support),
       'same_update_different_receiver':sum(x['response_nrmse']>thresholds['response_nrmse'] and max(x['receiver_rms'],x['alternate_rms'])>thresholds['minimum_response_rms'] for x in effects)>=2,
       'optimizer_memory_changes_action_and_response':sum(x['same_gradient_update_nrmse']>thresholds['response_nrmse'] and x['same_gradient_response_nrmse']>thresholds['response_nrmse'] and max(x['native_response_rms'],x['altered_response_rms'])>thresholds['minimum_response_rms'] for x in memory)>=2,
       'finite_amplitude_nonlinearity':sum(x['nonlinear_count'] for x in curves)>=thresholds['minimum_nonlinear_fraction']*sum(x['number'] for x in curves),
       'distributed_functional_support':sum(x['distributed_fraction'] for x in support)/len(support)>=thresholds['minimum_distributed_fraction'],
       'persistent_support_reorganization':sum(x['support_reallocation_fraction'] for x in support)/len(support)>=thresholds['minimum_reallocation_fraction'],
       'query_conditioned_support':all(x['max_query_profile_distance']>thresholds['minimum_query_distance'] for x in support),
       'nonadditive_full_logit_combination':all(max(x['pair_full_logit_interaction_rms'])>thresholds['minimum_full_logit_interaction_rms'] for x in support),
       'exact_learning_version_rollback':all(x['max_full_logits_difference']>thresholds['minimum_rollback_effect'] and x['restoration_exact'] for x in rollback),
       'shapley_efficiency_identity':all(x['complete_shapley_sum_error']<=thresholds['maximum_shapley_efficiency_error'] for x in support),
    }


def run_seed(seed, output:Path, pilot=False):
    seed_everything(seed)
    train,test,data_digest=digits_dataset(seed)
    net=HardSpikingNetwork()
    initial=clone_state(net)
    optimizer=torch.optim.Adagrad(net.parameters(),lr=LR,lr_decay=0.0,weight_decay=0,eps=1e-10)
    rng=np.random.default_rng(seed+500)
    snapshots={}
    training_trace=[]
    for step in range(1,STEPS+1):
        ix=torch.tensor(rng.integers(0,len(train[1]),size=BATCH_SIZE))
        net.train()
        capture=step in SNAPSHOTS
        prior=clone_state(net) if capture else None
        memory=memory_state(net,optimizer) if capture else None
        optimizer.zero_grad(set_to_none=True)
        prediction=net(train[0][ix])
        loss=F.cross_entropy(prediction,train[1][ix])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),4.0)
        gradients={n:p.grad.detach().clone() for n,p in net.named_parameters()} if capture else None
        optimizer.step()
        if capture:
            assert prior is not None and memory is not None and gradients is not None
            post=clone_state(net)
            d={n:post[n]-prior[n] for n in prior}
            snapshots[step]={'pre':prior,'post':post,'delta':d,'gradients':gradients,
                            'accumulator_pre':memory,'optimizer_step':step,
                            'train_batch_ids':train[2][ix].tolist(),
                            'pre_sha256':state_hash(prior),'post_sha256':state_hash(post),
                            'delta_sha256':state_hash(d),'grad_sha256':state_hash(gradients)}
        if step in (1,*SNAPSHOTS,STEPS):
            logits=predict(net,test[0]); acc=classification(logits,test[1]);
            training_trace.append({'step':step,'heldout_accuracy':acc,
                                   'loss':float(F.cross_entropy(logits,test[1]))})
    initial_final=clone_state(net)
    if pilot:
        return {'seed':seed,'training_trace':training_trace,'dataset_digest':data_digest}
    measured,raw=metric_report(net,seed,train,test,initial,snapshots,initial_final)
    native={'seed':seed,'dataset_digest':data_digest,'initial':initial,'final':initial_final,
            'snapshots':snapshots,'heldout_indices':test[2],
            'heldout_labels':test[1],'fixed_query_ids':test[2][:QUERY_N],
            'raw_measurement_arrays':raw}
    output.mkdir(parents=True,exist_ok=True)
    torch.save(native,output/'NATIVE_STATES_AND_OUTCOMES.pt')
    measured['snapshots']=[{'step':s,'pre_state_sha256':snapshots[s]['pre_sha256'],
                            'post_state_sha256':snapshots[s]['post_sha256'],
                            'delta_sha256':snapshots[s]['delta_sha256'],
                            'gradient_sha256':snapshots[s]['grad_sha256'],
                            'batch_ids_sha256':hashlib.sha256(np.asarray(snapshots[s]['train_batch_ids'],dtype=np.int64).tobytes()).hexdigest()}
                           for s in SNAPSHOTS]
    measured['train_trace']=training_trace
    measured['dataset_digest']=data_digest
    (output/'MEASUREMENTS.json').write_text(json.dumps(measured,indent=2,sort_keys=True)+'\n')
    return measured


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--formal',action='store_true')
    parser.add_argument('--protocol',default='/mnt/data/snn_tlir_frozen_protocol.json')
    parser.add_argument('--output',default='/mnt/data/SNN_TLIR_formal')
    args=parser.parse_args()
    if args.pilot == args.formal:
        raise ValueError('Select precisely one of --pilot and --formal')
    folder=Path(args.output)
    if args.pilot:
        result=run_seed(PILOT_SEED, folder, pilot=True)
        print(json.dumps(result,ensure_ascii=False))
        return
    protocol_path=Path(args.protocol)
    protocol=json.loads(protocol_path.read_text())
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if protocol['source_sha256']!=source_sha or protocol['formal_seeds']!=list(SEEDS):
        raise RuntimeError('FROZEN_PROTOCOL_SOURCE_OR_SEEDS_MISMATCH')
    if folder.exists():
        raise RuntimeError('FORMAL_OUTPUT_ALREADY_EXISTS (do not overwrite)')
    folder.mkdir(parents=True)
    observed=[]
    start=time.monotonic()
    for seed in SEEDS:
        result=run_seed(seed,folder/f'seed_{seed}')
        gates=evaluation_gates(result,protocol['thresholds'])
        result['frozen_gates']=gates
        (folder/f'seed_{seed}'/'FROZEN_GATES.json').write_text(json.dumps(gates,indent=2,sort_keys=True)+'\n')
        observed.append({'seed':seed,'heldout_accuracy':result['heldout_accuracy'],
                         'gates':gates,'nonlinear_fraction':sum(c['nonlinear_count'] for c in result['finite_amplitude'])/(len(SNAPSHOTS)*QUERY_N)})
        print(json.dumps(observed[-1],sort_keys=True),flush=True)
    manifest={}
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            manifest[str(path.relative_to(folder))]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                                     'size':path.stat().st_size}
    summary={'schema':'snn_tlir_strict_adversarial_result_v1','status':'FORMAL_EXECUTED',
             'source_sha256':source_sha,
             'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
             'python':platform.python_version(),'torch':torch.__version__,
             'numpy':np.__version__,'sklearn_version':__import__('sklearn').__version__,
             'runs':observed,'file_manifest':manifest,
             'runtime_seconds':time.monotonic()-start,
             'claims_not_tested':['Universal TLIR across all architectures',
                                  'Native GFG Core v3 execution',
                                  'A standalone capability-learning causal mediation theorem']}
    (folder/'FORMAL_SUMMARY.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print('FORMAL_SUMMARY_SHA256',hashlib.sha256((folder/'FORMAL_SUMMARY.json').read_bytes()).hexdigest())

if __name__=='__main__':
    main()
