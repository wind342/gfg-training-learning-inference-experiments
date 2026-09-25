"""Small prospective amplitude sweep. Old evidence is read-only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import torch
from experiments.gfg_nanogpt_functional_online_controller_v1 import common as old
from experiments.gfg_controller_distributed_runtime_v2.adapters import NanoGPTAdapter
from experiments.gfg_controller_distributed_runtime_v2.identity import CanonicalParameterRegistry
from experiments.gfg_controller_distributed_runtime_v2.ownership import Ownership
from experiments.gfg_controller_distributed_runtime_v2.boundary import calculate
from experiments.gfg_controller_distributed_runtime_v2.scientific import endpoints

OUT = Path('E:/gfg-evidence/large-update-prediction-smoke-v1/attempt-3')
ALPHAS = (0., .25, .5, 1., 2., 4., 8., 16.)
TAG = 'paper-experiments-cross-system-feedback-release'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def dump_npz(p, **arrays):
    with p.open('xb') as f:
        np.savez_compressed(f, **arrays)


def choice(minima, indices):
    return max(indices, key=lambda j: (int((minima[j] > 0).sum()),
               float(np.median(minima[j])), -abs(ALPHAS[j]-1), -ALPHAS[j]))


@torch.no_grad()
def logits(model, x):
    model.eval()
    return model(x)[0][:, -1, :].detach().cpu().numpy().copy()


def case(row):
    path = OUT / row['entry_id']; path.mkdir()
    rt = old.runtime(row)
    try:
        model = rt.model
        pre = rt.snapshot()
        assert pre.commitment()['state_sha256'] == row['initial_state_sha256']
        b = old.batch(rt, row['source_step'])
        op = row['opportunities'][0]
        assert hashlib.sha256(b.inputs.numpy().tobytes()).hexdigest() == op['batch_inputs_sha256']
        assert hashlib.sha256(b.targets.numpy().tobytes()).hexdigest() == op['batch_targets_sha256']
        # Use original encoded inputs/labels, never reconstruct token arithmetic.
        ex, ey = rt.training_inputs.clone(), rt.training_targets.clone()
        old.select_batch(rt, b)
        vx, vy = rt.base.validation_inputs, rt.base.validation_targets
        assert len(vx) == 212
        validation_pairs = {tuple(z.tolist()) for z in vx}
        assert len(validation_pairs) == 212
        assert len(ex) == 317
        assert not validation_pairs.intersection(tuple(z.tolist()) for z in ex)
        x = torch.cat([vx.cuda(), ex])
        y = torch.cat([vy[:,-1].cuda(), ey[:,-1]])
        baseline = logits(model, x)
        # The native proposal is formed but its endpoint is NOT evaluated yet.
        torch.cuda.synchronize(); tick = time.perf_counter()
        evidence = rt.train_step(execute_optimizer=True, seed=row['initial_rng']['seed'])
        proposed = rt.snapshot()
        delta = {n: proposed.parameters[n]-pre.parameters[n] for n in pre.parameters}
        rt.restore(pre)
        owner = Ownership(model, CanonicalParameterRegistry(model))
        for n in delta:
            owner.proposal.delta[n].copy_(delta[n].cuda())
            owner.proposal.candidate[n].copy_(proposed.parameters[n].cuda())
        model.eval()
        adapter = NanoGPTAdapter(model)
        with torch.no_grad():
            native, hs = adapter.jets(owner, x)
            start, w, dw = owner.readout_shard(adapter.readout_name)
            adapter_logits = torch.nn.functional.linear(native,w)[:,-1,:]
            # Anchor g0 to actual model forward, not a different GEMM execution.
            z = torch.from_numpy(baseline).to(w.device)
            _, _, original_minima, coords, _ = calculate(z,y,w,dw,hs,audit=True)
            mask = y[:,None] == torch.arange(w.shape[0],device=y.device)[None,:]
            predicted = torch.stack([endpoints(*coords,a).masked_fill(mask,torch.inf).amin(1) for a in ALPHAS])
        assert torch.isfinite(predicted).all()
        # Check formula against the unmodified reference at shared amplitudes.
        for j,k in [(1,0),(2,1),(3,3)]:
            assert torch.equal(predicted[j],original_minima[k])
        assert rt.snapshot().commitment() == pre.commitment()
        p = predicted.cpu().numpy()
        selected = choice(p[:,:212], range(1,len(ALPHAS)))
        restricted = choice(p[:,:212], (1,2,3))
        torch.cuda.synchronize(); prediction_seconds = time.perf_counter()-tick
        dump_npz(path/'PREDICTIONS.npz', coordinates=coords.cpu().numpy(),predicted_minima=p,
                 inputs=x.cpu().numpy(),targets=y.cpu().numpy(),baseline_logits=baseline,
                 adapter_logits=adapter_logits.cpu().numpy())
        write(path/'PREDICTION_SEAL.json', {
            'time_ns':time.time_ns(),'predictions_sha256':sha(path/'PREDICTIONS.npz'),
            'selected_alpha':ALPHAS[selected],'restricted_alpha':ALPHAS[restricted],
            'all_predictions_before_post_update_forward':True,
            'adapter_vs_native_logit_max_abs':float((adapter_logits-z).abs().max()),
            'g0_anchor':'actual native baseline forward',
            'source_state':pre.commitment(),'proposal_state':proposed.commitment(),
            'source_manifest_sha256':sha(row['receiver_state_path']),
            'source_bundle_manifest_sha256':sha(Path(row['bundle_path'])/'manifest.json') if (Path(row['bundle_path'])/'manifest.json').exists() else None,
            'source_batch_hashes':{k:op[k] for k in ('batch_inputs_sha256','batch_targets_sha256')},
            'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in rt.optimizer.param_groups],
            'parameter_count':sum(v.numel() for v in model.parameters()),
            'native_proposal_and_prediction_seconds':prediction_seconds,
            'source_step':row['source_step'],'source_rng_seed':row['initial_rng']['seed']})
        # Every fork shares identical receiving state, delta and optimizer memory.
        actual = []; receipts=[]
        for a in ALPHAS:
            state = old.scaled_state(pre, proposed, delta, a)
            rt.restore(state)
            before=rt.snapshot().commitment()
            z=logits(model,x)
            assert rt.snapshot().commitment()==before
            assert before['optimizer']==proposed.commitment()['optimizer']
            actual.append(z)
            receipts.append({'alpha':a,'time_ns':time.time_ns(),'state':before,'finite_logits':bool(np.isfinite(z).all())})
        assert np.array_equal(actual[0],baseline)
        rt.restore(proposed)
        assert np.array_equal(actual[3],logits(model,x))
        rt.restore(pre)
        assert rt.snapshot().commitment()==pre.commitment()
        dump_npz(path/'ACTUAL.npz',logits=np.stack(actual))
        write(path/'EXECUTION.json',{'receipts':receipts,'actual_sha256':sha(path/'ACTUAL.npz'),
            'restoration_exact':True,'alpha0_exact':True,'alpha1_native_exact':True})
        print('COMPLETE',row['entry_id'],'selected',ALPHAS[selected],flush=True)
    finally:
        rt.close()


def main():
    old.setup()
    assert git('rev-parse','HEAD')=='b026bc7c7f2d8d7665842cbc4137ae4369ca90ee'
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    assignment=old.HERE/'RUN_ASSIGNMENT.json'
    rows=old.read(assignment)['runs'][:3]
    write(OUT/'SEAL.json',{'time_ns':time.time_ns(),'base':git('rev-parse','HEAD'),
        'sources':{str(p.relative_to(REPO)):sha(p) for p in HERE.glob('*') if p.is_file()},
        'old_files':{n:sha(REPO/n) for n in git('ls-files').splitlines()},
        'trainer_sha256':sha(old.TRAINER/'model.py'),'assignment_sha256':sha(assignment),
        'entries':[r['entry_id'] for r in rows],'alphas':ALPHAS,
        'tag_object':git('rev-parse',TAG),'tag_commit':git('rev-parse',TAG+'^{}'),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    for row in rows:
        try:
            case(row)
        except Exception:
            write(OUT/(row['entry_id']+'-FAILURE.json'),{'traceback':traceback.format_exc()})
            raise
    print('ALL_THREE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
