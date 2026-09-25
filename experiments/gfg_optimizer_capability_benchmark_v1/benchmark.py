"""Small official-optimizer benchmark; no modifications to frozen experiments."""
import argparse
import copy
import dataclasses
import enum
import gc
import hashlib
import json
import logging
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for name in ('optimizers', 'Sophia', 'sls'):
    sys.path.insert(0, str(REPO / 'runtime' / name))
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import psutil
import torch
from sophia import SophiaG
from sls.sls import Sls
from distributed_shampoo import DistributedShampoo, AdamPreconditionerConfig
from experiments.gfg_nanogpt_functional_online_controller_v1 import common as old
from experiments.gfg_nanogpt_functional_online_controller_v1.selector import coordinates, select
from experiments.gfg_nanogpt_functional_online_controller_v1.independent_checker import independently_choose
from experiments.gfg_nanogpt_functional_online_controller_v1.runner import selection_boundary, guard
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import tensor_sha256

OUT = Path('E:/gfg-evidence/optimizer-capability-benchmark-v1')
METHODS = ('adamw', 'sophia', 'shampoo', 'sls', 'controller')
RATES = {m: [.0003, .001, .003, .01] for m in METHODS}
RATES['sophia'] = [.00003, .0001, .0003, .001]
RATES['sls'] = [.01, .1, 1., 10.]
KEYS = ('g0', 'first_gap', 'joint_rotation', 'hidden_curvature')
read, write, sha, identity, require = old.read, old.write_new, old.sha, old.identity, old.require


def sync():
    torch.cuda.synchronize()


def params(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def phash(values):
    return identity({n: tensor_sha256(v) for n, v in sorted(values.items())})


def structure(x):
    if isinstance(x, torch.Tensor):
        return {'tensor': tensor_sha256(x.detach().cpu()), 'shape': list(x.shape), 'dtype': str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): structure(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [structure(v) for v in x]
    if dataclasses.is_dataclass(x):
        return {'type': type(x).__name__, 'fields': structure(dataclasses.asdict(x))}
    if isinstance(x, enum.Enum):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (torch.dtype, torch.device)):
        return str(x)
    raise TypeError('unhandled optimizer state type: ' + str(type(x)))


def make_optimizer(model, method, lr):
    ps = list(model.parameters())
    if method in ('adamw', 'controller'):
        return torch.optim.AdamW(ps, lr=lr, betas=(.9, .98), weight_decay=0., foreach=False, fused=False)
    if method == 'sophia':
        return SophiaG(ps, lr=lr, weight_decay=0.)
    if method == 'sls':
        return Sls(ps, n_batches_per_epoch=1, init_step_size=lr)
    return DistributedShampoo(ps, lr=lr, betas=(.9, .98), epsilon=1e-12,
        weight_decay=0., max_preconditioner_dim=64, precondition_frequency=10,
        start_preconditioning_step=10,
        grafting_config=AdamPreconditionerConfig(beta2=.98, epsilon=1e-8))


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)[0][:, -1, :].cpu().numpy()
    target = y[:, -1].cpu().numpy()
    other = logits.copy()
    other[np.arange(len(y)), target] = -np.inf
    gap = logits[np.arange(len(y)), target].astype(np.float64) - other.max(1).astype(np.float64)
    require(np.isfinite(gap).all(), 'NONFINITE_EVAL')
    return {'logits': logits.tolist(), 'correct': (gap > 0).tolist(), 'accuracy': float((gap > 0).mean())}


def update(model, opt, method, lr, t, x, y, vx, vy, seed, cached_alpha, replay=False):
    model.train()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    details = {}
    if method == 'adamw':
        current_lr = lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * (t - 1) / 199)))
        for group in opt.param_groups:
            group['lr'] = current_lr
        details['lr'] = current_lr
    if method == 'sls':
        opt.zero_grad(set_to_none=True)
        before = opt.state['n_forwards']
        # Official code's wall-clock RNG wrapper is deterministically addressed.
        # No algorithm changes; dropout is zero in this model.
        with patch('sls.sls.time.time', return_value=float(seed)):
            loss = opt.step(lambda: model(x, y)[1])
        details.update(loss=float(loss.detach()), forwards=opt.state['n_forwards']-before,
                       step_size=float(opt.state['step_size']))
        return cached_alpha, details, None
    if method == 'sophia' and (t-1) % 10 == 0:
        opt.zero_grad(set_to_none=True)
        logits = model(x, y)[0]
        sampled = torch.distributions.Categorical(logits=logits.detach()).sample()
        sampled[y == -1] = -1
        torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), sampled.reshape(-1), ignore_index=-1).backward()
        opt.update_hessian()
        details['curvature_refresh'] = True
    before = params(model) if method == 'controller' else None
    opt.zero_grad(set_to_none=True)
    loss = model(x, y)[1]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if method == 'sophia':
        opt.step(bs=int((y != -1).sum()))
    else:
        opt.step()
    details['loss'] = float(loss.detach())
    c = None
    if method == 'controller':
        proposal = params(model)
        delta = {n: proposal[n] - before[n] for n in before}
        if (t-1) % 5 == 0:
            with selection_boundary():
                c = coordinates(model, before, delta, vx, vy)
                cached_alpha, scores = independently_choose(c) if replay else select(c)
            require(phash(params(model)) == phash(proposal), 'SELECTOR_MUTATION')
            details['scores'] = scores
        details['alpha'] = cached_alpha
        if cached_alpha != 1.:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_((before[n] + cached_alpha * delta[n]).to(p.device))
    return cached_alpha, details, c


def load_row(row):
    rt = old.runtime(row)
    start = params(rt.model)
    batches = []
    for t in range(200):
        b = old.batch(rt, row['source_step'] + t)
        op = row['opportunities'][t]
        require(hashlib.sha256(b.inputs.numpy().tobytes()).hexdigest() == op['batch_inputs_sha256'], 'SOURCE_BATCH_X')
        require(hashlib.sha256(b.targets.numpy().tobytes()).hexdigest() == op['batch_targets_sha256'], 'SOURCE_BATCH_Y')
        batches.append((b.inputs.to('cuda'), b.targets.to('cuda')))
    vx, vy = rt.base.validation_inputs.to('cuda'), rt.base.validation_targets.to('cuda')
    return rt, start, batches, vx, vy


def execute(row, loaded, method, ci, seal, mode='search', repeat=0):
    rt, start, batches, vx, vy = loaded
    root = OUT / mode / row['entry_id'] / (method + '-' + str(ci)) / str(repeat)
    if (root / 'TRAJECTORY.json').exists():
        return read(root / 'TRAJECTORY.json')
    require(not root.exists(), 'PARTIAL_OUTPUT_PRESERVED:' + str(root))
    root.mkdir(parents=True)
    model = rt.model
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(start[n].to('cuda'))
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    sync(); tick = time.perf_counter()
    opt = make_optimizer(model, method, RATES[method][ci])
    sync(); construction = time.perf_counter() - tick
    init_opt_hash = identity(structure(opt.state_dict()))
    initial = evaluate(model, vx, vy)
    records = []
    prev = None
    alpha = 1.
    original = read(OUT / 'search' / row['entry_id'] / (method+'-'+str(ci)) / '0/TRAJECTORY.json') if mode == 'timing' else None
    require(original is None or original['initial'] == initial, 'INITIAL_REPLAY')
    rss_peak = psutil.Process().memory_info().rss
    try:
        for t in range(1, 201):
            if (records[-1]['fact']['evaluation'] if records else initial)['accuracy'] >= .99:
                break
            x, y = batches[t-1]
            seed = old.seed_for(row, t-1, seal['source_rng_protocol_sha256'])
            sync(); tick = time.perf_counter()
            alpha, details, c = update(model, opt, method, RATES[method][ci], t, x, y, vx, vy, seed, alpha, mode == 'timing')
            sync(); compute = time.perf_counter()-tick
            coord_hash = None
            if c is not None:
                coord_hash = identity({k: tensor_sha256(torch.from_numpy(c[k])) for k in KEYS})
                if mode == 'search':
                    with (root / ('coords-%04d.npz' % t)).open('xb') as f:
                        np.savez_compressed(f, **{k: c[k] for k in KEYS})
                else:
                    coordpath = OUT / 'search' / row['entry_id'] / (method+'-'+str(ci)) / '0' / ('coords-%04d.npz' % t)
                    with np.load(coordpath, allow_pickle=False) as saved:
                        require(all(np.array_equal(saved[k], c[k]) for k in KEYS), 'COORD_REPLAY')
            # Commit the choice before reading its target endpoint.
            decision = {'t': t, 'details': details, 'coordinates_sha256': coord_hash}
            write(root / ('decision-%04d.json' % t), decision)
            post_hash = phash(params(model))
            sync(); tick = time.perf_counter()
            ev = evaluate(model, vx, vy)
            sync(); eval_time = time.perf_counter()-tick
            fact = {'t': t, 'seed': seed, 'batch_sha256': identity(row['opportunities'][t-1]),
                    'decision': decision, 'post_parameters_sha256': post_hash,
                    'evaluation': ev, 'previous_sha256': prev}
            prev = identity(fact)
            records.append({'fact': fact, 'sha256': prev, 'compute_seconds': compute, 'evaluation_seconds': eval_time})
            if original is not None:
                require(fact == original['steps'][t-1]['fact'], 'NATIVE_REPLAY:' + row['entry_id'] + ':' + method + ':' + str(t))
            rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
        final = {'parameters': params(model), 'optimizer': opt.state_dict()}
        final_hash = identity(structure(final))
        if original:
            require(len(records) == len(original['steps']) and final_hash == original['final_state_sha256'], 'FINAL_REPLAY')
        with (root / 'FINAL.pt').open('xb') as f:
            torch.save(final, f)
        result = {'entry_id': row['entry_id'], 'method': method, 'config': ci, 'lr': RATES[method][ci],
            'protocol_sha256': seal['files']['PROTOCOL_FREEZE.md'], 'initial_parameters_sha256': phash(start),
            'target_ids': vy[:, -1].cpu().tolist(), 'training_batch_shape': list(batches[0][0].shape),
            'initial_optimizer_sha256': init_opt_hash, 'initial': initial, 'steps': records,
            'construction_seconds': construction, 'wall_seconds': time.perf_counter()-wall_start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'peak_process_rss_bytes': rss_peak, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'final_state_sha256': final_hash, 'final_file_sha256': sha(root/'FINAL.pt'),
            'native_replay_exact': mode == 'timing'}
        write(root / 'TRAJECTORY.json', result)
        print(mode, row['entry_id'], method, ci, repeat, 'steps', len(records), 'acc', (records[-1]['fact']['evaluation'] if records else initial)['accuracy'], flush=True)
        return result
    except Exception as ex:
        write(root / ('FAILURE-'+str(time.time_ns())+'.json'), {'error': repr(ex), 'utc': old.utc()})
        raise
    finally:
        del opt


def crossing(result, threshold):
    if result['initial']['accuracy'] >= threshold:
        return 0
    return next((s['fact']['t'] for s in result['steps'] if s['fact']['evaluation']['accuracy'] >= threshold), None)


def selections(rows):
    entries = {}
    for row in rows:
        entry = row['entry_id']
        entries[entry] = {}
        for method in METHODS:
            ranks = []
            for ci in range(4):
                costs = []
                for threshold in (.99, .95):
                    values = []
                    for other in rows:
                        if other['entry_id'] == entry:
                            continue
                        r = read(OUT/'search'/other['entry_id']/(method+'-'+str(ci))/'0/TRAJECTORY.json')
                        hit = crossing(r, threshold)
                        if hit != 0:
                            values.append(200 if hit is None else hit)
                    costs.append(float(np.mean(values)) if values else 0.)
                ranks.append([*costs, ci])
            entries[entry][method] = {'config': min(ranks)[-1], 'ranks': ranks}
    return {'rule': 'leave-one-history-out, restricted99 then restricted95 then index', 'entries': entries}


def dependency_manifest():
    out = {}
    for name in ('Sophia', 'sls', 'optimizers'):
        root = REPO/'runtime'/name
        tracked = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
        out[name] = {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                     'files': {p: sha(root/p) for p in tracked if p.endswith('.py') or p.startswith('LICENSE')}}
    return out


def smoke():
    from experiments.gfg_nanogpt_full_network_receptive_state_v1.analysis import _model
    torch.manual_seed(123)
    model = _model(old.TRAINER).cuda()
    start = params(model)
    x = torch.randint(0,23,(317,3),device='cuda')
    y = torch.full_like(x, -1); y[:,-1] = torch.randint(0,23,(317,),device='cuda')
    outcomes = {}
    for m in METHODS:
        endings = []
        for rep in range(2):
            with torch.no_grad():
                for n,p in model.named_parameters(): p.copy_(start[n].to('cuda'))
            opt = make_optimizer(model,m,RATES[m][1])
            a = 1.
            for t in range(1,12):
                a,d,c = update(model,opt,m,RATES[m][1],t,x,y,x[:212],y[:212],900+t,a,rep==1)
            endings.append(identity(structure({'parameters':params(model),'optimizer':opt.state_dict()})))
        require(endings[0] == endings[1], 'SMOKE_REPLAY:'+m)
        outcomes[m] = {'eleven_steps_exact_replay':True,'final_sha256':endings[0]}
    return outcomes


def sealcheck():
    seal = read(HERE/'FREEZE_SEAL.json')
    for p,h in seal['files'].items(): require(sha(HERE/p)==h,'SEALED_FILE:'+p)
    require(dependency_manifest()==seal['dependencies'],'DEPENDENCY_CHANGED')
    require(sha(old.HERE/'RUN_ASSIGNMENT.json')==seal['assignment_sha256'],'ASSIGNMENT_CHANGED')
    old.verify_seal()
    return seal


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['smoke','freeze','search','timing','bridge'])
    args=ap.parse_args()
    old.setup(); sys.addaudithook(guard); logging.basicConfig(level=logging.ERROR)
    if args.action=='smoke':
        print(json.dumps(smoke(),indent=2)); return
    if args.action=='freeze':
        require(not OUT.exists(),'OUTPUT_ALREADY_EXISTS')
        old.verify_seal()
        report=smoke(); write(HERE/'SELF_TEST.json',report)
        write(HERE/'FREEZE_SEAL.json',{'files':{p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')},
            'dependencies':dependency_manifest(),'assignment_sha256':sha(old.HERE/'RUN_ASSIGNMENT.json'),
            'source_rng_protocol_sha256':sha(old.HERE/'PROTOCOL_FREEZE.md'),'base_commit':'0dc78f0c509ed9ce09b09b585151e41ecd9a6f71',
            'torch':torch.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'utc':old.utc()})
        print('FROZEN'); return
    seal=sealcheck()
    rows=read(old.HERE/'RUN_ASSIGNMENT.json')['runs']
    if args.action=='bridge':
        from experiments.gfg_nanogpt_functional_controller_cadence_v1 import experiment as previous
        report=previous.run(rows[0],'controller_5',previous.check_seal(),replay=True)
        write(HERE/'SOURCE_BRIDGE_CHECK.json',report); return
    smoke() # synthetic warmup excluded from results
    chosen=selections(rows) if args.action=='timing' else None
    if chosen and not (HERE/'LOO_SELECTION.json').exists(): write(HERE/'LOO_SELECTION.json',chosen)
    for i,row in enumerate(rows):
        loaded=load_row(row)
        try:
            if args.action=='search':
                for ci in range(4):
                    order=METHODS[(i+ci)%5:]+METHODS[:(i+ci)%5]
                    for m in order: execute(row,loaded,m,ci,seal)
            else:
                for repeat in range(3):
                    order=METHODS[(i+repeat)%5:]+METHODS[:(i+repeat)%5]
                    for m in order: execute(row,loaded,m,chosen['entries'][row['entry_id']][m]['config'],seal,'timing',repeat)
        finally:
            loaded[0].close(); del loaded; gc.collect()


if __name__=='__main__': main()
