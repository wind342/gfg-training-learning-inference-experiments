"""Compact, exclusively new cadence experiment; frozen source imports unchanged."""
import argparse
import gzip
import platform
import sys
import time

from experiments.gfg_nanogpt_functional_online_controller_v1 import common as old
from experiments.gfg_nanogpt_functional_online_controller_v1.common import *
from experiments.gfg_nanogpt_functional_online_controller_v1.selector import coordinates, select
from experiments.gfg_nanogpt_functional_online_controller_v1.runner import selection_boundary, guard
from experiments.gfg_nanogpt_functional_online_controller_v1.independent_checker import independently_choose
from experiments.gfg_nanogpt_support_redundancy_v1.runtime import tensor_sha256

HERE = Path(__file__).resolve().parent
OUTPUT = Path('E:/gfg-evidence/nanogpt-functional-controller-cadence-v1')
ARMS = ('native', 'fixed', 'controller_1', 'controller_2', 'controller_5')
KEYS = ('g0', 'first_gap', 'joint_rotation', 'hidden_curvature')


def fresh(arm, t):
    return arm.startswith('controller_') and (t - 1) % int(arm.rsplit('_', 1)[1]) == 0


def tensor_map_hash(values):
    return identity({k: tensor_sha256(v) for k, v in sorted(values.items())})


def synchronize():
    torch.cuda.synchronize()


def evaluate(rt, t):
    rt.model.eval()
    before = state_hash(rt.snapshot())
    with torch.no_grad():
        logits = rt.base.forward().numpy()
    y = rt.base.validation_targets[:, -1].numpy()
    other = logits.copy()
    other[np.arange(len(y)), y] = -np.inf
    margins = logits[np.arange(len(y)), y].astype(np.float64) - other.max(1).astype(np.float64)
    require(before == state_hash(rt.snapshot()), 'EVAL_MUTATION')
    return {'t': t, 'state_sha256': before, 'logits': logits.tolist(),
            'correct': (margins > 0).tolist(), 'accuracy': float((margins > 0).mean())}


def install(rt, pre, proposal, delta, alpha):
    if alpha != 1.0:
        with torch.no_grad():
            for name, p in rt.named_parameters.items():
                p.copy_((pre.parameters[name] + alpha * delta[name]).to(rt.device))


def check_seal():
    old.verify_seal()
    seal = read(HERE / 'FREEZE_SEAL.json')
    for p, h in seal['files'].items():
        require(sha(HERE / p) == h, 'CADENCE_FREEZE_CHANGED:' + p)
    require(sha(old.HERE / 'RUN_ASSIGNMENT.json') == seal['source_assignment_sha256'], 'ASSIGNMENT')
    return seal


def run(row, arm, seal, replay=False):
    root = OUTPUT / row['entry_id'] / arm
    if replay:
        original = read(root / 'TRAJECTORY.json')
        require(original['protocol_sha256'] == seal['files']['PROTOCOL_FREEZE.md'], 'PROTOCOL_BINDING')
    else:
        require(not root.exists(), 'NO_OVERWRITE:' + str(root))
        root.mkdir(parents=True)
    rt = runtime(row)
    wall_start = time.perf_counter()
    stats = {'training': 0., 'selection': 0., 'application': 0., 'setup': 0., 'evaluation': 0.}
    records, evaluations = [], []
    previous = None
    alpha, selected_at = (1.0 if arm == 'native' else .25), 0
    old_arm = {'native': 'native', 'fixed': 'fixed', 'controller_1': 'controller'}.get(arm)
    try:
        rt.set_rng(row['initial_rng']['seed'])
        require(rt.rng_commitment(row['initial_rng']['seed']) == row['initial_rng'], 'INITIAL_RNG')
        require(state_hash(rt.snapshot()) == row['initial_state_sha256'], 'INITIAL_STATE')
        start = time.perf_counter()
        e = evaluate(rt, 0)
        stats['evaluation'] += time.perf_counter() - start
        evaluations.append(e)
        if replay:
            require(e == original['evaluations'][0], 'INITIAL_EVAL_REPLAY')
        for t in range(1, 201):
            synchronize()
            start = time.perf_counter()
            pre = rt.snapshot()
            b = batch(rt, row['source_step'] + t - 1)
            select_batch(rt, b)
            seed = seed_for(row, t - 1, seal['source_rng_protocol_sha256'])
            rt.set_rng(seed)
            synchronize()
            setup_seconds = time.perf_counter() - start
            op = row['opportunities'][t - 1]
            require(hashlib.sha256(b.inputs.numpy().tobytes()).hexdigest() == op['batch_inputs_sha256'], 'BATCH_INPUT')
            require(hashlib.sha256(b.targets.numpy().tobytes()).hexdigest() == op['batch_targets_sha256'], 'BATCH_TARGET')
            synchronize()
            start = time.perf_counter()
            native = rt.train_step(execute_optimizer=True, seed=seed)
            proposal = rt.snapshot()
            synchronize()
            training_seconds = time.perf_counter() - start
            scores, coord_hash = None, None
            selection_seconds = 0.
            if fresh(arm, t):
                synchronize()
                start = time.perf_counter()
                with selection_boundary():
                    c = coordinates(rt.model, pre.parameters, native.parameter_updates,
                                    rt.base.validation_inputs.to(rt.device), rt.base.validation_targets.to(rt.device))
                    alpha, scores = independently_choose(c) if replay else select(c)
                synchronize()
                selection_seconds = time.perf_counter() - start
                selected_at = t
                require(state_hash(rt.snapshot()) == state_hash(proposal), 'COORDINATE_MUTATION')
                coord_path = root / ('coords-%04d.npz' % t)
                if replay:
                    with np.load(coord_path, allow_pickle=False) as saved:
                        require(set(saved.files) == set(KEYS), 'COORD_KEYS')
                        require(all(np.array_equal(saved[k], c[k]) for k in KEYS), 'COORDINATE_REPLAY')
                else:
                    with coord_path.open('xb') as f:
                        np.savez_compressed(f, **{k: c[k] for k in KEYS})
                coord_hash = sha(coord_path)
            decision = {'t': t, 'entry_id': row['entry_id'], 'arm': arm,
                        'pre_state_sha256': state_hash(pre), 'proposal_state_sha256': state_hash(proposal),
                        'delta_sha256': tensor_map_hash(native.parameter_updates), 'alpha': alpha,
                        'selected_at': selected_at, 'scores': scores, 'coordinate_file_sha256': coord_hash,
                        'protocol_sha256': seal['files']['PROTOCOL_FREEZE.md'], 'previous_record_sha256': previous}
            selection_path = root / ('selection-%04d.json' % t)
            if replay:
                require(read(selection_path) == decision, 'SEALED_DECISION_REPLAY')
            else:
                write_new(selection_path, decision)
            synchronize()
            start = time.perf_counter()
            install(rt, pre, proposal, native.parameter_updates, alpha)
            synchronize()
            application_seconds = time.perf_counter() - start
            post = rt.snapshot()
            require(post.commitment()['optimizer'] == proposal.commitment()['optimizer'], 'MOMENTS_CHANGED')
            if alpha == 1.:
                require(state_hash(post) == state_hash(proposal), 'NATIVE_ENDPOINT_BYTES')
            measurements = {'training': training_seconds, 'selection': selection_seconds,
                            'application': application_seconds, 'setup': setup_seconds}
            for k, value in measurements.items():
                stats[k] += value
            fact = {'t': t, 'decision_sha256': sha(selection_path), 'seed': seed,
                    'batch_sha256': identity(op), 'post_state_sha256': state_hash(post),
                    'raw_gradients_sha256': tensor_map_hash(native.raw_gradients),
                    'clipped_gradients_sha256': tensor_map_hash(native.clipped_gradients),
                    'rng_before': native.rng_before, 'rng_after': native.rng_after,
                    'loss': native.loss, 'decision': decision}
            previous = identity(fact)
            record = {'fact': fact, 'sha256': previous, 'seconds': measurements}
            if replay:
                require(fact == original['steps'][t - 1]['fact'], 'NATIVE_STEP_REPLAY:%s:%s:%s' % (row['entry_id'], arm, t))
                require(previous == original['steps'][t - 1]['sha256'], 'CHAIN_HASH')
                if old_arm:
                    archived = read(old.OUTPUT / row['entry_id'] / old_arm / 'steps' / ('%04d.json' % t))
                    require(state_hash(post) == archived['application']['state']['commitment']['state_sha256'], 'OLD_CONTROL_CHANGED')
            records.append(record)
            if t % 10 == 0:
                start = time.perf_counter()
                e = evaluate(rt, t)
                stats['evaluation'] += time.perf_counter() - start
                evaluations.append(e)
                if replay:
                    require(e == original['evaluations'][t // 10], 'EVALUATION_REPLAY')
            if t % 100 == 0:
                print('REPLAY' if replay else 'RUN', row['entry_id'], arm, t, 'accuracy', e['accuracy'], flush=True)
        if not replay:
            with (root / 'FINAL.pt').open('xb') as f:
                torch.save({'parameters': post.parameters, 'optimizer': post.optimizer}, f)
            stats['wall'] = time.perf_counter() - wall_start
            write_new(root / 'TRAJECTORY.json', {'entry_id': row['entry_id'], 'arm': arm,
                'protocol_sha256': seal['files']['PROTOCOL_FREEZE.md'], 'initial_state_sha256': row['initial_state_sha256'],
                'steps': records, 'evaluations': evaluations, 'seconds': stats, 'final_state_sha256': state_hash(post),
                'final_file_sha256': sha(root / 'FINAL.pt')})
        else:
            require(sha(root / 'FINAL.pt') == original['final_file_sha256'], 'FINAL_FILE_HASH')
            final = torch.load(root / 'FINAL.pt', map_location='cpu', weights_only=True)
            require(state_hash(StateSnapshot(final['parameters'], final['optimizer'])) == state_hash(post), 'FINAL_BYTES')
        return {'entry_id': row['entry_id'], 'arm': arm, 'steps': 200, 'coordinate_replays': sum(fresh(arm, t) for t in range(1, 201)), 'status': 'PASS'}
    except Exception as ex:
        prefix = 'REPLAY_FAILURE-' if replay else 'FAILURE-'
        write_new(root / (prefix + str(time.time_ns()) + '.json'), {'error': repr(ex), 'utc': utc()})
        raise
    finally:
        rt.close()


def self_test():
    from experiments.gfg_nanogpt_functional_online_controller_v1.self_test import synthetic_runtime, _model
    for k in (1, 2, 5):
        require([t for t in range(1, 201) if fresh('controller_' + str(k), t)] == list(range(1, 201, k)), 'CADENCE')
    torch.manual_seed(719)
    model = _model(TRAINER).cuda()
    x, y = (torch.randint(0, 23, (212, 3), device='cuda') for _ in range(2))
    rt = synthetic_runtime(model, x, y)
    rt.train_step(execute_optimizer=True, seed=719)
    for t in range(6):
        pre = rt.snapshot()
        n = rt.train_step(execute_optimizer=True, seed=720 + t)
        proposal = rt.snapshot()
        c = coordinates(model, pre.parameters, n.parameter_updates, x, y)
        require(select(c) == independently_choose(c), 'SELECTOR_INDEPENDENCE')
        for a in (.25, .5, .75, 1.):
            rt.restore(proposal)
            install(rt, pre, proposal, n.parameter_updates, a)
            require(state_hash(rt.snapshot()) == state_hash(scaled_state(pre, proposal, n.parameter_updates, a)), 'INSTALL')
        rt.restore(proposal)
    return {'status': 'PASS', 'synthetic_steps': 6, 'cadences': [1, 2, 5], 'formal_data_read': False}


def freeze():
    require(not OUTPUT.exists(), 'OUTPUT_EXISTS_BEFORE_FREEZE')
    old.verify_seal()
    test = self_test()
    write_new(HERE / 'SELF_TEST.json', test)
    files = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md', '.json')]
    write_new(HERE / 'FREEZE_SEAL.json', {'files': {p.name: sha(p) for p in files},
        'source_assignment_sha256': sha(old.HERE / 'RUN_ASSIGNMENT.json'),
        'source_rng_protocol_sha256': sha(old.HERE / 'PROTOCOL_FREEZE.md'),
        'source_commit': '706795db4a838fd1487d6ce278dec93b57b5f2e2', 'created_utc': utc(),
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'gpu': torch.cuda.get_device_name(0), 'source_code_manifest_sha256': sha(old.HERE / 'CODE_MANIFEST.json')})
    print('FROZEN', test, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('freeze', 'run', 'check'))
    args = ap.parse_args()
    setup()
    sys.addaudithook(guard)
    if args.action == 'freeze':
        freeze()
        return
    seal = check_seal()
    self_test()  # synthetic warmup, excluded from formal timing
    results = []
    for i, row in enumerate(read(old.HERE / 'RUN_ASSIGNMENT.json')['runs']):
        for arm in ARMS[i % 5:] + ARMS[:i % 5]:
            root = OUTPUT / row['entry_id'] / arm
            if args.action == 'run' and (root / 'TRAJECTORY.json').exists():
                require(read(root / 'TRAJECTORY.json')['protocol_sha256'] == seal['files']['PROTOCOL_FREEZE.md'], 'RESUME_PROTOCOL')
                continue
            results.append(run(row, arm, seal, replay=args.action == 'check'))
    if args.action == 'check':
        write_new(HERE / 'INDEPENDENT_CHECK.json', {'status': 'PASS', 'branches': results,
            'total_native_steps': sum(x['steps'] for x in results),
            'total_coordinate_replays': sum(x['coordinate_replays'] for x in results),
            'old_control_native_step_matches': 7200, 'created_utc': utc()})


if __name__ == '__main__':
    main()
