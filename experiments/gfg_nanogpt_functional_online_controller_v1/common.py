from pathlib import Path
import hashlib
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch

from experiments.gfg_nanogpt_autonomous_capability_discovery_v1.common import canonical_bytes, file_sha256, payload_sha256, require
from experiments.gfg_nanogpt_support_transition_v1.runtime import CausalTrainingRuntime, StateSnapshot
from experiments.gfg_nanogpt_stepwise_support_transition_v1.runtime import StepwiseTrainingRuntime
from experiments.gfg_nanogpt_stepwise_support_transition_v1.storage import restorable_state_from_manifest
from experiments.gfg_nanogpt_adjacent_response_transport_v1.inventory import load_named_array
from experiments.gfg_nanogpt_support_transition_v1.storage import TensorStore

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = Path('E:/gfg-evidence/nanogpt-adjacent-response-transport-v1/submission')
STATES = Path('D:/gfg-evidence/nanogpt-stepwise-support-transition-formal-v3')
BUNDLES = Path('D:/codex/source-information-continuity-gfg-nanogpt-autonomous-capability-v1/data_private/gfg_nanogpt_mechanism_discovery_archive/gfg')
TRAINER = Path('D:/codex/dependencies/nanoGPT-3adf61e')
OUTPUT = Path('E:/gfg-evidence/nanogpt-functional-online-controller-v1')
BRANCHES = ('native', 'fixed', 'controller')
HORIZON = 200
CADENCE = 10
SCHEMA = 'gfg-functional-online-controller-v1'
sha = file_sha256
identity = payload_sha256

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def utc():
    return datetime.now(timezone.utc).isoformat()

def setup():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(), 'CUDA_REQUIRED')

def load_source(row):
    doc = read(row['receiver_state_path'])
    for key in ('parameters', 'optimizer_exp_avg', 'optimizer_exp_avg_sq', 'optimizer_steps'):
        load_named_array(STATES / row['entry_id'], doc['state'][key])
    return restorable_state_from_manifest(STATES / row['entry_id'], doc['state'])

def runtime(row):
    rt = CausalTrainingRuntime(Path(row['bundle_path']), TRAINER, device='cuda')
    rt.restore(load_source(row))
    return rt

def batch(rt, step):
    return StepwiseTrainingRuntime.load_batch(rt, step)

def select_batch(rt, b):
    rt.training_inputs = b.inputs.to(rt.device)
    rt.training_targets = b.targets.to(rt.device)

def seed_for(row, offset, protocol_hash):
    if offset == 0:
        return row['initial_rng']['seed']
    return CausalTrainingRuntime.derive_seed(protocol_hash, row['entry_id'], row['source_step'] + offset, 1)

def pack_state(store, state):
    return {
        'commitment': state.commitment(),
        'parameters': store.put_named(state.parameters, representation='complete_named_parameters'),
        **{dest: store.put_named({n: s[src] for n, s in state.optimizer.items()}, representation=dest)
           for dest, src in [('optimizer_exp_avg', 'exp_avg'), ('optimizer_exp_avg_sq', 'exp_avg_sq'), ('optimizer_steps', 'step')]},
    }

def pack_rng(store, rt, seed):
    ns = np.random.get_state()
    return {'commitment': rt.rng_commitment(seed),
            'cpu': store.put(torch.get_rng_state(), representation='torch_cpu_rng_state'),
            'cuda': [store.put(s, representation='torch_cuda_rng_state') for s in torch.cuda.get_rng_state_all()],
            'numpy': store.put(ns[1], representation='numpy_rng_words'),
            'numpy_metadata': [ns[0], ns[2], ns[3], ns[4]],
            'python_state': repr(random.getstate())}

def state_hash(state):
    return state.commitment()['state_sha256']

def scaled_state(pre, proposed, delta, alpha):
    # Alpha=1 retains native bytes. Float32 subtraction and readdition are NOT
    # an identity, as already established in the frozen amplitude experiment.
    parameters = proposed.parameters if alpha == 1.0 else {
        name: pre.parameters[name] + alpha * delta[name] for name in pre.parameters}
    return StateSnapshot(parameters, proposed.optimizer)

def norm(values):
    return float(sum(float((v.double() ** 2).sum()) for v in values.values()) ** 0.5)

def verify_seal():
    seal = read(HERE / 'FREEZE_SEAL.json')
    for name, digest in seal['files'].items():
        require(sha(HERE / name) == digest, 'FREEZE_CHANGED:' + name)
    for path, digest in read(HERE / 'CODE_MANIFEST.json')['files'].items():
        require(sha(REPO / path) == digest, 'CODE_CHANGED:' + path)
    require(sha(TRAINER / 'model.py') == read(HERE / 'EXPERIMENT_CONTEXT.json')['trainer_model_sha256'], 'TRAINER_CHANGED')
    for path, digest in read(HERE / 'RUNTIME_MANIFEST.json')['optimizer_sources'].items():
        require(sha(Path(path)) == digest, 'OPTIMIZER_IMPLEMENTATION_CHANGED')
    return seal
