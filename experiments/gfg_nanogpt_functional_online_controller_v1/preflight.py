"""Read source identities and inputs only; never execute a formal training step."""
import argparse
import subprocess
import sys
import shutil
import numpy as np
import torch
from .common import *

def audit():
    setup()
    inv = read(SOURCE / 'RESOLVED_INVENTORY.json')
    sources = read(SOURCE / 'SELECTION_MANIFEST.json')['entry_sources']
    identities = read(SOURCE / 'IDENTITY_MATERIAL.json')['entries']
    rows = []
    grouped = {}
    for section in inv['sections']:
        grouped.setdefault(section['entry_id_audit_only'], []).append(section)
    require(len(grouped) == 12, 'RUN_COUNT')
    manifest = {str(SOURCE / n): sha(SOURCE / n) for n in ['RESOLVED_INVENTORY.json', 'SELECTION_MANIFEST.json', 'IDENTITY_MATERIAL.json']}
    for entry, sections in sorted(grouped.items()):
        chosen = min(sections, key=lambda s: (read(s['receiver_state_path'])['optimizer_step'], s['section_id']))
        state_doc = read(chosen['receiver_state_path'])
        original = sources[entry]['original_training_gfg']
        bundle = BUNDLES / original['bundle_manifest_sha256']
        row = {'entry_id': entry, 'source_step': state_doc['optimizer_step'],
               'section_id': chosen['section_id'], 'receiver_state_path': chosen['receiver_state_path'],
               'evaluation_input_path': chosen['evaluation_input_path'],
               'initial_rng_source_path': chosen['transition_path'],
               'initial_rng': read(chosen['transition_path'])['step']['rng_before'],
               'bundle_path': str(bundle), 'original_training_gfg': original,
               'evaluation_identities': identities[entry], 'opportunities': []}
        rt = runtime(row)
        try:
            require(rt.set_rng(row['initial_rng']['seed']) == row['initial_rng'], 'INITIAL_RNG_NOT_EXACT:' + entry)
            row['initial_state_sha256'] = state_hash(rt.snapshot())
            expected = np.array([r['target_group'] for r in identities[entry]])
            require(np.array_equal(rt.base.validation_targets[:, -1].cpu().numpy(), expected), 'TARGET_IDENTITIES')
            require(np.array_equal(rt.base.validation_inputs.cpu().numpy(), np.load(chosen['evaluation_input_path'])), 'EVALUATION_INPUTS')
            for offset in range(HORIZON):
                b = batch(rt, row['source_step'] + offset)
                row['opportunities'].append({'offset': offset, 'source_step': b.optimizer_step,
                    'batch_inputs_sha256': hashlib.sha256(b.inputs.numpy().tobytes()).hexdigest(),
                    'batch_targets_sha256': hashlib.sha256(b.targets.numpy().tobytes()).hexdigest(),
                    'source_rows': b.source_rows, 'selection_order_disposition': b.selection_order_disposition})
            for path in [Path(chosen['receiver_state_path']), Path(chosen['transition_path']), Path(chosen['evaluation_input_path']), bundle / 'capture_manifest.json', bundle / 'participant_gfg.sqlite3']:
                manifest[str(path)] = sha(path)
            require(manifest[str(bundle / 'participant_gfg.sqlite3')] == original['declared_gfg_database_sha256'], 'SOURCE_DB_IDENTITY')
            capture = read(bundle / 'capture_manifest.json')
            declared = capture.pop('manifest_sha256')
            require(identity(capture) == declared == original['capture_manifest_sha256'], 'CAPTURE_IDENTITY')
            for key in ('parameters','optimizer_exp_avg','optimizer_exp_avg_sq','optimizer_steps'):
                p = STATES / entry / state_doc['state'][key]['locator']
                manifest[str(p)] = sha(p)
            rows.append(row)
            print('SOURCE_READY', entry, row['source_step'], len(row['opportunities']), 'RNG_EXACT', flush=True)
        finally:
            rt.close()
    write_new(HERE / 'RUN_ASSIGNMENT.json', {'selection': 'minimum receiver optimizer step then section ID among six existing smoke sections per run; no new outcome selection', 'runs': rows})
    write_new(HERE / 'SOURCE_MANIFEST.json', {'files': manifest, 'status': 'HASH_VERIFIED_INPUTS_ONLY'})
    write_new(HERE / 'EXPERIMENT_CONTEXT.json', {'schema': SCHEMA, 'created_utc': utc(),
        'horizon': HORIZON, 'cadence': CADENCE, 'branches': BRANCHES, 'targets_per_run': 212,
        'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
        'python_executable': sys.executable, 'gpu': torch.cuda.get_device_name(),
        'trainer_model_sha256': sha(TRAINER / 'model.py'),
        'parent_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        'frozen_research_commit': '36dab5ce347dbbdac157ef23205f556606d18294',
        'output_root': str(OUTPUT), 'available_output_bytes': shutil.disk_usage(OUTPUT.parent).free,
        'formal_outcomes_accessed': False, 'initial_rng_seed_reconstruction_matches_all_saved_hashes': True,
        'control_and_evaluation_targets_are_same': True})

def freeze():
    require(not OUTPUT.exists(), 'FORMAL_OUTPUT_ALREADY_EXISTS')
    require(read(HERE / 'SELF_TEST.json')['status'] == 'PASS', 'SELF_TEST_REQUIRED')
    import inspect
    import scipy
    import importlib
    optimizer_modules=[importlib.import_module(n) for n in ('torch.optim.adamw','torch.optim.adam')]
    write_new(HERE / 'RUNTIME_MANIFEST.json', {
        'torch':torch.__version__, 'numpy':np.__version__, 'scipy':scipy.__version__,
        'optimizer_sources':{inspect.getfile(m):sha(Path(inspect.getfile(m))) for m in optimizer_modules},
        'cuda_runtime':torch.version.cuda, 'python':sys.version,
        'dependency_added_before_freeze':'scipy==1.16.3, no dependency upgrades',
    })
    files = {str(p.relative_to(REPO)).replace('\\','/'): sha(p) for p in sorted((REPO / 'experiments').rglob('*.py'))}
    write_new(HERE / 'CODE_MANIFEST.json', {'files': files, 'scope': 'all repository experiment Python sources, including imported dependencies'})
    names = ['PROTOCOL_FREEZE.md','EXPERIMENT_CONTEXT.json','SOURCE_MANIFEST.json','RUN_ASSIGNMENT.json','CODE_MANIFEST.json','SELF_TEST.json','RUNTIME_MANIFEST.json']
    seal = {'created_utc': utc(), 'status': 'FROZEN_BEFORE_FORMAL_EXECUTION', 'files': {n: sha(HERE / n) for n in names}}
    write_new(HERE / 'FREEZE_SEAL.json', seal)
    print('FROZEN', sha(HERE / 'FREEZE_SEAL.json'), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['audit','freeze'])
    args = parser.parse_args()
    audit() if args.mode == 'audit' else freeze()
