"""Post-execution read-only audit; independent of the runner/metric functions.

Added after formal execution. Does not select actions or change frozen rules.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, why):
    if not value:
        raise RuntimeError(why)


def main():
    manifest = read(HERE / 'ARTIFACT_MANIFEST.json')
    root = Path(manifest['output_root'])
    actual = {str(p.relative_to(root)).replace('\\', '/') for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']), 'Artifact set differs')
    for name, h in manifest['files'].items():
        require(digest(root / name) == h, 'File hash: ' + name)
    report = read(HERE / 'RESULTS.json')
    rows = read(HERE.parent / 'gfg_nanogpt_functional_online_controller_v1' / 'RUN_ASSIGNMENT.json')['runs']
    checked_evaluations = 0
    checked_steps = 0
    for arm in report['pooled_curves']:
        accuracies = []
        for row in rows:
            folder = root / row['entry_id'] / arm
            tr = read(folder / 'TRAJECTORY.json')
            y = np.array([e['target_group'] for e in row['evaluation_identities']])
            require(len(tr['steps']) == 200, 'Step count')
            require([e['t'] for e in tr['evaluations']] == list(range(0, 201, 10)), 'Evaluation grid')
            curve = []
            for e in tr['evaluations']:
                logits = np.asarray(e['logits'], dtype=np.float64)
                correct = [bool(all(logits[i, target] > logits[i, k] for k in range(24) if k != target)) for i, target in enumerate(y)]
                require(correct == e['correct'], 'Independent boundary correctness')
                acc = sum(correct) / len(correct)
                require(acc == e['accuracy'], 'Independent accuracy')
                curve.append(acc)
                checked_evaluations += 1
            accuracies.append(curve)
            previous = None
            for step in tr['steps']:
                fact = step['fact']
                text = json.dumps(fact, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
                require(hashlib.sha256(text.encode()).hexdigest() == step['sha256'], 'Record digest')
                decision = read(folder / ('selection-%04d.json' % fact['t']))
                require(decision['previous_record_sha256'] == previous, 'Chain link')
                require(digest(folder / ('selection-%04d.json' % fact['t'])) == fact['decision_sha256'], 'Decision link')
                previous = step['sha256']
                checked_steps += 1
            for q, data in report['thresholds'].items():
                threshold = float(q)
                found = next((i for i, acc in enumerate(curve) if acc >= threshold), None)
                target = data['arms'][arm]['per_run'][row['entry_id']]
                require(target['first_step'] == (None if found is None else found * 10), 'First threshold')
                t = 200 if found is None else found * 10
                require(target['restricted_steps'] == t, 'Censoring rule')
                seconds = sum(sum(s['seconds'][k] for k in ('training', 'selection', 'application')) for s in tr['steps'][:t])
                require(abs(seconds - target['restricted_core_seconds']) < 1e-10, 'Cost arithmetic')
                drops = [] if found is None else [i * 10 for i in range(found + 1, len(curve)) if curve[i] < threshold]
                require(drops == target['later_below_threshold_steps'], 'Drop accounting')
                stable = next((i * 10 for i in range(len(curve)) if min(curve[i:]) >= threshold), None)
                require(stable == target['stable_grid_step'], 'Sustained grid accounting')
        require(np.array_equal(np.mean(accuracies, axis=0), report['pooled_curves'][arm]), 'Pooled mean')
    result = {'status': 'PASS', 'artifact_files': len(actual), 'hash_chain_steps': checked_steps,
              'independent_boundary_evaluations': checked_evaluations, 'results_sha256': digest(HERE / 'RESULTS.json'),
              'manifest_sha256': digest(HERE / 'ARTIFACT_MANIFEST.json'), 'utc': datetime.now(timezone.utc).isoformat()}
    with (HERE / 'SUMMARY_AUDIT.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(result)


if __name__ == '__main__':
    main()
