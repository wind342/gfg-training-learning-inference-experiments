"""Frozen threshold/time summaries; all twelve runs retained, no fitted rule."""
from .experiment import *


def cost(trajectory, t, operational=False):
    fields = ('training', 'selection', 'application', 'setup') if operational else ('training', 'selection', 'application')
    return float(sum(sum(s['seconds'][k] for k in fields) for s in trajectory['steps'][:t]))


def crossing(trajectory, threshold):
    es = trajectory['evaluations']
    hits = [i for i, e in enumerate(es) if e['accuracy'] >= threshold]
    i = hits[0] if hits else None
    t = es[i]['t'] if i is not None else None
    stable = next((j for j in range(len(es)) if all(e['accuracy'] >= threshold for e in es[j:])), None)
    return {'first_step': t, 'first_core_seconds': None if t is None else cost(trajectory, t),
        'first_operational_seconds': None if t is None else cost(trajectory, t, True),
        'restricted_steps': 200 if t is None else t,
        'restricted_core_seconds': cost(trajectory, 200 if t is None else t),
        'restricted_operational_seconds': cost(trajectory, 200 if t is None else t, True),
        'later_below_threshold_steps': [] if i is None else [e['t'] for e in es[i + 1:] if e['accuracy'] < threshold],
        'stable_grid_step': None if stable is None else es[stable]['t']}


def comparison(xs, ys):
    differences = np.array(xs) - np.array(ys)
    return {'mean_difference': float(differences.mean()), 'lower': int((differences < 0).sum()),
            'equal': int((differences == 0).sum()), 'higher': int((differences > 0).sum())}


def summarize():
    check_seal()
    rows = read(old.HERE / 'RUN_ASSIGNMENT.json')['runs']
    trajectories = {arm: [read(OUTPUT / row['entry_id'] / arm / 'TRAJECTORY.json') for row in rows] for arm in ARMS}
    result = {'scope': '12 historical-state continuations; same labeled control/evaluation targets; 10-step observation grid',
              'thresholds': {}, 'total_seconds': {}, 'final_accuracy': {}, 'pooled_curves': {}}
    for arm, ts in trajectories.items():
        result['total_seconds'][arm] = {k: sum(t['seconds'][k] for t in ts) for k in ts[0]['seconds']}
        result['total_seconds'][arm]['core'] = sum(cost(t, 200) for t in ts)
        result['total_seconds'][arm]['operational'] = sum(cost(t, 200, True) for t in ts)
        result['final_accuracy'][arm] = float(np.mean([t['evaluations'][-1]['accuracy'] for t in ts]))
        result['pooled_curves'][arm] = np.mean([[e['accuracy'] for e in t['evaluations']] for t in ts], axis=0).tolist()
    for threshold in (.95, .99):
        data = {'arms': {}, 'paired': {}}
        for arm, ts in trajectories.items():
            xs = [crossing(t, threshold) for t in ts]
            curve = result['pooled_curves'][arm]
            pooled_t = next((i * 10 for i, a in enumerate(curve) if a >= threshold), None)
            data['arms'][arm] = {'per_run': {r['entry_id']: x for r, x in zip(rows, xs)},
                'reached': sum(x['first_step'] is not None for x in xs),
                'regressed_after_first': sum(bool(x['later_below_threshold_steps']) for x in xs),
                'mean_restricted_steps': float(np.mean([x['restricted_steps'] for x in xs])),
                'mean_restricted_core_seconds': float(np.mean([x['restricted_core_seconds'] for x in xs])),
                'mean_restricted_operational_seconds': float(np.mean([x['restricted_operational_seconds'] for x in xs])),
                'pooled_first_step': pooled_t,
                'pooled_first_core_seconds_mean_per_run': None if pooled_t is None else float(np.mean([cost(t, pooled_t) for t in ts]))}
        for arm in ('controller_2', 'controller_5'):
            for reference in ('native', 'fixed', 'controller_1'):
                a = list(data['arms'][arm]['per_run'].values())
                b = list(data['arms'][reference]['per_run'].values())
                data['paired'][arm + '_minus_' + reference] = {key: comparison([x[key] for x in a], [x[key] for x in b])
                    for key in ('restricted_steps', 'restricted_core_seconds', 'restricted_operational_seconds')}
        result['thresholds'][str(threshold)] = data
    return result


def main():
    result = summarize()
    write_new(HERE / 'RESULTS.json', result)
    files = {str(p.relative_to(OUTPUT)).replace('\\', '/'): sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    write_new(HERE / 'ARTIFACT_MANIFEST.json', {'output_root': str(OUTPUT), 'files': files,
        'total_bytes': sum((OUTPUT / p).stat().st_size for p in files)})
    for threshold, data in result['thresholds'].items():
        print('THRESHOLD', threshold)
        for arm, a in data['arms'].items():
            print(arm, {k: v for k, v in a.items() if k != 'per_run'})
    print('TOTAL_SECONDS', result['total_seconds'])


if __name__ == '__main__':
    main()
