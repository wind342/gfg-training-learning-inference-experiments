"""Reuse the sealed native executor; add only fixed displacement and LOO design."""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import gc
import statistics as st
import subprocess
import traceback
import torch
import numpy as np
from experiments.gfg_controller_matched_adamw_v1 import run as native
from .policy import StaticControl

b = native.b
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = Path('E:/gfg-evidence/controller-fixed-shrinkage-control-v1')
BASE = 'ab10b96129c3139d3fa4c50532a474aebfeeeca9'
TAG = 'paper-experiments-cross-system-feedback-release'
FROZEN = '36dab5ce347dbbdac157ef23205f556606d18294'
RATES = (.001, .003, .01)
FIXED = {'fixed1': 1., 'fixed075': .75, 'fixed05': .5, 'fixed025': .25}
POLICIES = (*FIXED, 'dynamic')
ARMS = ('native', 'loo_fixed', 'dynamic')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def location(policy, mode, entry, ci, rep=0):
    # Native executor calls every engine 'controller'; outer path supplies policy.
    return OUT/policy/mode/entry/f'controller-{ci}'/str(rep)/'TRAJECTORY.json'


def record(policy, mode, entry, ci, rep=0):
    return b.read(location(policy, mode, entry, ci, rep))


def execute(row, loaded, policy, ci, rng, mode, rep=0):
    factory = native.Controller if policy == 'dynamic' else lambda model, *args: StaticControl(model, FIXED[policy])
    with patch.object(native, 'OUT', OUT/policy), patch.object(native, 'RATES', RATES), patch.object(native, 'Controller', factory):
        return native.run_case(row, loaded, 'controller', ci, rng, mode, rep)


def hit(r, threshold):
    return native.hit(r, threshold)


def select_fixed(rows):
    choices = {}
    for row in rows:
        entry = row['entry_id']; choices[entry] = {}
        for ci in range(3):
            ranks = []
            for p, a in FIXED.items():
                costs = []
                for threshold in (.99, .95):
                    values = [hit(record(p, 'search', o['entry_id'], ci), threshold) for o in rows if o['entry_id'] != entry]
                    costs.append(st.mean(200 if h is None else h for h in values if h != 0))
                ranks.append([*costs, -a, p])
            choices[entry][str(ci)] = {'policy': min(ranks)[-1], 'ranks': ranks}
    return choices


def verdict(regimes):
    support = [regimes[str(lr)]['extra_value_supported'] for lr in RATES]
    fixed_dominates = []
    native_dominates = []
    any_strict = False
    for lr in RATES:
        arms = regimes[str(lr)]['thresholds']['0.99']; d = arms['dynamic']
        for name, output in [('loo_fixed', fixed_dominates), ('native', native_dominates)]:
            f = arms[name]
            output.append(f['reached'] >= d['reached'] and f['restricted_steps_mean'] <= d['restricted_steps_mean'])
            any_strict |= f['reached'] > d['reached'] or f['restricted_steps_mean'] < d['restricted_steps_mean']
    if any(support[1:]): label = 'ADAPTIVE VALUE SUPPORTED IN AGGRESSIVE REGIME'
    elif all(fixed_dominates) and all(native_dominates) and any_strict: label = 'NO BENEFIT'
    elif all(fixed_dominates[1:]): label = 'FIXED SHRINKAGE EXPLAINS MOST/ALL BENEFIT'
    else: label = 'MIXED / HISTORY-DEPENDENT'
    return {'label': label, 'aggressive_specific_role_consistent': any(support[1:]) and not support[0]}


def summarize(rows, choices):
    regimes = {}
    for ci, lr in enumerate(RATES):
        thresholds = {}
        for threshold in (.99, .95):
            arms = {}
            for arm in ARMS:
                entries = []
                for row in rows:
                    entry = row['entry_id']
                    p = 'fixed1' if arm == 'native' else 'dynamic' if arm == 'dynamic' else choices[entry][str(ci)]['policy']
                    reps = range(3, 6) if arm == 'loo_fixed' else range(3)
                    records = [record(p, 'timing', entry, ci, rep) for rep in reps]
                    h = hit(records[0], threshold)
                    if h == 0: continue
                    def cost(r, key):
                        return r[key] if h is None else r['steps'][h-1]['cumulative_'+key]
                    entries.append({'entry': entry, 'policy': p, 'hit': h, 'restricted_steps': 200 if h is None else h,
                        'compute_seconds': st.median(cost(r, 'compute_seconds') for r in records),
                        'checked_wall_seconds': st.median(cost(r, 'checked_wall_seconds') for r in records)})
                arms[arm] = {'eligible': len(entries), 'reached': sum(e['hit'] is not None for e in entries),
                    **{k+'_mean': st.mean(e[k] for e in entries) for k in ('restricted_steps', 'compute_seconds', 'checked_wall_seconds')}, 'entries': entries}
            fixed, dynamic = arms['loo_fixed']['entries'], arms['dynamic']['entries']
            gains = [f['restricted_steps']-d['restricted_steps'] for f, d in zip(fixed, dynamic)]
            arms['paired'] = {'wins': sum(v > 0 for v in gains), 'losses': sum(v < 0 for v in gains), 'ties': gains.count(0), 'gains': gains}
            thresholds[str(threshold)] = arms
        pairs = list(zip(thresholds['0.99']['loo_fixed']['entries'], thresholds['0.99']['dynamic']['entries']))
        def better(ps):
            return (sum(d['hit'] is not None for f, d in ps) >= sum(f['hit'] is not None for f, d in ps)
                and sum(f['restricted_steps']-d['restricted_steps'] for f, d in ps) > 0)
        deletion = [better(pairs[:i]+pairs[i+1:]) for i in range(len(pairs))]
        supported = better(pairs) and thresholds['0.99']['paired']['wins'] >= 2 and all(deletion)
        behavior = []; counts = Counter(); decisions = Counter()
        for row in rows:
            r = record('dynamic', 'search', row['entry_id'], ci)
            alphas = [s['details']['alpha'] for s in r['steps']]
            refresh = [s['details']['alpha'] for s in r['steps'] if (s['t']-1)%5 == 0]
            counts.update(map(str, alphas)); decisions.update(map(str, refresh))
            behavior.append({'entry': row['entry_id'], 'alphas': alphas, 'decision_alphas': refresh,
                'intervention_rate': sum(a < 1 for a in alphas)/len(alphas) if alphas else None,
                'mean_alpha': st.mean(alphas) if alphas else None})
        regimes[str(lr)] = {'thresholds': thresholds, 'extra_value_supported': supported,
            'delete_one_history_checks': deletion, 'alpha_counts': dict(counts), 'decision_alpha_counts': dict(decisions),
            'intervention_rate': sum(v for a,v in counts.items() if float(a)<1)/sum(counts.values()),
            'selected_fixed_distribution': dict(Counter(choices[r['entry_id']][str(ci)]['policy'] for r in rows)),
            'selected_fixed_eligible99_distribution': dict(Counter(e['policy'] for e in thresholds['0.99']['loo_fixed']['entries'])),
            'behavior': behavior,
            'all_fixed_first99': {r['entry_id']: {p: hit(record(p,'search',r['entry_id'],ci),.99) for p in POLICIES} for r in rows}}
    return {'status': 'COMPLETE_PENDING_INDEPENDENT_AUDIT', 'regimes': regimes, 'verdict': verdict(regimes)}


def unit_test():
    results = []
    for alpha in FIXED.values():
        torch.manual_seed(709)
        x = torch.randn(4, 3, device='cuda'); y = torch.randn(4, 2, device='cuda')
        class Model(torch.nn.Linear):
            def forward(self, x, y):
                output = super().forward(x)
                return output, (output-y).square().mean()
        model = Model(3,2).cuda(); reference = Model(3,2).cuda()
        reference.load_state_dict(model.state_dict())
        opt = native.optimizer(model,.003); other = native.optimizer(reference,.003)
        engine = StaticControl(model,alpha)
        for t in range(1,4):
            before = b.params(reference)
            reference.train(); torch.manual_seed(100+t); torch.cuda.manual_seed_all(100+t)
            other.zero_grad(set_to_none=True); reference(x,y)[1].backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.); other.step()
            proposal = b.params(reference)
            with torch.no_grad():
                if alpha!=1.:
                    for n,p in reference.named_parameters(): p.copy_((before[n]+alpha*(proposal[n]-before[n])).cuda())
            engine.update(model,opt,t,x,y,None,None,100+t,1.)
            assert b.phash(b.params(model)) == b.phash(b.params(reference))
            assert b.structure(opt.state_dict()) == b.structure(other.state_dict())
        results.append({'alpha':alpha,'three_steps_exact':True})
    return results


def bridge(r, entry, policy, ci):
    oldpolicy = 'adamw' if policy=='fixed1' else 'controller'
    old = b.read(native.OUT/'search'/entry/f'{oldpolicy}-{ci+1}'/'0/TRAJECTORY.json')
    assert r['initial_logits_sha256']==old['initial_logits_sha256'] and r['final_state_sha256']==old['final_state_sha256']
    assert len(r['steps'])==len(old['steps'])
    for a,z in zip(r['steps'],old['steps']):
        for key in ('t','seed','lr','post_parameters_sha256','logits_sha256','accuracy','coordinates_sha256'):
            assert a[key]==z[key],('baseline bridge',key)
        details=dict(a['details'])
        if policy=='fixed1':details.pop('alpha')
        assert details==z['details']


def main():
    assert not OUT.exists(), 'REFUSE_EXISTING_EVIDENCE_DIRECTORY'
    assert not (HERE/'RUN_SEAL.json').exists(), 'REFUSE_EXISTING_SEAL'
    assert git('rev-parse','HEAD')==BASE and git('rev-parse',TAG+'^{}')==FROZEN
    b.old.setup()
    with patch.object(b,'REPO',native.resident.previous.DEPENDENCIES.parent): oldseal=b.sealcheck()
    tracked=git('ls-files').splitlines()
    seal={'base_commit':BASE,'utc':b.old.utc(),'frozen_tag_object':git('rev-parse',TAG),'frozen_tag_commit':FROZEN,
        'sources':{str(p.relative_to(REPO)):b.sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md')},
        'baseline_python':{p:b.sha(REPO/p) for p in tracked if p.endswith('.py')},
        'assignment_sha256':b.sha(b.old.HERE/'RUN_ASSIGNMENT.json'),'source_rng_protocol_sha256':oldseal['source_rng_protocol_sha256'],
        'prior_dependency_seal':oldseal,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    b.write(HERE/'RUN_SEAL.json',seal)
    OUT.mkdir()
    b.write(HERE/'UNIT_TESTS.json',unit_test())
    rows=b.read(b.old.HERE/'RUN_ASSIGNMENT.json')['runs'];rng=seal['source_rng_protocol_sha256'];bridges=[]
    for i,row in enumerate(rows):
        loaded=b.load_row(row)
        try:
            for ci in range(3):
                order=POLICIES[(i+ci)%5:]+POLICIES[:(i+ci)%5]
                for p in order:
                    r=execute(row,loaded,p,ci,rng,'search')
                    if p in ('fixed1','dynamic'):
                        bridge(r,row['entry_id'],p,ci);bridges.append([row['entry_id'],p,ci])
        finally:loaded[0].close();del loaded;gc.collect()
    b.write(HERE/'BASELINE_BRIDGE.json',{'status':'PASS','exact_trajectories':bridges})
    choices=select_fixed(rows);b.write(HERE/'SELECTION.json',choices)
    selection_hash=b.sha(HERE/'SELECTION.json')
    for i,row in enumerate(rows):
        loaded=b.load_row(row)
        try:
            for ci in range(3):
                for rep in range(3):
                    order=ARMS[(i+ci+rep)%3:]+ARMS[:(i+ci+rep)%3]
                    for arm in order:
                        p='fixed1' if arm=='native' else 'dynamic' if arm=='dynamic' else choices[row['entry_id']][str(ci)]['policy']
                        execute(row,loaded,p,ci,rng,'timing',rep+3 if arm=='loo_fixed' else rep)
        finally:loaded[0].close();del loaded;gc.collect()
    assert b.sha(HERE/'SELECTION.json')==selection_hash
    b.write(HERE/'RESULTS.json',summarize(rows,choices))
    print('COMPLETE_PENDING_INDEPENDENT_AUDIT',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        failure=HERE/('FAILURE_'+str(__import__('time').time_ns())+'.txt')
        with failure.open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
