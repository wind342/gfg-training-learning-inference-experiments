from .common import *
from .preflight import install,verify
from collections import Counter

def records(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]

def distribution(a):
    if not a:return None
    return dict(n=len(a),min=float(np.min(a)),median=float(np.median(a)),mean=float(np.mean(a)),p90=float(np.percentile(a,90)),max=float(np.max(a)))

def summarize(budget):
    folder=ROOT/f'budget{budget}';updates=records(folder/'UPDATES.jsonl.gz');sweeps=records(folder/'SWEEPS.jsonl.gz')
    active=[u for u in updates if u['unsafe_native']];rounds=[r for u in active for r in u['rounds']]
    safe=[r for r in rounds if r['first_safe_sweep'] is not None];caps=[r for r in rounds if r['original_solver_status']=='INNER_LIMIT']
    first=[u['first_safe_cumulative_sweeps'] for u in active];original=sum(u['original_sweeps'] for u in active)
    plain=sum(u['original_solver_seconds'] for u in active);prefix=sum(u['prefix_solver_seconds_estimate'] for u in active)
    checks=sum(u['prefix_check_seconds'] for u in active);oldchecks=sum(u['original_check_seconds'] for u in active)
    result=dict(budget=budget,unsafe_updates=len(active),local_solves=len(rounds),original_sweeps=original,
        first_safe_prefix_sweeps=sum(first),potential_sweeps_avoided=original-sum(first),potential_sweep_reduction=1-sum(first)/original,
        first_safe_per_update=distribution(first),first_safe_local=distribution([r['first_safe_sweep'] for r in safe]),
        first_safe_histogram=dict(sorted(Counter(first).items())),
        local_rounds_ever_safe=len(safe),local_rounds_never_safe=len(rounds)-len(safe),
        local_rounds_safe_before_original_stop=sum(r['first_safe_sweep']<r['original_sweeps'] for r in safe),
        safe_to_unsafe_transitions=sum(r['safe_to_unsafe_transitions'] for r in rounds),
        ever_safe_but_final_unsafe=sum(r['first_safe_sweep'] is not None and not r['last_safe'] for r in rounds),
        cap_solves=len(caps),cap_ever_safe=sum(r['first_safe_sweep'] is not None for r in caps),
        cap_first_safe=distribution([r['first_safe_sweep'] for r in caps if r['first_safe_sweep'] is not None]),
        original_plain_solver_seconds=plain,prefix_solver_seconds_estimate=prefix,estimated_solver_seconds_saved=plain-prefix,
        every_sweep_prefix_check_seconds=checks,original_terminal_checks_seconds=oldchecks,
        additional_check_seconds=checks-oldchecks,estimated_net_seconds_saved=plain-prefix-(checks-oldchecks),
        updates_estimated_net_positive=sum(u['estimated_net_seconds_saved']>0 for u in active),
        first_safe_candidate_differs_from_executed=sum(u['first_safe_same_as_executed'] is False for u in active),
        first_safe_vs_executed_l2=distribution([u['first_safe_vs_executed_l2'] for u in active]),
        all_probe_check_seconds=sum(s['check_seconds'] for s in sweeps),probe_check_seconds=distribution([s['check_seconds'] for s in sweeps]),
        probe_reconstruction_seconds=sum(s['reconstruction_seconds'] for s in sweeps),
        probe_materialization_seconds=sum(s['materialization_seconds'] for s in sweeps),
        probe_endpoint_counts_seconds=sum(s['endpoint_counts_seconds'] for s in sweeps),
        original_trajectory_checks=read(folder/'CHECKS.json'))
    write(folder/'SUMMARY.json',result);return result

def main():
    access=install();verify();assert read(ROOT/'CHECKS.json')['status']=='PASS';summaries=[summarize(b) for b in (210,630)]
    write(ROOT/'SUMMARY.json',dict(status='PASS',reports=summaries,diagnostic_only=True,
        limitation='Fixed original update states. Prefix solver time is estimated using measured sweep profiles normalized to plain solver time. No early-stop training trajectory or end-to-end speedup is measured.',access=access.report()))
    dst=HERE/'results';dst.mkdir(exist_ok=False)
    for name in ('FREEZE.json','NUMERICS.json','CHECKS.json','SUMMARY.json'):
        with (dst/name).open('xb') as f:f.write((ROOT/name).read_bytes())
    for b in (210,630):
        for name in ('CHECKS.json','SUMMARY.json'):
            with (dst/f'{b}_{name}').open('xb') as f:f.write((ROOT/f'budget{b}'/name).read_bytes())
    lines=['# Task2 first-actual-safe sweep diagnostic','',
        'Source: accepted Stage1 A/B/C from f0b6aaf. Every original sweep is observed; early-safe candidates are never committed. Both original trajectories, native proposals, optimizer moments, predictor outputs, complete solver decisions and final checkpoint states are unchanged. First-safe and original terminal candidates are independently reconstructed and forwarded; committed heads are independently forwarded.','',
        '| Quantity | 210 | 630 |','|---|---:|---:|']
    def row(label,fn):lines.append('|'+label+'|'+'|'.join(str(fn(r)) for r in summaries)+'|')
    for label,key in [('Unsafe native updates','unsafe_updates'),('Original local solves','local_solves'),('Original sweeps','original_sweeps'),('First-safe prefix sweeps','first_safe_prefix_sweeps'),('Potential avoided sweeps on fixed path','potential_sweeps_avoided'),('Local solves safe before original stop','local_rounds_safe_before_original_stop'),('Safe → unsafe transitions','safe_to_unsafe_transitions'),('Ever-safe but final-unsafe local solves','ever_safe_but_final_unsafe'),('Cap solves','cap_solves'),('Cap solves ever safe','cap_ever_safe'),('Earlier candidate differs from executed head','first_safe_candidate_differs_from_executed')]:row(label,lambda r,k=key:r[k])
    row('First-safe sweep per update median / mean / p90',lambda r:' / '.join(f"{r['first_safe_per_update'][k]:.2f}" for k in ('median','mean','p90')))
    row('Potential sweep reduction',lambda r:f"{100*r['potential_sweep_reduction']:.2f}%")
    lines+=['','| Cost on fixed update states (seconds) | 210 | 630 |','|---|---:|---:|']
    for label,key in [('Plain original solver, measured','original_plain_solver_seconds'),('First-safe prefix solver, estimated','prefix_solver_seconds_estimate'),('Solver time saved, estimated','estimated_solver_seconds_saved'),('Check every sweep until first safe, measured checks on original states','every_sweep_prefix_check_seconds'),('Original one terminal check per outer, matched probes','original_terminal_checks_seconds'),('Additional check cost','additional_check_seconds'),('Net saving, estimate (positive saves time)','estimated_net_seconds_saved')]:row(label,lambda r,k=key:f'{r[k]:.6f}')
    row('Full candidate check median (ms)',lambda r:f"{1000*r['probe_check_seconds']['median']:.6f}")
    row('Full candidate check mean (ms)',lambda r:f"{1000*r['probe_check_seconds']['mean']:.6f}")
    lines+=['','Each candidate check includes CPU reconstruction, FP32 conversion, parameter transfers and exact buffer256 forward/counts. Audit/log/hash work is excluded. The matched plain solver is run without callbacks; its total normalizes observed per-sweep CPU profiles. GPU/CPU interleaving and cache effects make savings estimates approximate. Skipped outer geometry/diagnostic work is not credited.','',
        'These are per-update counterfactuals on the fixed historical states, not a measured early-stop controller. Different early candidates can change later native updates, floors and learning. No check interval, residual threshold, early-stop policy or new solver was selected or deployed.','',
        'All original committed endpoints remain floor-safe; old CE backward is zero. Frozen BERT/cache/input/source hashes remain unchanged. Test was never accessed, and tasks3–11 were not run.','',
        f'Full per-sweep observations, per-update summaries and audit identities: `{ROOT}`. Compact evidence is in `results/`.']
    with (HERE/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    write(HERE/'ARTIFACT_MANIFEST.json',dict(authority=str(ROOT),files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file()}))
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
