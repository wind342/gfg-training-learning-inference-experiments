"""Post-execution reporting only; not imported by any training/selection code."""
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=Path('E:/gfg-evidence/optimizer-capability-benchmark-v1')
METHODS=('adamw','sophia','shampoo','sls','controller')


def main():
    results=json.loads((HERE/'RESULTS.json').read_text())
    selection=json.loads((HERE/'LOO_SELECTION.json').read_text())['entries']
    times={}; tuning={m:{'configuration_history_runs':0,'updates':0,'compute_seconds':0.,'wall_seconds':0.} for m in METHODS}
    for p in OUT.glob('*/*/*/*/TRAJECTORY.json'):
        r=json.loads(p.read_text()); m=r['method']
        if p.parts[-5]=='search':
            acc=tuning[m]; acc['configuration_history_runs']+=1; acc['updates']+=len(r['steps'])
            acc['compute_seconds']+=r['construction_seconds']+sum(s['compute_seconds'] for s in r['steps'])
            acc['wall_seconds']+=r['wall_seconds']
        elif r['initial']['accuracy']<.99:
            times.setdefault((r['entry_id'],m),[]).append(r['wall_seconds'])
    wall99={m:statistics.mean(statistics.median(v) for (entry,method),v in times.items() if method==m) for m in METHODS}
    outcome={'search_cost_by_method':tuning,'mean_restricted_end_to_end_evidence_wall99_seconds':wall99,
        'scope':'Wall includes evidence hashes, per-step JSON, final checkpoint and evaluation, but excludes source loading and synthetic warmup. Compute-only timings are in RESULTS.json. Tuning cost is not silently amortized into a single selected run.',
        'no_energy_measurement':True}
    with (HERE/'COST_ACCOUNTING.json').open('x',encoding='utf-8') as f: json.dump(outcome,f,indent=2)
    rows=[d for d in results['per_history'] if d['threshold']==.99 and d['first_step']!=0]
    lines=['# Per-history first 99% and cost accounting','','Each cell is first-hit updates / median synchronized compute seconds. A dash means not reached within 200 updates; that full cost remains in aggregate results. Histories initially >=99% are omitted here.','',
           '| History | AdamW | Sophia-G | Shampoo | SLS | Controller |','|---|---:|---:|---:|---:|---:|']
    for entry in sorted({d['entry'] for d in rows}):
        cells=[]
        for m in METHODS:
            d=next(d for d in rows if d['entry']==entry and d['method']==m)
            step='not reached' if d['first_step'] is None else str(d['first_step'])
            cells.append(f"{step} / {d['median_compute_seconds']:.4f}s")
        lines.append('| '+entry+' | '+' | '.join(cells)+' |')
    lines+=['','## Selected initial learning rates','','| History | AdamW | Sophia-G | Shampoo | SLS | Controller |','|---|---:|---:|---:|---:|---:|']
    for entry in sorted({d['entry'] for d in rows}):
        lines.append('| '+entry+' | '+' | '.join(str(next(d['lr'] for d in rows if d['entry']==entry and d['method']==m)) for m in METHODS)+' |')
    lines+=['','## All four-configuration search costs','','These are additional experimental/tuning costs, not the cost of executing one selected controller.','',
           '| Method | Updates | Compute seconds | Evidence wall seconds |','|---|---:|---:|---:|']
    for m,a in tuning.items(): lines.append(f"| {m} | {a['updates']} | {a['compute_seconds']:.3f} | {a['wall_seconds']:.3f} |")
    lines+=['','## End-to-end wall to first 99% or cap','','Includes auditing and disk I/O, so it is not pure optimizer speed. Excludes source loading and synthetic warmup.','',
           '| Method | Mean restricted wall seconds |','|---|---:|']
    for m,v in wall99.items(): lines.append(f'| {m} | {v:.4f} |')
    lines+=['','Electricity consumption was not measured; do not infer energy savings from these timings.','']
    with (HERE/'PER_HISTORY_AND_COSTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':main()
