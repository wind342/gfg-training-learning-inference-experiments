from .common import *
from .preflight import install,verify_inputs
import statistics

def main():
    access=install();verify_inputs();timing=read(ROOT/'TIMING.json');accepted=timing['accepted_variants'];variants=['baseline','stage1','stage2']
    results=HERE/'results';results.mkdir(exist_ok=False)
    files=[ROOT/'FREEZE.json',ROOT/'NUMERICS.json',ROOT/'TIMING.json']
    if (ROOT/'D_DIAGNOSTIC.json').exists():files.append(ROOT/'D_DIAGNOSTIC.json')
    for v in variants:
        for name in ('IDENTITY.json','AUDIT.json'):
            if (ROOT/v/name).exists():files.append(ROOT/v/name)
        for b in (210,630):
            for name in ('CHECKS.json','IDENTITY.json'):
                if (ROOT/v/f'budget{b}'/name).exists():files.append(ROOT/v/f'budget{b}'/name)
    for p in files:
        name=p.relative_to(ROOT).as_posix().replace('/','__')
        with (results/name).open('xb') as out:out.write(p.read_bytes())
    rows=[]
    for budget in (210,630):
        for v in variants:
            if not (ROOT/v/f'budget{budget}/CHECKS.json').exists():continue
            validation=read(ROOT/v/f'budget{budget}/CHECKS.json');identity=read(ROOT/v/f'budget{budget}/IDENTITY.json')
            samples=[next(r for r in x['reports'] if r['budget']==budget) for x in timing['reports'] if x['variant']==v]
            def measured(fn):
                values=[float(fn(x)) for x in samples]
                return dict(values=values,median=statistics.median(values),min=min(values),max=max(values)) if values else None
            metrics={k:measured(lambda x,k=k:x[k]) for k in ('total_algorithm_wall','control_wall','predictor_wall','peak_cuda_bytes')}
            for k in ('endpoint','hildreth_ball','parameter_geometry','jacobian_gram_guard','reconstruction','materialization'):
                metrics[k]=measured(lambda x,k=k:x['region_seconds'][k])
            rows.append(dict(budget=budget,variant=v,identity=identity['status'],checkpoint_serialized_identical=identity['serialized_checkpoint_identical'],
                solver_sweeps=validation['inner_sweeps'],operations=validation['operation_counts'],timing=metrics,
                flop_ledger=validation['flop_ledger'],current_dense_flops=validation['current_forward_backward_dense_flops'],
                floor_violations=validation['floor_violations'],old_ce_backward_examples=validation['old_ce_backward_examples']))
    summary=dict(status='PASS' if 'stage1' in accepted else 'FAIL',accepted_variants=accepted,rows=rows,task2_only=True,test_files_opened=0,access=access.report())
    write(ROOT/'SUMMARY.json',summary)
    with (results/'SUMMARY.json').open('xb') as out:out.write((ROOT/'SUMMARY.json').read_bytes())
    lines=['# Banking77 cutter repeated-computation experiment','',
        'Base `534130d8995fbf9f6cec176afb915f1a56845702`. Only task2 210/630. No changes to the predictor, floor, selection rule, solver tolerance/caps, trust-region rule, reserve, AdamW proposal or schedule. Historical evidence is preserved. See [PROTOCOL.md](PROTOCOL.md).','',
        'Stage1 removes duplicate post-ball G@t, skips the first zero-direction apply, and reuses already-checked executed endpoints. Stage2 adds unique-control directional GEMM separately. Independent replay always computes committed-head endpoints again.','',
        '| Budget | Version | Trajectory identical | Sweeps | Exact endpoints | Post-ball G@t calls | Apply calls | Solver wall | Endpoint wall | Control wall | Total wall |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def t(r,k):return f"{r['timing'][k]['median']:.4f}" if r['timing'][k] else 'not promoted'
    for r in rows:
        o=r['operations'];lines.append(f"|{r['budget']}|{r['variant']}|{r['identity']}|{r['solver_sweeps']}|{o['exact_endpoint_calls']}|{o['post_ball_Gt_calls']}|{o['apply_calls']}|{t(r,'hildreth_ball')}|{t(r,'endpoint')}|{t(r,'control_wall')}|{t(r,'total_algorithm_wall')}|")
    lines+=['','Times are seconds, medians of exactly three prespecified rotated full replays; every timed replay is checked against authority. These are instrumented algorithm timings, not bare deployment latency. Historical validation times are retained separately; no best-of runs. Existing internal cutter hashes/diagnostics are inside all variants’ control timers; outer commit hashes, logs, train evaluation and independent audit are excluded.','',
        '| Budget | Version | Predictor s | Parameter geometry s | Jacobian/Gram/guard s | Reconstruction s | Materialization s | Peak GPU bytes |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+str(r['budget'])+'|'+r['variant']+'|'+'|'.join(t(r,k) for k in ('predictor_wall','parameter_geometry','jacobian_gram_guard','reconstruction','materialization','peak_cuda_bytes'))+'|')
    lines+=['','| Budget | Version | Geometry pairs | Geometry unique controls | Actual apply pair rows | Actual apply unique rows | All Gram GEMVs | Skipped zero applies | Eliminated duplicate G@t |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        o=r['operations'];lines.append('|'+str(r['budget'])+'|'+r['variant']+'|'+'|'.join(str(o[k]) for k in ('geometry_active_pairs','geometry_unique_controls','apply_pair_rows','apply_unique_rows','all_gram_matvec_calls','skipped_zero_apply','eliminated_Gt'))+'|')
    lines+=['','| Budget | Version | Predictor GFLOPs | Endpoint GFLOPs | Directional apply estimated GFLOPs | Removed duplicate GEMV GFLOPs | Total covered GFLOPs |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        f=r['flop_ledger'];lines.append('|'+str(r['budget'])+'|'+r['variant']+'|'+'|'.join(f'{x/1e9:.6f}' for x in (f['predictor_dense'],f['endpoint'],f['jacobian_directional_dense_estimate'],-f['eliminated_duplicate_Gt'],sum(f.values())+r['current_dense_flops']))+'|')
    lines+=['','Dense multiply-add FLOPs are lower bounds; the inherited analytic/solver estimates exclude many elementwise, GELU and optimizer operations. Known eliminated work is explicitly subtracted. Post-ball G@t counters exclude the other two Gram-vector products for ball/cycle norms. Raw timing repetitions/ranges, all checks, checkpoint identities and FLOP ledgers are in `results/`.','',
        'All accepted variants have zero floor violations and zero old CE backward. Task1 state, frozen BERT/cache files and source hashes are checked unchanged. No tasks3–11 or test access.','',
        'The sweep counts remain fixed. This experiment measures redundant computation only; it does not establish how early a candidate would have become functionally safe.','',
        f'External create-only authority: `{ROOT}`. Full trajectories and checkpoints remain external.']
    with (HERE/'README.md').open('x',encoding='utf8',newline='\n') as out:out.write('\n'.join(lines)+'\n')
    write(HERE/'ARTIFACT_MANIFEST.json',dict(authority=str(ROOT),files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file()}))
    print('Report created:',accepted,flush=True)

if __name__=='__main__':main()
