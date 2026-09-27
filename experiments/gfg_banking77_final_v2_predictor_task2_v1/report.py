"""Create-only compact evidence; all full trajectories stay in external authority."""
from .common import *
from .preflight import install,verify_inputs
import shutil

def main():
    access=install();guard();verify_inputs();audit=read(ROOT/'AUDIT.json');assert audit['status']=='PASS'
    smoke=read(ROOT/'PREDICTOR_SMOKE.json');reports=[read(ROOT/f'budget{b}/CHECKS.json') for b in (210,630)]
    dst=HERE/'results';dst.mkdir(exist_ok=False)
    for name in ('FREEZE.json','PREDICTOR_AUTHORITY.json','NUMERICS.json','PREDICTOR_SMOKE.json','AUDIT.json','SMOKE.json'):
        with (dst/name).open('xb') as f:f.write((ROOT/name).read_bytes())
    for b in (210,630):
        with (dst/f'CHECKS_{b}.json').open('xb') as f:f.write((ROOT/f'budget{b}/CHECKS.json').read_bytes())
    status='PASS' if all(r['status']=='PASS' and r['predictor']['status']=='PASS' for r in reports) else 'FAIL'
    write(ROOT/'FINAL.json',dict(status=status,task2_only=True,no_continuation=True,audit_status='PASS',
        native_predictor_status=smoke['status'],controlled_predictor_status=[r['predictor']['status'] for r in reports],
        controller_gate_status=[r['status'] for r in reports],access=access.report(),test_files_opened=0))
    with (dst/'FINAL.json').open('xb') as f:f.write((ROOT/'FINAL.json').read_bytes())
    artifacts={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file()}
    write(HERE/'ARTIFACT_MANIFEST.json',dict(authority=str(ROOT),files=artifacts))
    lines=['# Banking77 Final V2 nonlinear predictor: task2 only','',
        'Every scheduled update uses the historical actual-update JVP/nested-JVP predictor. The existing nonlinear cutter, trust-region rule, reserve, numerical tolerance, inner/outer caps and cap endpoint acceptance remain unchanged. Initial selection uses predicted deficits; Jacobian geometry uses actual logits. Predicted-safe native candidates also receive exact rechecks. See [PROTOCOL.md](PROTOCOL.md).','',
        f'Final status **{status}**. Independent native-only and controlled replay **PASS**. Stopped after task2; no tasks3–11, test or new baselines. Historical evidence remains unchanged.','',
        '| Controlled task2 | 210 | 630 |','|---|---:|---:|']
    def row(label,fn):lines.append('|'+label+'|'+'|'.join(str(fn(r)) for r in reports)+'|')
    row('Predictor calls',lambda r:r['predictor']['calls'])
    row('Predicted projection triggers',lambda r:r['predicted_projection_triggers'])
    for label,key in [('Actual unsafe native','unsafe_native_steps'),('Actual native-safe','native_safe_steps'),('Ordinary PASS + safe','ordinary_pass_actual_safe'),('INNER_LIMIT events','inner_limit'),('Cap-safe accepts','inner_limit_actual_safe_accept'),('Cap-unsafe continuations','inner_limit_actual_unsafe_continue'),('Rollback','final_rollback'),('Floor violations','floor_violations'),('Outer iterations','outer_iterations'),('Inner sweeps','inner_sweeps'),('Jacobian pairs','jacobian_pairs'),('Correct → wrong','correct_to_wrong'),('Wrong → correct','wrong_to_correct')]:row(label,lambda r,k=key:r[k])
    row('Cap-safe acceptance rate',lambda r:f"{100*r['cap_safe_acceptance_rate']:.2f}%")
    row('Executed/native norm median',lambda r:f"{r['norm_ratio_triggered']['median']:.6f}")
    row('Cosine median',lambda r:f"{r['cosine_nonzero_triggered']['median']:.6f}")
    for group in ('old','current','all'):row(group+' train macro (%)',lambda r,g=group:f"{100*r['train_metrics'][g]['macro']:.4f}")
    row('Current train CE entry → final',lambda r:f"{r['entry_current_train']['ce']:.6f} → {r['train_metrics']['current']['ce']:.6f}")
    row('Actual endpoint calls',lambda r:r['endpoint_calls']['total'])
    row('Predictor wall (s)',lambda r:f"{r['predictor_wall']:.4f}")
    row('Control wall incl predictor (s)',lambda r:f"{r['control_wall']:.4f}")
    row('Total algorithm wall (s)',lambda r:f"{r['total_algorithm_wall']:.4f}")
    for label,key in [('Predictor','predictor_flops'),('Exact endpoints','endpoint_flops'),('Cutter estimates','cutter_flops'),('Current CE dense','current_forward_backward_dense_flops')]:row(label+' GFLOPs',lambda r,k=key:f"{r[k]/1e9:.6f}")
    row('Total covered GFLOPs',lambda r:f"{sum(r[k] for k in ('predictor_flops','endpoint_flops','cutter_flops','current_forward_backward_dense_flops'))/1e9:.6f}")
    row('Old CE backward examples / FLOPs',lambda r:'0 / 0')
    row('Control-bank logical row reads',lambda r:r['control_bank_reads'])
    lines+=['','| Predictor | Native210 | Native630 | Controlled210 | Controlled630 |','|---|---:|---:|---:|---:|']
    measurements=smoke['reports']+[r['predictor'] for r in reports]
    for label,key in [('Calls','calls'),('Margin MAE','margin_mae'),('Margin max error','margin_max_error'),('Correctness agreement','sign_agreement'),('Argmax agreement','argmax_agreement'),('Balanced accuracy','balanced_accuracy'),('Floor-safe agreement','floor_safe_agreement'),('False-safe updates','false_safe_updates'),('False-unsafe updates','false_unsafe_updates'),('Predicted-safe updates','predicted_safe_updates'),('False-safe / all updates','false_safe_per_update'),('False-safe / predicted-safe','false_safe_given_predicted_safe')]:
        lines.append('|'+label+'|'+'|'.join(str(m[key]) for m in measurements)+'|')
    for i,label in enumerate(('wrong→wrong','wrong→correct','correct→wrong','correct→correct')):
        lines.append('|'+label+' recall|'+'|'.join('N/A (no events)' if m['transition_recall'][i] is None else str(m['transition_recall'][i]) for m in measurements)+'|')
    lines+=['','The same fixed gates are measured on both native-only and controlled trajectories. Missing transitions have undefined recall, never imputed as 1. Prediction does not certify safety; exact FP32 counts are the final authority.','',
        'Wall times include synchronization/instrumentation and have one execution per arm; no best-of runs. Hashing, logs, accuracy observers, train-set evaluation and independent audits are excluded. Dense matmul FLOPs exclude GELU transcendental operations, optimizer operations and other elementwise kernels; cutter costs are the existing estimates, not full hardware instruction counts. Predictor cost is measured with FlopCounterMode.','',
        'The independent audit recomputes every native proposal, optimizer moment/counter, prediction component, initial selection, local solver coefficient/dual, materialized candidate, cap acceptance/continuation and final head. Capped candidates additionally receive explicit dense-Jacobian reconstruction checks. BERT/cache/input hashes are verified unchanged. Python access logs cover experiment file reads; native library/OS internals are outside the hook.','',
        f'External authority: `{ROOT}`. All compact checks are in `results/`, with full artifact identities in `ARTIFACT_MANIFEST.json`. No automatic 11-task continuation exists.']
    with (HERE/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print('Report',status,HERE/'README.md',flush=True)

if __name__=='__main__':main()
