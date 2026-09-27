# Final V2 safe screening — preserved task10 failure

**Status: STOPPED_NUMERICAL_MISMATCH. Phase A gate NOT PASSED.**

Parent: `41ca46bef060815d2f106f8235764fb8b03e67b7`.
One independent create-only experiment. Old scientific sources/results, main,
paper tag and input authorities remain unchanged. Banking77 was not started.

## What happened

The numerical unit preflight passed, including original-controller differential
checks with the original full evaluator. The formal task10 continuation then
completed 11 of 210 scheduled updates and stopped on step12's **uncommitted
native proposal**. No automatically modified rule, retry or final test evaluation.

Across 87 observer calls there were **zero observed false-safe certificates**.
However, one uncertified frontier row changed sign between the gathered-batch
FP32 GEMM and the historical full-batch FP32 GEMM. Its relation count differed,
triggering the predeclared abort. This does not establish a successful certificate
implementation for all future states, and it is not a successful 210-step run.

| Quantity | Result |
|---|---:|
| Old protection bank | 10072 |
| Completed updates | 11 / 210 |
| Committed heads/moments byte-identical to history | 11 / 11 |
| Entire paired projection traces identical | 9 / 11 |
| Observed false-safe certificates | 0 |
| Candidate relation-count mismatches | 1, at step12 |
| Executed relation-floor violations | 0 (792 relation-step checks) |
| Completed skips | 0 |
| Completed Hildreth sweeps | 52 |
| Final old/new/all | Not evaluated; no completed task10 endpoint |

Steps7 and11 already had intermediate selection-trace differences, although
their executed heads, final constraints and moments matched. These differences
are preserved in STEPS.jsonl.gz with the paired reference trace.

## Reproduced numerical cause

Old bank row541, dataset ID2641, relation3, competitor13. Exactly the same actual
FP32 candidate hash was independently reconstructed using the original production
controller for the preceding eleven updates.

| Evaluation | Margin |
|---|---:|
| Historical full FP32 batch (10152 rows including audit-only rows) | -2.86102294921875e-6 |
| Gathered FP32 frontier batch (3506 rows including audit-only rows) | +9.5367431640625e-7 |
| FP64 diagnostic only | +7.035244875552848e-7 |

The row was **not certified**: its conservative lower bound was approximately
-0.24439. Thus the failure is not a sample incorrectly skipped by the safety
certificate. It is a changed finite-precision decision in the exact-frontier
stage. FP64 is diagnostic here, not substituted as the correctness authority.

GEMM batch shape can change reduction/rounding; the historical algorithm's
strict margin>0 decision makes this significant near zero. A batch-invariant
arithmetic path could define a new reproducible evaluator, but would not by
itself reproduce historical cuBLAS rounding. Alternatively a full original-shape
fallback for numerically ambiguous decisions can preserve the original authority
at additional cost. Neither repair was silently introduced or benchmarked.

## Partial timing and workload (first 11 updates only)

| Quantity | Result |
|---|---:|
| Original full candidate row evaluations | 856120 |
| Screened exact candidate row evaluations | 269359 |
| Reduction in those candidate rows | 68.54% |
| Additional FP64 anchor refresh rows | 69532 |
| Relation refreshes including initialization | 497 |
| Frontier fraction median / p90 / max, all 87 calls | 32.91% / 34.82% / 35.45% |
| Screened algorithm including setup/refresh | 2.5884 s |
| Paired unchanged-reference algorithm | 2.1326 s |
| Paired speedup | 0.8239x (screened about21.4% slower) |
| Completed-call independent observer | 0.2905 s, excluded from algorithm |

Times use CUDA synchronization. Both sides share the measured current-batch
update/proposal cost; the reference runs the original production projection on
the same receiving/native states. Interleaved short-prefix timing is diagnostic,
not a completed-task throughput result. Row counts are not FLOPs/traffic; FP64
refresh work is additional, and all screening/refresh work is included in wall.
No claim is made that full-bank scanning alone was the dominant bottleneck.

## Implementation boundary

`core.py` uses Cauchy-Schwarz with conservative FP64/FP32 error allowances;
it does not mislabel FP64 computation as exact real arithmetic. It retains
original fixed relation floors, wrong-row selection, Hildreth coordinate and
original padding. Previously selected rows are always evaluated because their
pair threshold is1e-6 even when a positive-sign certificate exists. No old CE
backward, replay, optimizer changes or per-sample locking were introduced.

Anchors refresh per relation at scheduled boundaries if previous endpoint
frontier fraction>25% or age>=100. This rule was frozen before execution.
It was not changed after the negative result.

The certificate assumes normal IEEE FP32/FP64 arithmetic with TF32/autocast off,
finite bounded inputs and an underflow allowance. Its derivation and numerical
preflight are not formal verification of CUDA hardware. Every formal candidate
was independently full-scanned, and the mismatch was acted on rather than hidden.

## Evidence and reproduction

- `PROTOCOL.md`: pre-run rule freeze.
- `results/PREFLIGHT.json`: original source/input hashes.
- `results/NUMERICAL_PREFLIGHT.json`: synthetic and production differential tests.
- `results/CALLS.jsonl.gz`: every candidate's certificate bitmask/full observer.
- `results/STEPS.jsonl.gz`: completed heads/moments identities, projection traces,
  paired mismatches and timings.
- `results/FAILURE_*.json`: preserved exception.
- `results/FAILURE_DIAGNOSIS.json`: independent reconstruction of the offending
  native candidate; no test read and no formal rerun.
- `results/PARTIAL_RESULTS.json`, `results/FAILURE_AUDIT.json`: explicit failed
  gate and verification of preserved records, not a success CHECKS.json.
- `ARTIFACT_MANIFEST.json`: byte sizes and SHA-256 of copied local authorities.

Full local authority: `E:/gfg-evidence/final-v2-safe-screening-v1`.
No BERT/checkpoint/feature banks were duplicated in this experiment or published.
Public logs contain IDs/numbers, not raw examples or secrets. Inputs reuse the
previous compact comparison authority, whose checks passed before this run.

Preflight can run independently:

```text
python -B -m experiments.gfg_final_architecture_v2_safe_screening_v1.tests
```

`run.py` requires the declared parent HEAD and refuses existing output ROOT.
It intentionally does not resume/retry this stopped authority. `check.py` is for
a completed run and is NOT reported as passed. Post-failure reconstruction and
packaging are under `diagnostics/`; those were added after the formal source
freeze without changing any frozen-run source. They also create outputs only.

**Conclusion:** substantial partial scan reduction, but original FP32 decision
equivalence failed and the observed prefix was slower. Preserve this negative
engineering result; do not advance to Phase B on this gate.
