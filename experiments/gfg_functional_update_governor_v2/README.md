# Functional update governor v2

Base: `25ef6dbdde7ad6b31a2807e6323c785a29190dc1`. This directory is an
isolated engineering experiment, not a change to any frozen paper result.
All 2,339 previously tracked files and the annotated paper tag remain unchanged.
The v1 instrument and the distributed-runtime-v2 numerical kernels are imported,
not copied or edited. No new model, optimizer direction or multi-GPU implementation.

## Result first

**Do not promote this policy as an acceleration improvement over governor v1.**
The micro sensor and per-target latch work as implemented, but preservation-first
selection frequently suppresses formation. SAFE is slower to reach99; NEAR does
not reach99 within the unchanged 200-update budget. Thresholds were not retuned.

| Case / method | First update >=99% | Wall to99 | Full instrument calls |
|---|---:|---:|---:|
| SAFE v2 | 131 | 15.63 s | 26 |
| SAFE governor v1 | 119 | 14.96 s | 29 |
| SAFE historical dynamic runtime v2 | 143 | 22.67 s | 29 |
| SAFE same-LR native | 121 | 7.65 s | 0 |
| SAFE tuned native | 106 | 6.78 s | 0 |
| NEAR v2 | **Not reached** | **Not reached** | 63 over 200 updates |
| NEAR governor v1 | 191 | 19.95 s | 32 |
| NEAR historical dynamic runtime v2 | 158 | 24.99 s | 32 |
| NEAR same-LR native | 174 | 11.00 s | 0 |
| NEAR tuned native | 106 | 6.82 s | 0 |

NEAR v2 finished 200 updates at **93.3754%**, taking **29.70 s**; this is a
budget-end time, NOT a time to99. SAFE finished at99.0536%. Both peaks were
1,892,453,888 allocated CUDA bytes (1.7625 GiB). Historical times were not rerun
contemporaneously. V2 additionally synchronizes CUDA at timing-segment boundaries.

Fixed setup: seed201, 71,405,312 parameters, sequence3, vocabulary24, FP32,
AdamW(.9,.98), decay0, clip1, batch128; SAFE LR3.7688918072220455e-6 and NEAR
LR6.030226891555273e-5. Same original212 control /317 evaluation queries and
training batches; evaluation queries are excluded from control, not training.
Two throwaway warmup updates precede exact reinitialization. Timed work includes
training, monitoring/control and every-update evaluation, not initialization.
This is modular-addition scale testing, not natural-language pretraining.

## Running paths and defaults

1. Normal training forward/backward and unchanged native gradient clipping.
2. On first step or50-update maximum gap, directly measure the full bank.
3. Otherwise, every step observes the next deterministic shard of at most8 probes
   on the actual receiving model, before constructing any optimizer proposal.
4. The micro observation is one eval forward: target logit minus best competitor,
   with no JVP, backward, proposal copy, swap or simulated optimizer update.
5. New capability-risk events request the existing full response instrument.
6. That instrument measures the actual native displacement, all competitors and
   the unchanged complete quadratic coordinates for {.125,.25,.5,1,2}.
7. The new governor selects a current-step amplitude; no future-step carryover.
8. Without full measurement, native `optimizer.step()` executes directly.

Paths are `FULL_INSTRUMENT`, `MICRO_PROBE` and `NATIVE_FAST`. The last is available
when configured `micro_every>1`; the fixed tests use1, so all non-full steps are
micro steps. No claim that those steps have zero monitoring overhead.

The expensive instrument remains a direct update-response calculation, not a
probe trained on labels. No loss/gradient-norm anomaly triggers remain in v2;
these values are telemetry and nonfinite safety checks only.

### Identity, anchors and hysteresis

The ordered caller-supplied bank is bound by tensor identity/version and shape;
changing/reordering it fails closed. Logged indices address that fixed order.
There is no hardcoded bank size in the scheduler/runtime. Complete coverage takes
27 micro observations for212 targets; skipped full-measurement steps increase the
elapsed update interval. Unsampled failures are not detected instantaneously.

Each target has a separate last full baseline, excursion anchor and latch level.
Observed newly correct targets are also remembered. Soft erosion is positive
margin <.5 times anchor (only if anchor >=1e-6); hard crossing is margin <=0 after
a positive reference. Full refresh never replaces a latched pre-excursion anchor.
Repeated unchanged risk is logged without retriggering. A different target or
soft-to-hard escalation can trigger. Positive recovery to at least.75 times
max(anchor,1e-6) rearms the target. Recovery is a STRONGER condition than the
erosion threshold, not a more permissive one. Full refresh also observes all
targets for latch maintenance. Periodic checks may still occur while latched.

The sensor detects degradation, not proof that the current LR caused overshoot.
Stable but slow formation is only reconsidered at full checks; this policy may
miss acceleration opportunities between them. Exact zero remains the hard
boundary; ordinary micro forward and the reused adapter baseline can have small
floating-point differences. The soft floor is not a tolerance on the hard test.

### Selection

Remove weighted gain-minus-damage utility. Native alpha1 is the reference.
Preservation damage is minimized, then formation gain is maximized. If both
counts tie native, retain native regardless of a tiny margin advantage.
Only count-improving alternatives can use frontier progress as a tie-breaker.
For an all-correct bank, exclude alpha>1 and prefer the least change from1 at
equal minimum damage. Direction and objective are separate reason fields because
shrinking can improve formation and amplification could improve preservation.

This is a strong preservation priority, NOT a mathematical guarantee of fewer
updates. In particular, currently-correct probes can be correct at initialization;
the policy does not establish that every such probe is a robustly learned skill.

## Measured overhead and actions

| Metric | SAFE | NEAR |
|---|---:|---:|
| Full instrument fraction | 19.85% | 31.50% |
| V1 full instrument fraction | 24.37% | 16.75% |
| Call count change vs v1 | -10.34% | +96.88% |
| Micro forward calls | 130 | 199 |
| Total micro forward wall | 0.8844 s | 1.3946 s |
| Mean micro forward | 6.80 ms | 7.01 ms |
| Instrument-only wall (JVP + boundary) | 4.9287 s | 11.9190 s |
| Instrument-only fraction of total wall | 31.54% | 40.13% |
| Proposal wall | 1.6191 s | 3.9225 s |
| Ordinary native updates | 105 | 137 |

NEAR's count comparison is against v1's OWN first99 at191, whereas v2 exhausted
200; the fractions additionally normalize for different run lengths. Instrument
wall is combined JVP plus boundary calculation, not an isolated JVP benchmark.

Full-measurement alpha counts:

| Alpha | SAFE | NEAR |
|---|---:|---:|
| .125 | 14 | 43 |
| .25 | 6 | 9 |
| .5 | 4 | 5 |
| 1 | 2 | 5 |
| 2 | 0 | 1 |

Executed-step alpha1 counts are107/142, including unmeasured native steps.
SAFE reasons:24 SHRINK_PRESERVATION,1 NO_BETTER_ACTION,1 NATIVE_SAFE.
NEAR:56 SHRINK_PRESERVATION,1 SHRINK_FORMATION,1 AMPLIFY_FORMATION,
3 NATIVE_SAFE,2 NO_BETTER_ACTION. The first/full maximum policy is unchanged:
these trajectories generated enough actual target-risk events that MAX_INTERVAL
did not trigger after FIRST_STEP.

### Why this did not accelerate

The logs show24 SAFE and54 NEAR decisions selecting fewer predicted new correct
targets than native. For example, SAFE step1 has6 currently-correct probes:
native predicts gain7/damage5; alpha.125 predicts gain1/damage1, so preservation
priority chooses.125. This is the requested policy tradeoff, not a solver error.
It is evidence of conservative selection, not an ablation proving the sole cause
of the slower trajectory. The new selector and sensor changed together.

Read-only event replay matches every trigger, anchor/latch transition and rearm.
Across micro plus full observations,1,042 SAFE and5,112 NEAR persistent-risk
observations were suppressed;221/824 rearm events occurred. Risk events involved
149/211 distinct target identities. Those are observation/event counts, not
independent targets or instrument calls. Full refresh events do not request an
extra same-step measurement. Calls may still be frequent due to other targets,
escalation or new excursions after genuine threshold recovery. Raising a
threshold to conceal this outcome was not attempted.

The micro forward is much cheaper per call than full measurement, but did not
replace enough full calls in NEAR. Full measurements and proposal formation
remain the dominant added costs. A cheap sensor alone does not fix a policy
that trades away too much formation or alternates current-step shrinking with
uncontrolled native updates.

## Files, checks and preserved entry-point failure

- `governor.py`: count-constrained selection.
- `scheduler.py`: ordinary-forward sensor, deterministic shards and target latches.
- `runtime.py`: current-step execution; imports v1 ownership/instrument.
- `smoke.py`: scoped reuse of v1's identical run harness and result summaries.
- `tests.py`:27 ordinary tests, including real tiny CUDA adapter execution.
- `verify.py`: read-only raw-logit, trigger replay, hash and regression checks.
- `results/`: compact run records, comparisons, CHECKS and byte-identical copies
  of v1's previously untracked SAFE/NEAR JSON. No raw NPZ/checkpoints in Git.

An initial harness bug passed `create` to `patch.multiple`, where it is a reserved
argument, so the v1 constructor remained active. Both attempt-1 jobs actually ran
v1 despite the intended v2 configuration. They are INVALID AS V2 RESULTS and
excluded, with original files retained unchanged and their paths/hashes recorded.
The corrected constructor injection has a dedicated regression test. Only the
subsequent attempt-2 jobs are v2 results; no settings were tuned after observing
them. This was an execution-wiring failure, not an unsuccessful v2 trial selected
away. Corrected run logs contain micro/latch records and no loss/norm triggers.

Raw data: `E:/gfg-evidence/functional-update-governor-v2/{safe,near}/attempt-2/`.
Verification recomputes every recorded accuracy and first99 from raw logits,
replays the scheduler, checks hashes, and runs27 tests. All pass. A dependency
emits a torch.jit.script deprecation warning. No OOM occurred. World>1 is rejected;
no NCCL/distributed validation is claimed. On instrument failure, receiving
parameters are not committed but native optimizer moments may have advanced:
abort and restore both model/optimizer in a new runtime, never silently resume.

```powershell
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_functional_update_governor_v2.tests
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_functional_update_governor_v2.verify
```

`smoke safe` / `smoke near` create new append-only attempts. The summary explicitly
selects the audited entry-point failure plus corrected run, and refuses any extra
attempt rather than silently choosing a best result. Raw-dependent verification
requires the local artifacts. Source tests do not require those 70M raw records.
