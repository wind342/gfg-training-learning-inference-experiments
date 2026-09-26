# Functional update governor v3

Base: `ba9790eb212148ae703f93456f5d0d3941e7afca`. Isolated engineering work;
v1/v2, the frozen paper tag, main and existing evidence are unchanged. This adds
group-level control, not a new optimizer direction or a new prediction model.

## Results

Both fixed runs reached 99%. Compared with governor v2, SAFE required fewer steps
and substantially fewer full measurements; NEAR recovered successful attainment
within the 200-update budget. Neither run beat its historical same-LR native
AdamW baseline in steps or wall time. Do not present these runs as acceleration
over native AdamW.

| Case / method | First update >=99% | Wall to99 | Full calls |
|---|---:|---:|---:|
| SAFE governor v3 | 123 | 9.66 s | 3 |
| SAFE governor v2 | 131 | 15.63 s | 26 |
| SAFE governor v1 | 119 | 14.96 s | 29 |
| SAFE historical dynamic runtime v2 | 143 | 22.67 s | 29 |
| SAFE same-LR native | 121 | 7.65 s | 0 |
| SAFE tuned native | 106 | 6.78 s | 0 |
| NEAR governor v3 | 182 | 16.05 s | 12 |
| NEAR governor v2 | Not reached | Not reached | 63 |
| NEAR governor v1 | 191 | 19.95 s | 32 |
| NEAR historical dynamic runtime v2 | 158 | 24.99 s | 32 |
| NEAR same-LR native | 174 | 11.00 s | 0 |
| NEAR tuned native | 106 | 6.82 s | 0 |

V3 endpoint accuracy: SAFE 99.0536%, NEAR 99.6845%. V2 NEAR ended at 93.3754%
after 200 updates and 29.70 s; that time is not time to99. Baselines are existing
historical runs, not contemporaneous reruns. No parameter sweep or post-result
threshold change. Exactly one actual v3 attempt per case, with no failed attempts.

Setup remains seed201, 71,405,312 parameters, sequence3, vocabulary24, FP32,
batch128, AdamW(.9,.98), decay0, clip1. SAFE LR3.7688918072220455e-6;
NEAR LR6.030226891555273e-5; tuned native LR7.537783614444091e-6.
Same 212 control /317 evaluation queries and original training batch generator.
Evaluation queries are excluded from the controller, not from training.
Two discarded warmup steps precede exact reinitialization. Wall includes
training, control and per-update evaluation, but excludes initialization.

## Canonical grouping and hierarchy

TL-E06 `gfg_nanogpt_support_redundancy_v1/builder.py` groups by the correct
target token and uses linear-interpolated q10 margins. TL-E07
`gfg_nanogpt_support_reallocation_audit_v1/analysis.py` aggregates correctness
within each group. TL-E08 `gfg_nanogpt_identity_aligned_margin_crossing_v1/experiment.py`
distinguishes target boundary transitions from aggregate capability.

V3 implements target margin/transition -> group capability -> overall capability.
The current bank uses direct modular-addition labels; older scientific runs used
opaque task token relabeling. This reuses their grouping definition, not their
historical sample/token identities. The task harness explicitly checks the 23
correct target labels, 0..22. Generic policy code accepts caller-supplied groups.
Ordered bank identity/version/shape binding fails closed on replacement/reordering.

Group sizes in label order:
`11,7,14,8,12,11,11,11,7,12,9,7,12,9,10,10,7,5,7,13,4,7,8`.
Twelve groups have fewer than ten probes: the fixed 90% threshold then requires
all probes correct. Group protection does not imply one crossing can never alarm.
The grouping rule and margin q10 are checked directly against the original code.

## Control rule

- Keep the v1 actual-update/JVP/boundary instrument unchanged. Aggregate its
  predicted margins for each alpha in `{.125,.25,.5,1,2}`.
- Currently protected groups have measured capability >=90%. Historical degraded
  groups do not become extra recovery constraints in candidate selection.
- With no protected groups, maximize overall predicted correct count, then macro
  group capability. Losses of individual correct targets are diagnostic only.
- If native preserves all protected groups, restrict candidates to those that
  also preserve them, then maximize overall and macro capability.
- If native violates protection, minimize violation count, worst deficit and
  total deficit, then maximize overall and macro capability.
- Native wins equal overall/macro scores among the best allowed candidates.
  q10 alone cannot justify a nonnative action. In formation mode, new protected
  groups and q10 break ties among nonnative candidates already superior to native.
- Only control the measured current update. `release_steps=0`; otherwise use
  native `optimizer.step()`. No stale amplitude carryover.

SAFE step1 has six correct probes, all in group7 (6/11, not a protected group).
V3 chooses alpha2, predicting eight correct, rather than v2's alpha.125 preserving
six. This is a net capability decision, not protection of six separate abilities.

## Group sentinel

Observe one entire group per ordinary step, round-robin. Configurable
`group_micro_every>1` is supported; these fixed runs use1. This is one eval forward,
without JVP/backward/proposal/parameter swap/optimizer simulation. Track each
group's FORMING/PROTECTED/DEGRADED state, capability, q10, anchor and observation age.

Full triggers: first step, maximum50-update gap, protected capability below90%,
or protected q10 below50% of a positive anchor (anchor floor1e-6). Nonpositive
current q10 also counts as erosion after a positive anchor. FORMING fluctuations
do not alarm. A group excursion latches; continued risk does not repeatedly
trigger. Different groups and soft-to-hard escalation remain independent.
Rearm requires capability >=90% and q10 >=75% of the positive pre-excursion anchor;
without a valid positive anchor, capability recovery suffices. Full refresh
preserves latched anchors. Coverage takes23 micro observations, not necessarily
23 updates because scheduled full measurements replace the micro observation.

## Recorded decisions and cost

| Quantity | SAFE | NEAR |
|---|---:|---:|
| Full fraction | 3/123 (2.44%) | 12/182 (6.59%) |
| Full-call reduction from v2 | 26 ->3 (88.46%) | 63 ->12 (80.95%) |
| Group micro calls | 120 | 179 |
| Micro wall total | 0.855 s | 1.210 s |
| Full instrument wall | 0.608 s | 2.314 s |
| Proposal wall | 0.191 s | 0.792 s |
| Peak CUDA allocation | 1,892,453,888 bytes | 1,892,453,888 bytes |

Executed alpha counts:

- SAFE: `.125:1, .5:1, 1:120, 2:1`.
- NEAR: `.125:2, .25:1, .5:4, 1:174, 2:1`.

SAFE full triggers: FIRST_STEP1, MAX_INTERVAL2. NEAR: FIRST_STEP1,
MAX_INTERVAL2, PROTECTED_GROUP_DEGRADATION9. No erosion-only trigger occurred in
these two runs; its behavior is covered by tests, not claimed as a measured gain.
NEAR full decisions: FORMATION_PROGRESS2, NATIVE_SAFE4,
SHRINK_PROTECTED_GROUP5, AMPLIFY_FORMATION1.

Full-measurement current protected-group counts:
SAFE `0,0,10`; NEAR `0,1,1,0,3,1,2,3,7,10,17,20`.
These are current counts, not cumulative historical protected counts. Per-step
historical state counts/ages, every candidate's predicted capability and transition
counts are retained in RESULT files. CHECKS records selected predicted gains and
violations. NEAR step131 has one selected predicted protected violation: the best
available candidate still violates one group. The policy is constrained selection,
not a guarantee that every protected group can be preserved.

## Remaining issue

The group correction substantially reduces measurement overhead, but does not
yet produce a step advantage over same-LR native training. SAFE formation is
reconsidered only at steps1,51,101; sparse observation can miss useful control
opportunities. That is an implementation consequence, not a proven explanation
of the performance gap. Micro monitoring still costs0.855/1.210 s, and NEAR full
measurement/proposal adds about3.11 s. A single seed cannot establish general
performance. No extra experiments were run to attribute the remaining gap.

## Files, checks and rerunning

New files only: `groups.py`, `governor.py`, `scheduler.py`, `runtime.py`,
`smoke.py`, `tests.py`, `verify.py`, this README and four result JSON files.
All imported v1/v2/distributed-runtime source files remain byte-identical.
34 tests pass, including CPU/CUDA execution, native optimizer moments, micro
non-mutation/JVP avoidance, group latching/coverage, tie behavior, finite checks
and existing variable-alpha backends. Recorded scheduler/decisions and raw
evaluation logits were replayed/recomputed successfully; source/raw hashes pass.
This is not an independent rerun of training or all predicted derivatives.

```text
python -m experiments.gfg_functional_update_governor_v3.tests
python -m experiments.gfg_functional_update_governor_v3.smoke safe
python -m experiments.gfg_functional_update_governor_v3.smoke near
python -m experiments.gfg_functional_update_governor_v3.smoke summary
python -m experiments.gfg_functional_update_governor_v3.verify
```

Raw append-only attempts remain outside Git at
`E:/gfg-evidence/functional-update-governor-v3/{safe,near}/attempt-1/`.
Their paths and SHA-256 identities are in the compact checked-in results.
Verification of raw logits requires those external files. The summary deliberately
rejects multiple attempts, and output writers do not overwrite reports. Do not
delete historical attempts or select a favorable run to bypass those checks.
