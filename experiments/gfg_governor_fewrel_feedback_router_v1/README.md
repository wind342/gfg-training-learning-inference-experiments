# FewRel Functional Feedback Router V1: completed60-step check

## Architecture

The new router selects old training feedback before backward. Native AdamW then
forms the real mixed update, and the unchanged boundary calculation serves a
new emergency-only gate. Generic code is in `../gfg_functional_feedback_router_v1/`;
this directory contains data-role validation, the FewRel harness and result checker.
See PROTOCOL.md for fixed choices and the conditional extension gate.

Router modes and allocation: historical old DEGRADED=>RECOVERY; otherwise
PROTECTED level1=>BALANCE; otherwise FORMATION/no replay. Risk clearance emits
RELEASE immediately. At most8 groups x2 exemplars, deterministic per-tier and
per-group rotation; reserve one stable anchor if available and fill spare slots
with anchors. Current groups cannot be selected from old memory. No alpha2,
formation maximization, recovery alpha tie-break, new optimizer, or learned router.

## Integrity

- Base6fc8f88e9dcc0320ba0bae7599f5f87c4350c9c7; only the two new directories.
- Same task1 checkpoint/optimizer and first960 task2 training occurrences.
  The checkpoint already has DEGRADED groups0 and4; no artificial FORMATION reset.
-80 fixed old TRAIN exemplars, ten/relation, append-only. Control and test IDs
  are excluded from backward; exact encoded control/test duplicates are also
  excluded from memory. Future/current groups cannot become old feedback.
- The existing inherited control/test duplicate at test index7561 remains
  disclosed; secondary exclusion does not change conclusions. No dataset rewrite.
-24 new tests+42 existing v4 regression tests+4 existing FewRel adapter tests
  passed. Includes native CUDA execution, exact two-loss gradient/moment check,
  gate restrictions, history timing, cursor/role checks and memory identity.
- The result checker recomputes routing/cursors from recorded history, gates from
  predicted endpoints, metrics from saved test predictions, source/checkpoint/
  dataset hashes and data-role identities. It is not an independent full training
  rerun. Micro-observation histories are consumed as recorded, not re-forwarded.
- Both new training attempts completed without failure. The cache-release-only
  workaround from the prior experiment is applied before full measurements and
  included in timing. Original failure records remain untouched in the prior study.
- Raw records: `E:/gfg-evidence/functional-feedback-router-v1/`; compact report and
  raw hashes: `results/SMOKE_CHECKS.json`. No model/data/checkpoint is added to Git.

## 60-step smoke

Same initial old-test accuracy97.1429%; new0%. Test:1120 old+1120 new, all16 classes
active, no task identifier. All arms see960 identical new training samples.
Always-replay is the previously completed exact-protocol native reference, checked
against its frozen result hash. New-only native is newly run, not renamed v4.

| Arm | Old accuracy | New accuracy | All accuracy | Old exposures | Wall | Peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| New-only native AdamW |74.1071%|84.5536%|79.3304%|0|70.803s|5,292,684,288|
| Always-replay native reference |93.3929%|79.0179%|86.2054%|960|85.873s|5,753,887,232|
| Router V1 |93.6607%|79.0179%|86.3393%|960|131.262s|6,629,485,056|

Router versus new-only retains19.5536 percentage points more old capability,
while new capability is5.5357 points lower. This is a substantial preservation
effect with new learning still present, not a time-only objective. Router versus
always-replay changes old accuracy by only0.2679 points (three examples), with
identical final new accuracy. It does not establish a meaningful incremental
capability advantage. Feedback membership/counts are identical to always-replay
in this eight-old-group case; ordering differs and affects dropout/sample pairing.
The tiny endpoint difference must not be attributed to a proved routing benefit.

Router behavior:

- FORMATION0 / BALANCE0 / RECOVERY60; active60/60=100%; activation1; RELEASE0.
- Each of the8 old relations contributes120 occurrences:960 total,80 unique items.
- Full predictor3/60 at offsets1,37,55. All three gates NATIVE_SAFE; shrink0;
  NO_SAFE_ALPHA0; alpha1 on all60 steps. Selected endpoint agreement480/480,
  with just one actual correct/incorrect transition, correctly predicted.
- **Router collapsed to always-replay behavior in this window.** Old-example
  saving versus fixed replay:0%. No claim of replaying the entire old dataset:
  it repeatedly uses the bounded80-example memory, not all old training data.
- Resource diagnostic only: old-accuracy gain over new-only per100 old exposures
  is2.0368 percentage points router versus2.0089 fixed replay. These descriptive
  ratios are not a new score, a linear scaling law or evidence of superiority.

## Why no RELEASE, and why no 3-task run

Old group0 remained in the inherited degraded excursion. At offset20 its latest
observed control accuracy reached90%, but q10=.06872 was below the existing
rearm requirement .75 x .153187=.11489; this did not clear its history. At the
last observation its control accuracy was80% and q10 negative. This is a local
ten-probe observation, not proof that the group's entire test population cannot
recover. Other old groups also showed margin risk later. We did not lower
thresholds, reset anchors or change probes after observing the result.

Any persistent old risk activates replay; with exactly8 eligible old groups,
anchor filling then includes all8. This explains the observed fixed-replay-like
allocation. Persistent activation is consistent with this policy's recorded
risk, but fails its hoped-for resource-saving behavior.

The preregistered extension gate required integrity PASS, activation, RELEASE,
and final new accuracy>=70%. RELEASE was absent, so `three_task_gate=false`.
The3-task harness is implemented but NOT executed/validated as an experiment;
no ten-task run or sweep. All gate conditions are reported rather than tuned.

## Actual engineering value

Relative to new-only/alpha-only control, access to old feedback preserves much
more old capability. Relative to already effective fixed replay, this V1 has
not demonstrated added practical value: no old-example saving, no clear extra
capability gain, more monitoring/prediction cost. The implemented separation of
feedback construction and emergency checking is tested, but is not by itself
evidence of a better continual-learning controller.

```text
python -m experiments.gfg_functional_feedback_router_v1.tests
python -m experiments.gfg_governor_fewrel_feedback_router_v1.run native
python -m experiments.gfg_governor_fewrel_feedback_router_v1.run router
python -m experiments.gfg_governor_fewrel_feedback_router_v1.check
```

Create-only outputs intentionally reject overwriting an existing run/report.
Use the existing governor-v4-bench environment and pinned parent model/data.
The original B reference is reused, not regenerated. Wall includes fixed test
evaluation, monitoring and actual endpoint audits; model loading is excluded.
