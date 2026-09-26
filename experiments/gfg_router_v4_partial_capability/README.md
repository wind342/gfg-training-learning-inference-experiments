# Router V4: completed PARTIAL capability maintenance

Base: `c67de1af5ca0379b0db9f2fe20e624344e324451`.
Only new files are added. All V3 code, smoke/three-task evidence and frozen tags
remain unchanged. Three arms were rerun once with the same checkpoint and
protocol; historical V3 is a reference, not a fourth rerun. No sweep or10-task run.

## What changed

- At task expansion, completed groups still FORMING become PARTIAL. Current
  task groups remain FORMING; protected/degraded histories are not reset.
- PARTIAL reuses the original forming observer and0.9 promotion threshold.
  Below-threshold observations keep it PARTIAL, not automatically DEGRADED.
  After promotion, original degradation/q10/rearm rules apply unchanged.
- Higher-priority risk selection is unchanged. At most one PARTIAL is eligible
  on even global updates, with deterministic rotation. It shares the original
  auxiliary slot with the stable anchor (odd updates when both exist).
- If no anchor exists, PARTIAL uses only remaining capacity; it cannot displace
  eight higher-priority risks. No blanket filling with additional partial groups.

The live policy contains no special case for groups9/14. Thresholds, dose,
coalition limit8, two examples/group, memory10/group, predictor, emergency gate,
optimizer/LR and current batch ordering are inherited unchanged. Existing modules
are adapted in memory by the independent runner, not edited on disk.

## Setup and validation

Same shared task1 model/AdamW checkpoint, seed2021, LR5e-6, one epoch/task.
Tasks2/3 each have210 updates of16 current examples,6,720 new occurrences/arm.
Read-only V3 smoke PASS and prior three-task PASS admission is checked before
each arm. Outputs are create-only; no result-selected restart or tuning occurred.

119 preflight tests passed, including17 new PARTIAL tests and actual task-boundary
integration. An initial integration-test fixture recreated tensor views and
triggered the original identity guard; the fixture was corrected before any
training, without weakening that guard. Final regression results are recorded
separately. The final combined run then hit the same recreated-view defect in
an older frozen V3 fixture (which had passed preflight through object-ID reuse).
Its failure remains in results/FINAL_TESTS.json. `regression.py` explicitly skips
that one unstable fixture and executes its unchanged assertions with persistent
views:119 passed,1 old fixture skipped,120 entries total. No original test file
or production identity guard was edited. See
results/FINAL_TESTS_WITH_FIXTURE_ADAPTER.json.
The first adapter report export could not serialize the skipped TestCase object;
its partial output is retained as `.json.partial.txt`. The exporter now records
the test identifier as text; this did not affect any test or training outcome.

Task1 shared accuracy is97.1429%; it was not retrained separately for each arm.
No task oracle is used at readout; active classes expand8->16->24. Test data do
not select replay or stop training. The inherited control/test duplicate at
index7561 is disclosed and sensitivity retained; memory excludes encoded
control/test duplicates. Audits recompute metrics and routing from records,
not independent native inference reruns.

## Capability outcomes

| Task endpoint | Method | Old | New | All |
|---|---|---:|---:|---:|
|2|New-only native|5.6250%|92.5893%|49.1071%|
|2|Fixed replay native|87.8571%|90.0000%|88.9286%|
|2|V4 PARTIAL router|86.6071%|90.5357%|88.5714%|
|3|New-only native|5.8929%|98.0357%|36.6071%|
|3|Fixed replay native|82.5446%|97.2321%|87.4405%|
|3|Historical V3|77.4107%|97.3214%|84.0476%|
|3|V4 PARTIAL router|82.0982%|97.0536%|87.0833%|

V4 task2 endpoint matches V3. At task3, old accuracy improves4.6875 percentage
points over V3, new accuracy declines0.2679 points and all accuracy improves
3.0357 points. The old-accuracy deficit versus fixed replay narrows from5.1339
to0.4464 points (91.30% of the prior net gap closed). New accuracy is0.1786
points below fixed replay. This is a favorable result on this fixed fork, not
a multi-seed equivalence claim or complete recovery of every old capability.

After task3, old-cohort accuracy is:

| Method | Original task1 cohort | Task2 cohort |
|---|---:|---:|
|New-only|0.7143%|11.0714%|
|Fixed replay|86.3393%|78.7500%|
|Historical V3|86.8750%|67.9464%|
|V4|85.8036%|78.3929%|

The task2 cohort improves10.4464 points; the task1 cohort declines1.0714 points
relative to V3. Sharing anchor capacity is a real allocation trade-off.

## Previously omitted partial groups

At step420 exactly groups9/14 are converted; no group is converted at step210.
Both remain PARTIAL throughout task3, not promoted to high-frequency recovery.

| Group | Task3 start | V3 end | V4 end | Fixed replay end | V4 replay occurrences | Longest no-feedback span |
|---|---:|---:|---:|---:|---:|---:|
|9|57.1429%|2.1429%|35.7143%|42.8571%|106|3 steps|
|14|82.1429%|45.0000%|74.2857%|78.5714%|104|3 steps|

These are53 and52 rotation turns, two examples/turn. Previously both had zero
feedback for210 steps. The zero-feedback defect is fixed; some loss relative
to each group's task3 starting capability remains.

Task3 per-relation replay (occurrences, not unique examples):

| Group | V4 total | PARTIAL portion | Anchor portion | Maximum no-feedback span |
|---|---:|---:|---:|---:|
|0|420|0|0|0|
|1|24|0|24|19|
|2|168|0|14|19|
|3|420|0|0|0|
|4|196|0|16|19|
|5|420|0|0|0|
|6|24|0|24|19|
|7|30|0|30|19|
|8|26|0|26|33|
|9|106|106|0|3|
|10|56|0|18|35|
|11|196|0|16|17|
|12|24|0|24|29|
|13|420|0|0|0|
|14|104|104|0|3|
|15|58|0|18|31|

Every recorded high-priority risk is served when eligible: zero risk-unserved
steps. PARTIAL+anchor never exceeds one group per update. Some stable-anchor
gaps grow to35 steps; they are disclosed, not hidden by an average.

## Exposure and resource cost

| Method | Task2 old | Task3 old | Total old | Wall (tasks2/3) | Peak allocated |
|---|---:|---:|---:|---:|---:|
|New-only|0|0|0|193.309 s|4.929 GiB|
|Fixed replay|3360|3360|6720|314.127 s|5.359 GiB|
|Historical V3|1482|2556|4038|599.127 s|5.745 GiB|
|V4|1482|2692|4174|620.020 s|6.175 GiB|

V4 uses37.8869% less total old exposure than fixed replay (task3 saving19.8810%).
It uses136 more old occurrences than V3, as the closed-loop risk trajectory
changes; the auxiliary budget is shared, not expanded. There is no wall-time
advantage. Wall includes monitoring/evaluation, excludes initial model loading
and the shared historical task1 cost (~140.274 s). Per-task summed update times
are in CHECKS.json and are not substituted for total wall.

V4 feedback active fraction:207/210=98.5714% in task2 and210/210=100% in task3.
Task2 modes:194 RECOVERY,13 BALANCE,3 FORMATION,1 RELEASE event. Task3:210
RECOVERY,0 RELEASE events. RELEASE is a transition, not a fourth exclusive mode.

Old-loss weights and step counts:

- Task2:0:3;1/5:17;3/11:69;1/3:105;5/13:16 (unchanged from V3).
- Task3:5/13:34;3/7:77;7/15:78;1/2:21.
- Fixed replay:1/2 on every update; new-only:0 on every update.

Task3 batches10/12/14/16 occur34/77/78/21 times. Corresponding observed risk
counts are4/5/6/7, plus exactly one partial OR stable auxiliary. Thus21 full
batches reflect7 recorded risks, not blanket partial filling. Frequency remains
always-on; allocation is still targeted, not fixed equal replay.

Full predictor calls:9+11=20. Emergency shrink:1+2=3; alpha1 on417 updates,
alpha0.5 on2 and alpha0.25 on1. No NO_SAFE_ALPHA event. Selected endpoint
agreement4080/4080, including11 actual transitions, all predicted. This does not
guarantee preservation on unmeasured updates or eliminate the remaining losses.

## Repetition and integrity boundaries

Both newly rerun baseline arms match the historical endpoint accuracies and
per-example correct/incorrect outcomes. Final wrong-answer identities differ
on3 new-only examples and1 fixed-replay example; this is not claimed as bitwise
deterministic execution. The first60 V4 control decisions/weights match V3, and
the task2 endpoint is unchanged. All actual records remain preserved.

Checker PASS includes source/checkpoint/data/log hashes, common current batches,
memory/cursors, boundary lifecycle, priority/cadence, per-sample dose, gate
decisions and metrics from saved predictions. New observer transitions are also
unit-tested. Micro histories are taken from recorded execution; no claim of
independent full native re-execution is made.

```text
python -m experiments.gfg_router_v4_partial_capability.tests
python -m experiments.gfg_router_v4_partial_capability.run native
python -m experiments.gfg_router_v4_partial_capability.run replay
python -m experiments.gfg_router_v4_partial_capability.run router
python -m experiments.gfg_router_v4_partial_capability.check
python -m experiments.gfg_router_v4_partial_capability.regression
```

The pinned local FewRel/V3 environment is required. Outputs reject overwrites.
Raw records: E:/gfg-evidence/router-v4-partial-capability/.
Public summaries: results/PREFLIGHT.json, results/CHECKS.json,
results/FINAL_TESTS.json (original failure) and
results/FINAL_TESTS_WITH_FIXTURE_ADAPTER.json (explicit fixture correction).
No dataset, model weights or checkpoint is committed.
