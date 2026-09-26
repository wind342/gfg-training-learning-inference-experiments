# Router V3: fixed FewRel three-task continuation

Base: `62a658ca07060f7be76436b6ba10072233b25456`.
No existing file or Router V3 strategy was changed. This directory supplies an
independent admission/continuation harness and read-only audits. All three arms
completed once: no failed training attempts, tuning, sweeps or ten-task runs.

## Admission and fixed setup

The old runner expected `SMOKE_CHECKS.json` with a `three_task_gate`, whereas V3
publishes `CHECKS.json` with a `router_v3` summary. `gate.py` validates the actual
V3 authority against its committed copy, rechecks raw hashes and 60-step records,
routing, dose and metrics, and creates a derived compatibility gate only under
the NEW output root. It never writes to the original V3 evidence directory.
Every arm repeats that read-only admission before loading the model.

All arms share the existing task1 model/AdamW checkpoint (not three independently
trained task1 models), seed2021, LR5e-6, FP32, clipping1, and current-task order.
Tasks2/3 each contain one epoch: 210 updates of16 current examples. Thus every
arm processes the same 6,720 new-example occurrences. No task oracle at readout:
the active class scope expands from8 to16 to24. Router history, memory and cursor
continuity are verified across the task2->3 boundary. V3 uses its exact existing
states, coalition rules, sample-count dose, q10/threshold, predictor and gate.

Fixed replay contributes16 old examples/update:2 from each of8 rotating groups.
In task3 its16 old relations are covered in alternating8-group batches, not all
16 relations per update. All methods use the same pinned local dependencies.

## Endpoint capability

Accuracy is over the held-out split with140 examples/relation. Task3 old accuracy
averages16 old relations, so all accuracy is (2*old+new)/3, not (old+new)/2.

| Endpoint | Method | Old | New | All |
|---|---|---:|---:|---:|
| Task1 shared prefix | All three | N/A |97.1429%|97.1429%|
| Task2 | New-only native AdamW |5.6250%|92.5893%|49.1071%|
| Task2 | Fixed replay native |87.8571%|90.0000%|88.9286%|
| Task2 | Router V3 |86.6071%|90.5357%|88.5714%|
| Task3 | New-only native AdamW |5.8929%|98.0357%|36.6071%|
| Task3 | Fixed replay native |82.5446%|97.2321%|87.4405%|
| Task3 | Router V3 |77.4107%|97.3214%|84.0476%|

After task3, separate old-cohort accuracy is:

| Method | Original task1 cohort | Task2 cohort | Current task3 cohort |
|---|---:|---:|---:|
| New-only |0.7143%|11.0714%|98.0357%|
| Fixed replay |86.3393%|78.7500%|97.2321%|
| V3 |86.8750%|67.9464%|97.3214%|

Against new-only, V3 retains much more old capability (+71.5179 percentage
points) at a0.7143-point reduction in new accuracy. This is not Pareto superiority
on both separate metrics. Against fixed replay, new accuracy is almost equal
(+0.0893 points), but old accuracy is5.1339 points lower and all accuracy3.3929
points lower. The deficit is concentrated in the task2 cohort, not the original
task1 cohort. This run does NOT establish equivalent retention to fixed replay.

## Feedback dose, state and cost

| Method | Task2 old exposures | Task3 old exposures | Total | Active task2/task3 | Continuation wall | Peak allocated |
|---|---:|---:|---:|---|---:|---:|
| New-only |0|0|0|0% / 0%|193.152 s|4.929 GiB|
| Fixed replay |3360|3360|6720|100% / 100%|313.123 s|5.359 GiB|
| V3 |1482|2556|4038|98.5714% / 100%|599.127 s|5.745 GiB|

V3 reduces old exposure55.8929% in task2,23.9286% in task3 and39.9107% overall.
Exposure counts include repeated memory examples, not unique examples. It does
not save wall time. Continuation wall includes audit/monitoring and fixed
evaluations, excludes initial loading and the shared task1 prefix. Shared task1
historical wall is140.274 s; it must not be charged as three new training runs.
Summed update seconds (excluding evaluations and outer-loop overhead) are:
native63.535/63.831, fixed124.445/123.024, V3 218.410/305.718 for tasks2/3.

V3 state counts:

| Task | FORMATION | BALANCE | RECOVERY | RELEASE events | Activations |
|---|---:|---:|---:|---:|---:|
|2|3|13|194|1|2|
|3|0|0|210|0|0|

RELEASE is a transition event, not a fourth mutually exclusive state. Task3
inherits an already active router; it does not reset recovery debts.

Effective old-loss weight distribution (weight: number of updates):

- Task2 V3:0:3;1/5:17;3/11:69;1/3:105;5/13:16.
- Task3 V3:5/13:54;3/7:84;7/15:72.
- Fixed replay:1/2:210 in each task; new-only:0:210 in each task.

These correspond to V3 old batch sizes0/4/6/8/10 in task2 and10/12/14 in task3.
Task3 never reaches the fixed16-example dose, but feedback is active on every
update; average old batch size is12.1714 versus fixed16. It is not fixed replay
in allocation, but the frequency is identical and the exposure saving narrows.

Full predictor calls:9 in task2 and10 in task3. Emergency shrink:1 and2,
respectively. All three use alpha0.5; the other417 steps use alpha1. No
NO_SAFE_ALPHA event occurred. Native-safe full calls:8+8. Baselines have no
predictor calls or shrink. Selected-endpoint prediction agrees on3840/3840
boundaries, including14 actual transitions (14/14); this does not audit every
unmeasured update or imply that the controller prevents all forgetting.

## Task3 allocation and coverage gap

At allocation time the recorded risk count is4 for54 updates,5 for84 and6 for72.
Every selected non-anchor group belongs to the recorded DEGRADED/at-risk set;
exactly one stable anchor is added on each task3 update. All recorded risk groups
receive feedback whenever eligible: zero risk-unserved steps for every group.
Thus there is no capacity/priority starvation of already-recognized risks in
this run. These are probe-history risk states, not privileged test truth.

However, groups9 and14 remain FORMING for all210 task3 updates. They receive
zero replay despite having partial held-out capability. Their accuracies fall
57.14%->2.14% and82.14%->45.00%. They are outside V3's eligible risk/anchor sets,
not omitted because the coalition was full. All other old relations receive
feedback; their longest consecutive no-feedback span is at most17 updates.

| Group | V3 task2 replay | V3 task3 replay | Task3 longest no-feedback | V3 task3 start accuracy | V3 task3 end accuracy | Fixed task3 end accuracy |
|---|---:|---:|---:|---:|---:|---:|
|0|414|420|0|72.86%|71.43%|60.00%|
|1|80|46|9|97.86%|95.71%|97.14%|
|2|78|174|9|97.86%|90.71%|95.71%|
|3|384|420|0|55.00%|54.29%|52.14%|
|4|104|234|9|83.57%|92.14%|93.57%|
|5|268|372|9|93.57%|95.71%|94.29%|
|6|78|48|9|97.86%|97.14%|100.00%|
|7|76|46|16|94.29%|97.86%|97.86%|
|8|N/A|50|9|100.00%|98.57%|98.57%|
|9|N/A|0|210|57.14%|2.14%|42.86%|
|10|N/A|46|17|100.00%|97.14%|98.57%|
|11|N/A|186|9|100.00%|80.00%|86.43%|
|12|N/A|48|9|100.00%|100.00%|100.00%|
|13|N/A|420|0|95.71%|45.71%|39.29%|
|14|N/A|0|210|82.14%|45.00%|78.57%|
|15|N/A|46|15|89.29%|75.00%|85.71%|

Fixed replay contributes420 occurrences per old group in task2 and210 per old
group in task3; new-only contributes zero. N/A means a group was current, not
old, during task2; all current groups still receive their full training epoch.
All per-relation metrics/counters are machine-readable in results/CHECKS.json.

The two omitted groups account for104 of115 fewer correct old-test answers
relative to fixed replay (90.43% of the net aggregate gap). This is a descriptive
decomposition, not an intervention establishing that adding their replay alone
would recover the gap. Some heavily replayed groups also retain incomplete
capability; for example group13 receives420 occurrences yet falls95.71%->45.71%.
Feedback receipt and accurate prediction are not guarantees of recovery.

## Verification and numerical disclosure

Preflight:97 tests passed, including10 new harness tests. Data-role checks passed
for both scopes, with80 then160 old-memory examples. Memory excludes encoded
control/test duplicates. The inherited control/test duplicate at test7561 is
not repaired or hidden; exclusion sensitivity remains in the metric records.

The in-progress extra strict prefix check found float-only differences from
the V3 smoke starting at step217. Its maximum absolute float difference over
the first60 steps is8.165183288610933e-6. Sample identities, discrete router
states/capabilities, coalitions, doses, alpha/gate decisions and measured boundary
signs are identical. This is NOT a bitwise floating-point reproduction.

The original pre-run `check.py` remains unchanged and hash-verified, including
its overly strict bitwise smoke-prefix assertion. Use `audit.py` for the final
audit: it was added after that diagnostic, reports every floating discrepancy,
requires discrete decisions and signs to agree, and uses no fitted tolerance.
It also validates all three runs' records, source/checkpoint/data hashes,
identical current batches, boundary-state continuity, router/cursor decisions,
sample-count dose and metrics from saved predictions. This is a record audit,
not an independent native inference rerun. The final auditor's own hash is
included in CHECKS.json. No training source or old evidence was changed to
accommodate the floating differences.

## Reproduction and artifacts

```text
python -m experiments.gfg_router_v3_three_task_continuation.tests
python -m experiments.gfg_router_v3_three_task_continuation.run native
python -m experiments.gfg_router_v3_three_task_continuation.run replay
python -m experiments.gfg_router_v3_three_task_continuation.run router
python -m experiments.gfg_router_v3_three_task_continuation.audit
python -m experiments.gfg_router_v3_three_task_continuation.audit_tests
```

Commands require the existing pinned local FewRel/V3 environment. Runner and
auditor use create-only outputs and reject overwriting the completed run.
Raw records: E:/gfg-evidence/router-v3-three-task-continuation/.
Compact reports: results/PREFLIGHT.json, results/CHECKS.json and
results/FINAL_TESTS.json. No dataset, model or checkpoint is committed.
