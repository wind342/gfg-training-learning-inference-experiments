# Final Architecture v2: frozen ten-task validation

Engineering base: 03a17843a8c7ad82b21eecf4b7b0360feb542c56.
Historical comparison: 115534a09cd554fcd75fd46dde711fdb98988519.
Production policy is imported WITHOUT MODIFICATION from
gfg_relation_capability_floor_task8_v1.core.floor_projection, validated in
7ae5104 and continuation f24d37c. No changes to any prior source or artifact.

Exactly two new arms, v2_210 and v2_630. Both start from the identical verified
task1 checkpoint (BERT/head/AdamW state at step210), then independently and
continuously train tasks2 through10. v2_210 performs 210 scheduled updates/task;
v2_630 performs 630 (three repetitions of the fixed 210-batch task order).
Total7560 new updates. Never initialize one arm from the other's endpoint.
Task1 is NOT retrained; this is not the fresh-pretrained task1 timing smoke.

Frozen/eval/no-gradient BERT, existing80-row biased linear head, seed2021,
batch16, LR5e-5, fused AdamW betas(.9,.999), weight_decay0, gradient clip1,
no scheduler/reset/replay/old loss/Router/distillation/feature expansion/adapter.
RNG seed2021*100000+continuous optimizer scheduled counter, including skipped
updates. Counter starts210 and ends2100 / 5880 respectively. Projection must
not change moments. TF32 disabled, same existing CPU/GPU numerical settings.

Use the existing clean validation bank and primary test exclusions unchanged.
Reuse existing frozen validation features; append80 new-task audit-only
controls exactly as in existing harnesses. Only old controls enter constraints.
Do not give earlier tasks future validation labels/features or test outcomes.

At each task boundary record old-control counts before and after expanding
seen classes from8*(t-1) to8*t. K[r] is the count with strict FP32 margin>0
AFTER expansion, before any task update. Freeze K for all epochs of that task;
never ratchet it upward mid-task or recompute at epoch boundaries. Report any
expansion losses separately, never conceal them by declaring in-task safety.
Next task gets its own K from that arm's endpoint, as specified by the user.

Use unchanged nearest-boundary/tie-by-input-ID deficit selection and cumulative
active constraints within each update. Existing Hildreth coordinate/reserve,
100 TOTAL sweeps/update, first relation-floor-safe stopping, and actual FP32
recheck. Counts use margin>0. Selected-pair constraints target1e-6 plus existing
roundoff reserve, but final acceptance is the unchanged relation-count test,
not a newly imposed per-sample1e-6 requirement. On failure skip and advance
scheduled optimizer moments once. Do not claim globally nearest feasible
projection or infer infeasibility from the sweep cap.

Every committed step must satisfy C[r]>=K[r] for all old relations. Preserve
all flips, gains, selected recoveries, constraints, sweeps, failures/skips,
norm ratios/cosines, floor deficits, parameter/gradient/moment records and wall.
Save per-epoch tensor audits to bound memory; do not copy BERT every epoch.
Record training-loop wall (including instrumentation), update compute wall,
projection wall, serialization and task wall separately. Read-only disk guard
requires6GiB remaining before an epoch; resource failure is retained, not an
algorithm change or permission to rerun. Expected additional disk18-24GiB.

Test barrier: only after both arms finish a task and seal train records, evaluate
once read-only on the primary seen test set. Record old/new/all/previous-task
macro and per-relation accuracy. During that SAME endpoint forward also compute
predictions with the next task's expanded readout, on current-task-and-older
test examples only. This gives the next task's entry diagnostic without a
second training-time evaluation. The already-completed task1 checkpoint gets
the same read-only initial diagnostic. Neither diagnostic enters training.
Report prior endpoint -> expanded entry -> task end separately. Never compare
different competitor sets as if they were identical readout conditions.

No test-driven early stop, tuning, changed budget, or additional arm/seed/task.
Do not evaluate test between the630 arm's epochs. At task2, the two arms'
first210-update trajectories must be byte-identical; check before any further
task. Compare independent checker against the existing task8 checker on stored
states and production at varying seen sizes before formal execution.

Independent audit reconstructs all native AdamW updates from stored clipped
gradients/moments and separately replays constraint selection and Hildreth.
Verify all7560 actual endpoints, continuous arm histories, original per-task K,
task expansion, BERT identity, data/cache/source hashes and read-only barriers.
Recompute metrics and summaries, preserving historical Native210 and V1 records
without rerunning them. Mean old/new/all is unweighted over nine task endpoints.

Interpretation: V2 vsV1 tests the whole fixed relation-level policy (both sample
replacement and removal of within-task count ratcheting), not replacement alone.
V2_630 vsV2_210 tests a longer-budget continuous trajectory, not identical later
task starting states or equal wall. Prior task8 isolated continuation provides
separate within-task evidence. Control-test divergence demonstrates imperfect
generalization of the finite-bank guarantee; it does not identify coverage as
the unique cause. This is one-seed final architecture validation, not a claim
of universal performance. No policy changes after any observed outcomes.
