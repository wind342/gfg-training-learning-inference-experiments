# Fixed Router V3 three-task continuation

Base: 62a658ca07060f7be76436b6ba10072233b25456. This is an independent harness,
not a policy revision. Preserve every existing source, smoke result and frozen tag.
Read V3 CHECKS.json, recompute its smoke integrity checks and compare raw hashes
before admitting any arm. Write a derived compatibility gate only in the new
output directory, never in the V3 directory. A bare PASS string is insufficient.

Run exactly three arms: native (new-only AdamW), replay (fixed native replay),
router (unchanged V3). Share the existing task1 model/AdamW checkpoint, seed2021,
LR5e-6, FP32, clipping1, one epoch/task and deterministic current batch order.
Task1 is a shared historical prefix, not independently retrained per arm.
Tasks2 and3 each use 210 updates of16 current examples; 420 new updates per arm.
Fixed replay selects8 old groups cyclically and2 examples/group, including when
task3 has16 old groups. V3 keeps its exact state, coalition, sample-count dose,
memory, q10, thresholds, predictor, emergency gate and microbatch backward.
Keep state/memory continuity across task boundaries and expand seen-class scope.

No sweep, policy tuning, ten-task run, extra arm or result-selected rerun.
Use the inherited 3600s cap per arm and create-only output. Preserve any failure.
Check data-role/memory collisions for both task scopes before GPU execution.

Report task1/2/3 endpoint old/new/all accuracy, old-example exposure, active
fraction, FORMATION/BALANCE/RECOVERY counts and RELEASE events (not a fourth
exclusive state), per-relation replay, effective old-loss weights, full calls,
alpha/gate/shrink counts, wall and peak allocated GPU memory. Separate shared
task1 cost, continuation wall and summed update time per task.

For task3, report risk/anchor allocation, batch-size and risk-count histograms,
per-relation exposure and maximum consecutive no-feedback/risk-unserved spans.
Report UNFORMED old groups explicitly; risk means the router's observed state,
not privileged full-test truth. Compare task3 retention on the original task1
cohort and task2 cohort separately as well as pooled old accuracy. Correlate
coverage with per-relation endpoint accuracy without using tests for control.
Do not declare superiority from lower exposure if capability retention fails.
No ad-hoc success thresholds: report continuous gaps against both baselines.

Raw output: E:/gfg-evidence/router-v3-three-task-continuation.
Commit compact results, validators and harness to a new codex branch after
checking; do not commit datasets, model weights or checkpoints.
