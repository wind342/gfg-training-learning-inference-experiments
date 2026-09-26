# V4: only completed-partial capability maintenance

Base c67de1af5ca0379b0db9f2fe20e624344e324451; preserve all prior sources/results.
At task boundaries change only completed groups still FORMING to PARTIAL.
Current groups remain FORMING. PROTECTED/DEGRADED histories are preserved.
PARTIAL uses the existing forming observation and0.9 promotion threshold; being
below0.9 alone must not turn it into DEGRADED. After promotion all existing
protected degradation/q10/rearm rules apply. No new partial-decline threshold.

Priority: DEGRADED, at-risk PROTECTED, eligible PARTIAL, stable anchor.
Freeze the following engineering schedule before observing results: at most ONE
PARTIAL on even global update steps, deterministic rotation; on odd steps use
the original stable anchor. They share one auxiliary slot, never two. If no
partial exists, V3 behavior is exact (apart from new descriptive fields).
Keep V3 risk selection/reservation:7 risk slots if anchors exist, otherwise8.
Without anchors, a partial can use an available slot but cannot displace eight
higher-priority risks; report any such starvation instead of weakening priority.
With only partials and no anchors, odd steps have no old feedback. Mode is
BALANCE when partials exist but no higher risk; cadence is separate from mode.

Unchanged: thresholds, q10/rearm, actual sample-count weighting, coalition cap8,
2 samples/group, memory10/group, predictor, emergency gate, AdamW/LR5e-6,
seed2021, clipping, dropout, shared task1 checkpoint and current batch order.
Run only native, fixed replay and V4 router, one fixed three-task experiment.
Tasks2/3 each210 updates (one epoch). No tuning/sweep, no10 tasks, no extra arm.
Historical V3 remains a hash-verified reference, not a fourth rerun.

Tests before GPU: transition isolation, partial remains partial below threshold,
promotion and subsequent degradation, shared-slot/cadence/cap, high-risk priority,
no-partial V3 equivalence, task3 coverage and memory roles. Read-only V3 smoke
PASS and prior three-task PASS admission. Create-only new evidence directory;
3600s inherited limit per arm. Preserve failures, no selected restarts.

Report old/new/all by task; task1/task2 cohort retention; exposures/active fraction;
states/release events; per-relation partial/anchor/risk replay; weight/batch
histograms; full/shrink counts; wall/peak memory. Specifically examine groups9/14,
blanket replay and whether retention improves without losing new capability.
Do not infer causality beyond this fixed fork or replace capability with cost.
Float reruns need not be bitwise equal; validate actual records and disclose
any differences, not fitted tolerances. No new native execution during audit.

Output E:/gfg-evidence/router-v4-partial-capability, new codex branch, one commit.
