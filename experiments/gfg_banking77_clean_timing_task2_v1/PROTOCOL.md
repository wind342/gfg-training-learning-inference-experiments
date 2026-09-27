# Task2 clean training timers, unchanged trajectories

Base d131c383945719a250d7ec01daa889f22167b13f. Independent directory and
create-only evidence. Task2 budgets 210/630, fixed task1 checkpoints/schedules.
No test, later tasks, other baselines, tuning, policy or solver changes.

Move SHA256, detailed record formatting, dual/list serialization, layer/cumulative
diagnostic norms and safe-candidate diagnostic repair calculations outside the
training timer. Retain all actual FP32 endpoint floor checks, selection, solver
math, numerical/trust-ball guards, and unsafe-candidate repair ratio calculations
needed by the unchanged rho update. Keep minimal reference retention/dual snapshot
cost inside control wall. No arithmetic/buffer optimization bundled here.

Perform deferred evidence work after the control timer ends on each step. Recreate
EVERY prior decision field and compare full per-step records and checkpoint tensors
against the sealed baseline/firstcheck policies. Independent replay of both clean
policies must pass before benchmarks. Evidence work is moved, not silently deleted;
report its observer time separately. Algorithm wall is entry + native + control;
it is not end-to-end experiment process wall (logging/audit/evaluation excluded).

Use nested exclusive timers rooted at entry/native/control. Include synchronization
overheads in timed regions; split child time from parent instead of summing nested
inclusive profiles. For EVERY run, exclusive parts must sum to root elapsed to 1e-7s.
The remaining parent-exclusive control region explicitly includes bookkeeping,
counting, dispatch, commit and timer overhead. Manual geometry subregion counters
are descriptive inclusive fields, not part of the additive ledger.

Freeze source/protocol/input hashes before numerical checks or validation runs.
Check unchanged solver/geometry/predictor/optimizer/schedules, deferred-record
identity and independent full native/moment/endpoint replay. Old CE backward zero,
no test access, frozen BERT/cache hash unchanged. Preserve all historical artifacts.

Benchmark both policies in both recorded (original evidence-in-controller) and clean
modes on this same runtime. Three fixed repetitions, four arms per repetition:
1. recorded baseline, clean baseline, recorded firstcheck, clean firstcheck
2. clean firstcheck, recorded firstcheck, clean baseline, recorded baseline
3. clean baseline, recorded baseline, clean firstcheck, recorded firstcheck
Each arm runs 210 then 630. All trajectories must match their original sealed policy.
No speed-based rerun or interval tuning. Report median/range and paired samples for
algorithm/control wall; isolate moving evidence via same-policy recorded vs clean,
then compare clean baseline vs clean firstcheck for actual policy speed benefit.
For additive presentation use one actual run per arm (the median-algorithm run),
not a sum of separately selected per-region medians. Explain timing noise/limits.

Stop after measurement, independent audit and report. No automatic full training
or promotion; new results describe identical per-policy trajectories with different
evidence placement. Both local solver stopping policies themselves remain unchanged.
