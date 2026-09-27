# Task2 one first-sweep endpoint check per outer

Independent experiment from 90ca749 (accepted Stage1 arithmetic from f0b6aaf).
Task2 only, budgets 210 and 630 fixed before either run, same original task1
checkpoints and current-batch schedules. No test, tasks3-11, ER/RanPAC, tuning,
old CE backward or historical artifact modification.

Only policy change: after the first complete coordinate + ball sweep of each
outer solve, reconstruct/materialize the candidate and perform the original
buffer256 FP32 endpoint against all fixed task-entry intent floors. If safe,
return FIRST_SWEEP_ACTUAL_SAFE and commit immediately. This is NOT solver PASS;
original PASS tolerance remains 1e-8 and its other conditions remain unchanged.
If unsafe, continue all original solver state (including halfspace and ball
corrections) without reset to original PASS or 200 sweeps. Then the original
endpoint/outer continuation/rho/8-outer rollback logic applies. Reuse the first
candidate if the solver terminates at sweep 1. No later periodic inner checks.

Keep model, optimizer, moments, predictor, selection, analytic Jacobian, reserve,
EPS, rho, limits, cached arithmetic and buffer256 unchanged. A distinct training
trajectory is expected. No simultaneous buffer or geometry optimization.

Record every first check: coef, dual hash, materialized candidate hash, endpoint
hash, counts, safety; also all ordinary/cap/outer decisions and executed heads.
Independent replay must reexecute native AdamW, moments/counters, predictor,
selection, independent geometry/solver, EVERY first check including failures,
all continuation and committed endpoints. Solver failures of the first check
must leave solver state intact. Frozen BERT/cache/input and source hashes checked
before and after. Evidence create-only; failed results remain recorded.

Run synthetic numerical equivalence checks before training. Run baseline Stage1
and firstcheck once with complete logs; independently audit both budgets for
both arms. Baseline must also exactly match sealed Stage1 records. No requirement
that new heads equal baseline. Safety/state failure blocks benchmark; learning
gate failure is reported, not retuned. Existing smoke gate: zero floor violations,
>=20 unsafe steps, rollback <=10%, successful repair >=90%, new train macro >=50%,
final current CE <=90% of entry, triggered norm median >=0.1, old CE backward zero,
all state/moment checks pass. Report train Old/New/All and CE without test access.

Benchmark three fixed paired repetitions in orders [baseline,firstcheck],
[firstcheck,baseline], [baseline,firstcheck], both budgets for every arm.
Each replay must match its own validation trajectory and final state. Report
median and range of control/total algorithm wall, solver core, reconstruction,
materialization, endpoint, predictor, geometry. First-check inclusive time overlaps
reconstruction/materialization/endpoint and is not additive. Exclude log/hash
observer and independent audit from algorithm wall, as in sealed baseline.
Include all new candidate reconstruction, checks and instrumentation performed
inside controller in control wall. FLOPs retain existing dense/analytic estimates,
with eliminated duplicate Gram GEMV accounted separately; not hardware counters.
No result-driven reruns, policy promotion or automatic full experiment.
