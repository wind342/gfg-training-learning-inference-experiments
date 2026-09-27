# Task2 offline first-actual-safe sweep diagnosis

Base f0b6aaf39bd4525ef958d9c4519045f1facc1c7a, accepted Stage1 A/B/C trajectory.
Independent directory/branch and create-only authority. Task2 210/630 only, from the
same frozen task1 checkpoints. Never access test or change any historical evidence.

Observe every completed local Hildreth/Dykstra sweep, after coordinate/ball corrections
and before the unchanged convergence test. Reconstruct that sweep's candidate using
the same current geometry, coef/dual, FP32 materialization and buffer256 endpoint.
Count all old controls against fixed K. Continue the solver even if the observer finds
safe: only the original stopping condition and original outer checks control training.
Record first safe sweep, full safe/unsafe sequence, reversions, original stopping sweep,
cap status and original endpoint. Do not test a new early-stop training trajectory.

Native AdamW proposal, moments, predictor components, selection, geometry, coef/duals,
all final solver info/decisions, original candidates and executed heads must match sealed
Stage1 at every update. Callback may not mutate solver inputs/state. Independently rebuild
and forward first-safe candidates and each original terminal candidate, using the
existing independent analytic factors/reconstruction/endpoint implementation. Recheck
all committed heads independently. Preserve hashes and original checkpoint identity.

Time separately: same-state plain solver (no callback), observer solver core, candidate
reconstruction, FP32 materialization/transfers, endpoint+counts. Observer hashes/logging
and independent checks are excluded from candidate-check costs. Interleaving perturbs
cache/timing, so use plain solver total to normalize the measured per-sweep core profile.
Report solver-only savings as an estimate on these fixed update states, not a measured
speedup of a new controller. Include full candidate-check cost, not only GPU forward.

For each fixed original update, consider the first-safe point anywhere on its original
outer path. Sum the prefix sweeps/check costs; compare with original full solves and
their one check per outer. Report estimated solver saving minus additional check cost,
with skipped outer geometry/diagnostic work explicitly not credited. Also report the
check-cost vs entire plain solver budget, distributions and per-budget results. No
selection of sweep-check interval, residual trigger, rho/tolerance or other policy.

A candidate may be safe early and unsafe later; record this rather than assuming safety
is monotone. Report how often the early candidate differs from the original executed
head. Any early commit could change subsequent updates, so fixed-path counterfactuals
cannot establish 11-task learning, later floors or an end-to-end training speedup.

Freeze code before diagnostic replay. Stop and report after both task2 arms. No training
extension, tasks3-11, ER/RanPAC, test, controller implementation change or promotion.
