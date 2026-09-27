# Task2-only cap endpoint acceptance verification

Base 3e57e593f0d4af4098dd203c72af2191739225b2. New create-only authority
E:/gfg-evidence/banking77-v2-cap-recheck-smoke-v1. Preserve every prior file.
This is a disclosed verification after previous v1/v2 outcomes were known;
it is not a fresh hypothesis or previously unseen benchmark claim.

Exactly two new trajectories: task2, 210 and630 scheduled current-only AdamW
updates, starting from the SAME respective frozen task1 checkpoints as v1.
No Native arm, tasks3-11, ER, RanPAC, test reading/hashing/evaluation, tuning,
rho/tolerance sweep, solver replacement or result-driven repeat training.
An independent auditor replays both trajectories solely for validation.

Inherit v1 verbatim: Banking77 fixed source/cache, seed2021, class order and
controls10/intent, frozen BERT CLS768, MLP768->256->GELU->77 with biases/no
dropout/no LayerNorm, fused AdamW lr5e-5 beta(.9,.999),eps1e-8,wd0,clip1,
batch16, PCG64(4023) task2 schedule repeated3 for630, continuous moments.
K is strict FP32 count at expanded task2 entry, fixed throughout task2.
Keep nearest-boundary selection, analytic Geometry, reserve,EPS1e-6,buffer256,
all caches/settings, Hildreth/Dykstra implementation and 200 sweep cap.
Keep outer cap8, rho_initial=2*||native-before||, shrink0.5 when repair<0.25
or predicted repair<=0, rho lower bound initial/16. Do not change any solver
PASS tolerance: halfspace1e-8,ball1e-10,cycle1e-10+1e-8*rho.

ONLY algorithm modification: INNER_LIMIT does not immediately roll back.
Reconstruct/materialize via the same code as PASS, then exact old endpoint.
Record INNER_LIMIT_ACTUAL_SAFE_ACCEPT for actual floor-safe first-stop;
INNER_LIMIT_ACTUAL_UNSAFE_CONTINUE for unsafe and outer<8; at unsafe outer8
record OUTER_LIMIT_ROLLBACK. Also record ordinary PASS safe/continue states.
Actual finite-bank C>=K is acceptance authority. Preserve v1 trust checks;
nonfinite numerical outputs are implementation failure, not floor relaxation.
Accepted cap iterates are not claimed exact local QP optima.

Only current examples enter CE/backward/clip. Native shadow advances AdamW
once; cutter never touches moments. Check moments before/after cutter and
every final scheduled counter. Actual committed heads, including rollback,
are independently endpoint checked. Input batch IDs must exclude controls
and must all belong to task2. Geometry uses NumPy, no old CE/autograd.

Freeze before running: exposure>=20 unsafe proposals; repair>=90% of unsafe;
rollback<=10% of scheduled; violations0; new train macro>=50%; final current
CE<=90% entry; median executed/native norm>=0.1 on triggered steps including
rollback zeros; old CE backward0; all input/state/continuity audits PASS.
Low exposure is INCONCLUSIVE. Stop after both budgets, regardless of gate.
No path from this runner to formal training or test evaluation.

INNER_LIMIT is counted per local solve/outer round, not scheduled step.
cap-safe acceptance = cap-safe accepts / INNER_LIMIT events. Cap events
partition into safe accepts, unsafe continues and cap-unsafe outer rollbacks.
Also report ordinary safe accepts, native safe, all rollbacks, all exact
calls split entry/native/projected/commit, Jacobian calls/pairs, inner/outer
iterations, norm/cosine, correctness flips, train old/new/all macro/current CE,
control/algorithm wall, endpoint/cutter FLOP coverage and old CE backward0.
Train metrics exclude controls and are not held-out test capabilities.

Audit is a separate implementation: in-place AdamW, independent FP32 endpoint
counts, selection, local geometry/solver replay, rounded candidate rebuilding,
decision and trust transitions. It must not call production project, select,
Geometry or reconstruction. For every cap, additionally check explicit dense
Jacobian reconstruction in the auditor, never in production. Trace coef/dual,
current/candidate/endpoint hashes allow every continuation to be checked.
Production and audit are separate processes; audit overhead excluded from
training timing. Evidence instrumentation is disclosed, no speedup claim.

Python input-access guard allows only TRAIN_FEATURES,CACHE,numerical cost
metadata, the two task1 checkpoints, two v1 task2 CHECKS, frozen BERT files,
and v1 source/protocol. No recursive hashing of old evidence. Test data and
test metrics are never opened, even for hashes. Verify allowed-input hashes
before/after and tracked old sources via git diff. Existing v1/v2 are untouched.
