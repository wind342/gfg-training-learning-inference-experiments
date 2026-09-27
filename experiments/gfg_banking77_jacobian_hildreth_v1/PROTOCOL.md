# Banking77 nonlinear V2: sequential Jacobian-Hildreth with trust ball

Engineering base aaf66dbfb8f28c890b2645f22992226becc9f273. New authority
E:/gfg-evidence/banking77-jacobian-hildreth-v1; no old source/evidence changes.
Original Banking77 task1/native/ER/RanPAC authorities are read-only and hashed.
This is a follow-up after prior Banking77 test results were observed, not a
claim that this benchmark has never been inspected. No new test access for
smoke or tuning. No scalar alpha grid, transition screening or warm-QP actuator.

## Fixed shared data/model/optimizer and budgets

Reuse verified Banking77 source57ec275d8078af65b7731c2a98be812d844a6d6b and
BERT86b5e0934494bd15c9632b12f734a8a67f723594 CLS768 cache. Seed2021, same
77-intent permutation,11x7 stream and10 reserved controls/intent. Remaining
train only for current CE. MLP768->256->GELU(exact)->77, both biased layers
plastic, dropout0/no LayerNorm. Batch16, AdamW lr5e-5, betas(.9,.999), eps1e-8,
weight_decay0, fused/foreach=False, clip1. Continuous moments and step count.
Same fixed210-batch schedule, repeated three times for630. Both budgets are
declared now; no budget choice after outcomes. Start task2 from respective
verified task1 endpoint/moments; task1 is common and immutable. If smoke passes,
continue the new task2 endpoints through task11 independently at210/630.
Reuse already sealed Native/ER/RanPAC baselines only after identity/hash checks;
smoke replays Native task2 independently. No new baseline tuning or RP variants.

## Unchanged outer V2 semantics

Strict FP32 margin>0 correctness. K fixed at task entry after seen-class
expansion; log expansion loss. Current CE/clip creates a real AdamW SHADOW
candidate and advances moments once. Actual buffer256 old endpoint first;
native-safe commits unchanged with no Jacobian/cutter. Otherwise select D[r]
wrong rows nearest zero per deficient relation, tie by stable original row ID.
Selected rows and violating pairs (actual FP32 target-minus-competitor<EPS)
accumulate only inside this update, reset next update. All currently active
pairs are relinearized together at each new candidate. Relation membership
replacement is allowed; no permanently protected individual samples.

EPS1e-6. Local problem: minimize .5||d-(native-current)||^2 with A d>=b and
||d||<=rho; b=EPS+reserve-actual_FP32_pair_margin. Each local correction is
materialized in FP32 and exact-evaluated. First relation-floor-safe endpoint
commits. Actual commit is independently rechecked. Numerical failure or caps
roll back to receiving head, retaining the once-advanced native moments.

## Derivatives and engineered solver (no old CE backward)

With u=W1 h+b1, q=GELU(u), s=(W2[y]-W2[c])*GELU'(u), the Jacobian blocks are
s h^T, s, (e_y-e_c) q^T, e_y-e_c. Use FP64 analytic factors and structured
Gram/reconstruction; no pair-by216653 dense Jacobian in production. Cache
immutable h, labels, IDs, relation indices, input Gram once per task. FP32
endpoint bank pads once to256 and reuses input/hidden/output buffers. Selected
row activation/derivative work only when native is unsafe. Geometry changes
each outer round: never reuse stale S/q/lengths. Full old CE backward=0;
Jacobian computation is NOT claimed free or derivative-free.

Local projection uses cyclic Hildreth halfspace updates plus Dykstra ball
projection with its own persistent correction inside that local solve. Keep
the displacement in span{native-current,active Jacobian rows}; reconstruct
parameters once per outer round. No post-hoc clipping of a solved QP. Reset
dual/ball corrections on relinearization. Maximum200 inner sweeps; accept
local solve only with halfspace residual<=1e-8, ball residual<=1e-10, full-cycle
step change<=1e-10+1e-8*rho. Otherwise record inner failure and roll back.
Numerical tests compare the small synthetic constrained objective to an
independent SciPy solver and compare structured rows/Gram/reconstruction to
autograd and explicit Jacobians. Infeasible/cap cases are recorded, not global
infeasibility certificates.

## Fixed trust and numerical guard, no search

rho_initial=2*||native-before||_2. Maximum8 outer rounds. After unsafe exact
recheck, compare summed positive active-margin deficits against EPS+reserve.
If actual repair / predicted repair <0.25 (or no positive predicted repair),
halve rho for the next round, lower bounded by rho_initial/16. Otherwise leave
rho unchanged. Never expand rho, change the native objective anchor or tune
after smoke. Record per-round and cumulative displacement, model errors, ball
hits, predicted/actual repair, inner residuals and all failure causes.

Reserve is an explicitly engineering FP32 guard, NOT a rigorous GELU-library
error certificate or a nonlinear remainder bound. For each selected row:
eu=gamma769*(|W1| |h|+|b1|), eq=1.13*eu+8*u32*(1+|u|+|q|),
ez=|W2|eq+gamma257*(|W2||q|+|b2|), u32=2^-24. Pair reserve=max(ez_y+ez_c,
abs(actual_FP32_pair_margin-FP64_pair_margin))+8*u32. The1.13 factor bounds the
smooth exact GELU derivative. The library-error cushion is heuristic, frozen
in advance; exact FP32 rechecks alone certify the finite-bank floor. Do not
carry the old linear roundoff formula or call local linearization globally safe.

## Prospective smoke gate

Run task2 Native and nonlinear V2 for both210/630 schedules. No test reader.
Gate per budget: all scheduled counters/moments correct, floor violations0,
at least20 unsafe native proposals, >=90% unsafe proposals repaired without
rollback, total rollback<=10% of scheduled updates, final current-train macro
>=0.50 and current-train CE <=90% of its task-entry CE. Also require median
executed/native update norm ratio>=0.10 on triggered updates (zeros on rollback).
All gates must pass BOTH budgets. Low exposure is INCONCLUSIVE, not PASS.
Fail/inconclusive stops tasks3-11; no rho/guard/cap/threshold repair after seeing
outcomes. Record old controls, current training and arrived-training cohort
metrics separately; these are not held-out test capabilities. Do not use
old-control accuracy as an unbiased generalization metric.

## Reporting and validation

Report native unsafe/safe, trigger/repair/rollback, selected controls/pairs,
Jacobian rows and calls, outer/inner sweeps, actual rechecks, replacement flips,
norm ratios/cosines, layerwise/cumulative corrections, trust hits, and costs.
Separate current backward, old CE backward(0), endpoint, Jacobian, Gram,
coordinates, reconstruction/materialization, control/algorithm wall. FLOP
coverage explicitly includes analytic derivative/Gram/reconstruction work;
dense arithmetic lower bounds and elementwise operation estimates are distinct.
Do not claim exact hardware Total FLOPs from an incomplete counter.
Safety replay task2 checks every executed head/moment/decision independently;
source/input/checkpoint/artifact hashes sealed. No simultaneous GPU workloads
during measurements. Full testing only after both new trajectories finish;
otherwise test remains unopened. Preserve all negative results.
