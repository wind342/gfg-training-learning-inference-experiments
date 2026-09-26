# Fixed numerical-roundoff repair, one new Hildreth arm

Base 405702d51b0fff8f281e59c65b8fb2fbd273ce7c. New independent branch/directory;
arm 140_controls_hildreth_roundoff. Do not change or rerun old experiments, main,
frozen tags or historical results. One formal run only. No test-driven changes.

All parent Hildreth settings are retained: FewRel task2, seed2021, LR5e-5, batch16,
210 scheduled updates, same task1 checkpoint/head/AdamW moments, clip1, same current
batches and RNG schedule, frozen/eval BERT, same exact1200 cached features including
1120 old validation controls. No replay, Router, scalar shrink, new model, LR sweep
or extra seed. Moments advance once even on skipped parameter execution.

## The sole changed mechanism: an internal numerical reserve

Actual acceptance is UNCHANGED: native-safe means every receiving-correct old
control remains strictly correct. Such proposals are committed byte-for-byte.
After triggering Hildreth, the actual materialized FP32 head must satisfy every
eligible target/competitor gap>=1e-6. Same maximum100 sweeps, active-pair discovery
at actual FP32 gap<1e-6, persistent dual coefficients, deterministic order and
first-safe stopping. Still skip after100 unsafe sweeps. No tolerance in acceptance.

Only the dual-coordinate RHS changes from eps to eps+eta(i,t,c). Compute once per
scheduled update from cached augmented features H=[h,1] and the receiving/native
augmented heads B,N (16 seen rows):

u = np.finfo(float32).eps/2 = 2^-24
k = H.shape[1]+2 =1539 for this head
gamma = k*u/(1-k*u)
E = gamma * (abs(H) @ maximum(abs(B),abs(N)).T)
eta(i,t,c) = E[i,t]+E[i,c]

Hold E fixed for all sweeps in this update. Use the identical Hildreth update
lambda_new=max(0,lambda+(eps+eta-gap)/norm2), then apply the dual difference to
the two parameter rows. No fitted multiplier, hand-selected margin, or sweep.
The two additional operations reserve space for parameter casting and margin
subtraction beyond the augmented inner product. FP64 dual/primal arithmetic and
FP32 final materialization are unchanged. State-based numerical reserves can
vary across updates/pairs; this is a frozen deterministic rule, not parameter tuning.

The scale estimate uses the receiving/native envelope, not an a priori bound on
all later projected parameters. Therefore it is a conservative roundoff-inspired
estimate, NOT a universal certified GPU error bound or proof that every projection
will pass. Actual FP32 recheck remains authoritative. Standard gamma_n analysis:
https://nhigham.com/wp-content/uploads/2023/10/high97t.pdf (matrix-vector rounding
error). This experiment-specific reserve is not claimed to be a theorem from that
reference. It can be conservative and may alter the retention/plasticity tradeoff.

## Before formal training

Run unit tests for the unchanged head/optimizer invariants and Hildreth logic,
including dual release, dynamic constraints,100-sweep skip, strict epsilon handling,
numerical-rule formula/scaling and positive internal padding. Perform a small
numerical audit using exactly the first, middle and last failed projection calls
from the historical Hildreth ledger, in that order. Reuse their saved receiving
and native heads and old control features; no model training or test inputs.
Apply the ONE predetermined rule above, checking observed writeback/forward error
against the computed pair budget. No searching alternative constants. Record
NUMERICAL_CHECK.json even if it fails; only a passing numerical check authorizes
formal training. Preserve source/artifact hashes before formal execution.

## Reporting and integrity

Keep the parent's requested metrics: native-safe/triggered, active counts, sweeps,
safe/failed, skips, old protected correct-to-wrong, norm ratio/cosine, projection
wall; final protected correctness, full task2 train16 accuracy and old/new/all
primary test macro. Test only after the complete sole training result is hashed.
Compare against historical Hildreth and A without rerunning them. All failures
and overhead retained. A successful result is not a global optimum claim.

Per-update evidence additionally records the fixed numerical-budget hash and
active pair reserves, allowing independent recreation of the padded constraints.
Independent checker replays every sweep without calling production Hildreth or
the budget helper, verifies parameter execution and source/record identities.
Old evidence, main and frozen tag stay unchanged. Publish code and generated
records only; original data/features/weights remain local.
