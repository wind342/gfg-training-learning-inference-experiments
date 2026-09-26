# One-arm Hildreth head-only smoke

Base 9e0dc71c7692fa16ebeb81dbf73197aad93add4b. Only new arm:
140_controls_hildreth. Historical A_140_current is referenced without rerunning.
No old files/results, main or paper freeze tag changes. No second arm, sweep,
extra seed, replay, Router, BERT update, scalar shrink, or post-result rerun.

FewRel task2, seed2021, LR5e-5, 210 scheduled updates, batch16, original task1
checkpoint/head/AdamW moments, clip1, same current batches and per-step RNG seeds.
BERT frozen/eval. The EXACT A cache and ordered IDs are reused: 1120 old validation
controls (140/relation), plus the unchanged 80 new controls for audit only.
No old-example training loss. Moments advance once per scheduled update, even
when parameter execution is skipped. Projection never modifies moments or RNG.

Eligibility is receiving FP32 margin>0. If all eligible old native endpoints are
strictly positive, commit native parameters byte-for-byte, as in A. This native
fast path does not impose epsilon. Otherwise solve the head Euclidean projection
toward the original native candidate subject to every eligible old target versus
each of 15 seen competitors having endpoint gap>=1e-6, including head bias.

Hildreth uses FP64 CPU sparse two-row arithmetic over augmented cached features
[h,1], not full flattened constraint vectors or an external solver. Let P start
at the native FP32 head cast to FP64 (16 seen rows); each active pair has lambda=0.
For pair i,t,c: gap=dot(H_i,P_t-P_c), norm2=2*sum(H_i**2),
lambda_new=max(0,lambda+(1e-6-gap)/norm2), a=lambda_new-lambda,
P_t+=a*H_i, P_c-=a*H_i. Keep lambda across sweeps, including its decreases.
This preserves the native origin and dual corrections; never restart a sweep
from the native proposal and never replace Hildreth with one-sided clipping.

Initial active pairs are all eligible pairs with native FP32 gap<1e-6. Process
active pairs in deterministic (sample index,competitor) order. After each complete
sweep, cast P to original FP32 head dtype, preserve unseen native rows unchanged,
and perform a full-bank native FP32 affine check. Newly failing pairs are added
with lambda0 and processed starting next sweep; existing lambdas are not reset.
Finite native FP32 pair gaps, cast to FP64 for comparison with the literal 1e-6,
must all be>=1e-6 to accept. Do not silently replace this with >0 or a tolerance.

Stop at the first safe full-bank endpoint, or at 100 complete sweeps. If still
unsafe, skip parameters and record HILDRETH_FAILED_SKIP. No epsilon inflation,
rounding repair, relaxation, increased budget, solver substitution, or scalar
fallback. The FP64 optimization iterate is retained between sweeps; it is not
rebased to rounded FP32 parameters. Thus FP64 affine convergence and FP32 endpoint
safety can differ. Record both residual types, retaining finite-precision failures
as failures. Reaching the sweep cap is not evidence of infeasibility. Early-safe
termination establishes feasibility, not global projection optimality.

Record native-safe/triggered, initial/final active counts, sweep count and summary
per sweep, final safe/failed, skip, executed old correct-to-wrong, attempted and
executed norm ratio/cosine, projection wall. Minimal audit records additionally
include pair IDs/final duals, native/receiving/projected/executed margin vectors,
FP64 head and dual SHA-256 at each sweep, and fixed source/data/state identities.
Local full head histories enable independent reproduction of every solver call.
No original text, token arrays, BERT weights or feature tensors in public records.

Final metrics: old protected correct/1120, task2 full train16 accuracy, primary
old/new/all test macro accuracy. Test is accessed by model only after the sole
training result is complete and its hash is fixed. Same primary test IDs as A.
No test-based changes. All formal failures are retained; no automatic retries.

Preflight tests include analytic sparse gradient/bias, dual updates and release,
a small analytic closest-point oracle, native-safe identity, epsilon and tie
handling, dynamic active addition, dtype/rounding rejection, 100-sweep cap/skip,
input/RNG immutability, inherited optimizer and frozen backbone, bank isolation.
Checker independently reconstructs all sweeps from saved native heads without
calling production Hildreth, verifies actual head execution and all reported
metrics, and validates original baseline/source hashes. One commit on the new
branch after integrity checks, then push that branch only.
