# Final V2 safe screening: task10 only

Parent 41ca46bef060815d2f106f8235764fb8b03e67b7. Phase A only; no
Banking77 execution or new scientific controller. Old sources/artifacts, main,
and frozen tags remain unchanged. Outputs are create-only.

## Fixed execution

Start from the verified cached-comparison v2_210 task9 CHECKPOINT.pt. Task10,
210 updates, seed2021, inherited head/moments, current batch order, LR5e-5,
clip1, fused AdamW, frozen cached BERT, 80 seen classes, 10072 old controls,
and original fixed task-entry K. Production Hildreth coordinate, reserve,
materialization, deficit selection, 100-sweep cap, first-floor-safe stopping,
and moment advancement are unchanged. No test-based decisions or retries.

## Certificate and numerical contract

For augmented x=[h,1], q is an upper bound on ||x||. Per relation anchor A
stores a lower bound on the real minimum target-competitor gap, not an assertion
that FP64 is exact. For actual materialized FP32 candidate V, use
min_gap(V) >= anchor_lower - q * max_c ||(V_y-A_y)-(V_c-A_c)||.

All certificate arithmetic is FP64. Set k=8*d+64 (d includes bias),
u_p=eps_p/2, gamma_p=k*u_p/(1-k*u_p). This deliberately overcounts scalar
products, sums, sqrt, bias, margin subtraction, and certificate assembly.
Norm upper bounds divide computed norms by (1-gamma64), rounded upward.
Anchor gap subtracts gamma64*q*(||A_y||+max_c||A_c||), with an additional
gamma64 assembly allowance. D/subtraction norm error is bounded by
2*gamma64*(max||V_c||+max||A_c||), added to the computed norm bound.
FP32 forward reserve is gamma32*q*(||V_y||+max_c||V_c||), plus
k*tiny32*(1+q+||V_y||+max||V_c||) to cover flushed-underflow terms.
An additional gamma64*(1+absolute magnitudes of certificate terms) is
subtracted when assembling the lower bound; nextafter rounds down.
Only a strictly positive lower bound certifies correctness.

Contract: IEEE FP32/FP64, TF32/autocast disabled; finite input/candidate
entries bounded by 1e10 to exclude overflow in this fixed dimension. This
is a floating-point error-model certificate, not formal verification of CUDA
hardware/library implementation. The full independent observer tests its
actual implementation. Unsupported/nonfinite certificates never certify.
Original Hildreth before/native padding remains unchanged and is NOT used as
the screening proof for projected candidates.

Every uncertified old row and every audit-only new row gets actual FP32
logits. All previously selected Hildreth rows always remain in the evaluated
set, even if certified positive: production failing_pairs uses 1e-6, not zero.
Correctness of certified rows supplies only a Boolean, never a fabricated
margin. All wrong rows are evaluated; choose() therefore receives exact
wrong-row margins and original dataset-ID tie breaks. Re-screen each new
projected candidate. No per-sample locking is introduced.

## Anchor refresh

Initialize every relation at task10 receiving state. At subsequent scheduled
step boundaries only, refresh a relation at the actual receiving head if its
previous executed endpoint's certificate-frontier fraction exceeded 25%, or
its anchor age is >=100 scheduled updates. Refresh once per relation per step,
never inside Hildreth. Union refreshed rows in one FP64 forward, update the
associated per-relation anchor heads. Count all refresh/setup work and rows.
No threshold tuning, sorting optimisation, or alternate certificate after
seeing formal results. Failed/inefficient screening is an admissible result.

## Observer and reference

For EVERY receiving/native/projected/commit screening call, a separate full
original-shape FP32 observer runs after controller output is materialized.
It checks all certified signs, all row signs, relation counts, and wrong-row
margin ordering. It never supplies replacements for controller logits or
decisions. Record float/ordering mismatches. Abort on false-safe or count
mismatch (preserve failure); do not silently repair or tune. Sparse GEMM may
not reproduce original FP32 reductions byte-for-byte.

For each scheduled update, call unchanged production floor_projection on
the same receiving/native pair as a read-only paired reference. Compare
deficits, every selection, active pairs, dual/primal sweep identities, skip,
executed head and counts. Continue and report benign numeric/trajectory
mismatches, but never claim historical equivalence if not achieved.
Also compare each executed head/moment against historical task10 logs.
Final test evaluation only after 210 scheduled updates, compare historical
predictions. On early abort do NOT report a completed test result.

## Timing and records

CUDA-synchronised screened algorithm time includes screening, full refreshes,
normal training/proposal, Hildreth, required transfers, and commit check.
Independent full observers, hashes/serialization, and production reference
calls are timed separately. Paired current-machine reference time uses the
same current update time plus its unmodified projection and actual recheck.
These interleaved timings are diagnostic, not a standalone throughput claim.
Report original candidate forward rows, frontier exact rows, FP64 anchor
refresh rows separately; none is called FLOPs or hardware traffic.

Save source/input hashes, compact per-call and per-step records, all projection
selections and mismatch records, small final head/moments, and predictions.
No duplicate BERT/features, no per-step model checkpoint bank. Numerical unit
preflight before formal source freeze; no automatic formal retry. Phase B is
not authorised by a merely completed run: report Phase A first.
