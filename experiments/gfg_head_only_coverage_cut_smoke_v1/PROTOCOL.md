# Frozen coverage x cut mechanistic smoke

Base: df98982d476b57976872d9f6fb21b6a62404a4a3.
Historical 10/current is referenced, never retrained or overwritten. Three new
arms, in fixed order: A=140/current, B=10/tight, C=140/tight. No other arms,
reruns, sweep, additional seeds, replay, Router, scalar shrink, or BERT updates.

FewRel task2, seed2021, LR5e-5, batch16,210 updates, same original task1 head and
AdamW moments, same current batch order, clipping1, optimizer and RNG schedule as
the parent. BERT remains frozen/eval. Moments advance once even on skipped
parameter updates. Test evaluation begins only after ALL three arms complete;
completion hashes are frozen in TRAINING_COMPLETE.json. No test-driven changes.

The 10 bank is the original first10 validation examples per old relation. The
140 bank is every validation example of each of the8 old relations, in ascending
relation/example order. No test samples used for protection. All1120 encoded
inputs must be unique and disjoint from all training and the existing primary
test set. Historical160 cached features are reused exactly; the1040 additional
old validation features are computed once using the same frozen BERT, batches16.
These labels/features are active control data, not held-out validation outcomes.

Eligibility is receiving old margin>0, evaluated afresh each update. Violation
is candidate margin<=0 (including ties). All16 competitors and head bias count.
A calls the parent's current-cut function unchanged, except old_count=1120.
B/C enumerate violating pairs at the start of each pass, ordered by sample then
competitor. Before processing EACH pair, reconstruct the current FP32 candidate
head and recalculate its two logits with a single-example affine forward.
If current pair margin m<=0, apply delta += ((1e-6-m)/norm2)*g, where g has +h,+1
in target and -h,-1 in competitor; norm2=2*(h.h+1), float64 accumulation.
If already positive, skip the pair. Store corrections in original FP32 dtype.
This is a single-constraint boundary correction, NOT a joint minimum-distance
solution. Single-example and batched FP32 affine rounding can differ: only the
actual full-bank FP32 exact recheck authorizes execution. Do not enlarge epsilon.

After each pass reconstruct receiving+delta and check every eligible old example.
Max2 passes; enumerate fresh pairs on pass2. If still unsafe, skip parameters,
record CUT_FAILED_SKIP; never increase the pass limit. Safe native proposals are
committed byte-for-byte. Later corrections can invalidate earlier constraints;
skip saturation or float32 boundary failures are valid outcomes, not grounds to
rerun. No extra safety margin, solver, QP, or hidden threshold.

Public summary: old/new/all test macro accuracy; final old protected correct/total;
projection steps; native violating sample-update occurrences; executed old
correct-to-wrong; skips; mean norm ratio, cosine, and relative vector displacement
on projection-triggered steps; training wall. Attempted-cut and actual-executed
geometry are distinguished because skipped updates have zero executed norm and
undefined cosine. Raw step certificates and immutable head snapshots support
independent recomputation. Initially incorrect protection samples are not forced
correct, but become eligible if later learned. 140/current vs historical isolates
coverage;10/tight vs historical compares the combined tighter cut and within-pass
recheck change, not either change alone. No universal/generalization claims.

Preflight tests analytic gradient, single-constraint formula, safe-pair recheck,
ties, safe-native identity, two-pass skip, unchanged parent/current behavior,
data isolation, baseline source/data/head/optimizer identity. Independent checker
reconstructs each native/cut/executed endpoint from saved states, recomputes the
projection without calling either production cut, checks all masks, pass limits,
coefficients, geometry, current batches and test counts. Formal wall cap2h/arm.
Failures are retained; no automatic rerun. Original evidence, main and frozen tag
remain unchanged. Local tensors and original data are not published.
