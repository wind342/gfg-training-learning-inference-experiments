# Safe screening v3: engineering evaluation, task10 only

Base 2f5e45beb5bdd2b4031ddb8461d56beb34f0ee3f. Old experiments/results,
main and frozen release tag remain untouched. Separate create-only authority:
E:/gfg-evidence/final-v2-safe-screening-v3. No Banking77.

Same cached FewRel task9 V2-210 checkpoint, task10 order, seed2021, LR5e-5,
210 updates, inherited fused AdamW moments, frozen BERT, 10072 old controls,
fixed task-entry K. No old loss/backward, replay, model or protection change.

Reuse v2 fixed 256-row FP32 affine numerical authority. Keep exact original
Hildreth coordinates/reserve/100 sweeps/first-safe stopping/selection tie breaks.
Screen stores correct-count LOWER BOUNDS for safe relations, never claims these
are complete counts. Deficient relation counts and wrong-margin ordering must
be exact. Existing selected constraints are evaluated regardless of floor.

Replace relation-specific full-head anchors by one previous candidate head and
cumulative class-row displacement norm paths. Triangle inequality bounds
target-competitor changes using target displacement plus maximum displacement.
No groups x classes x hidden tensor, no FP64 feature/head matrix product.
Per-row exact margins refresh their own cache at no extra forward cost; no
periodic or 25%-frontier relation refresh. Conservative roundoff bounds retain
v1 gamma assumptions and FP32 underflow allowances; cumulative transport adds
FP64 addition/subtraction allowances. Zero false-safe is empirical verification
under that error model, not a formal GPU hardware proof.

Per candidate: count certified controls; batch up to K minus certified count
unproven rows per relation, plus previously selected rows. Stop safe relations.
Fully resolve remaining uncertain rows only in still unresolved relations.
No per-row GPU forward or per-relation CPU loop in certificate/floor screening.

Preflight: all-positive/all-negative certificate paths, direct exact affine
comparisons, full fixed-reference first-update identity, cumulative candidate
path changes and return transitions. Run only after tests PASS.

Every formal candidate has an independent read-only full-bank observer. Its
values never feed cache, selection, K or optimizer. Full-reference replay of
every scheduled update verifies executed head, moments, selection, active pairs,
primal/dual sweeps. Full observer records actual counts separately from bounds.
Test only after training seals. Preserve failures; no automatic scientific retry.

Record useful exact rows, actual padded rows/GEMM calls, cache-refresh rows
(overlap with exact rows explicitly), additional refresh forward rows, FP64
class vector work, wall time, certificate/forward/cache/mask CUDA timeline,
false-safe, floor violations, optimizer/head identities. GPU timeline intervals
can include host launch gaps and are NOT pure SM arithmetic time. Algorithm
wall excludes independent observers/serialization; include all screening work.
Fresh paired reference measurement, plus historical v2 time explicitly marked
noncontemporaneous. Separate standalone timed replays must agree with verified
heads before performance claims. Profiling may diagnose but never alter policy.
