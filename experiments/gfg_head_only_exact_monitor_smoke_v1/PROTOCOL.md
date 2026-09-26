# Frozen head-only exact-monitor smoke

Base: e84014c2417e631025e9a8dc1b9a69d23997856c.
Only this new directory may change. Original experiments, data, checkpoints,
Router, main and the frozen paper tag remain unchanged. Freeze source hashes in
PREFLIGHT.json before either formal arm. No tuning, reruns, replay or interventions.

## Question and design
Run head_only and head_only_exact_monitor once each, task2 only, seed2021,
shared task1 checkpoint, seen16, 210 updates, batch16, LR5e-6, inherited AdamW
head moments/step, betas(.9,.999), weight_decay0, fused=True, clip_norm1.
Use exactly the inherited task2 batch order and authorized data isolation.
Historical full-FT native results are external references, never rerun.

BERT parameters are frozen, gradients cleared and always eval, including after
model.train(). Current batches get a fresh no_grad BERT feature forward every
step. No current-training feature cache. Only fixed160 control features are
cached, with BERT eval and no_grad, once per arm before training.

Both arms use the same candidate path: a head-only shadow Linear plus native
AdamW initialized from the exact checkpoint head states. Copy current head and
gradients to shadow, advance shadow optimizer once, then copy its endpoint to
the real head. The monitor is read-only, before that commit, always alpha1.
No BertInstrument, JVP, propagation, projection, shrink or rejection.
Only the monitor arm computes baseline/candidate control logits each step with
F.linear and cached features; original class scope16 is applied before margins.
All competing16 logits participate. Exact refers to actual affine evaluation,
not symbolic real-arithmetic exactness or a generalization guarantee.

## Observation definitions
At each monitor step, full-refresh the inherited GroupSentinel on receiving
control margins (step recorded correctly), then snapshot state/anchor. Use this
one fixed receiving anchor for BOTH baseline and candidate safety:
capability>=.9 AND (anchor<1e-6 OR q10>=.5*anchor).
q10 uses numpy quantile(method=linear), correctness is margin>0; argmax accuracy
is reported separately on held-out tests. Report already-unsafe and newly-unsafe
for ALL8 old groups, plus the receiving-PROTECTED subset. Do not omit DEGRADED.
After guaranteed commit, observe endpoint with source=endpoint_observation
(no extra full_refresh anchor reset). Inherited .75 rearm rules remain.
Per-step full_refresh may track gradual erosion; save checkpoint anchors and
all margins so cumulative loss is not mistaken for absence of newly-unsafe.
Counts of newly-unsafe updates describe this uncontrolled trajectory, NOT the
number of interventions a future closed-loop policy would require.

## Evaluation and integrity
Before training, evaluate old test/control at seen8 and again at seen16.
Store per-relation metrics and native predictions, separate scope expansion
from later head interference. Final evaluate original2240 task2 test examples;
authorized primary has2239. Full80-task isolation uses existing seven exclusions
and control50124->50130, never changes source data. Test never controls training.
Compare final head parameters, head moments/step and predictions byte-for-byte
between arms. Save only head/optimizer final states, referencing immutable base
checkpoint for BERT; do not duplicate a full frozen BERT checkpoint.

Preflight tests: frozen flags, eval persistence, no BERT grad/update, head grad/
update, head moments exact migration, shadow/native optimizer equivalence,
cached feature immutability, exact-head endpoint equivalence, monitor parameter/
moment/RNG immutability, full safety transition semantics and pairwise arm
identity on a small real BERT. Formal checker additionally checks210 shared
batches, all data/source hashes, BERT identity, final states and native test replay.
Independent record checker recalculates controls from saved small head snapshots
and cached H; it is not an independent full retraining.

## Costs and reporting
CUDA synchronize timing boundaries. Separately time current BERT forward, head
forward/loss, head backward/clip, proposal, commit, cached endpoint calculation,
capability/q10/state statistics, integrity bookkeeping and JSON I/O. Include
control feature-cache setup separately and in all-in wall, not monitor kernel.
Report monitor per-step cost, update-time ratio, peak allocated/reserved after
setup and process lifetime peaks. Hashes and native endpoint audit are separate.
One serial run per arm; no statistically precise speedup claim from two timings.
2h safety cap per arm. All failures are retained. No threshold of success triggers
new work in this experiment. Report results even if new capability is poor.

Head-only versus full-FT changes optimization trajectory and dropout; differences
are not an additive causal estimate of backbone-change contribution. A low score
only applies to this frozen representation, inherited optimizer, LR and budget.
