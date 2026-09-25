# Fixed shrinkage versus state-conditioned amplitude control

Pre-outcome protocol, base ab10b96129c3139d3fa4c50532a474aebfeeeca9.
Only this new directory and a new external evidence directory may be written.
No old evidence, main or frozen tag changes. Frozen release commit remains
36dab5ce347dbbdac157ef23205f556606d18294.

## Scientific comparison

Reuse all 12 historical starts, ordered batches, RNG identities and 212 labeled
control/evaluation targets. Fresh fused AdamW (foreach=False), betas=(.9,.98),
eps=1e-8, weight_decay=0, clip norm=1, deterministic CUDA FP32, TF32 off.
LRs .001/.003/.01 each use multiplier .1+.9*.5*(1+cos(pi*(t-1)/199)).
Stop at first strict-gap accuracy >=.99 or 200 updates. First .95 is secondary.
Exclude initially-at-threshold cases separately; failure costs 200, retain reach
counts to distinguish failure from attainment at step 200.

Five policies per LR/start: fixed alpha 1/.75/.5/.25 and unchanged ab10b96 dynamic
controller. Dynamic retains quadratic_complete, all targets/competitors, four
alphas, cadence5, cached intervening decisions and identity-step shortcut.
Static uses native AdamW followed by before+alpha*(proposal-before); alpha1
retains native bytes. Moments follow native recurrence, not alpha scaling;
numerical moments need not agree across diverging trajectories. Fixed scaling
is equivalent to scaled LR in exact arithmetic for this zero-decay optimizer;
the specified operation order is retained for floating-point causal matching.

For each held-out history and LR, rank fixed alphas using only other histories:
mean restricted first99, mean restricted first95, then larger alpha. Exclude
initial zeros. This selects among FOUR specified constants, not a global best
learning rate. No held-out future outputs select its constant. Histories are
held out for selection; targets are NOT held out from the controller. This is
a follow-up on an already observed witness family, not new independent data.

## Prespecified verdict rules

For each LR compare dynamic with LOO-fixed on the 99% eligible histories.
Define paired gain = fixed restricted steps - dynamic restricted steps.
A regime supports extra value only if: dynamic reach >= fixed reach; mean
gain >0; at least TWO histories have positive gain; AND removal of ANY one
history leaves reach >= fixed and mean gain >0. This is a descriptive robustness
criterion, not a population-level significance claim. Report all wins/losses/ties.
Aggressive support label requires this criterion in .003 OR .01. Always report
the other regime; conflicts are explicitly qualified. Aggressive-specific role
is consistent only when aggressive support holds and .001 does not meet it.
If no aggressive support and fixed has >= reach and <= mean cost in BOTH
aggressive regimes, label FIXED SHRINKAGE EXPLAINS MOST/ALL BENEFIT, unless
dynamic is likewise weakly dominated by both native and fixed at all three LRs
(with some strict deficit), in which case label NO BENEFIT. Otherwise MIXED /
HISTORY-DEPENDENT. Labels concern this comparison, not the underlying theory.
Matching fixed does not prove the dynamic mechanism absent; beating constants
does not isolate state feedback from every possible time-dependent schedule.

## Execution, timing and evidence

180 search trajectories (12*3*5). Freeze LOO selection, then 324 timing replays
(12*3*3 arms*3 repeats), all three LRs. Native and selected fixed are independently
timed even when both alpha1. Rotate arm order by history/LR/repeat. Stop at
capability attainment, never train past .99. Search costs reported separately.
Per-history median of three times then mean over threshold-eligible histories.
Compute includes construction and synchronized updates; checked wall includes
evaluation, coordinate checker and parameter hash audit. Exclude input loading,
reference loading, warmup and final artifact writing. Prior search warms CUDA.
No reuse of earlier implementation speedup as an algorithmic benefit.

All artifacts written exclusively, refuse existing evidence directory. Seal new
protocol/code/checker plus all baseline tracked Python sources, old assignment,
dependency seal, environment and frozen tag before any scientific outcomes.
Software fixes retain failed seal/output and require a new seal/output location.
Native exact replay checks logits, parameters, optimizer, seeds, LR, decisions,
coordinates/scores, lengths and ledger chain. Independent offline audit reopens
raw logits/coordinates, independently recomputes endpoints, selection, statistics,
timing medians and verdict, verifies source/old-file identities and manifests.
Hash identities alone do not reconstruct parameter states: actual native timing
re-execution supplies the post-state and final-optimizer verification.

Unit check fixed policy against separate direct displacement arithmetic and
optimizer-state equality for a small synthetic model, for all four alphas.
Bridge native/dynamic search facts to prior ab10b96 trajectories, stripping only
the new native alpha annotation; require exact final identity and logits.
Report pooled and per-history dynamic intervention/decision rates and alpha counts.
