# Frozen update-direction isolation smoke

Base f35b595c4e4582bd3a52e938b29e2cf253446d25. New directory only; no prior
source, evidence, tag, optimizer or Router V4 policy changes. Three one-shot arms:
native, scalar, projection; seed2021; original shared task1 checkpoint; task2 only;
210 updates, batch16, LR5e-6, AdamW betas(.9,.999), weight_decay0, fused=True,
clip1. Same fixed batches and per-update dropout seeds as the ten-task harness.
No old-example replay loss. Old control probes ARE optimization-control data:
projection uses their labeled margin gradients; this is not old-data-free learning.
Use the previous authorized isolation and primary/full-test definitions unchanged.

Detection offsets are exactly1,11,...,201 (21 sites); native has no detections.
Scalar/projection use identical sites; other updates are ordinary AdamW alpha1.
Test evaluation only at task2 endpoint. No sweep, tuning, result-selected rerun or
extra seed. Cap6h/arm; preserve any failure and partial records; do not retry.

At each detection form the native AdamW proposal in that arm's receiving state.
Across diverged arms proposals need not be numerically identical. Moments advance
once when the proposal is formed and are never changed by projection/scaling.
The inherited group sentinel is advanced consecutively, but its adaptive trigger
is NOT used. Refresh its history with the measured receiving margins, then freeze
the receiving PROTECTED set among groups0..7 and its anchors for all candidate
comparisons. Initial checkpoint states0/4 are DEGRADED. Record every pre-refresh
state, baseline, refreshed state and post-update state. Losses of protection
between detections and at8->16 class expansion are explicitly recorded separately
from damage caused by measured/committed updates. Existing DEGRADED/FORMING groups
are reported, not given a new protection rule. State updates reuse the existing
GroupSentinel rules, threshold.9, erosion.5, recovery.75 and floor1e-6 unchanged.

Threat: candidate capability<.9, or receiving anchor>=floor and candidate
q10<erosion_ratio*anchor. Evaluate ALL receiving PROTECTED old groups, including
ones not originally threatened, for safety after projection/scaling. Candidate
predictions never update the persistent sentinel. Choose largest safe alpha from
1,.5,.25,.125; if none, execute.125 and label NO_SAFE_SCALAR_ALPHA or
NO_SAFE_PROJECTED_ALPHA. Candidate/anchor comparisons are frozen per detection.

For projection, sort originally threatened groups by native predicted alpha1
minimum margin, ties by group id. In each group choose the receiving minimum-
margin probe (canonical first on ties), its current strongest competitor (first
token on ties), and differentiate that signed margin in eval mode over all trainable
parameters. Compute one group gradient at a time; no gradient library. This is
a local candidate direction, not guaranteed to represent the group's capability
or q10. Record whether the chosen probe is currently wrong and whether it is an
original-predicted correct->wrong crossing. No replacement of this heuristic.

One sequential pass only: if dot(g,d)<0, d<-d-dot(g,d)/(norm(g)^2+1e-12)*g.
Tensorwise products/norms, float64 accumulation, no flattened whole-model vector.
Each later projection can reintroduce earlier conflicts; re-predict once after
the entire pass. All four alphas are along the NEW direction. Commit receiving
parameters+alpha*new_delta, never the stale native candidate at alpha1. No second
projection. If there is no original threat, preserve native proposal exactly.
Clear current loss gradients before margin differentiation. Extra eval/gradient
operations preserve module modes and RNG; never feed control gradients into AdamW.

Record native/projected/executed predictions for all160 controls, old/new group
aggregates, newly formed new-group counts relative to the current receiving
capability, absolute correct-count gains and their ratio only when the native
gain is positive (otherwise null). Store original/new direction hashes and
norm/cosine/removal-fraction summaries. Zero norms produce null ratios, not NaN.
Readonly actual endpoint audit at each measured update follows the commit and
cannot reselect the action. Record boundary-sign agreement and actual crossings.
No additional native counterfactual parameter update is executed: comparisons
to unexecuted native proposals are explicitly predictions, not causal outcomes.

Final primary macro old/new/all, per relation, and original-test secondary metrics.
Also report threat/constraint counts, successful alpha1 projections, actual conflict
cuts, shrink/no-safe counts, eligibility loss between detections, measurement
counts, update/predictor/projection-gradient/projection-pass/second-prediction/
endpoint-audit/evaluation/wall time, CUDA allocated/reserved memory. Time and
memory are descriptive single executions, not isolated repeated benchmarks.
Only final checkpoints; publish generated records, not weights or dataset text.

Preflight tests and source/protocol hashes freeze before training. Checker replays
state/threat/alpha choices from records and recomputes endpoint metrics, identities,
batch/RNG schedule, zero replay, projection algebra certificates and hashes. Full
gradient/native directional recomputation is not claimed by a record checker.
Interpret a negative endpoint alongside measured-step repairs and unguarded losses;
do not claim forgetting solved, capacity sufficient, or replay universally needless.
