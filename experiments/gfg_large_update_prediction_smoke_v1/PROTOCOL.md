# Large-update endpoint prediction smoke v1

Exploratory one-step matched intervention; NOT a complete training-speed test.
Base: b026bc7c7f2d8d7665842cbc4137ae4369ca90ee. Original sources and frozen tag
must remain unchanged. No training from scratch; no push authorized.

Before observing new endpoints, take the first three entries in the existing
functional-online-controller RUN_ASSIGNMENT.json, in its stored order.
Restore their recorded parameter AND Adam moment/step state. Use their recorded
next batch and initial RNG seed. Generate one proposal using the existing
CausalTrainingRuntime.train_step (including its optimizer configuration and
clipping), recording hyperparameters explicitly. This differs from the recent
70M/160M freshly initialized optimizer experiment; no cross-scale claim follows.

For each receiving state test alpha = 0, .25, .5, 1, 2, 4, 8, 16. Alpha 0 is
the restoration control. Alpha 1 uses exact native proposal bytes; other endpoints
use FP32 theta + alpha * (theta_native - theta). Optimizer moments are those of
the one native proposal, unchanged across endpoint forks. No chained endpoint
training. Preserve every tested multiplier, including failures/nonfinite values.

Compute the existing exact complete-quadratic coordinates once at the receiving
state along the realized native displacement. Do NOT refit coefficients at large
alpha. Mask only the target-vs-itself coordinate; require every competing endpoint
gap > 0. Keep frozen FP32 coordinate rounding/FP64 endpoint association.
Use the historical 212 validation queries for selection. Evaluate their 317-query
complement separately, never select on it. This is selection-disjoint, not a new
unseen-task/generalization claim. Compute both arrays before evaluating endpoints.

Seal all predicted minima and selected alpha to disk before any post-update
forward. Select over .25,.5,1,2,4,8,16 by predicted correct count, median minimum
gap, then closest to 1 (remaining tie smaller alpha). Separately report restricted
.25,.5,1 selection and retrospective best ACTUAL endpoint (oracle diagnostic only).

Report actual accuracy, gains/losses, predicted-vs-actual boundary agreement,
four-transition macro recall over observed classes, balanced correct/incorrect
recall, optimism error and minimum-gap absolute error. A high raw agreement with
uniformly wrong endpoints is not evidence of useful prediction. Preserve raw logits,
coordinates, predictions and hashes. Verify alpha-0 identity, alpha-1 native result,
optimizer immutability, restoration, and independently recompute statistics in NumPy.

Interpret a useful preliminary signal as alpha > 1 with better actual accuracy
than native alpha 1 and useful predictive discrimination. No post-outcome threshold
will be called preregistered; no statistical confirmation or general conclusion
from three selected historical receiving states. Do not run a full training race.

Pre-endpoint setup correction: attempt-1 stopped at the input-identity assertion,
before baseline evaluation or optimizer execution. Historical inputs are
[permuted_left, operator, permuted_right], not the recent scale fixture's encoding.
Attempt-2 directly reads the original 317 training inputs and encoded labels as
the selection-disjoint set. It is NOT held out from historical training. No labels
are reconstructed using arithmetic on token IDs. Preserve attempt-1 and its source.

Pre-endpoint arithmetic correction: attempt-2 stopped before endpoint evaluations
because the adapter's full-position readout was not byte-identical to the model's
native last-position forward. Attempt-3 anchors g0 to the actual native forward,
retains both raw logit arrays and reports their difference (no relaxed tolerance).
Directional terms remain the unmodified v2 complete-quadratic computation. This
g0 anchoring is explicit, not a claim of bit-identical v2 prediction semantics.
Attempt-2's code and failure remain preserved; no endpoint outcome was observed.
