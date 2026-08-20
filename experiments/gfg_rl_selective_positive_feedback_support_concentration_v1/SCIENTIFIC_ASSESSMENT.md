# Scientific assessment

## Did correct selective positive feedback produce a detectable trade-off?

Yes, in the executed shared finite GRU.  The reinforced capability remained at
100% in all 12 seeds, while the other three capabilities averaged 73.26%; their
balanced and frozen counterparts remained at 100%.  Because rewards were exact
task consequences and the frozen branch used identical episodes without a
persistent update, neither proxy-reward error nor mere inference exposure
explains this result.

## Was the proposed functional-support mechanism established?

The central concentration and causal-trade-off relations were established, but
the full preregistered composite criterion was not.  Selective feedback raised
the reinforced capability's support share by 6.54 percentage points from
baseline and by 6.52 points relative to balanced feedback.  New primary-support
capture occurred in 11/12 seeds.  Exhaustive version rollback supplied
directional causal evidence in 12/12 seeds.

The failed element was the frozen temporal gate.  A large 0.03 support-share
event preceded or coincided with the first accuracy drop in only 2/12 seeds.
The formal status must therefore remain `NOT_SUPPORTED`.  Diagnostic analysis
showed smaller support changes before or at the first drop in 12/12 seeds, so
the failure is consistent with a threshold/timescale mismatch, but that
post-hoc observation is not allowed to redefine success.

## Does the result prove that positive feedback consumes a fixed support budget?

No.  Positive support became more concentrated and other capabilities lost
boundary correctness, but their absolute positive Shapley mass was not driven
below baseline on average.  The defensible description is relative support
capture, altered functional geometry and crowding relative to balanced
formation—not literal conservation or physical exhaustion of a fixed quantity.

## What does the experiment add to the training--learning--inference theory?

It adds a bounded consequence of the theory.  Because inference projects
already formed support, selectively rewarding one class of correct projected
actions repeatedly re-enters training through the same capability.  In a shared
policy this can reorganize component versions toward that capability, strengthen
its future projection and reduce the boundary reliability of other capabilities.
The loop is therefore genuinely double-edged: the same feedback closure that
forms a desired capability can also concentrate learning around it.

## What is not established?

- that every reinforcement-learning algorithm or architecture exhibits this
  effect;
- that concentration is inevitable under adequate rehearsal, modularization,
  replay or explicit anti-interference constraints;
- that the observed support quantities obey a conservation law;
- that all real-world model collapse is caused by this mechanism;
- that the preregistered large support event always precedes behavioural loss.

## Next falsification target

The strongest next experiment would keep selective positive feedback fixed
while independently varying support isolation and balanced rehearsal.  If
modular support or minimal rehearsal prevents both support capture and boundary
loss without weakening the reinforced capability, that would identify a causal
control mechanism rather than merely repeating the present effect.
