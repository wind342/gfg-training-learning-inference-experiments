# RL-E05 formal results

## Frozen verdict

The preregistered composite hypothesis is **NOT_SUPPORTED**.  Nine of ten
decision gates passed.  The failed gate required a 0.03 increase in the
reinforced capability's positive-support share to occur before or at the first
accuracy loss of an unreinforced capability in at least 9 of 12 seeds.  This
occurred in 2 of 12 seeds.  All 12 formal seeds are retained.

This failure does not mean that the proposed concentration and crowding
relations were absent.  It means that their frozen temporal operationalization
was too strong for the observed ordering at the chosen checkpoint resolution.

## Primary preregistered results

All four capabilities reached exact mastery before branching.  Selective,
balanced and frozen branches began from byte-identical parameter--AdamW states
and used equal episode and update budgets.  Every positive consequence was
adjudicated by the exact two-action task rule; there was no proxy reward,
negative feedback or entropy bonus.

| Quantity | Formal result |
| --- | ---: |
| Formal seeds retained | 12/12 |
| Reinforced capability accuracy after selective feedback | 100.00% |
| Mean unreinforced accuracy after selective feedback | 73.26% |
| Mean unreinforced accuracy after balanced feedback | 100.00% |
| Accuracy deficit relative to balanced feedback | 26.74 percentage points |
| Baseline reinforced-capability support share | 26.23% |
| Selective final support share | 32.77% |
| Balanced final support share | 26.26% |
| Selective support-share increase from baseline | 6.54 percentage points |
| Selective support-share excess over balanced | 6.52 percentage points |
| Seeds with a newly captured primary-support component | 11/12 |
| Seeds with a directional proper-subset version rollback | 12/12 |
| Seeds passing the preregistered temporal gate | 2/12 |

The cross-capability support HHI increased from 0.25172 at baseline to 0.25901
after selective feedback, while balanced feedback ended at 0.25130.  Thus the
support profile became more concentrated specifically in the selective branch.
The frozen branch remained exactly identical to its sealed initial
parameter--optimizer state in every seed.

All 15 non-empty component-version rollback subsets were executed for every
seed.  At least one proper subset in every seed reduced the reinforced
capability's mean margin while improving the unreinforced capabilities' mean
margin or accuracy.  These interventions connect the selectively trained
component versions to the observed trade-off; they do not rely only on a
correlation between training time and performance.

## Diagnostic-only analysis of the failed temporal gate

This analysis was performed after the frozen verdict and cannot replace its
decision criterion.  A positive support-share change of any size was visible
before the first observed unreinforced accuracy loss in 11 seeds and at the same
checkpoint in the remaining seed.  At the first accuracy loss, however, the
mean increase was only 0.01685, below the preregistered 0.03 event threshold.
The pooled association between reinforced support share and unreinforced
accuracy was -0.882; the mean within-seed association was -0.949.

The best directional proper-subset rollback restored 24.83 percentage points of
unreinforced accuracy on average while reducing the reinforced capability's
mean margin by 2.525.  The reinforced capability's absolute positive Shapley
support mass increased by 2.663 from baseline.  The unreinforced capabilities'
absolute positive support mass did not literally decline from baseline; it rose
slightly by 0.281, but remained 2.158 below the balanced branch.  Accordingly,
the evidence establishes relative concentration, a balanced-formation deficit
and a causal functional trade-off.  It does not establish a conserved physical
support budget that was literally depleted.

## Evidence integrity

The independent checker reloaded and recomputed all baseline and final model
states, all support profiles and all 180 component-subset rollbacks.  It also
verified identical selective/frozen cue batches and sampling uniforms, exact
frozen-state preservation and the experiment GFGs.  Each of the 12 GFGs contains
2,446 complete generation facts, including 2,400 feedback-update facts.  The
independent check status is **PASS**.

## Scientific conclusion

The strict composite claim, including the preregistered 0.03 temporal event,
was not established.  A narrower, architecture-bounded conclusion is supported:
in this shared finite attention-free GRU, prolonged correct selective positive
feedback preserved the reinforced capability, concentrated its relative
functional support, caused failures in other previously mastered capabilities,
and produced component versions whose rollback reversed the trade-off.  This is
evidence for reinforcement-induced functional-support concentration and
crowding in the executed system, not a claim that every reinforcement-learning
system must collapse.
