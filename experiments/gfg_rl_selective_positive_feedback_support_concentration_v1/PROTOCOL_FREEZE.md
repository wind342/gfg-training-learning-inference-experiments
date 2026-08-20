# RL-E05 selective-positive-feedback support-concentration protocol

## Scientific question

Does prolonged, correct, selective positive feedback strengthen the distributed
functional support of the reinforced capability while concentrating shared
support away from other already formed capabilities?

The proposed mechanism does not depend on a misspecified proxy reward.  Every
positive consequence in this experiment is adjudicated by the exact task rule.

## Claim under test

Four capabilities are first formed to exact behavioural mastery in a shared,
finite, attention-free GRU policy.  Starting from the byte-identical
parameter--AdamW receiving state, three branches then receive equal episode and
update budgets:

1. **selective:** only capability 0 is encountered; successful complete action
   chains produce a positive terminal consequence and a REINFORCE update;
2. **balanced:** all four capabilities are encountered equally under the same
   correct positive-consequence rule;
3. **frozen:** capability 0 produces actions and physical consequences under
   the same batches as the selective branch, but no persistent update occurs.

Incorrect action chains receive zero, not negative, learning credit during the
feedback phase.  The feedback phase therefore isolates repeated positive
reinforcement rather than punishment or proxy-reward error.

## Functional-support measurement

The GRU hidden state is divided, before execution, into four equal component
groups.  At each frozen checkpoint all 16 component coalitions are executed on
the complete four-capability cue space.  The scalar readout is the
identity-aligned correct-action margin.  Exact Shapley values over the 16 real
interventions define the signed contribution of each component to every
capability, cue and decision stage.

The primary support statistics are:

- positive support mass for each capability and component;
- the reinforced capability's share of total positive support;
- cross-capability support concentration (HHI);
- the primary capability supported by each component;
- target margins and four capability-level behavioural results.

Weights, gradient magnitude and component size are not treated as functional
support.  Post-update outputs do not enter any earlier training action.

## Causal adjudication

The baseline component versions are preserved.  At the final selective state,
all 15 non-empty subsets of the four trained component versions are exhaustively
rolled back to their pre-feedback versions.  The complete baseline restoration
is retained as an endpoint authority.  A rollback is directionally consistent
with support capture only when it decreases the reinforced capability's mean
margin and improves either the mean margin or the chain accuracy of the three
unreinforced capabilities relative to the final selective state (numerical
tolerance `1e-6`).  No difficult seed, component or rollback is deleted.

## Frozen evidence order

For each formal seed:

1. form all four capabilities under balanced supervised training;
2. seal the parameter--optimizer receiving state and baseline support profile;
3. restore three byte-identical branch states;
4. execute the selective, balanced and frozen feedback branches;
5. materialize evaluations and all-coalition support interventions at the
   frozen checkpoints;
6. execute all 15 non-empty component-subset rollbacks;
7. compile the real occurrences and results into the experiment GFG;
8. run the independent checker from the sealed artifacts.

## Falsification

The hypothesis is not supported if selective positive feedback improves the
reinforced capability without a reproducible increase in its share of positive
functional support, if the unreinforced capabilities do not differ from the
balanced and frozen controls, if the support transition does not precede or
coincide with the behavioural transition, or if version interventions do not
connect the detected support changes to capability outcomes.

One executed system cannot establish that every reinforcement-learning system
must collapse.  A positive result establishes a receiving-state- and
architecture-bounded mechanism of reinforcement-induced functional-support
concentration in this shared finite policy.
