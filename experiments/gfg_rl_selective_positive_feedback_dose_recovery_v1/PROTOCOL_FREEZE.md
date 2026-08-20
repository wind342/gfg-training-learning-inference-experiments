# RL-E06 selective-positive-feedback dose, duration and recovery protocol

## Scientific question

Does the concentration and duration of correct positive feedback causally control
how a shared learned system reallocates distributed functional support, and can
redistributing that feedback reverse the associated loss of unreinforced
capabilities?

The experiment does not test whether reinforcement learning is intrinsically
harmful.  It tests a bounded mechanism: in a finite shared policy, repeated
positive feedback directed toward a narrow subset of already formed capability
space can create a functional trade-off through the same support-reorganization
process that strengthens the reinforced capability.

## Starting state

Four exact Boolean capabilities are first formed to complete behavioural mastery
in the shared attention-free GRU policy used by RL-E05.  The complete
parameter--AdamW state, its optimizer memory and its baseline functional-support
profile are sealed.  Every experimental branch starts from this byte-identical
receiving state.

## Feedback-dose intervention

Every updating branch receives exactly 60 episodes and one optimizer opportunity
per feedback update.  The terminal consequence is always adjudicated by the
exact two-action task rule: a correct complete chain receives 1 and every other
chain receives 0.  Negative reward, proxy reward and entropy bonuses are
forbidden.

Only the allocation of those 60 episodes across capability identities changes:

| condition | capability 0 | capability 1 | capability 2 | capability 3 |
|---|---:|---:|---:|---:|
| balanced | 15 | 15 | 15 | 15 |
| mild concentration | 30 | 10 | 10 | 10 |
| high concentration | 45 | 5 | 5 | 5 |
| exclusive | 60 | 0 | 0 | 0 |

A frozen branch receives the same cue, action-uniform and capability ledger as
the exclusive branch but performs no persistent state update.  Cue occurrences,
sampling uniforms, batch size, update budget, optimizer and initial state are
otherwise held fixed.

The dose branches run for 3,200 feedback updates.  This is 192,000 environment
episodes per updating branch, rather than treating the update count alone as the
exposure duration.

## Temporal measurements

Target-specific chain accuracy, stage accuracy, mean margin and minimum margin
are evaluated after every feedback update.  Exact functional support is measured
at all frozen checkpoints, densely during the first 100 updates and more sparsely
thereafter.  The measurements therefore preserve continuous internal movement
even when a discrete correct/incorrect boundary has not yet changed.

No rule requires a globally large support-share change to precede a behavioural
transition.  A target close to its readout boundary may change outcome after a
small target-specific response.  The temporal question is evaluated through the
joint trajectories of feedback dose, support, continuous margins and discrete
boundary results, not through an arbitrary minimum global-share event.

## Recovery interventions

At exclusive-feedback update 800, the exact parameter--AdamW state is sealed and
used to start two matched recovery branches:

1. **rebalance recovery:** 15 episodes from each of the four capabilities;
2. **repair recovery:** no capability-0 episodes and 20 episodes from each of
   capabilities 1--3.

Both recovery branches receive 2,400 further updates, ending at the same global
feedback exposure as the continued exclusive branch.  Their results are compared
with continued exclusive training at update 3,200.  Recovery is not inferred
from a separate classifier: it must occur in the native capability margins and
boundary outcomes and be accompanied by functional-support reorganization.

## Functional-support authority

At every support checkpoint, all 16 component coalitions are executed on the
complete four-capability cue space.  Exact Shapley values of the identity-aligned
correct-action margin define signed component contributions.  Positive support
mass, task support share, cross-task HHI, within-task HHI, support overlap and
primary-support identities are derived from those real interventions.

Weights, parameter norms and gradient norms are recorded but are not substituted
for functional support.

## Primary causal decisions

The hypothesis requires all of the following under the frozen numerical gates:

1. complete starting mastery and exact frozen-state preservation;
2. preserved capability under balanced feedback;
3. a reproducible positive association between feedback concentration and
   capability-0 support share;
4. a reproducible negative association between feedback concentration and the
   continuous margins of capabilities 1--3;
5. a material exclusive-versus-balanced deficit in the unreinforced capability
   outcomes together with excess capability-0 support share;
6. recovery of unreinforced capability after feedback is redistributed, with
   the rebalance branch retaining the originally reinforced capability;
7. recovery accompanied by movement of the functional-support state away from
   the continued-exclusive state.

Formal thresholds are selected only from independent development seeds and are
sealed before formal execution.  All formal seeds, difficult trajectories and
failed recovery branches are retained.

## Falsification

The proposed double-edged mechanism is not supported if increasing feedback
concentration does not systematically alter support and unreinforced margins, if
equal-budget balanced feedback produces the same deficit, or if redistributing
feedback fails to change the capability/support trajectory relative to continued
exclusive training.

Even a positive result remains system-bounded.  It establishes a controllable
mechanism in the executed shared GRU policy; it does not establish that every
reinforcement-learning method or architecture must lose unrelated abilities.

