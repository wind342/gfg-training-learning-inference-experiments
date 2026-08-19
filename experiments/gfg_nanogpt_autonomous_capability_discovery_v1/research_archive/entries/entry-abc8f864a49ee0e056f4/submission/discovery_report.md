# Discovery report

## Result

The selected mechanism is formation of a cyclic-composition circuit, not a
late improvement in sample memorization. The training facts alone constrain a
token coordinate \(\phi\), unique up to a cyclic automorphism, such that every
observed training relation satisfies
\(\phi(a)+\phi(b)-\phi(y)=0\). The model generalizes when its complete
evaluation logit tensor becomes approximately circulant in
\(\phi(k)-\phi(a)-\phi(b)\), and its predicted operation table becomes
associative and Latin. No token-coordinate map or answer table is embedded in
the candidate; `initialize` solves the coordinate constraints from each new
prefix and retains only aggregate state.

The formal transition in this graph is step 800. There is an earlier
evaluation at or below 0.3, and the evaluations at steps 800, 900 and 1000
all have train accuracy at least 0.99 and validation accuracy at least 0.9.
The frozen cut rule selects step 500: the last three train accuracies are 1.0,
the current validation accuracy is 0.6745283, and no qualifying window has
yet occurred.

## Exact GFG derivation

The source training relation is not reconstructed by a temporal join.
Batch occurrence
`occ_10bace233ceff9d389539539b8f34521532eac7311b793176f7a386784e52382`
realizes fact block
`factblock_f737ae1c404863732f0745f177c5aeac13edc631940cdbcc6cb8f563da386b97`.
Its exact task source forms input object
`obj_4de1ef4587dcec33e039dbdd50bd886653fca7b6a3add99c4a56f3a6b8b09685`
and target object
`obj_c96a77414dc2571f9caec1a0da2e36719a26a3befe205ae0c2ccd5c8b3c648c9`.
Expanding this fact block supplies the `realizes_fact`, incidence and
`reads_from` records used by the coordinate constraints.

At the cut, optimizer occurrence
`occ_886866270bfec2fdd11ddb3503a4432a42bdacf134767be3c20f8c9c8efdf0df`
forms the exact version-500 parameter results. For example,
`factblock_ce26b21d1b2dcb98d7be5293b472dd6e883a2f4bc9b1eee12ef0a4f083d0cd70`
keeps the prior parameter, clipped gradient, optimizer configuration and
three optimizer-state sources together with its parameter/state outcomes.
Primitive edge
`edge_62b9a0b2c6433c76e6324f361f8e19304afeaa927a4185357bbcf6440ab900ce`
records the corresponding `GeneratedOrigin` parameter-version link.

Evaluation occurrence
`occ_bbebcada347726f9ca26d54cf5242f5eeaea722a7daca9dc2703c71e1d24f4bb`
then realizes
`factblock_5d11d8f4d4128ad3064b41f4c07ffb3a2a51318f3442b8b0280d7d027e7344a2`.
That atomic block reads the exact version-500 parameter objects and the task
and forms, among its separate outcomes, validation-logit object
`obj_3804ee3d181f301b2c744efe854d87e87cbf6dcc69d97eae2fc28b67b82870ae`
and metric object
`obj_50caf8fde4ced3c7f9e6e4570c390696ffc166e2d1e5474d4b35c150e51de883`.
The logit content hash is
`49eb8a84bdf63c4b8cfe9bbfcf0e2fec87009715404d33c33cfee141bff01596`.
Thus the state is tied to a concrete parameter result and evaluation
occurrence, rather than to timestamp proximity or equal values.

The 317 training equations over 23 operand symbols have rank 22, leaving one
coordinate degree of freedom; the recovered coordinates are a permutation of
the 23 cyclic values. Using this training-only coordinate, selected
falsification checkpoints give:

- At step 400, circulant explained-logit variance is 0.462588 and predicted
  table associativity is 0.26038.
- At the step-500 cut, they are 0.742113 and 0.61700; commutativity is
  0.81096 and final-layer outcome-cluster variance is 0.720315.
- At the transition point they are 0.852345 and 0.93877; commutativity is
  0.97353 and final-layer outcome-cluster variance is 0.857810.
- At the next point, associativity is 0.99310 and the circulant statistic is
  0.895419.

These aggregates are computed over exact materialized tensors while keeping
each evaluation occurrence and outcome identity separate.

## Mechanisms proposed and falsification attempts

### Selected: cyclic-composition circuit formation

Prediction: a training-only cyclic coordinate must be identifiable before
the transition; held-out logits and late hidden states should increasingly
factor through the composed coordinate; current predicted tables should
approach the algebraic laws rather than merely fit the 317 observed entries.

Attempted falsification: solve the coordinate equations without held-out
labels and measure the full current logit tensor at several exact evaluation
occurrences. The constraints are identifiable (rank 22 of 23), and the
circulant, associative, commutative and hidden-clustering measures rise
together through the transition. The final complete prediction table is
closed, Latin, commutative and associative with a single identity. This
hypothesis was not falsified.

### Rejected: accumulation of new sample coverage

Prediction: generalization begins when new training samples or new batch
membership supply missing relations.

Falsification: all 10,000 `before_batch` occurrences contain exactly the same
317 distinct sample identities, in the same order and with one input content
hash. The full training set is already present in the first batch occurrence.
Across all 2,929 evaluation objects, there are zero established
`GeneratedOrigin` edges into a non-evaluation block. This does not turn a
missing edge into a logical negation; it shows that the complete validated
graph establishes no such source path. Coverage therefore cannot explain the
late state change.

### Rejected: a training-loss threshold

Prediction: crossing a scalar loss threshold changes memorization into
generalization.

Falsification: evaluation occurrence
`occ_0c1b875c44c35453540a0fae81dc1a585bdeb5ca25f12a089c63dc12366f97af`
already has train accuracy 1.0 and loss 0.0121466 at step 100 while validation
accuracy is 0.0235849. At step 300 the loss is 0.0019980 while validation is
only 0.113208. Conversely, transition-point loss is higher, 0.00615557.
Low loss is a memorization condition, not the executable state variable.

### Rejected: gradient clipping is the phase switch

Prediction: the first clipping event near the capability boundary causes the
transition.

Falsification: clipping changes tensor contents in 461 distinct optimizer
steps, including many at steps 3--76, long before rule expression, and later
bursts after the rule is locked. Near the boundary, bursts at 446--460 and
775--786 are not unique first events. At the materialized 400, 500, 600, 700,
800 and 900 checkpoints, the raw-gradient and clipped-gradient outcomes are
distinct facts but have equal content hashes and equal aggregate norms.
Clipping can constrain an update, but its occurrence is neither sufficient
nor the finite state that forecasts generalization.

## Forecast and intervention

The state machine labels the cut `RULE_ASSEMBLY`. It forecasts the remaining
distance from current validation expression and cyclic-circuit organization,
emits an inclusive transition interval, generates every future 100-step
validation point algorithmically, and advances through `RULE_EXPRESSED` to
`RULE_LOCKED`. It performs no future GFG read.

The intervention predicts `DELAY` with a shift interval of 1000--1800
optimizer steps. For 1200 updates it uses only allowed hooks to zero current
gradients and set current optimizer-group learning rates to zero. Because the
evaluation at step \(t\) reads the exact parameter Result \(\theta_t\), this
holds the circuit's parameter result and `RULE_ASSEMBLY` progress fixed while
program order and data order continue. It then restores the captured learning
rates. The predicted interval allows for optimizer-moment rebuilding after
release and exceeds the required 600-step effective shift.
