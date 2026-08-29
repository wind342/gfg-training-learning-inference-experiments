# FA-E01 protocol freeze

## Experiment identity

**Name:** Frozen-Evidence Single-Relation Functional Invariance (FA-E01)

**Repository base:** `36dab5ce347dbbdac157ef23205f556606d18294`

**Protocol status:** frozen before reading or computing any FA-E01 outcome.

This experiment performs analysis only. It must not train a model, create a new
task, execute a CUDA forward pass, rerun an intervention, or modify a frozen
source bundle. It reads only the already validated nanoGPT source GFGs, CSRG-4C
support bundles and INF-E01 projection GFGs described by
`EXISTING_NANOGPT_AUDIT.md`.

## Scoped scientific question

The exact task contains one learned operation-level relation: cyclic addition
modulo 23. FA-E01 asks only whether its causal functional-support fingerprint
shows stable common organization when concrete operands and correct output
identity both vary, and whether that organization covaries with the already
selected historical capability phases.

The experiment cannot test across-relation selectivity. No result may be
reported as proof of full abstraction, human-like abstraction or a general
abstraction mechanism.

## A. Eligibility

A run is eligible only if all of the following are materialized and pass their
existing validators:

1. one linked source bundle containing 317 step-0 training rows and 212 unique
   step-100 validation token-embedding inputs;
2. a rank-22/nullity-1 cyclic mapping certificate that reproduces all observed
   training targets and uniquely completes the 212 validation targets up to a
   non-zero field-scalar automorphism;
3. all 23 correct-output groups represented in the validation rows;
4. one INF-E01 record for each of `pre_formation`, `formed`, `decline` and
   `recovered` (or the mechanically selected non-recovered replacement, if a
   frozen record has that name);
5. at every phase, complete row-order baseline logits, four single-gate logits
   and six pair-gate logits, each with shape `212 x 24`;
6. exact agreement among source-bundle identity, target tensor identity and
   row count across the linked records.

Eligibility is structural and hash-based. It cannot depend on an FA-E01 effect.
An ineligible run is reported with its failing condition and is not replaced.
At least 10 eligible runs are required for a non-`INCONCLUSIVE` verdict.

## B. Per-example causal fingerprint

The fixed component order is:

```text
h0.attn, h0.mlp, h1.attn, h1.mlp
```

The fixed pair order is the lexicographic order induced by that component list.

For example `x`, let `m0(x)` be the baseline true-target margin (correct-target
logit minus the largest non-target logit). Let `mc(x)` and `mcd(x)` be the same
margin under the already executed single gate `c` and pair gate `(c,d)`.

To preserve the existing INF-E01 sign convention, define causal margin
degradation as:

```text
E_c(x)  = m0(x) - mc(x)
E_cd(x) = m0(x) - mcd(x)
```

The primary primitive fingerprint is the ten-dimensional vector:

```text
F_raw(x) = [four E_c values, six E_cd values]
```

The six non-additive interaction terms reuse the existing INF-E01 definition:

```text
I_cd(x) = E_cd(x) - E_c(x) - E_d(x)
```

Interactions are a predeclared component-level decomposition and robustness
analysis. They are not appended to the primary fingerprint because doing so
would deterministically duplicate information already present in the ten
primitive coordinates.

Auxiliary outcomes, fixed before analysis, are baseline margin, baseline
prediction correctness, gate-induced prediction flips and raw fingerprint
amplitude. Rollback outputs are retained as provenance/phase evidence but are
not part of the primary fingerprint.

## C. Normalization

For ordinary within-run/phase comparisons, each primitive coordinate is divided
by its run-phase root-mean-square magnitude, floored at `1e-12`. Coordinates are
not mean-centred, so a common signed causal direction is not removed. Each
resulting example vector is then divided by its L2 norm for direction-based
metrics. Rows with post-scaling norm at or below `1e-12` are direction-
ineligible and are reported; they remain in raw-amplitude summaries.

For leave-one-output-out analysis, coordinate RMS scales and the consensus are
computed using only the 22 discovery output groups. The frozen discovery scales
are then applied to the held-out group. No held-out fingerprint contributes to
its own scale or consensus.

Raw, coordinate-scaled and unit-direction fingerprints are all saved.

## D. Metrics

The primary similarity metric is cosine similarity of the coordinate-scaled,
unit-normalized primitive fingerprints.

Predeclared robustness metrics are:

- Pearson correlation between coordinate-scaled primitive fingerprints;
- normalized Euclidean distance between unit directions, with smaller values
  indicating greater agreement;
- the same cosine analyses on the four single effects and six interaction
  effects separately.

No metric may be selected or dropped because of its outcome.

## E. Pair and control classes

Pairs are unordered and formed within one run and one phase. The fixed classes
are:

1. **main different-output unmatched:** different correct output, different
   left operand and different right operand;
2. **same-output different-instance:** same correct output and different
   ordered operand pair, excluding commuted pairs;
3. **commuted:** `(a,b)` versus `(b,a)`, `a != b`;
4. **shared-left:** different output, same left operand, different right
   operand;
5. **shared-right:** different output, same right operand, different left
   operand;
6. **fully unmatched random control:** a deterministic sample from class 1,
   frequency-matched to the compared control's output-group-pair counts.

Every class is constructed from recovered source inputs and derived targets,
not predictions. To prevent target-frequency dominance, class means first
average within output group (same-output) or unordered output-group pair
(different-output), then average those group-level values equally.

A predeclared baseline-confidence robustness analysis retains only pairs whose
two baseline margins fall in the same quintile of the run-phase margin
distribution. It is secondary and cannot replace the unfiltered primary class
means.

## F. Cross-output primary analysis

The primary pair effect for each run and phase is the equally output-pair-
weighted mean cosine in class 1. Cross-output invariance is assessed against:

- the same-output class;
- shared-left and shared-right controls;
- the coordinate-independence null in Section J.

The predeclared practical-equivalence margin is `0.10` cosine units. A run-phase
meets cross-output equivalence when its class-1 mean is no more than `0.10`
below the same-output mean and no more than `0.10` below either shared-operand
mean. This threshold is an effect-size rule, not a claim of universal
equivalence.

## G. Cross-run aggregation

Token identities are never aligned across runs. All calculations occur inside
each run, followed by aggregation of run-level effects in common functional
coordinates.

The run is the independent replication unit; individual pairs are never used
as independent replicates. Reports include all 13 run-level values, the median,
the number of positive/equivalent runs and a deterministic 95% bootstrap
interval over runs using 10,000 resamples and seed `73129`.

## H. Historical-phase analysis

The primary historical contrast is:

```text
formed - pre_formation
```

for the leave-one-output-out null-excess score defined below. The secondary
contrasts are `decline - formed` and `recovered - decline`. All run trajectories
are retained; strict 13/13 monotonicity is not required.

The phase result is an association with mechanically selected historical
capability state, not a new causal intervention on phase identity. Raw
fingerprint amplitude, baseline margin and baseline accuracy are reported by
phase to expose confidence/scale confounding.

## I. Held-out output consensus test

For each eligible run and phase, each of the 23 correct-output groups is held
out once. For held-out output `o`:

1. calculate coordinate RMS scales from the other 22 groups only;
2. scale and unit-normalize the 22-group discovery fingerprints;
3. define `C_-o` as their mean direction, renormalized to unit length;
4. apply the frozen scales to the held-out fingerprints;
5. score the held-out group by its mean cosine to `C_-o`.

The run-phase score is the equal mean of the 23 held-out-output scores. Outputs
with no direction-eligible held-out row make the run-phase ineligible.

This test measures cross-output generalization of a common causal pattern. It
does not predict a new relation.

## J. Null and permutation controls

All random controls use NumPy `PCG64` with master seed `20260829`. Exactly 1,000
replicates are used per run-phase unless a control is exhaustively enumerable.

1. **Primary coordinate-independence null:** independently permute example rows
   within each of the ten primitive coordinates, inside a run and phase, then
   repeat the complete leave-one-output-out procedure. This preserves every
   coordinate's empirical marginal distribution and group sizes while
   destroying the example-level multivariate organization. The primary
   null-excess is observed score minus the null median; the observed percentile
   in the null distribution is also reported.
2. **Output-label permutation:** permute output-group labels across examples
   while preserving group sizes, then repeat the held-out-output analysis. This
   tests dependence on the registered output partition; it is not treated as a
   different-relation null.
3. **Input-identity permutation:** permute the mapping from fingerprints to
   recovered input rows, then rebuild shared-operand and commuted classes. This
   tests claims tied to concrete-input overlap. A bare row permutation without
   rebuilding pair classes is prohibited because it would leave overall
   similarity unchanged.
4. **Phase-label permutation:** for the primary historical contrast, enumerate
   all 24 phase-label permutations within each run and construct 10,000 global
   run-level permutation replicates using the master RNG. The observed median
   formed-minus-pre contrast is located in this distribution.

Real effects, null medians, 2.5/97.5 percentiles and observed percentiles are
reported. P-values alone are insufficient.

## K. Component-level decomposition

The report separately provides:

- four-coordinate single-gate directional agreement and amplitude;
- six-coordinate pair-gate directional agreement and amplitude;
- six-coordinate non-additive interaction agreement and amplitude;
- the fraction of total raw squared primitive-effect magnitude attributable to
  singles versus pair gates.

The decomposition assigns one of `single-component-dominated`,
`pair-intervention-dominated` or `distributed` using frozen magnitude-share
cutoffs: at least `0.75`, at most `0.25`, or between those bounds,
respectively. This is descriptive and cannot determine the main verdict alone.

## L. Verdict criteria

The following frozen constants apply:

- run replication threshold: at least 9 of 13 eligible runs, scaled
  proportionally by ceiling(`0.69 * eligible_runs`) if fewer than 13 are
  structurally eligible;
- practical cosine equivalence margin: `0.10`;
- minimum median primary null-excess: `0.05`;
- primary null percentile threshold: `0.95`;
- minimum median formed-minus-pre null-excess change: `0.05`.

`SINGLE_RELATION_FUNCTIONAL_INVARIANCE_SUPPORTED` requires all of:

1. formed-phase held-out-output null-excess is at least `0.05` in the median,
   and the observed score exceeds its coordinate-independence 95th percentile
   in at least the run replication threshold;
2. formed-phase cross-output equivalence to same-output, shared-left and
   shared-right classes holds in at least the run replication threshold;
3. the result repeats under independent token relabellings at the run level;
4. formed-minus-pre null-excess change has median at least `0.05` and is
   positive in at least the run replication threshold;
5. the direction-eligible fraction is at least `0.90` in every eligible formed
   phase;
6. cosine, correlation and normalized-distance robustness checks do not give
   opposite directional conclusions.

`PARTIAL_FUNCTIONAL_INVARIANCE_SUPPORTED` applies when formed-phase cross-output
commonality and run replication meet items 1-3, but historical association,
eligibility or robustness fails one of items 4-6.

`OUTPUT_CONDITIONED_CONVERGENCE_ONLY` applies when same-output similarity
exceeds main different-output similarity by more than `0.10` in at least the
run replication threshold and the main null criterion fails.

`INPUT_STRUCTURE_DOMINATED` applies when a shared-left or shared-right class
exceeds the main class by more than `0.10` in at least the run replication
threshold and the main null criterion fails.

`STATIC_ARCHITECTURAL_PATTERN_ONLY` applies when formed-phase commonality meets
items 1-3 but formed-minus-pre median change is below `0.05` and no predeclared
phase control supports systematic historical association.

`NO_FUNCTIONAL_INVARIANCE` applies when formed-phase median null-excess is
non-positive and fewer than half of eligible runs exceed the primary null 95th
percentile.

`INCONCLUSIVE` applies when fewer than 10 runs are eligible, required frozen
objects cannot be reconstructed, controls conflict without satisfying another
verdict, or no rule above applies.

Verdict precedence is the order shown above, except that
`STATIC_ARCHITECTURAL_PATTERN_ONLY` is evaluated before
`PARTIAL_FUNCTIONAL_INVARIANCE_SUPPORTED` when historical association fails.

## M. Failure and reporting rules

The experiment stops without outcome interpretation if a source hash fails,
row identities do not align, a required tensor is absent, validation targets
cannot be reconstructed, or existing independent validators fail.

Every negative result, all run-level trajectories, direction-ineligible rows,
all null distributions and every predeclared metric are retained. The final
claim ceiling is:

> Existing frozen nanoGPT evidence supports single-relation causal functional
> invariance across concrete-input and output variation.

Any positive result must also state:

> The existing task does not test across-relation selectivity because only one
> underlying operation is present.

