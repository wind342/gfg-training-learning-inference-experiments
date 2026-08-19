# Native pre-branch left-history protocol v1

Status: `FROZEN_BEFORE_MODEL_RESULTS`

## Scientific question

This experiment tests whether facts already available immediately before a
finite-amplitude functional response, together with the current native update
that has been formed but not yet applied, contain a transportable signal of the
local response branch that will subsequently occur.

The primary branch is whether the strongest incorrect competitor changes over
the five positive response amplitudes. Secondary labels are severe structural
conflict, primary-support handoff, turnback, sign reversal, correct-to-wrong,
and wrong-to-correct. Labels are evaluation targets only.

## Frozen event order

1. Establish the receiver state `X_t`, including the alpha-zero logits,
   margins, support state, parameter/Adam receiver state, and left history.
2. Form the actual native update `U_t`; do not apply it to the receiver.
3. Seal branch risk and response-curve predictions.
4. Execute the positive-amplitude responses and establish their labels.
5. Compare sealed predictions with the newly established response facts.

Objects marked `target_only_after_cut`, all current alpha-positive responses,
current J/K functional probes, post-update logits/margins/support, and branch
labels are prohibited inputs.

## Identity and missingness

Target, competitor, and support-component identities are used only to align
the same object through time. Numeric class identifiers, run identifiers,
entry identifiers, section identifiers, absolute optimizer steps, and phase
labels are never model features. Raw missing values remain `NA` in the ledger.
Within each training fold only, numeric `NA` values are replaced by the
training median after adding an explicit missing indicator; they are never
silently replaced by zero as scientific values.

## Frozen feature spaces

- `X0`: the existing 49 numeric `F1 + F3 + F5` variables.
- `X1`: `X0` plus additional current competitor-crowding summaries and sampled
  identity-aligned prior competitor-switch/persistence history. Existing
  top-1/top-2 and top-1/top-3 gaps already present in `X0` are not duplicated.
- `X2`: `X1` plus identity-aligned competitor-gap and target-margin left
  histories at lags 1, 2, 5, and 10, their finite differences, and the
  lag-one left acceleration.
- `X3`: `X2` plus current-versus-prior native-update cosine and norm-ratio
  continuity, globally and for the four registered support components. These
  variables measure update continuation; they do not claim to reveal the
  unseen effect of the update on a competitor.
- `X4`: `X3` plus current support-component gaps, identity-aligned support
  histories, left differences, acceleration, and sampled handoff/persistence
  history. `X4` is admitted only if the availability audit passes.

All left differences are the already materialized
`input_available_at_cut` objects. For scale `m`,
`V^-_{t,m}=(S_t-S_{t-m})/m`; lag-one acceleration is
`(S_t-S_{t-1})-(S_{t-1}-S_{t-2})`.

## Frozen non-parametric estimators

The primary estimator is inverse-distance weighted KNN with `k=64`. Distances
are root-mean-square Euclidean distances after median/IQR scaling fitted only
on the outer training runs. Constant coordinates are removed. No neural
network, random forest, or parametric classifier is trained.

For threshold metrics, each outer fold selects a threshold using only its
training records: self records are excluded, and the threshold with maximum
recall subject to false-positive rate no greater than 0.10 is chosen. If no
threshold satisfies the constraint, the all-negative threshold is used.

The untrained approach score is the mean of four training-fold percentile
ranks with signs fixed before results: small current top-competitor gap,
negative lag-one gap velocity, negative lag-one gap acceleration, and high
current/prior-update cosine. It is diagnostic and is not tuned after viewing
test labels.

## Validation split and metrics

All 12 executions are evaluated by complete leave-one-run-out validation. A
test run supplies no scaler value, threshold, neighbor, hyperparameter, or
feature decision. The primary branch metrics are ROC-AUC, PR-AUC, Brier score,
calibration, quantile risk ratio, recall and false-positive rate. Run-wise
results and run-clustered bootstrap 95% intervals use 1,000 deterministic
resamples.

Historical F2 AUC 0.608 and F4 AUC 0.482 localized severe *pair conflicts*;
they are retained as contextual references, not misreported as prospective
competitor-switch baselines. Prospective baselines are prevalence, current
competitor gap, sampled past-switch count, and `X0` KNN.

## Response prediction and oracle definitions

Ordinary response KNN uses the frozen `X4` distance when `X4` is admitted,
otherwise `X3`. The primary oracle (`Oracle-S`) uses the same distance but
restricts historical neighbors to the test record's true competitor-switch
branch. It is diagnostic only. The executable routed version predicts the
test branch from pre-response risk and then restricts historical neighbors to
that predicted branch; historical branch labels are allowed because their
responses have already occurred. Support-handoff oracle results, if reported,
are separate and exclude unevaluable support labels.

Curve metrics are five-node displacement RMSE, endpoint-direction accuracy,
final-boundary accuracy, unchanged-target false-crossing rate, and
wrong-to-correct recall, both overall and on the severe-conflict subset.

## Frozen adjudication

`LEFT_HISTORY_BRANCH_SIGNAL_SUPPORTED` requires all of the following:

1. The best pre-response left-history space improves competitor-switch PR-AUC
   over `X0`, with a positive run-clustered 95% interval for the delta.
2. At least 8 of 12 runs have non-negative run-wise PR-AUC change.
3. Either severe-conflict PR-AUC also has a positive clustered delta interval,
   or executable routed response prediction reduces severe-subset curve RMSE
   by at least 3% without reducing overall boundary accuracy by more than one
   percentage point.
4. All leakage, source identity, replay, and GFG checks pass.

`PARTIALLY_SUPPORTED` applies when left history gives stable risk
stratification but does not satisfy the routing/downstream condition or is not
stable across runs. `NOT_SUPPORTED` applies when the gain is unstable,
non-positive, or invalid. If Oracle-S materially improves severe-subset curve
RMSE by at least 3% but executable pre-state routing is not supported, the
additional diagnosis is
`BRANCH_REAL_BUT_PRESTATE_NONIDENTIFIABLE_UNDER_CURRENT_OBSERVATION`.

No outcome threshold or scientific verdict may be changed after outer-fold
results are inspected.
