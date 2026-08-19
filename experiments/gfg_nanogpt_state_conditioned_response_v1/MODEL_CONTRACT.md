# State-Conditioned Nonlinear Response Model Contract v1

Status: `FROZEN_BEFORE_MODEL_RESULTS`

## Scientific question

Can facts available after the actual update vector has formed, but before any finite-amplitude response is evaluated, predict the complete target-level displacement curve

\[
d_{e,t}(\alpha)=m_{e,t}(\alpha)-m_{e,t}(0),
\qquad
\alpha\in\{0,0.125,0.25,0.5,0.75,1\}?
\]

Absolute margins are mechanically reconstructed as

\[
\widehat m_{e,t}(\alpha)=m_{e,t}(0)+\widehat d_{e,t}(\alpha).
\]

The primary regression target is displacement, not absolute margin. This prevents copying the pre-update margin from appearing to solve the response problem.

## Event boundary

The admitted order is:

1. establish pre-update parameter and Adam state `(theta_t, O_t)`;
2. execute the current batch forward/backward computation;
3. form the clipped gradient, Adam increment and actual parameter update `Delta theta_t`;
4. establish F1, natural F3 and pre-update F5 inputs and seal the prediction;
5. evaluate `theta_t + alpha Delta theta_t` at the frozen alpha nodes;
6. establish and adjudicate the true response curve.

No positive-alpha response, current-step functional J/K probe, future margin, post-update receiver state or endpoint result is an input.

## Admitted inputs

- **F1 boundary state:** current margin, current correct state, correct logit, top competing logits and gaps, crowding and distance from the decision boundary.
- **F3 natural update geometry:** actual update norm, component update norms/shares/concentration, and correct-versus-current-competitor embedding-row update norms, difference, cosine and cancellation. Every admitted F3 field is derived from the native actual parameter update. No finite-amplitude functional probe is admitted.
- **F5 pre-update parameter/Adam receiver summaries:** per-component parameter RMS, first-moment RMS, square-root second-moment mean, preconditioned RMS, cross-component imbalance and deterministic update/Adam summary interaction.
- **alpha:** the requested amplitude node.

`run_id`, `entry_id`, absolute optimizer step, phase labels, sample identifiers and target identifiers are grouping or audit identities only and are never model features.

## Feature sets

- `B0`: persistence, `d(alpha)=0`.
- `B1/M1`: F1.
- `B2`: F1 plus global update L2 norm only.
- `M2`: F1 + full natural F3.
- `M3`: F1 + F5.
- `B3/M4`: F1 + F3 + F5.

The primary attribution comparisons are M1–M4. F3 is additionally split descriptively into global/component and target-specific fields; no claim that the target-specific subset dominates is permitted without a separate within-F3 ablation.

## Model family

All learned models use the same deterministic nonlinear random-feature ridge regressor:

- robust training-fold median/IQR standardization;
- clipping standardized inputs to `[-10, 10]`;
- 96 deterministic tanh random features plus linear features and intercept;
- ridge coefficient `1e-2` with the intercept unpenalized;
- float64 fitting;
- no early stopping and no outcome-dependent feature selection.

Only the response representation is selected:

- `A_DIRECT`: predict the five positive-alpha displacement nodes jointly;
- `B_AMPLITUDE_SHAPE`: predict `log1p(max |d|)` and the normalized five-node shape, then reconstruct their product;
- `C_PCA3`: fit a three-component displacement basis inside the training fold and predict its coefficients.

All basis vectors, medians, IQRs and regression coefficients are learned inside the applicable training fold.

## Nested run isolation

The outer evaluation is 12-fold leave-one-run-out. All 1,272 target records from the held-out run remain together.

Within each outer training set, representation A/B/C is selected by deterministic three-fold grouped validation over the remaining runs. The fixed random-feature width and ridge coefficient are not tuned. The representation with the lowest mean validation displacement NRMSE is selected; ties are resolved alphabetically. The selected representation is then reused for B1, B2 and M1–M4 in that outer fold.

No outer-fold response is used for standardization, basis fitting, representation selection or threshold selection.

The final executable M4 artifact uses the representation selected by the majority of the 12 inner selections, with alphabetical tie-breaking, and is fitted only after all outer predictions have been frozen. Its fit uses all 12 development runs and is not itself an unbiased evaluation result.

## Evaluation

Primary metrics:

- displacement MAE and RMSE overall and at each alpha;
- displacement NRMSE using training-fold per-node IQR scales;
- normalized-shape RMSE and correlation;
- endpoint displacement MAE/RMSE and direction accuracy.

Derived classifications are computed only from predicted curves:

- response type: near-linear, saturating, accelerating, turnback, sign-reversal or other;
- boundary class: maintain-correct, correct-to-wrong, maintain-wrong or wrong-to-correct;
- per-class precision, recall and F1;
- unchanged-target false-crossing rate.

Local J and J/K are diagnostic baselines only. They may use the already completed `alpha=-0.125,+0.125` evidence, but their values are never inputs to a formal model.

Confidence intervals for model improvements use 2,000 deterministic bootstrap resamples of whole held-out runs. Per-target rows are not treated as independent replicates.

## Frozen success rule

The experiment is `SUPPORTED` only if all conditions hold:

1. M4 displacement RMSE improves over B0, B1 and B2 and the run-clustered 95% CI lower bound for improvement over B2 is positive;
2. M4 normalized-shape RMSE improves over B2 with a positive run-clustered 95% CI lower bound;
3. M4 improves displacement RMSE over B2 in at least 8 of 12 held-out runs;
4. M4 unchanged-target false-crossing rate is below both diagnostic J and J/K rates;
5. M4 does not obtain the result by reading any forbidden input.

If some but not all conditions hold, status is `PARTIALLY_SUPPORTED`. Otherwise it is `NOT_SUPPORTED`.

## Failure and challenge audit

An outer prediction is a frozen failure case if any condition holds:

- displacement NRMSE exceeds 1.0;
- boundary class is wrong;
- response type is wrong.

Failure classification is post-outcome and descriptive, never a model input. Support or competitor changes observed in the completed response may be used only to classify failure mechanisms.

The 1,477 previously selected matched-state divergent-response pairs are an outcome-selected challenge set. They are never used for model selection, fitting, standardization or thresholds. They are evaluated only after all outer predictions are sealed.

## Hidden-evaluation boundary

This experiment uses only the 12 development runs. A prior diagnostic access invalidated the old global-unseen execution as a strict hidden adjudicator. No ordinary strict `READY` may claim a global-unseen pass. A future hidden adjudication requires a newly generated execution never accessed by this analysis session.
