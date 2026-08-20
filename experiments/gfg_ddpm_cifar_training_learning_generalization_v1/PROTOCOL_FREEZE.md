# Frozen protocol: DDPM/CIFAR-10 cross-system generalization v1

Status: **FROZEN BEFORE FORMAL EXECUTION**.

## Pre-formal smoke amendment

The first non-decision smoke run used residual-candidate distance `0.5`. In a
3,072-dimensional residual field this produced a degenerate all-correct
boundary (16/16 maintained correct), as expected from concentration. Before
any formal run, the distance was replaced by the dimension-derived rule
`1 / sqrt(3 * 32 * 32)`. No formal outcomes existed when this amendment was
made. Boundary screening was also set to the complete first 128-example test
batch rather than a partial batch; the retained target count remains 24. No
formal outcomes existed when this amendment was made; all formal tests and
thresholds below remained unchanged.

## Scientific question

Do the primary training--learning relations identified in nanoGPT and
transported to ResNet classification remain present in a generative diffusion
system whose objective, architecture and functional output are different from
both earlier systems?

This experiment can support or falsify transport to the executed system. It
cannot, by itself, prove unconditional universality.

## System change

The executed system is an unconditional DDPM-style epsilon predictor trained
on CIFAR-10 with a time-conditioned compact U-Net and AdamW. Unlike the two
earlier systems, its native functional output is a spatial residual field and
its training objective is denoising score estimation across diffusion time.

## Concrete target identity and readout boundary

An evaluation target is the frozen tuple

```text
(CIFAR image identity, diffusion timestep, noise occurrence identity).
```

For each target, the true noise residual and seven frozen nearby residual
candidates are retained. Let `E_true` be mean-squared error to the true residual
and `E_k` the error to candidate `k`. The target margin is

```text
m = min_k(E_k - E_true).
```

The target is correct exactly when `m >= 0`, meaning that the predicted
residual is closer to the concrete true noise occurrence than to every frozen
competitor. This boundary is derived from the native epsilon-prediction task;
it is not a class label or an arbitrary post-hoc loss threshold.

## Event boundary and allowed prediction information

At a registered update, the actual training batch, timestep occurrences,
noise occurrences, gradients, pre-update parameters, pre-update AdamW memory
and actual parameter update are first formed. The prediction boundary is then
sealed before any post-update target residual, margin or correctness is read.

Allowed inputs are:

- F1: pre-update target margin, correctness, true/candidate errors, timestep
  and signal-to-noise state;
- F3: the actual parameter update, blockwise update geometry and target-
  specific directional action computed on the pre-update model by automatic
  differentiation;
- F5: pre-update parameter state, AdamW first and second moments,
  preconditioning summaries and their interaction with the actual update.

Forbidden inputs are listed in `MODEL_CONTRACT.json`. In particular, no
response at alpha greater than zero may enter the predictor.

## Finite-amplitude response

The exact realized update is replayed from the identical pre-update state at

```text
alpha in {0, 0.125, 0.25, 0.5, 0.75, 1}.
```

The complete target-margin path is retained. First and second directional
predictions are calculated only from the pre-update model and actual update.

## Receiving-state exchanges

Two exchanges are frozen:

1. parameter receiving-state exchange: apply the same realized update to a
   different earlier parameter receiving state;
2. AdamW-memory exchange: hold the current parameters, batch, gradients and
   hyperparameters fixed while substituting an earlier first/second-moment
   state to form a counterfactual actual update.

Both exchanges compare functional response on the same frozen target tuples.

## Distributed functional support

The four registered U-Net support routes are high-resolution skip,
low-resolution skip, bottleneck and decoder refinement. All 16 gate coalitions
are executed at the same materialized state. Exact four-player Shapley values,
pair interactions, distributed support, support reallocation and primary-
support switches are computed. Repeated ungated inference must be identical.

## Run isolation and controls

Three complete seeds are executed. Held-out prediction leaves one complete run
out; targets from one run may not be randomly split between training and test.
Hyperparameters and feature families are frozen before formal execution.

Negative controls are:

- target-outcome permutation;
- F3 update-geometry permutation;
- F5 AdamW-state permutation;
- unchanged-boundary prediction.

The controls audit identity pairing and do not replace native causal exchanges.

## Integrity

The run fails integrity if any of the following occurs:

- reconstructed AdamW update maximum error exceeds `2e-6`;
- alpha=0 differs from the native pre-update endpoint;
- alpha=1 differs from the native post-update endpoint;
- repeated ungated functional readout differs above `1e-7`;
- a compact GFG fails identity, incidence, evidence or coverage validation;
- a forbidden post-update field enters a prediction feature;
- the total experiment footprint exceeds 25 GiB.

## Frozen success tests

`CROSS_SYSTEM_GENERALIZATION_SUPPORTED` requires all of:

1. integrity passes;
2. at least 75% of parameter receiving-state exchanges have response NRMSE
   above 0.05;
3. at least 60% of AdamW-memory exchanges have response NRMSE above 0.05;
4. at least 10% of target paths are classified as saturating, accelerating,
   turnback or sign reversal;
5. at least 50% of targets have at least two active support components and at
   least 25% undergo support reallocation above 0.01;
6. on complete held-out runs, F1+F3+F5 four-way macro recall exceeds F1 by at
   least two percentage points and the direction of the gain is positive in
   every held-out run;
7. correctly paired F1+F3+F5 exceeds each frozen permutation control in
   four-way macro recall.

The other frozen verdicts are `CROSS_SYSTEM_GENERALIZATION_PARTIALLY_SUPPORTED`,
`CROSS_SYSTEM_GENERALIZATION_NOT_SUPPORTED` and `INTEGRITY_FAILURE`.

## Storage

Only registered scientific occurrences, compact outcomes, hashes, bindings,
one final checkpoint per run and small generated sample grids may be retained.
Dense framework traces and duplicate checkpoints are prohibited. The complete
experiment must remain below 25 GiB.
