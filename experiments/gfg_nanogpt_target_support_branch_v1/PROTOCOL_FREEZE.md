# Target-level support-branch protocol freeze

## Question

Can target-specific distributed support and competitor-boundary structure,
already present before the current update response, distinguish response
branches that remain conflated by group-level CSRG and improve historical KNN
response transport?

## Frozen evidence boundary

- The response records, severe-conflict labels, original nearest-neighbour
  identities, development runs and confirmation runs are reused unchanged.
- Only the `alpha = 0` gate observations are allowed in executable query
  coordinates. They exist before the actual update response.
- `alpha > 0` logits, margins, predictions and support values are outcome
  material. They are forbidden as executable query inputs.
- Confirmation queries use only the eight development runs as history.
- Development queries use leave-one-development-run-out history.
- Candidate pools remain the same 64 F1/F3/F5 neighbours. New coordinates may
  only reweight this frozen pool.

## Frozen target-level coordinates

For each identity-aligned evaluation unit, the target-support block contains:

- four single-component necessities;
- six pair-backup values;
- four support-allocation shares;
- support concentration and effective-support count;
- minimum single- and double-gate margins;
- one support-defined indicator.

The competitor-boundary block contains, for each of the four single gates and
six pair gates:

- signed margin displacement from the ungated baseline;
- correct-logit displacement;
- displacement of the ungated leading competitor's logit;
- whether the leading competitor identity changes.

All fields are derived from the two baseline plus ten gate forwards at
`alpha = 0`. Raw class identities are not used as numeric coordinates.

## Frozen models

- `f1_f3_f5`: current 64-neighbour baseline.
- `target_support`: equal block mass for F1/F3/F5 and target support.
- `competitor_boundary`: equal block mass for F1/F3/F5 and the competitor
  boundary block.
- `target_support_competitor`: equal block mass for all three blocks.

Every block is robustly scaled using the current fold's training runs only.
Distances use inverse-distance weighting. No hyperparameter is selected from
confirmation results.

## Frozen evaluation

Report curve RMSE, endpoint-direction accuracy, boundary accuracy,
unchanged-target false-crossing rate and wrong-to-correct recall for:

- all evaluable records;
- all severe conflicts;
- the frozen 311-case group-level remainder, defined as severe and admitted by
  M4, below the prior group-level continuous-divergence threshold, and without
  a group-level primary-transition mismatch.

The 311-case result is not generalized to all nanoGPT training. Post-update
coordinates may be used only in separate diagnostic counts, never to weight a
prediction.
