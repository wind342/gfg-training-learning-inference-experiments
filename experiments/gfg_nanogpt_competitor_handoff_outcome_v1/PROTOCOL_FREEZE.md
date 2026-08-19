# Multi-competitor handoff outcome protocol freeze

## Question

Does the pre-outcome geometry of all currently competing classes distinguish
the response branches left unresolved by F1/F3/F5 and group-level support,
and improve direct prediction of the final correct/wrong result?

## Primary outcome

The primary target is the final correctness state after the already formed
actual update is applied. Curve RMSE is not an evaluation target. The reported
quantities are final-result accuracy, balanced accuracy, confusion counts,
per-transition recall, repaired baseline errors, newly broken baseline answers
and net repairs.

## Time boundary

Executable query inputs may use only material established after the current
actual parameter update has formed and before the updated model's evaluation
response is executed:

- current `alpha = 0` logits;
- current parameter state;
- the actual parameter update;
- F1/F3/F5 already legal at this boundary.

No `alpha > 0` logit, margin, prediction, support value, competitor identity or
response type may enter a query coordinate. Those values are labels or
post-outcome diagnostics only.

## Frozen competitor coordinates

For every evaluation unit, all 23 incorrect classes are ordered by their
current `alpha = 0` logits. Raw class identities are not numeric coordinates.

`all_competitor_gaps` contains the correct-minus-competitor logit gap at every
rank.

`all_competitor_geometry` contains, at every rank:

- norm of the current output-boundary row `W_y - W_c`;
- norm of its actual update `delta_W_y - delta_W_c`;
- cosine between the current boundary and its update;
- signed radial update ratio;
- direct row-space gap-effect estimate obtained from the minimum-norm hidden
  representation consistent with the current complete logit vector.

The final quantity is explicitly an output-row-space estimate, not a claim to
be the complete functional JVP.

## Frozen prediction methods

All methods reuse the same 64 candidates selected by robustly scaled F1/F3/F5.
They differ only in distance reweighting:

- `f1_f3_f5_outcome`;
- `all_competitor_gaps`;
- `all_competitor_geometry`;
- `all_competitor_combined`.

Each added block is robustly scaled on the current training fold only and has
equal block mass with F1/F3/F5. Historical final correctness is combined by
inverse-distance weighted KNN. The final-correct threshold is fixed at 0.5.

## Run isolation

Development runs use leave-one-development-run-out history. All four
confirmation runs use only the eight development runs as history. Candidate
selection, scaling and voting never read the held-out run's outcomes.

## Required subsets

Every result is reported separately for all 15,264 records, all 1,304 severe
conflicts and the frozen 311-case group-level remainder. A new method is
scientifically useful for the remainder only if it produces positive net
repairs without relying on curve error.
