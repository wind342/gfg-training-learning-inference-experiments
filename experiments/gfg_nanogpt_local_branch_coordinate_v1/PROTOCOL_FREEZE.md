# Local branch coordinate discovery protocol (frozen before outcome evaluation)

## Question

The existing F1/F3/F5 plus left-history coordinate space transports finite-amplitude
responses well for most records, but a small local conflict region remains. This
experiment asks whether one **pre-response relational coordinate** separates ordinary
evolution from that local branch and thereby lets the same KNN transport rule use a
more appropriate empirical neighbourhood.

The experiment does not train nanoGPT, run a new response probe, use the VM scientist,
or treat a post-response branch label as an input. Severe conflict is used only as the
adjudication label after every candidate value has been formed.

## Frozen run split

The split is the SHA-256 ordering of
`local-branch-coordinate-v1:<entry_id>`, frozen before candidate outcomes are read.

Development runs (candidate selection only):

- `entry-abc8f864a49ee0e056f4`
- `entry-54ba9566f731754a0e3f`
- `entry-7a22f51938059541de98`
- `entry-f34e7e61444c90976b36`
- `entry-7c1a3094f8acf9cf4bb0`
- `entry-5bb1186bc27eb82111fb`
- `entry-8fa6576fc7128f93a228`
- `entry-d8b1bf9cd00ddf314725`

Confirmation runs (untouched until one candidate is sealed):

- `entry-4ed462761347d6b87e61`
- `entry-d5b80ca9b9cd18fa343f`
- `entry-786d0a3628f6f791399f`
- `entry-481b86f81d58d496a687`

## Event boundary

Allowed inputs exist after the current batch, gradient, Adam increment and actual
parameter update have formed, but before that update is applied to the receiver and
before any alpha-positive response, endpoint margin, support change, competitor switch,
boundary transition, or severe-conflict result exists.

Identifiers, absolute optimizer step, run identity, response section identity and all
future facts are prohibited model inputs.

## Frozen candidate coordinates

Each candidate is a single scalar with a fixed semantic definition. Candidate
selection may choose one scalar, but may not synthesize a classifier from several
candidates or add candidates after confirmation data are read.

1. `batch_target_advantage`: current batch frequency of the evaluation target minus
   current strongest competitor frequency, divided by batch size.
2. `exact_context_target_advantage`: among batch rows whose three input tokens exactly
   equal the evaluation input, target count minus competitor count, divided by matched
   count plus one.
3. `update_receiver_alignment`: cosine between the target-minus-competitor receiver
   embedding vector and the corresponding target-minus-competitor actual update.
4. `adam_receiver_alignment`: cosine between the same receiver embedding difference
   and the target-minus-competitor pre-update Adam first moment.
5. `action_support_alignment`: dot product between the four component update shares
   and the current four-component support allocation.
6. `action_support_velocity_alignment`: dot product between component update shares
   and the lag-one left finite difference of component support allocation.
7. `action_support_necessity_alignment`: dot product between component update shares
   and normalized current component necessity.
8. `preconditioned_support_velocity_alignment`: dot product between component shares
   of the formed post-preconditioned direction and the lag-one left finite difference
   of component support allocation.

All tensor identities, hashes and source paths used to form these scalars are retained
in a source ledger. Undefined cosine values remain missing and are imputed from
development runs only.

## Candidate selection

The known coordinate block is X3: F1/F3/F5, competitor identity/gap history, left
finite differences and native-update continuity. Robust scaling is fitted only on the
relevant training runs. To keep one new coordinate from disappearing inside a large
feature block, distance is block-balanced:

`d^2 = mean((x_i-x_j)^2) + (q_i-q_j)^2`.

For each candidate, severe-branch risk is the distance-weighted fraction among 64
cross-run neighbours. Development score is complete leave-one-development-run-out
PR-AUC. The candidate with the greatest development PR-AUC is sealed; ties use the
lexicographic candidate name. No candidate reroll is allowed after confirmation.

## Confirmation and downstream test

The sealed coordinate is evaluated on all four confirmation runs using only the eight
development runs as neighbours. Report:

- severe-branch ROC-AUC, PR-AUC and Brier for X3 and X3+q;
- per-confirmation-run results;
- response-curve RMSE, endpoint-direction accuracy and boundary accuracy for ordinary
  X3 KNN and X3+q KNN;
- severe-conflict subset response metrics;
- a diagnostic oracle that restricts neighbours to the true same severe/non-severe
  branch (never an executable result);
- cross-run nearest-normal matched pairs for every confirmation severe record.

## Adjudication

`SUPPORTED` requires all of:

1. confirmation PR-AUC of X3+q exceeds X3;
2. at least three of four confirmation runs have non-negative PR-AUC change where the
   metric is defined;
3. severe-subset curve RMSE improves by at least 3%;
4. overall boundary accuracy falls by no more than one percentage point;
5. independent recomputation, source hashes, GFG validation and no-leakage checks PASS.

`PARTIALLY_SUPPORTED` applies when the coordinate improves confirmation branch risk or
matched separation but does not meet the downstream response threshold.

Otherwise the result is `NOT_SUPPORTED`. Failure is preserved; it does not authorize
post-hoc candidate changes.
