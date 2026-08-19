# Formal result

## Outcome

The frozen formal experiment passed every preregistered scientific gate and
every independent evidence check.

| condition | relation treatment | final chain accuracy | learning-curve AUC |
|---|---|---:|---:|
| A | true consequence binding and true temporal credit | 1.0000 | 0.8516 |
| B | consequence components permuted across episodes | 0.3438 | 0.1954 |
| C | true episode consequences, action-stage credit swapped | 0.2448 | 0.1837 |

Condition A reached 100% final chain accuracy in all 12 formal seeds and
completely abandoned the old rule.  B and C reached the 90% chain threshold in
zero of 12 seeds.  A won the paired AUC comparison against B and C in 12/12
seeds.  The paired mean AUC differences were 0.6562 (95% t interval
0.6397--0.6726) and 0.6679 (0.6455--0.6903), respectively.

Condition A required a mean and median 10,880 episodes to reach the frozen 90%
threshold (range 6,400--14,720).

## Direct one-step causal isolation

The longitudinal arms necessarily diverge after their updates begin: different
policies produce different later actions and physical consequences.  The 48
preregistered one-step forks isolate the immediate causal question before that
divergence.  In every fork, A, B and C used the same receiving
parameter--optimizer state, cue batch, sampling uniforms, action ledger and
physical consequence ledger.

- changing only episode binding (A versus B) changed the actual parameter
  update and post-update policy logits in 48/48 forks;
- changing only temporal credit target (A versus C) changed the actual update
  and post-update logits in 48/48 forks.

This rules out the explanation that only the quantity or sign distribution of
reward caused the formal difference.  The relation between a consequence and
the action to which it is bound and credited changed the real training action.

## Machine evidence

All 12 seeds produced independently revalidated Core v3 snapshots and GFGs.
Together they contain 75,600 atomic generation facts, 270,000 nodes and 334,764
edges.  The independent checker passed frozen-contract hashes, initial clone
identity, common batches and random streams, reward-multiset invariants,
temporal-credit identities, parameter-state continuity, metric recomputation,
Core validation and GFG validation.

The raw formal execution occupies 338,156,176 bytes and the Core/GFG evidence
694,311,545 bytes on the E drive.  They are not committed to Git; their exact
locations and hashes are recorded in `ARTIFACT_MANIFEST.json`.

## Development result retained

The first development setting failed: phase-A policies were trained until
overconfident and adapted poorly after target reversal.  It is retained at the
recorded artifact path.  Development-only calibration raised the fixed entropy
coefficient and selected a shorter phase-A budget; all formal parameters were
then hashed before any formal seed was opened.  This failure is part of the
audit trail, not a discarded formal run.

## Supported claim and boundary

The executed evidence supports the following operational claim:

> In this delayed two-decision parameter--optimizer system, reinforcement
> learning is a feedback closure in which frozen inference produces actions,
> the environment produces delayed consequences, binding and temporal credit
> form the actual training action, and that action changes the formation state
> that governs future inference.

It does not show that every reinforcement-learning algorithm must use this
particular estimator, network or task representation, nor that one experiment
covers every reinforcement-learning system.  It does show that the proposed
closure is more than a renaming of reward: breaking either of its two relation
links while retaining the declared marginals changes the actual update and
prevents the same policy re-formation in the executed system.
