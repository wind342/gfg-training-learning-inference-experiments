# GFG reinforcement-learning feedback-closure protocol v1

## Claim under test

Within the executed delayed two-decision parameter--optimizer system,
reinforcement learning is tested as a feedback closure:

\[
Z_t \xrightarrow{\text{frozen inference}} a_t
\xrightarrow{\text{environment consequence}} r_{t+1}
\xrightarrow{\text{binding and temporal credit}} U_t
\xrightarrow{\text{training formation}} Z_{t+1}.
\]

This experiment does not attempt to derive a unique reinforcement-learning
algorithm or generalize one execution to every reinforcement-learning system.
It tests whether correct action--consequence binding and temporal credit are
causally necessary for the expected policy re-formation in this system.

## Native task and event order

Every episode contains a four-bit cue and two sequential binary actions.  The
environment emits no reward after the first action.  Only after the second
action does it emit two signed consequence components, one for each decision.
The two phase-A targets are fixed Boolean functions of disjoint cue pairs.
Phase B complements both targets.

The frozen native event order is:

1. construct the pre-update parameter--optimizer receiving state;
2. execute decision-1 inference and sample action 1;
3. execute decision-2 inference, conditioned on action 1, and sample action 2;
4. emit both physical consequence components at the terminal event;
5. establish consequence-to-episode binding records;
6. establish condition-specific credit-to-action records;
7. form the scalar training objective, gradients and AdamW update;
8. establish the new parameter version;
9. evaluate future frozen inference on all cue identities.

Transient GRU state may evolve inside one frozen inference episode.  Frozen
inference means that parameter and optimizer state do not persistently change.

## Three causal conditions

All three conditions begin from byte-identical model and optimizer state and
receive the same cue batches and action-sampling uniforms.

- **A -- correct closure:** each physical consequence component remains bound
  to its actual episode and is credited to the action that produced it.
- **B -- reward binding permutation:** for each decision stage separately, the
  consequence records are permuted across episodes.  The component multiset,
  training budget and update count are unchanged.  Credit is then ordinary
  relative to this false episode binding.
- **C -- temporal credit swap:** physical consequence records remain bound to
  their actual episodes, but stage-1 consequence is credited to action 2 and
  stage-2 consequence to action 1.

The implementation must preserve two different relation types:
`produced_consequence` and `credited_to_action`.  A reward value alone is not
accepted as evidence of either relation.

## One-step causal forks

At the frozen update indices, the same receiving state, cue batch and random
uniforms are copied into A, B and C transformations.  Their actual gradient and
parameter-update hashes and immediate future policy displacement are recorded.
Fork states are discarded and never re-enter the main trajectories.

## Capture protocol

The declared unit is one real minibatch update.  Synchronous receipts contain
the complete per-episode cue, action, consequence-binding and credit ledger,
plus pre-state, loss, gradient, optimizer update, post-state and frozen policy
evaluation hashes.  Core v3 bindings establish atomic facts for each stage of
the update; typed GFG edges preserve program order, physical consequence
production, episode binding, temporal credit, optimizer application and
parameter-version continuity.

No reward, consequence or post-update result is permitted in the inputs of the
earlier inference occurrences.

## Primary adjudication

The main outcome is phase-B chain accuracy across the complete cue space.
Secondary outcomes are stage accuracies, old-rule retention, area under the
adaptation curve, episodes to the frozen threshold, per-stage action-logit
displacement and actual update identity.  The causal gates are fixed in
`EXPERIMENT_CONTRACT.json`.

The preregistered expectations are qualitative and directional:

- A should form the reversed policy;
- B should not form the same policy from the same reward multiset;
- C should not form the same policy when temporal credit targets are wrong;
- changing only binding or credit at a one-step fork should change the actual
  update and its immediate policy effect.

All seeds are retained.  Failure of a scientific gate is a scientific result,
not a platform failure.

## Evidence and replay

The raw ledgers and evidence bundles are written outside the Git repository on
the E drive.  The repository retains code, frozen hashes and compact aggregate
results.  An independent checker recomputes metrics from the raw event ledger,
verifies reward-multiset invariants, initial clone equality, Core snapshot and
GFG structure, and rejects any missing or mismatched identity chain.
