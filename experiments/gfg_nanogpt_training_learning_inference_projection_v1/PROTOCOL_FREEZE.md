# Training-learning-inference support projection protocol v1

## Claim under test

Within the executed opaque cyclic-task nanoGPT family, training establishes
parameter-versioned distributed functional support.  A frozen inference does
not copy the learning state: for a concrete query it executes a new occurrence
that calls and combines a query-conditioned projection of that established
support.

The operational form is

\[
S^{\mathrm{active}}_{q,c}=\Pi_{q,c}(L^*),\qquad
y=H_{q,c}(S^{\mathrm{active}}_{q,c}).
\]

This experiment does not claim that all reasoning is a literal static
projection, that attention weights alone are support, or that the result is a
universal law for every trained model.

## Frozen source family

- all 13 validated native nanoGPT training GFG bundles in the frozen research
  archive;
- the corresponding validated CSRG-4C-v1 support bundles;
- nanoGPT commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`;
- held-out validation rows, which are disjoint from the training-row split;
- components `h0.attn`, `h0.mlp`, `h1.attn`, and `h1.mlp` at the registered
  output-after-projection/before-residual-addition gate site.

No new training is run.  Every inference is a real CUDA forward of an exact
historical parameter checkpoint.

## Mechanical phase selection

For every run, using only the already validated 100-step evaluation series:

1. `pre_formation`: the checkpoint immediately before the first checkpoint
   with validation accuracy at least 0.90;
2. `formed`: that first checkpoint with accuracy at least 0.90;
3. `decline`: the right endpoint of the largest adjacent accuracy decrease
   after formation;
4. `recovered`: the first later checkpoint with accuracy at least 0.90, or the
   final checkpoint labelled `post_decline_not_recovered` if none exists.

The resulting selection is sealed before any new forward or rollback result is
read.

## Native inference capture

At each selected checkpoint:

- execute two ordinary frozen-model forwards and require byte-identical logits;
- synchronously capture the actual input and output tensor of each registered
  component call;
- execute all four single-component zero gates;
- execute all six pair-component zero gates;
- retain complete logits, predictions, per-example margins, group summaries,
  component parameter-version identities, and native call order.

## Version rollback control

At the `formed` checkpoint, replace exactly one registered component at a time
with that same component's `pre_formation` parameter versions.  All other
parameters remain at the formed checkpoint.  Execute a real forward, restore
the formed component bytes, and execute again.  Restoration must recover every
parameter hash and the complete baseline logits exactly.

The hybrid rollback state is a declared causal intervention, not a naturally
visited training checkpoint.

## Frozen adjudication

The four primary gates are:

1. **identity continuity**: every parameter read by inference matches an exact
   parameter-version object in the source training GFG;
2. **causal call**: every run has at least one registered component whose gate
   changes complete inference logits, and every captured component call has a
   nonzero output;
3. **query conditioning and combination**: every run has at least two distinct
   23-group support-effect profiles and at least one component pair/group with
   absolute non-additive interaction greater than `1e-6`;
4. **learned-version dependence**: in every run at least one pre-formation
   component rollback changes complete formed-checkpoint logits, and every
   restoration is byte exact.

Accuracy need not decrease for every component rollback.  All zero, positive,
and negative effects are reported.  No failed run is removed.

