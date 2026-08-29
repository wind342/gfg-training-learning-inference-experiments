# Existing nanoGPT audit for functional abstraction

## Scope and frozen evidence

This is a fact audit only. It does not train a model, create a task, run a new
inference intervention, or test the functional-abstraction hypothesis. The
audit reads the task generator, the frozen source GFGs, the validated CSRG-4C
bundles and the INF-E01 projection GFGs associated with frozen repository
commit `36dab5ce347dbbdac157ef23205f556606d18294`.

The audited code paths are:

- `experiments/gfg_nanogpt_autonomous_capability_discovery_v1/task_generator.py`;
- `experiments/gfg_nanogpt_support_redundancy_v1/`, including the cyclic
  mapping recovery in `task_mapping.py`;
- `experiments/gfg_nanogpt_training_learning_inference_projection_v1/`;
- the 13 source bundles linked by `source_bundle_id` from the INF-E01 GFGs.

The audit recovered each validation input from the source-GFG origin with role
`layer_input` and semantic position
`step:100:stage:evaluation_validation:module:token_embedding:input:0`. Training
inputs and targets were recovered from the step-0 `training_batch_inputs` and
`training_batch_targets` origins. The opaque cyclic mapping was then recovered
from those training facts by the already validated rank-22/nullity-1 procedure.

## Audit verdict

`RELATION_STRUCTURE_NOT_IDENTIFIABLE_IN_EXISTING_TASK`

The existing data support a narrower, no-new-training analysis of causal
support convergence across different operand pairs that produce the same
target token. They do **not** identify the requested abstraction claim because
the task contains only one operation-level relation, while its 23 registered
groups are correct-output-token groups. Relation identity and output-token
identity therefore cannot be varied independently.

This is an identifiability verdict, not negative evidence about functional
abstraction.

## 1. Exact task

Each run instantiates a randomly token-relabelled cyclic-addition task over the
prime field of order 23. If `pi` is the run-specific permutation of the 23
operand/result tokens, every concrete input is the three-token sequence

```text
[pi(a), 23, pi(b)]
```

for an ordered pair `(a,b)` in `Z_23 x Z_23`. Token `23` is the single fixed
operator token. The correct output is

```text
pi((a + b) mod 23)
```

and appears at the final target position; the other two target positions are
ignored. Thus the task has 529 possible ordered concrete inputs, one underlying
operation (`+ mod 23`), 23 possible outputs and no natural-language semantics.

The generator shuffles the 529 ordered pairs and assigns 317 to training and
212 to validation. Each row has a content-derived sample identity. No two
validation rows within a run are the same ordered token triple.

## 2. Organization of the 212 validation examples

Every audited run has exactly 212 unique validation rows and 317 training rows.
The 13 runs have 13 distinct validation-input hashes and 13 distinct derived-
target hashes, so the rows are not one shared 212-row array copied across all
runs. They are separately materialized run-specific held-out splits with
run-specific opaque token relabellings.

Within each run, the 212 rows cover concrete ordered operand pairs. The exact
correct target is recovered without reading validation labels by solving the
cyclic operation from the 317 observed training facts. The recovered target
table is invariant under every non-zero field-scalar automorphism, which is why
the opaque target completion is unique even though the selected latent residue
basis is not.

Across the 13 runs, every target token is represented. Per-run target-group
sizes range from 4 to 16 examples; the first linked run has 6 to 13 examples per
target group.

## 3. Meaning of the 23 target groups

The code defines the group label as the correct target token itself:

```text
target_group(example) = correct opaque output token in {0,...,22}
```

After recovering a latent cyclic coordinate system, this is equivalently the
result residue class `(a+b) mod 23`, up to the run-specific token permutation
and an irrelevant non-zero field-scalar automorphism.

The 23 groups are therefore **not 23 different underlying relations**. They are
23 output/readout classes under one shared modular-addition relation. Existing
CSRG group summaries (`target_group_q10_margin`, component necessity, backup,
slack and effective support) are organized by this output-token identity.

## 4. Does a natural relation x concrete-instance decomposition exist?

Two levels must be distinguished.

1. **Operation-level relation.** All 212 rows instantiate the same relation,
   cyclic addition modulo 23, using different ordered operand pairs. This gives
   many concrete realizations of one relation, but no different-relation
   comparison.
2. **Registered target-group level.** Every target group contains multiple
   concrete operand pairs producing the same output token. This supplies a
   target-conditioned instance decomposition, but target-group identity is
   exactly output-token identity. It cannot show that support is organized by
   relation rather than final output token.

Consequently, the existing task does not contain the factorial variation
needed to distinguish underlying relation, concrete input and output token.

## 5. Available pairs and controls

### Available

- **Same operation / different concrete instance:** available in abundance;
  all distinct validation rows within a run instantiate the same modular-
  addition operation.
- **Same target token / different operand pair:** available in every target
  group. Summed across groups, there are 910 to 966 unordered within-target
  pairs per run.
- **Commuted concrete pairs:** 30 to 45 held-out pairs per run have both
  `[pi(a),23,pi(b)]` and `[pi(b),23,pi(a)]` present. They share the operation and
  output while changing input order.
- **Different target / partially matched concrete structure:** 1,827 to 1,891
  unordered pairs per run have different target tokens while sharing the left
  operand or the right operand. These are valid syntactic/input controls.
- Baseline margin, target frequency and exact row identity can be matched from
  already materialized data.

### Not available

- **Different underlying relation / similar concrete structure:** unavailable,
  because only one underlying operation is present. Sharing an operand across
  different target groups changes the output class, not the operation.
- **Different underlying relation / same output token:** unavailable. A fixed
  output token identifies one target/result group, and there is no second
  relation producing that token under an independently recorded relation
  identity.
- **Same underlying relation with output token held non-identical as a control
  against relation identity:** not an independently identified comparison;
  all rows share the operation, so it cannot establish relation selectivity.

The within-target pairs can test output-conditioned causal convergence. They
cannot by themselves test whether convergence follows an abstract relation
rather than the target/readout class.

## 6. Existing outputs available for a per-example causal fingerprint

### CSRG-4C

For every materialized checkpoint in each of the 13 runs, the validated support
bundles retain complete row-order outputs for:

- two baseline inference repeats;
- four single-component gates (`h0.attn`, `h0.mlp`, `h1.attn`, `h1.mlp`);
- all six pair-component gates;
- complete logits with shape `212 x 24`;
- predictions and per-example true-target margins.

The existing raw row-order tensors are sufficient to derive, without any new
model execution, an example-level fingerprint containing the baseline margin,
four single-gate margin changes, six pair-gate margin changes, prediction
transitions and six pairwise non-additive interaction terms. The currently
stored necessity, backup, slack and effective-support tensors are 23-group
derivations; they must not be mistaken for already materialized example-level
quantities. Their example-level analogues can be calculated exactly from the
retained single/pair logits if later authorized.

### INF-E01

INF-E01 contains 13 runs x 4 mechanically selected historical phases = 52
phase observations: `pre_formation`, `formed`, `decline` and `recovered`. Each
phase retains the baseline, four single gates and six pair gates as complete
`212 x 24` logits, plus the actual component call inputs/outputs. The formed
phase also retains four pre-formation component-version rollback logits and
four restored-formed-version logits.

Thus the existing projection GFGs can support a no-new-training,
phase-conditioned, per-example causal fingerprint. Activation similarity is
not required. However, these richer causal measurements do not repair the
missing independent relation labels.

## 7. Can FA-E01 be completed with no new training?

**Not with the requested claim and controls.**

A narrower experiment can be completed entirely from the frozen evidence:
measure whether different operand pairs sharing a target token have more
similar causal gate fingerprints than matched different-target pairs, and how
that pattern changes across the four historical phases. Such a result would be
evidence about target/output-conditioned functional convergence.

Calling that result relation-level functional abstraction would be invalid
because the design cannot separate relation identity from output-token/readout
identity. The authorized first-phase verdict therefore remains
`RELATION_STRUCTURE_NOT_IDENTIFIABLE_IN_EXISTING_TASK`, and no FA-E01 execution
is started by this audit.

## 8. Unique missing condition

The unique necessary missing condition is an independently identifiable
`relation x concrete instance x output token` design: at least two underlying
relations, multiple concrete instances per relation, and output-token overlap
across relations so that relation identity can be compared while output token
is held fixed. Similar concrete/input structures must also occur across those
relations.

Until that condition is present in validated examples, causal fingerprints can
test convergence by input or output group, but cannot adjudicate the proposed
relation-level abstraction claim.
