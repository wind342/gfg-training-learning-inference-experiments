# Gradient-dependency oracle preimplementation audit

Status: `PROCEED_WITH_FALSIFICATION_GATES`

This read-only audit was completed and committed before implementation of the
native gradient-dependency oracle. The stacked maintenance branch starts at the
frozen PR #17 head
`19eac2a1c5435b378a19c6b37d17a2d275cf794c`, repository tree
`2a2c8ddb595dcb725c5ee6d6c7e72596097b07d3`, and experiment tree
`4c1bb11d178ae09825ff91713094696f5865eba5`.

## The shared semantic assumption

The Core capture function in `core_capture.py`, lines 154-202, walks actual
operation receipts backward from the loss and derives local gradient-value
dependencies by branching on `tracked_matmul`, `tracked_mul`, `tracked_relu`,
`tracked_pow`, `tracked_sin`, and the declared zero-dependency operators. Its
full-file SHA-256 is
`0fee02b3346de642157fe8eb1d35b70b5e24823b2f39cc146045e6123e94161f`;
the LF-normalized function-region SHA-256 is
`acf64cabb8896b1ca22dee188edbf51bd099d76267ed1fb35181761fb2a6c461`.

The receipt reference function in `independent_reference.py`, lines 107-151,
performs a separate implementation of the same receipt traversal, checkpoint
rewrite, and semantically equivalent local reverse rules. Its full-file
SHA-256 is
`02440d378b9a8fd645cedde6f75c4d46dbdda5c43be4a54278381127e872d769`;
the LF-normalized function-region SHA-256 is
`a1bbfe3ce8b4d16d617c25614c06cabc72dd38abeee4d18a43ad76e26e151b7d`.

The code paths are isolated, but they can share the same theoretical mistake.
Consequently, the v1 receipt reference is a useful preservation baseline but
is not a fully independent semantic oracle for `gradient_value_dependency`.
This limitation does not by itself invalidate the exact Autograd projection,
checkpoint numerical divergence, output orthogonality, or zero-versus-unused
results. It limits the strength of the v1 gradient-lineage independence claim.

## Frozen public PyTorch surface

The oracle will run the frozen CPython 3.12.10 / PyTorch 2.13.0+cpu build
`cf30153c4c131c8164ee7798e5022d810682e2cb`. It may use only public graph and
hook surfaces: `Tensor.grad_fn`, `Node.name`, `Node.next_functions`,
`Node.register_prehook`, `Node.register_hook`, `saved_tensors_hooks`, and
tensor gradient hooks. The public contract permits pack hooks to return an
arbitrary packed object, requires unpack hooks to return an equivalent tensor,
and permits Node pre/post hooks to observe without replacing gradients when
they return `None`.

The native oracle is forbidden from using private `_saved_*` or
`_raw_saved_*` fields, operation-specific derivative tables, the two existing
local-rule helpers, persistent object IDs, or random intervention search.

## Process and authority boundaries

- Candidate/Core may read validated Core snapshots and frozen profiles, but
  may not read the native oracle, saved-tensor observations, interventions, or
  receipt reference.
- The native oracle may execute real PyTorch and use public graph/hooks plus
  stable source registrations, but may not read Core, Candidate, validated
  snapshots, bindings, artifacts containing Core answers, or old reference
  answers.
- Comparison receives only the two normalized relation sets and cannot mutate
  or reconstruct either side.
- In-process object identity may temporarily align a live tensor to a stable
  ref, but it may not be serialized, hashed, or reused across runs.

## Frozen v1 evidence and tests

The committed v1 evidence has status
`PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED`, 22/22 gates, 41 manifest
artifacts with zero mismatches, and normalized two-run scientific SHA-256
`270d8901041778ee2a60cb493cbf5b99591db032213c32a7e12462099d5b0e17`.
The audit independently reran the original suites: 34/34 experiment tests and
24/24 unchanged Core tests passed. The original 32 negative controls remain a
separate frozen accounting set.

The protected tree hashes are unchanged:

- `src/generation_relation_core`:
  `bf9d63597d72a8632dee69a1fbc61a2b2a42e4ce`
- `protocol/core_v3`:
  `0b4a2608864e771ebca7cdbfad95aabaed2d0723`
- `compat/v2`: `7bbb49d18daf7ea99d7633b40c6df5bc002824ca`
- `tests/core`: `5d02044752e8346ed942e90d272d488eadae9071`

Implementation may proceed only inside
`experiments/pytorch_autograd_training_lineage_v1/`. If public hooks cannot
support reliable observation, stable token mapping, and topology-preserving
single-token intervention, or if the independently observed native relation
set differs from Core, the hardening result must fail closed rather than fall
back to handwritten derivative rules.
