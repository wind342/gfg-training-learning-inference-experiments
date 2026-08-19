# Preimplementation audit

Status: `PROCEED`

The audit was completed before candidate or training-lineage implementation.
The branch was created from the fetched `origin/main` head
`e00144b6b47504287c2d16f20b064da81e43f1cc`, the smallest public base that
contains frozen Core v3 without importing another experiment's scientific
result.

## Authority and environment

- CPython 3.12.10 on 64-bit Windows 11 build 26200.
- Official CPU wheel `torch-2.13.0+cpu-cp312-cp312-win_amd64.whl`.
- Wheel size: 121,933,498 bytes.
- Wheel SHA-256:
  `a8b450c1e58e5800e5b4691dac412f8d2d65a1dc3298166f91596603a3531e6f`.
- The digest equals the hash fragment published by the official PyTorch CPU
  wheel index.
- Installed version: `2.13.0+cpu`; build commit:
  `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Build reports MKL, MKL-DNN, OpenMP, LAPACK, AVX2, and `USE_CUDA=0`.
- Versioned 2.13 documentation pages and the wheel index were downloaded to an
  ignored local directory and hashed; their locators and digests are frozen in
  `artifacts/pytorch_authority_manifest.json`.

## Public surface and feasibility

The installed public API exposes `Tensor.grad_fn`, `Node.name`,
`Node.next_functions`, tensor gradient hooks, non-reentrant checkpoint
`context_fn`, `determinism_check`, and `early_stop`.

An actual preimplementation checkpoint probe used separate original and
recomputation context managers. The function ran once during original forward
with external scale 1 and once during backward recomputation with a replacement
scale tensor of value 2. Backward completed, the default determinism check did
not raise, and all gradients were finite. This removes the checkpoint
feasibility blocker without asserting the final experiment result.

Actual eager-mode graphs over the declared standard operators exposed the
frozen node names in the crosswalk. The probes also observed ordered edge
slots, a real `None` edge, repeated edges to one shared node, and distinct leaf
accumulators.

## Core boundary

Existing Core v3 can represent sources, concrete occurrences, outcomes,
precise bindings, generated-origin stage bridges, evidence, environment, and
explicit dispositions. Parameter versions will be distinct semantic support
records even when the underlying Python `Parameter` object is unchanged.

The candidate will receive only a `ValidatedSnapshot`, matching
`SnapshotValidation`, the frozen profile, the frozen crosswalk, and the
canonicalizer. It will not receive native graph objects, execution receipts, or
expected results.

No blocking condition is present. This audit authorizes implementation only
inside `experiments/pytorch_autograd_training_lineage_v1/`.
