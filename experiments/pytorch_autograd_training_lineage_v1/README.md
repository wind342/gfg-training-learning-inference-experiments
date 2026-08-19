# PyTorch Autograd Projection and Bidirectional Training-Update Lineage v1

This directory is an independent, falsification-first experiment over one
frozen PyTorch CPU profile. It does not modify Core v3 and it does not use any
other experiment's scientific result as a premise.

Frozen claim scope:

> the frozen PyTorch Autograd dependency profile over the declared
> deterministic workloads

The preimplementation audit freezes the repository base, protected trees,
official PyTorch wheel and documentation authority, supported public APIs,
declared workloads, and Core-to-Autograd crosswalk before candidate
implementation begins.

The local wheel, documentation snapshots, and virtual environment are stored
outside tracked experiment output. No downloaded binary or absolute local path
is committed.

## Reproduction

Use CPython 3.12 with the wheel frozen in
`artifacts/pytorch_authority_manifest.json`, install the repository development
dependencies, and run:

```console
python -m experiments.pytorch_autograd_training_lineage_v1.materialize_evidence
python -m experiments.pytorch_autograd_training_lineage_v1.materialize_gradient_oracle_evidence
python -m pytest experiments/pytorch_autograd_training_lineage_v1/tests -q
python -m pytest tests/core -q
```

The original materializer preserves the frozen v1 result. The gradient-oracle
materializer then executes two complete hardening runs, including real native
backward hooks, saved-tensor and registered-source interventions, v2 lineage,
both negative-control suites, and both test suites. It writes deterministic
JSON evidence and independently rehashes the final manifest. The v1 result
remains `PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_SUPPORTED` with 22/22 gates; the
hardened result is
`PYTORCH_AUTOGRAD_GENERATION_FACTS_V1_EVIDENCE_HARDENING_SUPPORTED` with 20/20
additional gates.
