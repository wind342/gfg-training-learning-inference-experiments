# Unified operational projection proof v2

This package reruns and integrates the frozen Database which-lineage, OpenTelemetry trace-shadow, and ordinary ECMA-426 Source Map experiments without changing Core or any source experiment artifact.

The only publishable entrypoint is:

```console
python -m experiments.operational_projection_proof_v2.scripts.run_all --full
```

The command verifies exact source heads and merge ancestry, preserves the v1 proof bytes, executes all P1/P2/P3 workloads twice, runs every required test suite twice, compares deterministic machine evidence, and emits a conjunctive status. Missing dependencies, count drift, skipped mandatory tests, authority leakage, new Core changes, or any mismatch fail closed.

The declared profiles are in `profiles/`; the preimplementation authority audit is in `audits/`; stable reports are in `artifacts/`. Transient complete runs remain under ignored `data_private/operational_projection_proof_v2`.
