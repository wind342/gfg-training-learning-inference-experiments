# Real order-refund-freeze inter-fact relations v1

This experiment executes four controlled order workflows with a real SQLite
WAL database, separate Python worker processes, `multiprocessing.Queue`,
`Barrier`, and `Event`. Transaction and message outcomes come from actual SQL
rowcounts, commits/rollbacks, and queue operations.

Atomic facts retain `f=(u,tau,omega_bar,z;rho)`. Result-level relations live in
the external sidecar `H_e=(Gamma_e,R_e)`.

The experiment is scoped to the declared controlled execution profile. It
does not replace SQLite transactions or OpenTelemetry and does not establish
general concurrency or global scheduler completeness.

## Reproduce

From the repository root:

```console
python -m experiments.order_refund_freeze_inter_fact_relations_v1.scripts.run_all
```

The command performs two complete 40-workflow scientific materializations,
executes all 30 negative controls once, runs experiment/Core/full-repository
tests, and writes canonical machine evidence to `artifacts/`.

Primary review files:

- `PREIMPLEMENTATION_AUDIT.md`
- `DESIGN.md`
- `ORDER_REFUND_FREEZE_REPORT.md`
- `artifacts/experiment_summary.json`
- `artifacts/query_comparison.json`
- `artifacts/artifact_manifest.json`
