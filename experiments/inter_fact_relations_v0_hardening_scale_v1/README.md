# Inter-fact relations hardening and scale v1

This append-only experiment hardens
`experiments/inter_fact_relations_v0/` without modifying or copying it. It
tests whether an external relation sidecar can:

- validate primitive evidence by receipt content and endpoint semantics;
- require exact adjacent program-order edge sets across occurrences,
  receipts, relations, and evidence;
- derive controlled-scope capture completeness from a declared contract plus
  measured audit;
- exercise a controlled versioned reads-from fixture;
- preserve fact-specific endpoints for multi-fact occurrences;
- distinguish semantic, Core content, concrete run, and sidecar identities;
- answer large sparse-DAG queries without materializing global closure;
- match a receipt-based reference running in a separate process.

It does not extend the Theory of Generation Facts or add a Claim Atlas claim.
The atomic fact remains `f=(u,tau,omega_bar,z;rho)`.

`CAPTURE_COMPLETE` means complete only for the declared controlled executor
profile and the audited controlled capture scope. It is not a machine proof
of global scheduler completeness for an operating system, thread library, or
distributed system. Concurrency conclusions are limited to
`CONTROLLED_CAPTURE_SCOPE_ONLY`.

## Reproduce

From the repository root:

```console
python -m experiments.inter_fact_relations_v0_hardening_scale_v1.scripts.run_all
```

The command performs two complete scientific runs, runs the experiment,
frozen Core, and full-repository test suites, then writes canonical machine
evidence to `artifacts/`.

The optional 50,000-occurrence scale is resource guarded. A guarded
non-execution is recorded explicitly and is never replaced by a smaller
workload.

## Primary files

- `PREIMPLEMENTATION_AUDIT.md`: frozen audit committed before implementation.
- `DESIGN.md`: candidate/reference boundary and lifting policy.
- `HARDENING_SCALE_REPORT.md`: conclusions and limitations.
- `artifacts/experiment_summary.json`: compact machine verdict.
- `artifacts/artifact_manifest.json`: independently rehashed artifact index.
