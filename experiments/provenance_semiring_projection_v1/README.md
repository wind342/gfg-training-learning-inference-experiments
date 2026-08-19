# Provenance Semiring Projection v1

This experiment evaluates one bounded claim:

```text
complete Core v3 generation facts Gamma_G(omega)
  -- exact strict projection --> canonical N[X]
  -- exact algebraic projections --> {bag N, Boolean B, PosBool(X)}
  -- exact task projections --> {flat source support, Vars(N[X]), existing Database which-lineage}
```

The claim is restricted to the frozen finite positive relational-algebra profile in `profiles/positive_relational_algebra_profile_v1.json`. Difference, negation, antijoin, aggregation, recursion, arbitrary SQL, probabilistic evaluation, universal query equivalence, and claims about all database provenance are excluded.

## Independent paths

- `scripts/native_nx_path.py` directly evaluates the frozen RA AST using an independent native polynomial oracle and does not read Core, Candidate data, `NXPolynomial`, or Candidate's variable helper.
- `scripts/core_capture_path.py` observes real operator emissions and materializes unmodified Core v3 facts.
- `scripts/candidate_nx_path.py` validates a Snapshot and projects it using only Core facts plus the frozen profile/crosswalk.
- `scripts/compare_nx_paths.py` reads only the two canonical outputs and never repairs either side.

## Reproduction

From the repository root with Python 3.12+:

```console
python -m experiments.provenance_semiring_projection_v1.scripts.run_p1_exact --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_p2_strictness --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_formal_semantics_audit --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_native_oracle_selftest --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_p3_hierarchy --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_database_which_bridge --repo-root . --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_lower_strictness --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_v1_preservation --repo-root . --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_negative_controls --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_two_scientific_executions --repo-root . --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m experiments.provenance_semiring_projection_v1.scripts.run_isolation_audit --repo-root . --artifact-root experiments/provenance_semiring_projection_v1/artifacts
python -m pytest experiments/provenance_semiring_projection_v1/tests -q
python -m experiments.provenance_semiring_projection_v1.scripts.run_report_statistics --artifact-root experiments/provenance_semiring_projection_v1/artifacts --report experiments/provenance_semiring_projection_v1/EXPERIMENT_REPORT.md
python -m experiments.provenance_semiring_projection_v1.scripts.finalize_artifacts --repo-root . --experiment-root experiments/provenance_semiring_projection_v1
```

The frozen authority PDF and its retrieval audit are under `audits/`. `EXPERIMENT_REPORT.md` answers the required scientific questions. `artifacts/final_status.json` is the machine-readable gate decision, and `artifacts/artifact_manifest.json` rehashes every experiment file except its own self-referential entry.
