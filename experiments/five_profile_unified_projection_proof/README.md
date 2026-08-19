# Unified five-profile projection proof

This package replays five existing frozen proofs from one fail-closed entrypoint:

1. Database which-lineage;
2. ECMA-426 ordinary Source Map;
3. OpenTelemetry occurrence trace;
4. W3C PROV generation profile;
5. PyTorch Autograd dependency profile, including the hardened 29-relation
   gradient-dependency oracle.

It does not redefine a profile, crosswalk, fixture, candidate, native/reference
implementation, expected count, or P2 witness. Each mechanism runs in its own
subprocess and returns a structured result. The orchestrator never supplies a
candidate result to a native path or vice versa, and it never treats an old
unified artifact as a current execution.

## One command

After preparing the environment described in `ENVIRONMENT.md`, run:

```powershell
.venv\Scripts\python.exe -m experiments.five_profile_unified_projection_proof.scripts.run_all
```

The command runs all five mechanisms and all required test suites twice. Any
missing dependency, mechanism, result field, P1 comparison, P2 witness,
artifact, test, deterministic hash, or zero-Core-change gate makes the command
return nonzero. `SKIP`, `BLOCKED`, `UNAVAILABLE`, and `NOT_INSTALLED` cannot be
successful mechanism states.

Generated evidence is under `artifacts/`:

- `five_profile_summary.json`: unified machine conclusion;
- `unified_manifest.json`: source lineage, environment, profile/crosswalk and
  artifact hashes;
- `determinism.json`: canonical hashes of both complete runs;
- `test_results.json`: two structured test executions;
- `runs/run_{1,2}/`: per-mechanism structured results and canonical summaries.

## Scope

The supported statement is limited to the five explicitly frozen profiles and
workloads. It is not coverage of whole standards or arbitrary programs, does
not assert that Core is a unique/minimal ontology, does not assert that a
crosswalk is unique, and does not upgrade the five different external
independence ratings.

The Database ProvSQL 1.4.0 attempt is reported separately as external
validation and is not used as a premise for the constructive frozen-profile
P1/P2 result.

