# Operational domain-projection proof framework

This experiment turns a domain “projection” claim into executable, falsifiable
checks. It does not claim a general mathematical theorem. For a declared
profile, the framework evaluates only these operational properties:

1. **P1 — Exact derivability:** a candidate produced only from a validated Core
   snapshot is canonically identical to an independent domain reference across
   every declared record and field.
2. **P2 — Strict partiality:** at least two controlled pairs have unequal
   complete Core facts but equal domain projections.
3. **P3 — Hierarchical consistency:** where a real hierarchy exists, direct and
   independently staged projections are canonically identical with
   multiplicity preserved.

## Trust boundary

`src/database_projection.py` is the database candidate path. It receives only a
`ValidatedSnapshot` and its matching validation token. It has no filesystem,
Oracle, DuckDB, ProvSQL, or native-result dependency. Its indexes are local,
rebuildable values and are never serialized as authority.

`src/database_reference.py` is the independent reference path. It adapts the
existing frozen hand-authored Oracle without importing the candidate or the
database executor. The two paths meet only in `projection_result.py`, which
performs exact, full-field comparison. Duplicate semantic keys fail before a
dictionary can overwrite them.

The frozen Core schema SHA-256 is checked against
`27c429695cffac8cea6cf52f2fd57e35fac3fe81bf251a5fd446f95d93bb4720`.
No database, OpenTelemetry, or Source Map authority table or field is added.

## Current evaluation boundary

- Database which-lineage P1: evaluated on the committed synthetic business and
  many-to-many workloads.
- Database which-lineage P2: evaluated with two pairs of actual, independently
  validated Core executions.
- Database→OpenTelemetry P3: `NOT_EVALUATED`; the OTel experiment is unfinished
  on the selected base commit.
- OpenTelemetry P1/P2 and Source Map P1/P2/composition: `NOT_EVALUATED`; missing
  prerequisites are recorded in their profiles and reports.

`NOT_EVALUATED` is not a pass. A missing mandatory database proof, isolation
failure, second authority, output mismatch, or failed negative control makes
the command exit nonzero.

## Reproduction

From the repository root with Python 3.12+ and the repository development
dependencies installed:

```console
python -m experiments.operational_projection_proof.scripts.run_all --full
python -m pytest experiments/operational_projection_proof/tests
```

Machine reports are written to `artifacts/`. Use `--artifacts-dir PATH` to
write a second isolated run for byte-level reproducibility comparison.

## Extension protocol

To activate a future domain, first complete and freeze its independent native
workload, reference result, dependencies, and tests. Then change the profile
from `NOT_EVALUATED` only after adding a candidate that reads solely from a
validated snapshot, an isolated reference adapter, full-field comparison,
two P2 counterexamples, applicable P3 paths, and fail-closed controls. A
conceptual field mapping alone must remain `CONCEPTUALLY_COMPATIBLE`, not a
projection proof.
