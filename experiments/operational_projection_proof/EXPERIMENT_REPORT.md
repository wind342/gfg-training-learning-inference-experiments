# Operational domain-projection proof v1

## Verdict

The framework constructively establishes two operational properties for one
declared domain profile:

- Database which-lineage P1, exact derivability: **`SUPPORTED`**.
- Database which-lineage P2, strict partiality: **`SUPPORTED`**.

OpenTelemetry P1/P2, Source Map P1/P2, Core→database→OpenTelemetry P3, and
Source Map composition P3 are **`NOT_EVALUATED`**. They are not counted as
passes, partial passes, or evidence for a paper claim.

The supported scope is the fixed deterministic bag-preserving relational
operators and committed synthetic business/many-to-many workloads. This report
does not claim all SQL, all database provenance, a complete DBMS replacement,
or a universal theorem about domains.

## Operational definitions

Let the complete validated generation facts be `Γ_G(ω)`, a declared domain
projection program be `Π_D`, and an independent domain result be `L_D(ω)`.

**P1 — Exact derivability** is accepted only when two isolated executions
produce canonically identical declared records:

```text
Core generation → ValidatedSnapshot → candidate Π_D
frozen workload → independent Oracle/reference
candidate bytes/records/fields == reference bytes/records/fields
```

**P2 — Strict partiality** is accepted only when at least two controlled cases
have `Γ_1 != Γ_2` and `Π_D(Γ_1) == Π_D(Γ_2)`. Projection equality does not imply
complete generation-fact equality.

**P3 — Hierarchical consistency** is evaluated only when a legitimate wider to
narrower domain path exists. Direct and hierarchical projections must preserve
the same record multiset. Cycles and mismatches fail closed. No arbitrary
domain hierarchy is invented here.

## P1 database result

The candidate was derived only from validated Core tables. The reference was
adapted from the existing hand-authored Oracle. Exact comparison covered:

| Section | Candidate | Reference | FP | FN | Field mismatch | Multiplicity mismatch |
|---|---:|---:|---:|---:|---:|---:|
| Direct relations | 62 | 62 | 0 | 0 | 0 | 0 |
| Backward lineage | 1 | 1 | 0 | 0 | 0 | 0 |
| Forward lineage | 19 | 19 | 0 | 0 | 0 | 0 |
| Derivation paths | 20 | 20 | 0 | 0 | 0 | 0 |
| Explicit dispositions | 7 | 7 | 0 | 0 | 0 | 0 |
| Multiplicity summaries | 2 | 2 | 0 | 0 | 0 | 0 |
| Duplicate-identity cases | 1 | 1 | 0 | 0 | 0 | 0 |

Total canonical records were 112/112. Candidate execution remained identical
after the serialized reference result was deleted and while both Oracle modules
were replaced by runtime traps.

## P2 database result

Two independently constructed Snapshot pairs passed:

1. equal workload and lineage with distinct execution run identity;
2. equal workload and lineage with distinct declared environment context.

For both pairs, the domain projections were equal while complete snapshots,
binding sets, and transformation contexts were unequal. This establishes that
the tested which-lineage profile is a strict narrow projection of the complete
generation facts. It is a scope result, not a defect in database lineage.

## Authority and isolation audits

- Frozen Core schema hash matched exactly.
- Domain-specific second-authority store count: 0.
- Candidate Oracle import count: 0.
- Candidate native-result read count: 0.
- Contract-on/off CSV and JSON bytes were identical.
- Control-plane fields in business output: 0.
- All 13 negative controls returned their exact expected reason code.

Negative controls cover extra/missing relations, field and multiplicity drift,
duplicate-key overwrite, Oracle/native-result leakage, second authority,
output contamination, direct/hierarchical disagreement, invalid inference from
projection equality, status escalation, and hierarchy cycles.

## Deferred domains

The OpenTelemetry work observed outside this branch was untracked and still
changing during the initial audit; it had no completed frozen artifact set on
the selected base. The Source Map branch contained no domain implementation.
This branch deliberately does not copy either work-in-progress tree. Their
profiles and machine reports state `NOT_EVALUATED`, and their tests use explicit
pytest skips with the missing prerequisite as the reason.

## Reproducibility identity

- Base commit: `03caa31b8a6abfe6e112a0544071618c689bb11f`
- Branch: `theory/operational-projection-proof-v1`
- Core schema SHA-256:
  `27c429695cffac8cea6cf52f2fd57e35fac3fe81bf251a5fd446f95d93bb4720`
- Full command:
  `python -m experiments.operational_projection_proof.scripts.run_all --full`
- Tests:
  `python -m pytest experiments/operational_projection_proof/tests`

Exact runtime and dependency versions are in `artifacts/environment.json`.
Every report hash and every framework/profile/test/document hash is recorded in
`artifacts/artifact_manifest.json`.

## Paper-safe statement

Within the declared deterministic database which-lineage profile, independent
implementation, exact semantic comparison, controlled counterexamples, and
fail-closed negative controls constructively establish exact derivability and
strict partiality. Hierarchical consistency and the OpenTelemetry/Source Map
profiles remain unevaluated until their independent native experiments are
complete.
