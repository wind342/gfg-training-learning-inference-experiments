# Database-lineage replacement-principle experiment

This is a falsification-oriented test of one narrow claim: whether Core v3's modality-independent generation facts can capture tuple-level which-lineage for fixed deterministic relational operators without a database-specific Core relation schema.

Recorded conclusions:

- principle: `SUPPORTED_IN_TESTED_SCOPE`;
- scale: `SCALABILITY_DEMONSTRATED_TO_462399_VALIDATED_BINDINGS`;
- original one-label protocol: `PARTIALLY_SUPPORTED`, solely because mandatory SF0.1/Q1 was not run after its projected 21.5 GB memory requirement exceeded the available 17 GB hardware.

The fixed manual Oracle and four SF0.01 queries passed. ProvSQL v1.4.0 is fully pinned, but its external comparison is recorded as unavailable because Docker Desktop requires a host restart after enabling WSL/VirtualMachinePlatform. No ProvSQL agreement is claimed.

## Tested scope

- Bag-preserving selection, projection, and derived columns.
- Inner equi-join, including one-to-many and many-to-many identity cases.
- Grouped/scalar `SUM`, `COUNT`, and exact `AVG` sum/count calculation.
- Deterministic sort/limit with explicit exclusions.
- Direct and transitive forward/backward tuple lineage.
- Relation-material, evidence, successful-operation, and Snapshot validation for every binding.
- Contract-on/off byte-identical CSV and JSON.
- DuckDB-generated SF0.01 TPC-H Q1, Q3, Q6, and Q10 fixed plans.

This is a research workload based on the TPC-H schema, generator, and fixed queries; it is not an audited TPC-H benchmark.

## Trust boundary

`relational_executor.py`, `operators.py`, `core_adapter.py`, and `core_lineage_reader.py` form the tested path. They do not import the manual Oracle, DuckDB evaluator, ProvSQL evaluator, or canonical expected lineage. Relations are captured when an operator creates a support or disposition. Intermediate supports enter the next stage only through Core's existing `GeneratedOrigin` bridge. Final lineage is read only from validated Snapshot facts.

The reader's indexes and Core evidence resolver's indexes are transient and rebuildable from authoritative Snapshot/Core tables. They contain no SQL, tuple, or lineage-specific Core field and are never authoritative storage. Frozen scan implementations exist only in the experiment evaluator for exact before/after comparison.

## Reproduction

Full run:

```sh
python -m experiments.database_lineage.scripts.run_all --full
```

Fast CI run:

```sh
python -m experiments.database_lineage.scripts.run_all --fast
```

Linux/CI convenience targets are `make database-lineage-experiment` and `make database-lineage-fast`.

Generated databases, CSVs, complete forward maps, and run-local snapshots live under ignored `runtime/` paths. Row counts and hashes are retained in committed artifacts. Dependencies are pinned in `requirements.lock`; ProvSQL is pinned by release, commit, platform manifest, and image-index digest.

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the executed/not-run matrix, raw counts, validation design, performance, limitations, and claim audit. JSON artifacts preserve numerators, denominators, hashes, failure types, timings, and the resource-stop record.
