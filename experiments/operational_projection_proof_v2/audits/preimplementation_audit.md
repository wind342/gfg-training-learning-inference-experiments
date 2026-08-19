# Unified Operational Projection Proof v2 — Preimplementation Audit

This audit was completed before any v2 proof code was written. It records the two required reading passes: first the scientific structure, then the authority and isolation boundaries.

## Frozen history and preservation

All fetched remote heads exactly match the task contract:

| Source | Exact commit |
|---|---|
| Database lineage | `03caa31b8a6abfe6e112a0544071618c689bb11f` |
| OpenTelemetry projection | `25a9d2a614d2d34d36c38f7c560b818cdbc4b179` |
| Source Map projection | `7dba987713da345453781e4b95130f1deb5f04d4` |
| Operational projection proof v1 | `bc0bca1eae513f72c4ba578a285dbb56c742eae6` |
| Historical main base | `e00144b6b47504287c2d16f20b064da81e43f1cc` |

The integration branch was created from the exact Database head. Its required merge commits, in order, are:

1. `652113055be4fb401106b53b9f8a8fa43470ef1c` — parents Database and OTel.
2. `672999026ca88ba5e5968f9ca48affd6cfbe5c72` — parents the first merge and Source Map.
3. `448170807967c70ee512877b26b01074d1433b61` — parents the second merge and v1.

All four source commits are ancestors of the integration head. No rebase, squash, cherry-pick substitution, or history rewrite was used.

The complete v1 directory has the same Git tree ID at its source commit and the integrated head: `88f98c574821bfb7c61d94b11301118d7cbe866a`. Its artifact subtree also matches exactly: `7e60b4fc4a3ad58bd91279f07eff15208c1729a2`. The historical `NOT_EVALUATED` results are therefore preserved byte-for-byte.

## Core authority

The authoritative relationship state is the set of authoritative `CoreV3Tables` included in a `ValidatedSnapshot`. Snapshot validation checks entity/schema identity, global ID uniqueness, payload digests, foreign keys, relation material reconstruction, evidence authority, related entities, exactly one primary evidence link per binding, and exactly one successful generator operation closing each binding. There is no fallback or sampled validation.

The Database branch inherits one general Core implementation change from the historical main base: `src/generation_relation_core/relation_evidence.py`. It constructs function-local indexes from the Snapshot tables for entities, produced entities, operations, evidence, and primary-evidence candidates. These indexes are rebuildable views, are never persisted, contain no database/SQL/tuple/lineage semantics, and do not become an authority store. The diff from the exact Database head to the preimplementation integration head contains no Core or Core schema file.

## Domain authority boundaries

### Database which-lineage

Generation facts are established synchronously by actual relational operator callbacks through `CoreAdapter.capture_output` and `capture_disposition`. The candidate projection imports no Oracle and performs no filesystem or native-result read; it accepts only a validated Snapshot, validation proof, and explicit profile. Its independent reference is built from the frozen hand-authored synthetic Oracle. The two paths do not compare the candidate with itself. CSV/JSON ordinary results are serialized independently of Core metadata.

### OpenTelemetry trace shadow

The capture contract dispatches each actual relational callback to Core and, independently, to the official OpenTelemetry Python SDK. The native path canonicalizes official SDK finished spans. The direct candidate reads only `ValidatedSnapshot` plus its exact `SnapshotValidation`.

For P3, the direct Core→OTel module and Database→OTel module have separate occurrence, binding, producer, and `GeneratedOrigin` traversals. They share only the frozen trace schema/canonicalizer, semantic-key formatting, and error type. `database_to_otel_projection.py` reads the immutable `DatabaseDomainProjection`; it does not import or read Core Snapshot types. Candidate causal links are created only for `generated_origin` bindings by following `prior_support_id` to the unique producer occurrence. Native capture does not import projection modules, and candidate projection does not import native capture.

### ECMA-426 ordinary Source Map

The tested Emitter creates a synchronous receipt for each actual emit or disposition. Independent executions feed those receipts to the official `source-map` generator and to the Core collector, and receipt/output identity is checked across modes. The native generator does not read Core. The collector does not import a Source Map parser. The tested transformer does not import the Oracle. The candidate projection accepts only a validated Snapshot, registry validation, and stage ID; it does not read receipts, native maps, or Oracle output.

The existing strict-partiality source code does construct and compare two real Snapshot documents for each of its three cases. However, its committed artifact retains only `generation_facts_equal=false` and a prose list of lost facts. v2 must rerun the cases and record Snapshot IDs, complete semantic hashes, and concrete source/occurrence/binding/disposition/operation/evidence differences.

## Negative-control depth

The Source Map experiment's 30 existing controls are not all end-to-end replays:

- Controls 1–19 are validator-unit controls over mapping records, schema/VLQ behavior, coordinates, or direct reason-code sensitivity.
- Controls 20–25 mutate Snapshot/bridge data but invoke specialized validators rather than replaying the complete candidate path; they remain validator-unit controls in the inherited evidence.
- Controls 26–30 are isolation-policy controls.

The machine-readable audit classifies every inherited control individually. v2 must continue to report actual execution depth and may call an item `END_TO_END` only when the mutated object is sent through the complete path.

## Output and storage audit

No candidate projection reads native/reference output as an answer. Native maps, native spans, frozen Oracle records, the immutable Database projection, and report artifacts are comparison objects or transient derived views, not Core authorities and not fallback query stores. No forbidden persisted secondary relationship store was found. The preimplementation secondary-authority count is zero.

The ordinary database CSV/JSON outputs, OTel four-mode CSV/JSON outputs, and Source Map generated JavaScript outputs are byte-identical across enabled/disabled capture modes in the inherited evidence. Control-plane identifiers and metadata are not embedded in normal output.

## Required v2 work and audit verdict

The audit found no blocker to implementation, but inherited reports are not sufficient as the v2 proof. The v2 runner must rerun all domain paths, enrich Database/OTel/Source Map P2 comparisons, perform exact formal frozen-count checks, preserve every source experiment and v1 artifact, classify negatives by execution depth, run twice for determinism, and conjunctively fail on any mandatory condition.

Audit status: `READY_FOR_V2_IMPLEMENTATION`.
