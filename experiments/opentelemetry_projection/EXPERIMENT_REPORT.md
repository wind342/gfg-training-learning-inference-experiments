# OpenTelemetry Trace Projection Experiment Report

## Research question

Within the declared deterministic relational-execution scope, can the
occurrence/execution/causal shadow represented by OpenTelemetry spans be
derived exactly from validated Core v3 generation facts, both directly and
through the broader database-domain projection?

The acceptance criterion was exact canonical equality, not count equality.
The experiment was designed to falsify the claim with native-SDK comparison,
two non-injectivity witnesses, independent hierarchical derivation, four-mode
ordinary-output comparison, and thirteen fail-closed mutations.

## Classification

| Claim | Classification |
| --- | --- |
| Direct projection | SUPPORTED |
| Strict (lossy/non-injective) projection | SUPPORTED |
| Hierarchical compositionality | SUPPORTED |
| Output orthogonality | SUPPORTED |
| No second authority | SUPPORTED |

These classifications apply only to the controlled scope below and depend on
the recorded artifacts passing all exact comparisons. They do not generalize
to arbitrary tracing systems.

## Frozen semantics

One query execution is a root span. Every concrete Core
`GenerationOccurrence` is exactly one occurrence span. Stages are attributes,
not an additional span layer. Every occurrence span is a child of the root.

The database executor establishes direct relations synchronously when it
creates an output or explicit disposition. A later stage reintroduces a prior
support through `GeneratedOrigin`. The projector follows this validated bridge
back to the unique producing occurrence and emits one Span Link per causal
input. It preserves repeated links. It does not guess a causal parent from
execution order and does not turn source-to-outcome bindings into span edges.

The selected trace schema retains span semantic identity, name, occurrence and
operator type, run and stage context, parent, all causal links, deterministic
logical order, status, and one selected event. It excludes source identities,
support and disposition IDs, direct bindings, evidence, operation closure, and
lineage answers.

## Methods

Native spans are created during actual relational executor callbacks with the
official OpenTelemetry Python SDK 1.44.0, `TracerProvider`,
`SimpleSpanProcessor`, and `InMemorySpanExporter`. Sampling is not used. Span
limits are set above the declared workload bound; any SDK-reported dropped
attribute, event, or link fails normalization.

The direct path accepts only a `ValidatedSnapshot` plus the exact
`SnapshotValidation` proof for that Snapshot. The hierarchical path first
builds immutable dataclass query results containing database source tuples,
produced tuples, exclusions, occurrences, bindings, GeneratedOrigin bridges,
and roles. A separate module then selects the OTel shadow without reading Core
or native spans.

The canonicalizer replaces random raw IDs with declared semantic keys and
replaces wall-clock comparison with `occurrence_index`. It does not discard any
declared semantic comparison field.

## Exact evidence

The machine-readable artifacts record:

- native/direct/hierarchical span counts and canonical SHA-256 values;
- span-set, parent-edge, link-edge, attribute, status, and event differences;
- the two strict-projection witnesses and binding/relation differences;
- four-mode CSV/JSON hashes, value/order/type comparisons, and forbidden fields;
- all thirteen negative-control reason codes;
- static/runtime isolation and authority checks;
- official DuckDB and TPC-H answer comparison for SF0.01 Q6;
- timing, memory, environment, dependency versions, code hashes, and artifact
  hashes.

The exact numeric results are in
[`artifacts/metrics.json`](artifacts/metrics.json) and
[`artifacts/formal_tpch_q6.json`](artifacts/formal_tpch_q6.json), rather than
being copied by hand into this report.

## Falsification controls

The validator fails closed for: a missing span; a fabricated span; wrong
parent; wrong causal link; wrong status; wrong operation type; wrong selected
transform attribute; duplicate span; merged occurrences; an unknown
occurrence; a binding-derived span; a prohibited Oracle/native dependency; and
direct/hierarchical disagreement. No mutation changes the valid input Snapshot
or ordinary business output.

The duplicate check operates on the candidate list before key lookup. There is
no fallback, sampling comparison, partial pass, or post-hoc repair.

## Strict projection witnesses

Two pairs hold run semantics, occurrence sequence, stage/operator topology,
span names, parents, status, and selected context constant while changing
source identities and direct generation relations:

1. two all-pass selections over distinct source identities;
2. two many-to-many joins over identical values but distinct left/right tuple
   identities.

For each pair, the Core Snapshot, binding set, direct relation set, and backward
lineage differ while the native and projected normalized traces are equal.
Therefore:

`Π_OTel(Γ₁) = Π_OTel(Γ₂)` does not imply `Γ₁ = Γ₂`.

This is the required evidence that the trace is a narrow projection, not a
failed complete contract.

## No second authority

No Core schema or Core source file is modified. Core contains no `trace_id`,
`span_id`, `otel_parent`, or OpenTelemetry-specific field. The native spans are
an independent comparison object and are never read by either projection.
The database-domain dataclasses are ephemeral immutable query results, not a
persisted relationship table and not a source for Core queries.

Static imports are checked. Runtime import traps are exercised. Direct
projection still succeeds after native exporter records are cleared. Native
capture succeeds when projection imports are trapped.

## Limits

The result does not replace OpenTelemetry, collectors, exporters, metrics,
logs, baggage, sampling, remote context propagation, or semantic conventions.
It does not prove causality for arbitrary asynchronous, cross-process, or
distributed executions. It does not show that every tracing system is a Core
projection. It does not allow complete Core facts to be reconstructed from a
trace.

The supported statement is narrower: in this deterministic in-process
relational executor, the frozen occurrence/execution/causal trace shadow is an
exact, lossy, compositionally derivable projection of validated Core facts.
