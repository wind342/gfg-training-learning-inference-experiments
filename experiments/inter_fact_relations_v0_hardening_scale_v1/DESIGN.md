# Hardening and scale design

## Frozen boundary

The five-coordinate fact is unchanged:

```text
f = (u, tau, omega_bar, z; rho)
```

Runtime clocks, capture audit state, relation records, and run membership are
sidecar data. The experiment imports no formal Core implementation copy and
writes only in its own namespace.

## Authoritative experimental inputs

The controlled executor emits runtime receipts for:

- occurrence actor and sequence;
- adjacent program order;
- message send/receive pairing;
- synchronization participants, generation, and release;
- GeneratedOrigin support use;
- versioned resource writes and reads;
- resource-access conflicts;
- exact executor occurrence coverage;
- unknown, external, or unclassified activity.

Primitive evidence points to one receipt and carries explicit occurrence/fact
endpoints. The semantic validator binds the relation row, evidence envelope,
payload, receipt, runtime entities, authority, and execution run.
For each actor, program order is recomputed by sorting concrete occurrences by
sequence index and requiring the exact adjacent edge set. Duplicate sequence
indexes fail closed. Receipt, relation, and evidence bindings must be
one-to-one and must agree on endpoints, actor, sequence, run, authority, and
establishment source.

## Capture completeness

Completeness has two inputs:

1. a declared contract describing intended capture; and
2. a measured audit computed from receipts and validated primitive relations.

The result is one of:

```text
CAPTURE_COMPLETE
CAPTURE_PARTIAL
CAPTURE_CONFLICT
CAPTURE_NOT_ESTABLISHED
```

Only an exact `CAPTURE_COMPLETE` scope permits concurrency inference. There is
no `allow_concurrency` input.

The scheduler completeness basis is
`DECLARED_CONTROLLED_EXECUTOR_PROFILE`;
`global_scheduler_completeness_machine_proved` is always `false`; and the
concurrency scope is `CONTROLLED_CAPTURE_SCOPE_ONLY`. Thus
`CAPTURE_COMPLETE` is a verdict about the declared controlled capture profile,
not global scheduler completeness for an operating system, thread library, or
distributed system.

## Relation provenance

Every relation or comparison conclusion is classified as:

- `generator_established`;
- `wrapper_established`;
- `inferred`; or
- `independent_reference`.

Candidate graph rows may use the first three. Independent-reference answers
are kept outside the candidate graph.

Each candidate relation record carries:

```text
establishment_source
authority_id
execution_run_id
evidence_refs
rule_id
input_relation_refs
```

## Mixed occurrence/fact endpoints

The selected policy is relation-type specific:

- occurrence `happens_before` can lift to all endpoint facts, with the narrow
  meaning that their producing occurrences are ordered;
- `generated_origin_dependency` and `reads_from` preserve exact evidence
  selected facts;
- `conflicts_with` preserves exact resource-access facts;
- fact-level `concurrent_with` is query-only and requires complete capture.

The multi-fact scenario demonstrates one occurrence producing three facts
while only its second fact is a data dependency of the consumer.

## Identity separation

The experimental representation distinguishes:

- `semantic_occurrence_key`;
- frozen Core-like content-addressed occurrence identity;
- `concrete_occurrence_instance_id`, which includes `execution_run_id`;
- sidecar run membership.

Consequently, equal ordinary output and equal semantic/Core projections do not
imply equal strict run-scoped Gamma.

## Candidate resolver

The candidate reads only:

- the validated primitive store and endpoint catalog;
- the measured capture audit;
- frozen lifting rules;
- pre-registered queries.

It builds disposable adjacency, reverse-adjacency, conflict, topological, weak
component, and per-query cache indexes. It rejects cycles and preserves every
primitive relation ID for a source-target pair. It supports:

```text
happens_before(a,b)
concurrent_with(a,b)
predecessors(a)
successors(a)
relation_path(a,b)
conflicts(a)
fact_to_occurrence(a)
occurrence_to_selected_facts(a)
```

It never materializes the global transitive closure.

## Reference and comparison isolation

The reference reads runtime receipts, the capture contract, and queries. It
reconstructs an event DAG independently and never reads primitive candidate
relations or candidate output. Small uses a complete eager receipt oracle;
Medium and Large use a query-local receipt-DAG oracle.

Candidate and reference execute in separate processes. A third compare
process reads only their normalized outputs. Source and serialized input
audits enforce this boundary.

## Determinism and diagnostics

Scientific results contain counts, query manifests, relation results, capture
statuses, and normalized process evidence. Two complete runs must have equal
canonical bytes and SHA-256.

Elapsed time, peak RSS, process IDs, temporary paths, and available memory at
the optional-scale guard are diagnostics and are excluded from the scientific
hash.
