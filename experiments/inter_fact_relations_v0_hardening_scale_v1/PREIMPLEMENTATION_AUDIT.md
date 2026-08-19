# Pre-implementation audit

Status: `FROZEN_BEFORE_IMPLEMENTATION`

This document records the read-only audit of
`experiments/inter_fact_relations_v0/` at frozen source commit
`ad19cbb701e7c9d6bc2426756a252039c3119601`. It freezes the observed risks,
experimental assumptions, implementation boundary, and pass/fail conditions
before hardening or scale code is written.

## 1. Baseline and protected boundary

The audit ran in a clean, independent worktree on branch
`maintenance/inter-fact-relations-v0-hardening-scale-v1`.

Environment and baseline checks:

- Python: `3.12.10`
- v0 plus frozen Core tests: `43 passed`
- `src/generation_relation_core/` tree:
  `03fbdce13249f84abe9d8fb605da31cdc36eda27`
- `protocol/core_v3/` tree:
  `0b4a2608864e771ebca7cdbfad95aabaed2d0723`
- `compat/v2/` tree:
  `7bbb49d18daf7ea99d7633b40c6df5bc002824ca`
- `tests/core/` tree:
  `280cb44d592ae48d986719638980c11e57aab1f9`
- v0 experiment tree:
  `fccb595dfc0a8c7272f3e6e2af6937a57f8168b7`

The following paths are read-only for this experiment:

```text
src/generation_relation_core/
protocol/core_v3/
compat/v2/
tests/core/
experiments/inter_fact_relations_v0/
```

All implementation and generated evidence will be confined to:

```text
experiments/inter_fact_relations_v0_hardening_scale_v1/
```

No Core copy is permitted. If the sidecar cannot implement a required
capability, implementation must stop rather than modify or copy Core.

The frozen atomic fact remains:

```text
f = (u, tau, omega_bar, z; rho)
```

Clocks, run membership, capture state, and relation graphs remain outside the
five coordinates.

## 2. Read-only findings

### R1. Capture completeness is caller-declared

Observed:

- `scenarios/common.py:12-31` constructs a profile by directly writing five
  completeness booleans as `True`.
- The same function derives
  `unobserved_scheduler_edges_possible` and
  `concurrency_inference_allowed` from the caller's `allow_concurrency`
  argument.
- `src/relation_resolver.py:26-36` trusts those values and an exact occurrence
  set; it does not consume measured receipts or an audit result.

Consequence: v0 correctly refuses concurrency when the boolean profile is
incomplete, but it does not establish that a claimed-complete profile is
true.

Required hardening: separate a declared capture contract from a measured
capture audit. Only a machine-produced `CAPTURE_COMPLETE` result may enable
concurrency.

### R2. Primitive evidence validation is kind-level

Observed:

- `src/validator.py:25-32` maps each relation type to an allowed evidence kind.
- `src/validator.py:54-68` checks evidence existence and kind membership.
- It does not validate the evidence payload against the declared relation
  endpoints, executor receipts, Core GeneratedOrigin content, message
  pairing, synchronization membership, resource versions, or access modes.

Consequence: correctly typed but semantically false evidence can pass the v0
primitive check.

Required hardening: independent content validators for all six primitive
relation types, with mutation controls that preserve the evidence kind while
corrupting its content.

### R3. Occurrence-to-fact lifting is a Cartesian product

Observed:

- `src/relation_resolver.py:114-129` lifts every occurrence-level
  happens-before path to every source fact times every target fact.
- `src/relation_resolver.py:151-163` applies the same all-pairs policy to
  concurrency.

Consequence: the rule is adequate for v0's one-fact-per-occurrence fixtures
but does not distinguish occurrence order from fact-specific dependency when
an occurrence emits semantically independent facts.

Required hardening: compare `ALL_FACTS_OF_OCCURRENCE`,
`EVIDENCE_SELECTED_FACTS`, and `RELATION_TYPE_SPECIFIC_LIFTING`. Generated
origin, reads-from, and conflict relations must retain exact fact endpoints.

### R4. Same-Gamma uses semantic occurrence identity

Observed:

- `src/fact_adapter.py:106-121` derives a Core occurrence from a stable
  instance key, occurrence index, and semantic payload without a run-instance
  identifier.
- `scenarios/same_output_different_relations.py:116-138` reports experimental
  Gamma and fact-ID equality while explicitly qualifying that strict runtime
  identity is not established.

Consequence: v0's witness is valid only under its semantic canonicalization;
it does not prove strict run-scoped Gamma equality.

Required hardening: represent `execution_run_id`,
`semantic_occurrence_key`, `concrete_occurrence_instance_id`, Core
content-addressed occurrence identity, and sidecar run membership separately.

### R5. Reads-from has no independent scenario

Observed:

- `reads_from` is present in the relation vocabulary, causal set, schema, and
  expected evidence-kind map.
- No v0 scenario emits a `reads_from` primitive.

Consequence: schema acceptance is established, but content semantics and
version-sensitive behavior are not.

Required hardening: add a real versioned write/read receipt scenario and
mutations for wrong version and resource-only inference.

### R6. Happens-before closure is fully materialized

Observed:

- `src/relation_resolver.py:39-61` runs breadth-first search from every
  occurrence and retains every reachable pair.
- `src/relation_resolver.py:98-129` materializes every occurrence-level
  closure row and its fact-level Cartesian lifting.
- Concurrency scans all unordered occurrence pairs.

Consequence: output and work may become quadratic, before additional
fact-level multiplication.

Required hardening: retain v0-like eager closure only as a small reference.
The candidate must use disposable adjacency/reverse indexes, DAG validation,
bounded graph search, and per-query caching without materializing global
closure.

### R7. Vector-clock code is separate, but scenario assumptions are shared

Observed:

- `src/logical_clock.py` compares clocks without reading relation edges.
- However, each v0 scenario constructs the vector-clock transitions and
  primitive relation evidence from the same scenario function and the same
  causal intent.
- `scenarios/common.py:87` embeds the reference directly in the candidate run
  object.

Consequence: the comparison function is independent, but the full reference
path is not isolated from fixture construction or candidate input.

Required hardening: candidate and reference must run in separate processes
from disjoint input packages. Reference reads runtime receipts; candidate
reads only primitive relations, measured capture audit, lifting rules, and
queries.

### R8. Derived comparison discards proof content

Observed:

- `src/relation_model.py:84-90` defines a relation key as endpoint level,
  relation type, source, and target.
- `src/validator.py:97-100` compares sets of only those keys.
- Rule ID, evidence references, input relation IDs, relation ID, duplicates,
  and shortest-path proof content are not compared.

Consequence: a derived relation with the correct endpoints but an invalid or
missing proof can pass the set comparison.

Required hardening: validate the full canonical derived row and independently
recompute the declared path and exact input relation IDs.

### R9. Multiple causal primitives for one endpoint pair can be overwritten

Observed:

- `src/relation_resolver.py:71` stores causal edges in a dictionary keyed only
  by `(source, target)`.
- Assignments at `src/relation_resolver.py:87` and
  `src/relation_resolver.py:92-94` overwrite an earlier relation ID for the
  same pair.

Consequence: distinct program-order, message, synchronization, GeneratedOrigin,
or reads-from evidence between the same occurrences can be hidden.

Required hardening: index each endpoint pair to all primitive relation IDs and
fail on unregistered ambiguity rather than use last-write-wins behavior.

### R10. Evidence endpoints are not semantically bound

Observed:

- `src/relation_collector.py:18-42` records optional occurrence and fact ID
  lists in evidence.
- `src/validator.py:54-68` does not compare those endpoint lists or payload
  endpoint fields with the relation's source and target.

Consequence: evidence can name unrelated occurrences or facts while retaining
the expected kind.

Required hardening: each semantic validator must bind the declared relation,
evidence envelope, payload, authoritative receipt, and actual endpoint
entities.

## 3. Additional audit findings

- Primitive source is collapsed to
  `generator_or_controlled_wrapper`; v1 must use exactly
  `generator_established`, `wrapper_established`, `inferred`, or
  `independent_reference`.
- Independent reference currently resides inside the run object. It must not
  enter the candidate relation graph.
- Duplicate evidence IDs would be hidden by the dictionary construction in
  `validate_run`; v1 must reject duplicates before indexing.
- v0 validates ordinary output hashes but does not execute every comparable
  scenario in capture-disabled, primitive-only, and fully-resolved modes.
- v0 does not measure resolver time or peak memory at the required scales.
- v0's protected-path audit is specific to its earlier baseline and namespace;
  v1 needs an audit against `ad19cbb701e7c9d6bc2426756a252039c3119601`
  and must also freeze the complete v0 tree.

## 4. Frozen experimental assumptions

1. The experiment evaluates an external analytical relation sidecar, not a
   new authoritative Core relation store.
2. Runtime receipts are the authoritative experimental observations for
   actors, program order, messages, synchronization, GeneratedOrigin
   dependencies, reads-from versions, and resource accesses.
3. A declared capture contract expresses intended coverage but cannot prove
   completeness.
4. A measured audit is computed from receipts and primitive evidence. It
   cannot be overridden by a caller boolean.
5. Concurrency means mutual causal unreachability only inside an exact scope
   with `CAPTURE_COMPLETE`.
6. Candidate indexes and caches are disposable and rebuildable from primitive
   relations, the capture audit, frozen lifting rules, and queries.
7. The reference path does not read candidate relations, indexes, caches, or
   outputs. The candidate path does not read runtime/reference receipts or
   scenario internals.
8. Independent reference relations are comparison evidence only and are never
   candidate graph rows.
9. Small comparison is exhaustive over all occurrence pairs. Medium and large
   comparisons use deterministic pre-registered query manifests.
10. Diagnostic elapsed time and peak RSS are excluded from the scientific
    hash.
11. Failure is atomic: a failed mutation produces one registered reason code
    and no partial success artifact.
12. No result from a reduced workload may be labeled as the mandatory large
    workload.

## 5. Frozen design decisions

- Relation records carry `establishment_source`, `authority_id`,
  `execution_run_id`, `evidence_refs`, `rule_id`, and
  `input_relation_refs`.
- Primitive stores retain all candidate rows; no dictionary assignment may
  silently collapse distinct primitives.
- `generated_origin_dependency`, `reads_from`, and `conflicts_with` always
  retain exact fact endpoints.
- Occurrence-level `happens_before` may be conservatively lifted to facts only
  with a rule that states that it means occurrence order, not fact-specific
  data dependency.
- Fact-level concurrency lifting is separately evaluated and cannot be
  assumed from occurrence incomparability alone.
- Candidate cycle rejection uses an indexed DAG check. Query answering uses
  bounded search and disposable per-run caches.
- The eager resolver is reference-only and limited to small input.
- Strict run identity is external to the frozen fact coordinates.
- Ordinary output is produced before relation capture and is compared
  canonically across disabled, primitive-only, and full modes.

## 6. Pass conditions

The experiment passes only if all of the following are true:

1. Protected Core, protocol, compatibility, Core tests, and v0 experiment
   trees are unchanged.
2. All six primitive semantic validators pass valid receipts and reject their
   registered content mutations.
3. Capture state is computed as one of `CAPTURE_COMPLETE`,
   `CAPTURE_PARTIAL`, `CAPTURE_CONFLICT`, or
   `CAPTURE_NOT_ESTABLISHED`; only the first enables concurrency.
4. Reads-from is exercised with a concrete resource version.
5. A multi-fact occurrence produces no false fact-specific dependency.
6. Semantic, Core, concrete run-instance, and sidecar run identities are
   reported independently.
7. Small candidate/reference comparison is exhaustive and exact.
8. Medium candidate/reference query cohort is exact.
9. Large contains at least 10,000 occurrences, 30,000 facts, mixed required
   structures, and 20,000 pre-registered bidirectional queries.
10. Large candidate results have `FP=0` and `FN=0` without a full transitive
    closure.
11. At least 30 unique, single-application mutations fail closed with their
    pre-registered reason codes.
12. Candidate/reference process isolation checks pass.
13. Ordinary output value, ordering, schema, and canonical bytes are equal
    across all capture modes.
14. Two complete scientific runs produce the same scientific hash.
15. Artifact bytes, sizes, and SHA-256 values independently match the
    manifest.
16. Experiment, frozen Core, and full-repository tests introduce no new
    failure.

## 7. Fail conditions

Any of the following fails the experiment:

- a protected or v0 file changes;
- a sixth fact coordinate appears;
- a caller boolean directly enables concurrency;
- a semantically corrupt primitive passes validation;
- duplicate or ambiguous primitives are overwritten;
- fact-specific relations are broadened by occurrence Cartesian lifting;
- semantic occurrence identity is reported as concrete run identity;
- candidate or reference reads a forbidden input;
- a derived proof omits or misstates its rule or input relations;
- large scale is reduced, full closure is materialized, or query errors are
  nonzero;
- a mutation is repaired, applied more than once, returns an unregistered
  reason, or emits partial success;
- ordinary output contains relation metadata or differs by capture mode;
- scientific reruns differ outside explicitly excluded diagnostics;
- a not-established conclusion is reported as established.

## 8. Permitted conclusion boundary

Passing may establish controlled construction, semantic primitive validation,
machine-audited capture completeness, scoped causal/concurrency queries,
selective multi-fact lifting, large sparse-DAG query executability, and no need
to modify Core.

Passing must not be reported as general distributed concurrency completeness,
distributed consistency, complete transaction/lock/memory/race semantics,
unique or minimal relation representation, a Theory of Generation Facts
extension, or Claim Atlas evidence.
