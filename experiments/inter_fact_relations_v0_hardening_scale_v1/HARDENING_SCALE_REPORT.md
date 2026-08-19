# Inter-fact relations v0 hardening and scale report

## 1. Verdict

Machine status:
`INTER_FACT_RELATIONS_HARDENING_SCALE_V1_SUPPORTED`.

Scientific SHA-256:
`0328eb1cc0d0f7c54b8f6629e09f79d36158abc55ea2573e05ab77ca4a2cc286`.

The experiment establishes a controlled, external relation sidecar with
content-validated primitive evidence, machine-audited capture completeness,
selective multi-fact lifting, and exact large-query agreement with an
independent receipt oracle. It does not establish a general concurrency
theory.

## 2. v0 conclusions retained

The following v0 conclusions remain supported:

- frozen five-coordinate facts can be relation endpoints without a sixth
  coordinate;
- clocks and relation graphs remain outside the atomic fact;
- occurrence-level causality and fact-specific dependency/conflict form a
  useful hybrid;
- GeneratedOrigin establishes one dependency family but not general
  happens-before;
- logical/event-DAG reference information is comparison evidence, not an
  atomic coordinate;
- equal ordinary outputs do not determine the relation sidecar.

## 3. v0 conclusions weakened or rewritten

The v0 `experimental_gamma_equal` result is not retained as strict equality.
After separating runtime identity:

- ordinary output equality: `true`;
- semantic fact projection equality: `true`;
- Core fact-ID projection equality: `true`;
- concrete run-scoped Gamma equality: `false`;
- relation graph difference: `true`.

The correct strict status is
`EXACT_GAMMA_EQUALITY_NOT_ESTABLISHED`.

Concurrency is no longer enabled by a caller boolean. It is inferred only
inside a scope whose measured result is `CAPTURE_COMPLETE`. That status means
complete under the declared controlled executor profile and only within the
controlled capture scope. It is not a machine proof of global scheduler
completeness for an operating system, thread library, or distributed system.

## 4. Primitive evidence content

All six primitive types have separate content validators:

- `program_order` binds actor, sequence indexes, endpoints, and executor or
  scheduler receipt; the exact per-actor adjacent edge set is recomputed and
  matched one-to-one across receipts, relations, and evidence; wall clock,
  duplicate sequence indexes, missing edges, and extra edges are rejected;
- `generated_origin_dependency` binds producer support, consumer
  GeneratedOrigin, producer/consumer facts, and prior support;
- `message_send_receive` binds unique message ID, channel, payload digest,
  send/receive occurrences, and one-time pairing;
- `synchronizes_with` binds synchronization identity, participants,
  generation, release, and pre/post phase;
- the controlled versioned reads-from fixture binds producer write, consumer
  read, resource, and exact observed version;
- `conflicts_with` binds both accesses, occurrence IDs, one resource version,
  and at least one write.

Evidence kind alone is insufficient. The valid semantic fixture exercises all
six types. All registered payload and endpoint mutations fail closed.

## 5. Capture completeness audit

A declared contract states intended coverage. A separate measured audit
counts actors, events, expected adjacent program edges, messages, pairings,
synchronization operations, GeneratedOrigin dependencies, reads-from
versions, resource accesses, unknown edges, unclassified operations, external
communication, unbound evidence, uncovered occurrences, and ambiguous or
duplicate edges.

Every scope records:

```text
scheduler_completeness_basis =
  DECLARED_CONTROLLED_EXECUTOR_PROFILE
global_scheduler_completeness_machine_proved = false
concurrency_scope = CONTROLLED_CAPTURE_SCOPE_ONLY
```

The declared `unobserved_scheduler_relation_ruled_out` field is retained, but
its meaning is limited to that controlled profile. `CAPTURE_COMPLETE` does not
claim global scheduler completeness outside it.

The states are:

```text
CAPTURE_COMPLETE
CAPTURE_PARTIAL
CAPTURE_CONFLICT
CAPTURE_NOT_ESTABLISHED
```

The Large workload contains both a complete scope and an intentionally
incomplete component. The latter remains `CAPTURE_PARTIAL` because of a
registered unknown edge and cannot produce a concurrency conclusion.

## 6. Mixed occurrence/fact endpoints and lifting

The mixed endpoint design remains reasonable only with relation-specific
lifting:

- `ALL_FACTS_OF_OCCURRENCE` is acceptable for `happens_before` when explicitly
  interpreted as order of producing occurrences;
- `EVIDENCE_SELECTED_FACTS` is required for
  `generated_origin_dependency`, `reads_from`, and `conflicts_with`;
- `RELATION_TYPE_SPECIFIC_LIFTING` is the recommended overall policy;
- fact-level concurrency is query-only and gated by complete capture.

In the multi-fact scenario, occurrence A produces `f_A1`, `f_A2`, and `f_A3`.
Only `f_A2 -> f_B` is a fact-specific dependency. The other two dependency
edges are absent, while occurrence A still happens before occurrence B.

## 7. Identity result

Four identities are reported separately:

1. semantic occurrence identity describes repeatable operation meaning;
2. Core content identity is a content-addressed projection without runtime
   membership;
3. concrete occurrence identity includes the execution run;
4. sidecar membership records which runtime relation graph owns the
   occurrence.

Semantic or Core equality must not impersonate concrete run equality.

## 8. Eager/lazy consistency and scale

| Scale | Occurrences | Facts | Primitive relations | Queries | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Small | 100 | 300 | 200 | 14,856 | 0 | 0 |
| Medium | 1,000 | 3,000 | 3,300 | 4,200 | 0 | 0 |
| Large | 10,000 | 30,000 | 28,900 | 22,240 | 0 | 0 |

Small compares every ordered occurrence pair plus every unordered concurrency
pair against complete eager closure. Medium and Large use fixed query
manifests and independent receipt-DAG queries. Candidate metrics report
`full_transitive_closure_materialized=false` and
`global_closure_pair_count=0` at every scale.

Diagnostic publication-run measurements:

| Scale | Total elapsed (s) | Peak parent+children RSS (bytes) | Candidate elapsed (s) | Reference elapsed (s) |
|---|---:|---:|---:|---:|
| Small | 0.703 | 75,042,816 | 0.198 | 0.152 |
| Medium | 1.018 | 87,285,760 | 0.207 | 0.177 |
| Large | 10.580 | 465,661,952 | 0.929 | 0.793 |

Times and RSS are diagnostics and are excluded from the scientific hash.

Repair before/after diagnostics use the same versioned fixtures and scales,
but come from different publication runs, so they are not presented as a
controlled performance benchmark:

| Scale | Before total (s) | After total (s) | Before capture audit (s) | After capture audit (s) | Before peak RSS (bytes) | After peak RSS (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| Small | 0.778 | 0.703 | 0.000 | 0.002 | 76,021,760 | 75,042,816 |
| Medium | 0.912 | 1.018 | 0.008 | 0.038 | 85,094,400 | 87,285,760 |
| Large | 6.452 | 10.580 | 0.231 | 3.691 | 455,606,272 | 465,661,952 |

Before repair, 30/30 controls and 11 hardening tests passed, but the
same-count/wrong-adjacency case was not tested. After repair, 31/31 controls
and 12 hardening tests pass, and that mutation fails closed with
`PROGRAM_ORDER_ADJACENCY_SET_MISMATCH`. Core remains 33/33; the full repository
remains 121 passed with 5 pre-existing skips. The exact machine record is
`artifacts/repair_before_after.json`.

The optional 50,000-occurrence run returned
`SCALE_NOT_EXECUTED_RESOURCE_GUARD`: available memory was below the
6,442,450,944-byte safety threshold. No smaller workload was substituted.

## 9. Relation establishment-source accounting

| Scale | Generator established | Wrapper established | Inferred query conclusions | Independent reference answers | Reference rows in candidate graph |
|---|---:|---:|---:|---:|---:|
| Small | 60 | 140 | 4,950 | 14,856 | 0 |
| Medium | 1,000 | 2,300 | 2,200 | 4,200 | 0 |
| Large | 10,000 | 18,900 | 6,000 | 22,240 | 0 |

Inferred conclusions are query results and are not globally materialized.
Independent-reference answers never enter the candidate relation graph.

## 10. Candidate/reference independence

Candidate, reference, and compare execute in distinct processes.

Candidate input contains only the validated primitive store, capture audit,
lifting rules, run ID, and queries. Reference input contains runtime receipts,
capture contract, run ID, queries, and reference mode. The reference does not
read candidate relations or output; the candidate does not read runtime or
reference receipts; compare reads only normalized process outputs.

Static source checks and serialized input/output SHA-256 checks all pass.

## 11. Negative controls

All 31 required mutations executed exactly once and returned their
pre-registered reason code. None was repaired, and none emitted partial
success. Controls cover semantic evidence corruption, completeness failures,
process-boundary violations, derived-proof corruption, duplicate primitive
loss, identity substitution, fifth-coordinate violations, cycles,
concurrency contradictions, cache contamination, output leakage, protected
paths, scale downgrade, and same-count wrong program-order adjacency.

## 12. Ordinary-output orthogonality

Each scale compares:

- relation capture disabled;
- primitive capture enabled;
- full relation resolution enabled.

Value, ordering, schema, and canonical bytes are equal. Relation IDs, fact
IDs, clocks, evidence, and profile tokens do not enter ordinary output.

## 13. Determinism, artifacts, and tests

Two complete scientific runs produced identical canonical results and the
same scientific SHA-256. Five diagnostic categories are explicitly excluded.

The manifest declares 17 artifacts; every byte size and SHA-256 was
independently rehashed.

Tests:

- hardening-scale experiment: `12 passed`;
- frozen Core: `33 passed`;
- complete repository: `121 passed, 5 skipped`.

The five full-repository skips are pre-existing Source Map external dependency
or ignored private-bootstrap prerequisites. There are no new failures.

## 14. Core decision

Core modification is not required. Protected Core, protocol, compatibility,
Core tests, and the complete v0 experiment tree have zero changed files.
There is no experimental Core copy.

## 15. Established

- content-level validation for all six primitive relation types;
- machine-audited capture states and fail-closed concurrency gating;
- selective fact-specific lifting for one occurrence producing many facts;
- explicit semantic/Core/concrete/sidecar identity separation;
- exact candidate/reference answers for the tested Small, Medium, and Large
  workloads;
- Large sparse-DAG query execution without candidate global closure;
- process-isolated reference comparison;
- ordinary-output orthogonality;
- no need to modify Core.

## 16. Partially established

- fact-level concurrency is supported only under the declared controlled
  executor profile and `CONTROLLED_CAPTURE_SCOPE_ONLY`;
- occurrence-order lifting to facts is meaningful only as producing-event
  order, not as universal data dependency;
- scale is established at 10,000 occurrences and 30,000 facts, not at the
  guarded optional level;
- runtime receipt semantics are constructive for the tested controlled
  wrappers, not all possible collectors.

## 17. Not established

The experiment does not establish:

- arbitrary-system global concurrency completeness;
- distributed consistency;
- complete lock, transaction, memory-model, or race semantics;
- uniqueness or minimality of the relation representation;
- strict Gamma equality across distinct concrete runs;
- that complete Gamma alone necessarily fails to determine every relation
  graph;
- that the current vocabulary expresses all fact relations;
- general operating-system or distributed-system coverage.

## 18. Research and theory boundary

Further independent research is worthwhile for real collector receipts,
additional synchronization forms, selective fact concurrency, and higher
guarded scale.

The result remains outside the frozen theory and Claim Atlas because it is an
experimental sidecar with controlled wrapper assumptions, partial external
validity, and no proof of unique or general relation semantics. Promoting it
would exceed the evidence.
