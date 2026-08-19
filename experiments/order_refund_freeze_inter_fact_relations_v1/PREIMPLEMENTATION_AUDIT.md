# Preimplementation audit: real order-refund-freeze experiment v1

## Frozen base and scope

- Repository: `wind342/source-information-continuity`
- Branch: `experiment/order-refund-freeze-inter-fact-relations-v1`
- Base branch:
  `maintenance/inter-fact-relations-v0-hardening-scale-v1`
- Repaired base head:
  `bd5354bc7a91327839b53600349490c621b6804c`
- Allowed write namespace:
  `experiments/order_refund_freeze_inter_fact_relations_v1/`
- History policy: append-only; no rebase, squash, amend, force push, or
  rewritten commits.

Protected paths are read-only:

```text
src/generation_relation_core/
protocol/core_v3/
compat/v2/
tests/core/
experiments/inter_fact_relations_v0/
experiments/inter_fact_relations_v0_hardening_scale_v1/
claims/
claim_atlas/
```

No Core copy is permitted. The frozen atomic fact remains:

```text
f = (u, tau, omega_bar, z; rho)
```

Inter-fact relations remain an external sidecar:

```text
H_e = (Gamma_e, R_e)
```

No clock, database version, conflict record, or relation graph becomes a
sixth coordinate or is hidden inside an existing coordinate.

## Scientific question and falsifier

The experiment tests whether a validated relation sidecar over atomic
generation facts can exactly answer result-level formation, conflict,
notification-dependency, impact, and compensation questions in the four
frozen workflows below.

The hypothesis fails if any of the following occurs:

- candidate and independent reference differ for any preregistered query;
- a false positive or false negative is observed;
- a transaction or message result is simulated instead of executed;
- a missing business result lacks an `ExplicitDisposition`;
- a notification is bound to a request or plan rather than a committed
  refund result;
- capture changes ordinary business output;
- concurrency is returned outside an audited `CAPTURE_COMPLETE` scope;
- Scenario B and C do not preserve the paired ordinary-output witness;
- any protected path changes;
- either complete scientific materialization differs in canonical bytes.

The experiment does not test or establish a general concurrency theory,
arbitrary database semantics, global scheduler completeness, distributed
consistency, race freedom, deadlock freedom, or replacement of SQLite or
OpenTelemetry.

## Frozen real-component profile

Every scientific execution uses:

- a real SQLite database file in WAL mode;
- real `SELECT`, conditional `UPDATE`, `INSERT`, `COMMIT`, and `ROLLBACK`;
- actual SQLite `rowcount`;
- separate orchestrator, refund, freeze, and notification OS processes;
- `multiprocessing.Queue`;
- `multiprocessing.Barrier` and `multiprocessing.Event`;
- actual queue `put` and `get`;
- deterministic preregistered business, result, message, and occurrence IDs.

Wall-clock values, process IDs, temporary paths, SQLite rowids, and random
UUIDs are diagnostics only and are excluded from scientific hashes. Sleeps
may protect a timeout but never establish scientific order.

## Frozen database model

The schema contains at least:

```sql
orders(
  order_id TEXT PRIMARY KEY,
  amount_cents INTEGER NOT NULL,
  status TEXT NOT NULL,
  version INTEGER NOT NULL
)

refunds(
  refund_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  amount_cents INTEGER NOT NULL,
  status TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE
)

notifications(
  notification_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  refund_id TEXT,
  status TEXT NOT NULL,
  notification_kind TEXT NOT NULL
)
```

Initial state is exactly:

```text
order_id = order-001
amount_cents = 5000
status = OPEN
version = 7
```

## Frozen scenarios

### A. CONCURRENT_REFUND_WINS

Refund and Freeze both read `OPEN/version 7`. The orchestrator releases
Refund first. Refund conditionally updates and commits `REFUNDED/version 8`;
Freeze later obtains `rowcount=0`, rolls back, and forms
`FREEZE_VERSION_CONFLICT_AFTER_REFUND`. Notification consumes the actual
`RefundCommitted` queue message and writes one notification.

### B. CONCURRENT_FREEZE_WINS

Refund and Freeze both read `OPEN/version 7`. Freeze commits
`FROZEN/version 8`; Refund later obtains `rowcount=0`, rolls back, and forms
`REFUND_VERSION_CONFLICT_AFTER_FREEZE`. Notification consumes the disposed
message and forms `NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND`.

### C. LATE_REFUND_AFTER_FREEZE

Freeze commits completely before Refund reads. Refund reads
`FROZEN/version 8`, does not attempt its conditional write, and forms
`REFUND_REJECTED_ORDER_ALREADY_FROZEN`. Notification is suppressed.

Scenarios B and C must have byte-equal ordinary final business views but
different formation answers. This is a paired result-level witness, not a
claim of strict Gamma equality.

### D. IDEMPOTENT_DUPLICATE_REFUND

The first refund commits. A second refund action uses the same idempotency
key, forms `IDEMPOTENT_DUPLICATE_REFUND`, creates no second refund row, and
causes no second notification.

## Frozen outcomes and dispositions

Ordinary business supports:

- `RefundCommitted`
- `OrderFrozen`
- `NotificationSent`

Required explicit dispositions:

- `REFUND_VERSION_CONFLICT_AFTER_FREEZE`
- `FREEZE_VERSION_CONFLICT_AFTER_REFUND`
- `REFUND_REJECTED_ORDER_ALREADY_FROZEN`
- `IDEMPOTENT_DUPLICATE_REFUND`
- `NOTIFICATION_SUPPRESSED_NO_COMMITTED_REFUND`

Known non-formation may not be represented as null, missing, unknown, or
simply failed.

## Frozen primitive relation policy

- `program_order`: exact adjacent concrete-occurrence set per worker, using
  the repaired Stage A validator.
- `reads_from`: exact order resource and version, read receipt, and consumer
  action result.
- `conflicts_with`: same order and competed version with at least one write;
  never read-read.
- `message_send_receive`: exact queue message ID, digest, sender, and receiver.
- `synchronizes_with`: exact Barrier generation and Event release.
- `generated_origin_dependency`:
  `RefundCommitted -> RefundCommittedMessage -> NotificationSent`, and
  `RefundDisposed -> NotificationSuppressed`.
- `happens_before`: derived only from validated causal primitives.
- `concurrent_with`: query-only and enabled only for
  `CAPTURE_COMPLETE` in `CONTROLLED_CAPTURE_SCOPE_ONLY`.

## Preregistered queries

| ID | Exact target |
|---|---|
| Q01 | Order version actually read by each refund result or disposition |
| Q02 | Whether RefundWorker and FreezeWorker are concurrent |
| Q03 | Whether RefundWorker and FreezeWorker conflict |
| Q04 | Action that actually committed order version 8 |
| Q05 | Exact reason the refund did not commit |
| Q06 | Exact reason the freeze did not commit |
| Q07 | Whether `NotificationSent` depends on RefundRequested, RefundPlanned, or RefundCommitted |
| Q08 | Exact `ExplicitDisposition` causing notification suppression |
| Q09 | Concrete downstream results depending on a given `RefundCommitted` |
| Q10 | Results that read or competed for order version 7 |
| Q11 | Directly affected results to compensate for a given `RefundCommitted` |
| Q12 | Why Scenarios B and C have equal ordinary views but different refund failures |
| Q13 | Results affected and unaffected by the refund-freeze conflict |
| Q14 | Whether duplicate refund created a second refund or notification |

Every normalized answer must carry query ID, exact target, candidate answer,
independent reference answer, status, evidence path, result IDs, relation
IDs, disposition IDs, false-positive count, and false-negative count.

## Frozen views and answerability claim

- View A: ordinary business result.
- View B: honest conventional native trace profile.
- View C: atomic generation facts only.
- View D: atomic generation facts plus validated relation sidecar.

The permitted conclusion is limited: some cross-action result-level questions
are `NOT_ESTABLISHED` in the frozen ordinary, native-trace, and atomic-only
views, while View D answers them exactly in this controlled profile. The
experiment will not claim that all logs fail, OpenTelemetry cannot represent
such data, or SQLite does not resolve conflicts.

## Frozen process isolation

- Candidate reads only validated facts, validated sidecar, capture audit,
  lifting rules, and queries.
- Reference reads only canonical SQLite dump and raw SQL, queue, sync, and
  worker receipts plus queries.
- Trace resolver reads only the frozen native trace export.
- Compare reads only normalized candidate, reference, trace, and baseline
  answers.

All four run in distinct processes. They may not share answer helpers,
relation builders, expected-answer registries, or hidden data.

## Frozen repetitions and orthogonality

Each scenario executes five capture-disabled and five capture-enabled runs:
at least 40 real workflow executions. Each paired run shares scenario,
initial database state, deterministic IDs, gate sequence, and business
inputs. Canonical database dump, action results, notification result,
ordering, schema, and ordinary-output bytes must match exactly, with zero
relation-metadata leakage.

## Capture-completeness audit

The machine audit covers worker occurrences, exact adjacent program order,
Barrier participants, Event releases, queue sends/receives and pairing,
SQLite reads/writes/commits/rollbacks and exact versions, GeneratedOrigin
bridges, evidence binding, unknown operations, external communication,
incomplete exits, timeouts, duplicate messages/results, and missing
dispositions.

Statuses are `CAPTURE_COMPLETE`, `CAPTURE_PARTIAL`, `CAPTURE_CONFLICT`, and
`CAPTURE_NOT_ESTABLISHED`. No caller `allow_concurrency` switch is permitted.

## Preregistered negative controls

Each mutation executes once, has one exact expected reason code, is never
repaired, has `partial_success=false`, and emits no partial scientific result.

| Mutation | Expected reason |
|---|---|
| mutation-01-program-order-same-count-wrong-adjacency | PROGRAM_ORDER_ADJACENCY_SET_MISMATCH |
| mutation-02-refund-read-version | REFUND_READ_VERSION_MISMATCH |
| mutation-03-freeze-read-version | FREEZE_READ_VERSION_MISMATCH |
| mutation-04-refund-relation-order-version | REFUND_RELATION_ORDER_VERSION_MISMATCH |
| mutation-05-conflict-order | CONFLICT_ORDER_MISMATCH |
| mutation-06-conflict-version | CONFLICT_VERSION_MISMATCH |
| mutation-07-read-read-conflict | READ_READ_CONFLICT_INVALID |
| mutation-08-queue-receiver | QUEUE_RECEIVER_MISMATCH |
| mutation-09-payload-digest | QUEUE_PAYLOAD_DIGEST_MISMATCH |
| mutation-10-requested-notification | NOTIFICATION_REQUEST_BINDING_FORBIDDEN |
| mutation-11-planned-notification | NOTIFICATION_PLAN_BINDING_FORBIDDEN |
| mutation-12-disposed-as-committed | REFUND_DISPOSITION_AS_COMMIT |
| mutation-13-rollback-as-commit | ROLLBACK_AS_COMMIT |
| mutation-14-zero-rowcount-success | ZERO_ROWCOUNT_AS_SUCCESS |
| mutation-15-version-conflict-as-frozen | VERSION_CONFLICT_DISPOSITION_MISMATCH |
| mutation-16-frozen-as-version-conflict | FROZEN_STATE_DISPOSITION_MISMATCH |
| mutation-17-second-refund | IDEMPOTENCY_SECOND_REFUND |
| mutation-18-second-notification | IDEMPOTENCY_SECOND_NOTIFICATION |
| mutation-19-notification-without-commit | NOTIFICATION_COMMIT_PREDECESSOR_MISSING |
| mutation-20-incomplete-concurrency | CONCURRENT_WITHOUT_CAPTURE_COMPLETE |
| mutation-21-wall-clock-happens-before | WALL_CLOCK_CAUSALITY_FORBIDDEN |
| mutation-22-candidate-sql-receipts | CANDIDATE_FORBIDDEN_SQL_RECEIPTS |
| mutation-23-reference-sidecar | REFERENCE_FORBIDDEN_SIDECAR |
| mutation-24-trace-reference | TRACE_FORBIDDEN_REFERENCE_INPUT |
| mutation-25-cross-run-result-id | CROSS_RUN_RESULT_ID_REUSE |
| mutation-26-disposition-deleted | EXPLICIT_DISPOSITION_MISSING |
| mutation-27-relation-output-leak | ORDINARY_OUTPUT_RELATION_LEAK |
| mutation-28-core-change | PROTECTED_CORE_PATH_MODIFIED |
| mutation-29-stage-a-change | PROTECTED_STAGE_A_PATH_MODIFIED |
| mutation-30-mocked-transaction | REAL_SQLITE_EXECUTION_REQUIRED |

## Artifact and acceptance plan

The implementation must materialize every artifact named in the task,
including SQLite/dependency identity, run manifest, transaction/queue/sync
receipts, facts, relations, four-view answerability, query comparisons,
impact and paired witnesses, dispositions, isolation, controls,
determinism, tests, protected paths, manifest, and summary.

Database and WAL binaries may remain ignored, but canonical table dumps and
DB, WAL, schema, transaction-receipt, and result hashes must be committed.

Acceptance requires real SQLite WAL, real processes and IPC, all four
scenarios, at least 40 executions, exact candidate/reference answers with
FP=FN=0, the B/C witness, correct notification origin, complete explicit
dispositions, idempotency, capture orthogonality, honest trace boundaries,
process isolation, 30/30 fail-closed controls, two canonically identical
scientific materializations, zero protected changes, no Core copy, no Claim
Atlas addition, and no theory change.
