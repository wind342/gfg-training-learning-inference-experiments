# Order-refund-freeze result-level relation report

## Verdict

Machine status:
`ORDER_REFUND_FREEZE_INTER_FACT_RELATIONS_V1_SUPPORTED`.

Scientific SHA-256:
`47400c460c62a04b60b3412c4de4dea2c9ad70f6fe73a889527927d6cce36b0d`.

The publication run used a real SQLite 3.49.1 database in WAL mode, actual
SQL reads and conditional updates, real commit/rollback results, separate
worker processes, a multiprocessing queue, barriers, and events. For the
frozen controlled profile, the external inter-fact relation sidecar answered
all 56 preregistered scenario-query pairs exactly against an independent raw
receipt reference.

This is not a general concurrency result and does not replace SQLite,
transaction control, locks, or OpenTelemetry.

## Real execution

Each scenario ran five times with relation capture disabled and five times
with capture enabled:

| Scenario | Disabled | Enabled | Total |
|---|---:|---:|---:|
| CONCURRENT_REFUND_WINS | 5 | 5 | 10 |
| CONCURRENT_FREEZE_WINS | 5 | 5 | 10 |
| LATE_REFUND_AFTER_FREEZE | 5 | 5 | 10 |
| IDEMPOTENT_DUPLICATE_REFUND | 5 | 5 | 10 |
| **Total** | **20** | **20** | **40** |

Every workflow execution contained the orchestrator and at least three child
worker processes. Transaction outcomes came from actual SQLite `rowcount`,
`COMMIT`, and `ROLLBACK`; messages crossed an actual Queue with matched
`put/get` receipts and payload digests. Barrier and Event receipts record the
controlled gate sequence. Wall-clock values establish no causal relation.

All 20 capture-on/off pairs had equal final canonical dumps, action results,
notification results, ordering, schema, and canonical business-output bytes.
Relation metadata leakage count was zero.

## Scenario outcomes

- A: Refund committed `REFUNDED/version 8`; Freeze rolled back with
  `FREEZE_VERSION_CONFLICT_AFTER_REFUND`; one notification was sent.
- B: Freeze committed `FROZEN/version 8`; Refund rolled back with
  `REFUND_VERSION_CONFLICT_AFTER_FREEZE`; notification was explicitly
  suppressed.
- C: Freeze committed before Refund read. Refund observed
  `FROZEN/version 8`, skipped the conditional write, and formed
  `REFUND_REJECTED_ORDER_ALREADY_FROZEN`; notification was suppressed.
- D: the first refund committed; the same idempotency key produced
  `IDEMPOTENT_DUPLICATE_REFUND`; exactly one refund row and one notification
  exist.

Scenario B and C have byte-equal ordinary final business views, while their
refund formation answers differ. This is a paired result-level witness, not
strict Gamma equality.

## Generation facts and relation sidecar

Every atomic fact has exactly:

```text
f = (u, tau, omega_bar, z; rho)
```

There is no sixth coordinate. Database versions, clocks, conflicts, and
relation graphs remain outside the coordinate structure. The sidecar
`H_e=(Gamma_e,R_e)` contains validated:

- exact adjacent `program_order`, using the repaired Stage A set validator;
- exact-version `reads_from`;
- same-order/same-version write `conflicts_with`;
- Queue `message_send_receive`;
- Barrier/Event `synchronizes_with`;
- result-to-message-to-notification
  `generated_origin_dependency`;
- actual `commits_version`.

`happens_before` is derived only from validated causal primitives.
`concurrent_with` is query-only and allowed only when capture is
`CAPTURE_COMPLETE` in `CONTROLLED_CAPTURE_SCOPE_ONLY`.

## Result-level queries

Candidate and independent reference ran in distinct processes with disjoint
serialized inputs. A third process resolved the conventional native trace; a
fourth compared normalized answers.

| Metric | Result |
|---|---:|
| Scenario-query pairs | 56 |
| Mismatches | 0 |
| False positives | 0 |
| False negatives | 0 |

The candidate used only validated facts, sidecar, capture audit, frozen
lifting policy, and queries. The reference used only canonical SQLite dumps
and raw SQL, Queue, synchronization, and worker receipts. The trace resolver
used only the frozen native export. Compare used only normalized outputs.
Static source isolation and serialized input hashes are committed.

`NotificationSent` resolves through:

```text
RefundCommitted
  -> RefundCommittedMessage
  -> NotificationSent
```

It is never bound to `RefundRequested` or `RefundPlanned`. Suppressed
notifications resolve to the concrete refund disposition. Impact and
compensation queries return concrete result and relation IDs.

## Four information views

The ordinary business view, honest conventional OpenTelemetry profile, and
atomic-fact-only view answer several local outcome questions. They do not,
under the frozen profiles, establish every exact cross-action version,
conflict, message-origin, downstream-impact, or compensation path.

The Gamma-plus-sidecar view establishes all preregistered queries in this
controlled profile. This does not mean logs are generally insufficient or
that OpenTelemetry cannot carry richer application-specific data; the native
profile intentionally contains ordinary refund, freeze, and notification
spans, action status, order ID, parent/links, and action type, without secret
sidecar facts.

## Capture completeness

The audit verifies worker occurrence coverage, exact adjacent program order,
Barrier participants, Event releases, Queue pairing, SQL reads/writes,
commit/rollback, exact versions, GeneratedOrigin bridges, evidence binding,
duplicate messages/results, explicit dispositions, unknown activity,
external communication, incomplete exits, and timeouts.

All four representative enabled scopes are `CAPTURE_COMPLETE`. This means
complete only for the declared controlled executor profile. It is not a
machine proof of operating-system, arbitrary-thread-library, or distributed
global scheduler completeness.

## Negative controls and determinism

All 30 preregistered mutations executed once and returned the exact expected
reason code. None was repaired; all have `partial_success=false`; no partial
scientific result was emitted.

Two complete scientific materializations produced identical canonical bytes
and scientific SHA-256. SQLite DB/WAL binary hashes are committed as
diagnostics but excluded from the scientific hash because WAL salts are not
scientific inputs. Canonical table dumps, transaction-receipt hashes, and
result hashes are in the deterministic scientific bundle.

## Tests and protected paths

- experiment tests: 9 passed;
- frozen Core tests: 33 passed;
- full repository: 121 passed, 5 existing external-dependency/private-data
  skips;
- Core changed files: 0;
- Stage A experiment changed files: 0;
- experimental Core copy: absent;
- Claim Atlas changes: none.

## Supported

- exact result-level formation answers for the four real controlled workflows;
- actual version reads and competed-version conflict binding;
- actual result-to-message-to-notification dependency;
- explicit non-formation dispositions;
- concrete downstream impact and compensation targeting;
- B/C equal ordinary state with different real formation;
- capture orthogonality;
- fail-closed controlled-scope concurrency;
- unchanged five-coordinate atomic fact.

## Partially supported

- native trace answerability is evaluated only for the frozen conventional
  profile;
- concurrency is supported only inside the controlled capture profile;
- SQLite behavior is supported only for the declared schema, WAL settings,
  and business rules;
- external validity is limited to these four scenarios and the preregistered
  query set.

## Not established

- general concurrency theory;
- arbitrary database or operating-system scheduling semantics;
- distributed consistency, race freedom, or deadlock freedom;
- replacement of SQLite, locks, transactions, or OpenTelemetry;
- uniqueness or minimality of the relation vocabulary;
- strict Gamma equality across runs;
- membership in the frozen Theory of Generation Facts or Claim Atlas.
