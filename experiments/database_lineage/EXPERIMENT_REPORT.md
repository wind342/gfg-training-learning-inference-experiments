# Core v3 database-lineage replacement-principle experiment

## Verdict

This experiment records three separate conclusions:

- **Principle replacement: `SUPPORTED_IN_TESTED_SCOPE`.** The fixed adversarial Oracle and all four SF0.01 query plans passed output, direct-relation, forward/backward traversal, evidence, operation-closure, Snapshot, orthogonality, and determinism checks without a database-specific Core schema.
- **Engineering scale: `SCALABILITY_DEMONSTRATED_TO_462399_VALIDATED_BINDINGS`.** The largest completely validated Snapshot contained 178,797 bindings; the four SF0.01 Snapshots contained 462,399 bindings in total.
- **Original one-label protocol: `PARTIALLY_SUPPORTED`.** The sole reason is: `mandatory SF0.1/Q1 was not executed because the projected memory requirement exceeded the available hardware.`

The external ProvSQL comparator was unavailable in this Windows session because enabling WSL/VirtualMachinePlatform requires a host restart. This is recorded as missing independent external evidence, not as a Core comparison failure. The independent fixed manual Oracle did execute and produced zero forward/backward false positives and false negatives.

The result does not claim that Core replaces a DBMS or ProvSQL as a complete system, covers all SQL or provenance classes, is faster than lineage systems, or scales to arbitrary inputs.

## 1. Question, definition, and scope

The falsification target was whether controlled deterministic relational executions can establish tuple-level which-lineage using Core v3's modality-independent generation facts, without a database-specific relation-establishment schema.

“Replacement” means only:

1. capture the actual input–operation–output-or-exclusion relation at execution time;
2. derive direct and transitive backward/forward lineage from those facts; and
3. compose multi-stage lineage only through actual intermediate facts.

The tested operators are bag-preserving selection, projection and derived columns, inner equi-join, grouped/scalar `SUM`, `COUNT`, exact `AVG` as sum/count, stable sort, limit, and explicit exclusions. Field-level where-provenance, complete why/how-provenance, arbitrary SQL, transactions, optimization, access control, signatures, and comparative performance are outside the claim.

## 2. Reproducibility identity and environment

| Item | Recorded value |
|---|---|
| Base `main` | `e00144b6b47504287c2d16f20b064da81e43f1cc` |
| Branch | `experiment/database-lineage-core-v3-native-v1` |
| Core / Python | 3.0.0 / 3.12.10 |
| OS / physical memory | Windows 11 build 10.0.26200 / 17,011,310,592 bytes |
| DuckDB / TPC-H extension | 1.5.4 / v1.5.4 |
| Docker client / Desktop | 29.6.2 / 4.83.0; server unavailable pending required host restart |
| ProvSQL / PostgreSQL | v1.4.0 / 17 |
| ProvSQL tag commit | `37fc44474b75d3d0594e44b794b744675457eb7d` |
| Image index digest | `sha256:57c7877fe86638f201bc26fc0cb8ef759aeb09e9bfc03789c2d3a2b315305268` |
| Linux/amd64 manifest | `sha256:a5e3326de148f1a021df8eec2ae9f71b1f5bf672dd82bbaa9652791e6dcfe09e` |
| Source build | Debian bookworm; `make -j$(nproc)` then `make install` |

Exact dependency versions, platform probes, source hashes, generated-data row counts, and export SHA-256 values are in `artifacts/environment.json` and `artifacts/experiment_manifest.json`. Large databases, CSVs, complete forward maps, and run-local snapshots remain under ignored `runtime/` paths.

## 3. Core mapping and absence of a lineage schema

| Execution fact | Existing Core v3 entity/relation |
|---|---|
| Base tuple with stable identity | `SourceInformationRecord` |
| Actual operator event | `GenerationOccurrence` |
| Produced tuple | `PerceptualSupportRecord` |
| Known excluded tuple | `ExplicitDisposition` |
| Actual input/event/outcome edge | `GenerationBinding` with a role |
| Intermediate output consumed later | existing `GeneratedOrigin` bridge |
| Direct edge bytes | recomputed `relation_material` plus exactly one primary evidence |
| Stage completion | one successful operation closing every binding/evidence produced by that stage |

No SQL, tuple, join, aggregation, provenance, or lineage field was added to Core. No `lineage_from`, `lineage_to`, input/output map, lineage table, provenance circuit, or second authoritative relationship store exists. Every binding, occurrence, support, and disposition remains independent; grouping bindings under one successful stage operation does not merge them.

## 4. Executor and evaluation isolation

The executor evaluates rows stage by stage. Join hash tables, group accumulators, and sort buffers are transient computation state. A relation is captured when an operator produces a support or explicit disposition, never by matching final values after execution. Every intermediate support becomes the next stage's `GeneratedOrigin`; no base-input-to-final-output shortcut is written.

The tested path is `relational_executor.py` → `operators.py` → `core_adapter.py` → authoritative Core Snapshot → `core_lineage_reader.py`. The fixed Oracle, DuckDB, and ProvSQL modules are evaluation-only. Static import checks and a runtime Oracle trap demonstrate that the tested path still executes when Oracle access raises. The ordinary serializer receives business columns only; tuple IDs, Core IDs, and provenance tokens are forbidden.

## 5. Adversarial fixtures and independent Oracle

The `Customers`, `Products`, `Orders`, `OrderItems`, and `Promotions` fixtures include identical-value/distinct-identity tuples, one-to-many and many-to-many matches, left/right unmatched inputs, selection and limit exclusions, zero/negative/Decimal values, equal sort keys, converging aggregation inputs, and equal-value outputs with different lineage.

All 13 named cases passed:

| Check | Exact / total | False positive | False negative |
|---|---:|---:|---:|
| Normal output | 1 / 1 | 0 | 0 |
| Backward lineage | 1 / 1 | 0 | 0 |
| Forward lineage | 19 / 19 | 0 | 0 |
| Complete direct relation set | 1 / 1 | 0 | 0 |
| Explicit dispositions | 7 / 7 | 0 | 0 |
| Aggregate contributor sets | 2 / 2 | 0 | 0 |

The 2×2 many-to-many fixture produced exactly four join outputs and eight direct input edges. Duplicate-valued product tuples remained distinct. The final aggregate retained all nine distinct base contributors and 20 derivation paths. Missing dispositions and fabricated pairings were both zero.

## 6. TPC-H-derived data and fixed plans

DuckDB generated the data with `CALL dbgen(sf = 0.01)` and `CALL dbgen(sf = 0.1)`. Query text came from `tpch_queries()` and official expected display text from `tpch_answers()`. This is **a research workload based on the TPC-H schema, generator and fixed queries**, not an audited TPC-H benchmark.

| SF | customer | lineitem | nation | orders | part | partsupp | region | supplier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 1,500 | 60,175 | 25 | 15,000 | 2,000 | 8,000 | 5 | 100 |
| 0.1 | 15,000 | 600,572 | 25 | 150,000 | 20,000 | 80,000 | 5 | 1,000 |

Every table export SHA-256 is retained in the generation manifest. The fixed plans preserve official semantics:

- Q1: ship-date selection → derived charge fields → grouped SUM/AVG/COUNT → result-type projection → sort.
- Q3: customer/order/lineitem selections → two equi-joins → revenue → group SUM → sort → limit 10.
- Q6: compound selection → revenue expression → scalar SUM.
- Q10: order/lineitem selections → three equi-joins → revenue → group SUM → projection → sort → limit 20.

Official display values are parsed into the exact query result types without epsilon or rounding.

## 7. Completed output and validation matrix

| SF/query | Status | Rows | Bindings | DuckDB | Official | CSV+JSON contract on/off | Snapshot/evidence/operation |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.01/Q1 | passed | 4 | 178,797 | exact | exact | byte-identical | passed |
| 0.01/Q3 | passed | 10 | 121,112 | exact | exact | byte-identical | passed |
| 0.01/Q6 | passed | 1 | 62,557 | exact | exact | byte-identical | passed |
| 0.01/Q10 | passed | 20 | 99,933 | exact | exact | byte-identical | passed |
| 0.1/Q6 | `RESOURCE_LIMITED` | — | estimated 623,808 | not completed | not completed | not sampled | no partial validation |
| 0.1/Q1 | `NOT_RUN_RESOURCE_BOUND` | — | projected 1,784,292 | not run | not run | not run | not run |

For every completed binding, validation recomputed relation material, required exactly one primary evidence, checked authority and related entities, and required membership in exactly one successful operation. All origins, occurrences, supports, dispositions, bindings, evidence, foreign keys, projections, hashes, and implementation identity were validated. No fallback, sampling, skipped binding, silent loss, or fabricated pairing was accepted.

## 8. Q3/Q10 direct-structure audit and lineage traversal

The audit compares each occurrence's recorded input tuple identities against its exact binding origins with multiplicity. Join outputs must contain exactly one left and one right participant, aggregation occurrences must have one binding per recorded participant, every bridge must target an existing prior support, and a binding from a base origin directly to a final support is forbidden.

| Check | Q3 | Q10 |
|---|---:|---:|
| Direct bindings audited | 121,112 | 99,933 |
| Origin/input mismatches | 0 | 0 |
| Join role failures | 0 | 0 |
| Aggregation contributor-count failures | 0 | 0 |
| Broken `GeneratedOrigin` bridges | 0 | 0 |
| Fabricated base-to-final bindings | 0 | 0 |
| Selection dispositions | 36,792 | 59,662 |
| Left/right unmatched dispositions | 1,750 / 37,393 | 1,142 / 13,643 |
| Limit dispositions | 128 | 379 |

Backward traversal follows final support → direct bindings → origins and recursively crosses only `GeneratedOrigin.origin_payload.prior_support_id`. Forward traversal uses the inverse generic indexes and the same bridge. Cycle checks fail closed. Contributor sets may deduplicate identities, while binding paths and derivation-path counts retain multiplicity.

## 9. Core evidence resolver optimization

Core was modified, but its schema, entities, requirements, and query results were not. At each validation, the resolver builds five disposable indexes from the authoritative `CoreV3Tables` input:

- produced entity → every candidate operation;
- operation candidate → produced-entity set;
- operation candidate → evidence set;
- binding → every primary-evidence candidate;
- entity ID → entity.

Operation candidates use `(operation_result_id, list_position)` keys, so duplicate operation IDs cannot disappear by dictionary overwrite. Primary evidence candidates remain lists, so duplicates remain visible and fail. Success filtering occurs only after all candidate operations are indexed. The indexes contain no SQL, tuple, database, or lineage semantics and are never serialized.

The frozen scan resolver and indexed resolver used the same 300-binding fixture. Both passed the valid case and returned the same error type for missing/duplicate primary evidence, unauthorized authority, missing related entity, relation-material mismatch, missing operation, and duplicate operation. Result/error equivalence was 8/8. The indexed algorithm uses more temporary memory on this small fixture; it is an asymptotic lookup optimization, not a claimed small-fixture speedup. Exact timing and traced peak-memory distributions are in `artifacts/resolver_benchmark.json`.

| Resolver, 20 runs | Mean / median / min / max seconds | Mean / max traced peak bytes | Validation result |
|---|---:|---:|---|
| Frozen full scan | 0.056145 / 0.055449 / 0.054426 / 0.061721 | 105,214 / 105,466 | reference |
| Transient indexes | 0.056272 / 0.055505 / 0.054249 / 0.065945 | 421,312 / 430,694 | exactly identical |

## 10. Snapshot reader index

The reader builds only transient, rebuildable indexes over existing Snapshot entities and bindings. Deleting the reader discards every index; constructing another reader from the same validated Snapshot reproduces them. No authority moves from the Snapshot and no domain field is stored.

On the fixed 54-binding multistage fixture, scan and indexed readers returned identical forward and backward tuple IDs, binding paths, and derivation counts. Forward and backward false positives/false negatives were all zero. The benchmark records construction, forward/backward query time, returned results, and traced peak memory for 20 repetitions in `artifacts/reader_index_benchmark.json`. Index construction costs memory/time once and reduced repeated traversal time on this fixture; no database-scale performance claim is made.

| Reader, 20 runs | Construction mean s | Backward mean s | Forward mean s | Mean / max traced peak bytes |
|---|---:|---:|---:|---:|
| Frozen scan | 0.000004 | 0.000581 | 0.001993 | 13,924 / 18,880 |
| Transient indexes | 0.000206 | 0.000325 | 0.001081 | 28,905 / 32,480 |

## 11. Two-run determinism

All synthetic entities, relation material, evidence, outputs, and lineage were identical across two complete runs. For SF0.01 Q1/Q3/Q6/Q10, CSV and JSON hashes, output order/values, entity counts, binding counts, backward lineage bytes, path counts, and forward lineage sets were identical.

The first TPC-H execution used Core commit `50a21fd`; the final resolver duplicate-candidate correction produced commit `4e758ab` before the second execution. Snapshot IDs intentionally bind tracked Core implementation hashes, so raw IDs differ. This is not normalized away silently: each second-run Snapshot envelope was re-finalized with the first run's exact tracked implementation hashes. The reconstructed ID equaled the recorded first-run ID for every query. That equality covers every authoritative and derived table count/hash, environment/manifest/operation ID, relation material, binding, evidence, support, occurrence, disposition, and origin. `artifacts/tpch_determinism.json` records both `raw_snapshot_id_equal: false` and `normalized_snapshot_content_equal: true`.

## 12. Resource bounds

SF0.1/Q1 has the exact status:

> NOT RUN — projected in-memory representation exceeded available physical memory.

The plan projects approximately 1,784,292 simultaneous bindings. SF0.01/Q1 measured 178,797 bindings and 2,158,182,400 bytes peak RSS; linear projection is about 21.5 GB before OS/process headroom, versus approximately 17 GB physical memory. The run was not launched to avoid known resource exhaustion. No input reduction, semantic change, batching that drops relationships, sampling, evidence disabling, or partial Snapshot validation was used. This is not a relation-model expression failure; it means the in-memory Python reference has not demonstrated million-binding engineering scalability.

SF0.1/Q6 preflight estimated 623,808 bindings, 10,084,621,030 bytes peak RSS, 1,476.87 seconds Snapshot build, and 720.63 seconds validation. The guarded run began from scratch and continuously sampled RSS. It stopped after 464.48 seconds when available system memory fell below the 2 GiB safety threshold. Maximum process RSS was 4,205,154,304 bytes. No partial result was kept or called valid. Full samples and estimates are in `artifacts/resource_bound_decision.json` and the ignored runtime monitor.

## 13. ProvSQL independent comparator

The originally specified ProvSQL 1.3.0 did not provide sr_which. The independent which-provenance comparison therefore used the first reproducible release containing that operation.

ProvSQL v1.4.0 is pinned by formal tag commit, Docker image index digest, linux/amd64 manifest digest, PostgreSQL major, and build commands. Its tagged source/release notes expose `sr_which`, `sr_why`, `sr_how`, and `sr_counting`; the evaluator requires `sr_which(provenance(), 'provsql_tuple_mapping')` and has no fallback.

The container did not start in this session. Docker Desktop requires a restart after WSL 2.7.10 and the Windows WSL/VirtualMachinePlatform features were enabled (DISM returned restart-required code 3010); the task did not reboot the user's machine. Therefore the exact result is `external_comparator_unavailable`: 0/0 ProvSQL output rows compared, and ProvSQL false-positive/false-negative counts are `null`, not zero. No independent ProvSQL agreement is claimed. The fixed manual Oracle evidence remains valid and separate.

Sources used to pin the comparator: [ProvSQL releases](https://provsql.org/releases/), [official v1.4.0 tag](https://github.com/PierreSenellart/provsql/tree/v1.4.0), and [Docker Hub v1.4.0 image](https://hub.docker.com/layers/inriavalda/provsql/1.4.0/images/sha256-a5e3326de148f1a021df8eec2ae9f71b1f5bf672dd82bbaa9652791e6dcfe09e).

## 14. Performance and storage

Performance is descriptive only. Complete per-query entity counts, per-table canonical bytes, relation/evidence bytes, output bytes, binding/source ratios, contributor averages/maxima, and path depths are retained in `artifacts/tpch_results.json`.

| Workload | Disabled s | Enabled s | Snapshot build s | Validate s | Backward s | Forward s | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| SF0.01/Q1 | 0.491 | 188.529 | 516.352 | 252.399 | 0.518 | 0.607 | 2,160,279,552 |
| SF0.01/Q3 | 0.455 | 152.827 | 381.397 | 150.496 | 0.002 | 0.400 | 1,570,525,184 |
| SF0.01/Q6 | 0.080 | 90.574 | 150.289 | 80.431 | 0.005 | 0.189 | 1,009,897,472 |
| SF0.01/Q10 | 0.093 | 121.108 | 310.533 | 160.058 | 0.003 | 0.825 | 1,358,639,104 |
| SF0.1/Q6 guarded partial execution | — | — | not completed | not run | not run | not run | 4,205,154,304 |

No comparison to Smoke or ProvSQL performance is made.

## 15. Failure-mode and test audit

The existing Core suite passed 33/33 tests; the experiment suite passed 31/31. Automated checks attack all requested classes: duplicate identity, false Cartesian joins, missing join side, incomplete aggregation, contributor-count mismatch, silent selection/join/limit loss, equal-value identity confusion, broken intermediate bridge, fabricated base-to-final edge, missing/duplicate evidence, wrong authority/related entities/material, missing/duplicate successful operation, Snapshot drift, output contamination, nondeterminism, Oracle leakage, forbidden token/tuple fields, Decimal/date mismatch, and value-based lineage lookup.

No multiplicity, aggregation, many-to-many, evidence-closure, or traversal defect was found in the executed scope.

## 16. Capability comparison

| Capability | Core v3 | Manual Oracle | ProvSQL | Result |
|---|---:|---:|---:|---|
| selection lineage | passed | exact | unavailable | demonstrated in tested scope |
| projection lineage | passed | exact | unavailable | demonstrated in tested scope |
| one-to-many join | passed | exact | unavailable | demonstrated in tested scope |
| many-to-many join | passed | exact | unavailable | demonstrated in tested scope |
| aggregation contributors | passed | exact | unavailable | demonstrated in tested scope |
| duplicate tuple identity | passed | exact | unavailable | demonstrated in tested scope |
| backward lineage | passed | exact, 0 FP/FN | unavailable | demonstrated in tested scope |
| forward lineage | passed | exact, 0 FP/FN | unavailable | demonstrated in tested scope |
| explicit exclusions | passed | exact | N/A | demonstrated |
| evidence/operation closure | passed | N/A | N/A | demonstrated |
| output orthogonality | passed | N/A | N/A | demonstrated |
| recursive stage composition | passed | exact direct/path facts | unavailable | demonstrated in tested scope |

## 17. Claim audit

| Claim | Directly demonstrated | Inferred only | Not supported |
|---|---:|---:|---:|
| Core can express tuple-level lineage | tested operators/workloads |  |  |
| Core can replace lineage relation capture in tested operators | yes |  |  |
| Standard which-lineage is a projection of complete generation facts | manual-Oracle scope | broader workloads |  |
| Core replaces ProvSQL as a complete system |  |  | yes |
| Core supports all SQL |  |  | yes |
| Core supports field-level where-provenance |  |  | yes |
| Core supports full how-provenance |  |  | yes |
| Core is more efficient than database lineage systems |  |  | yes |

## 18. Conclusion and limitations

In the four fixed SF0.01 TPC-H queries tested here, Core v3 represented and validated 462,399 direct generation bindings without a database-specific lineage schema. Conventional outputs matched DuckDB exactly and remained byte-identical with the contract enabled. No silent loss or fabricated pairing was observed. These results support the replacement, within the tested operators and workload, of domain-specific tuple-lineage relation establishment by the general generation-time information model.

The current in-memory Python reference implementation was not evaluated on SF0.1/Q1 because its projected memory requirement exceeded the available 17 GB of physical memory. This limits the demonstrated engineering scale, not the relation semantics established in the completed workloads.

The largest completely validated single query was 178,797 bindings and the completed total was 462,399. Million-binding in-memory Snapshot scalability remains unproved. Streaming persistence, compact representations, or external-memory indexes are possible future engineering work and are not required for the present relation-principle result. ProvSQL comparison also remains unexecuted until the host can restart and run the pinned container.

## 19. Reproduction

Full experiment:

```sh
python -m experiments.database_lineage.scripts.run_all --full
```

Linux/CI convenience target:

```sh
make database-lineage-experiment
```

Fast synthetic/Core/reader/resolver CI subset:

```sh
python -m experiments.database_lineage.scripts.run_all --fast
```

The full command regenerates data, runs both full test suites and independent Oracle, executes SF0.01 twice, compares outputs/Snapshot hashes/forward/backward lineage, records the SF0.1 resource decisions without lowering standards, attempts the pinned ProvSQL comparator, and regenerates every committed JSON artifact. It returns non-zero when an original mandatory run or external comparison remains unavailable; a non-zero exit is never rewritten as a pass.
