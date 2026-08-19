# Unified Operational Projection Proof v2

## Outcome

`UNIFIED_OPERATIONAL_PROJECTION_PROOF_V2_SUPPORTED`

Within three declared profiles, database which-lineage, ordinary ECMA-426 Source Maps, and an OpenTelemetry occurrence/execution/causal trace shadow are exactly derivable from validated occurrence-specific generation facts. Each is a strict partial view of the full fact space. OpenTelemetry additionally satisfies cross-domain hierarchical consistency through the database projection, while Source Map relations satisfy multistage composition consistency through GeneratedOrigin bridges.

## 1. Complete generation-fact space

The authority is the validated occurrence-specific Core Snapshot: sources and generated origins, occurrences, supports and dispositions, bindings, relation material, primary evidence, and successful operation closure. Function-local indexes and projection objects are rebuildable views, not authority stores.

## 2. Three operational properties

P1 is exact candidate/reference equality inside each declared profile. P2 is witnessed by real same-projection/different-Snapshot counterexamples. P3 is split deliberately: OpenTelemetry uses cross-domain hierarchical projection; Source Map uses multistage relation composition.

## 3. Database: wider relation projection

The integrated rerun produced 112 candidate and 112 reference records with 0 false positives, 0 false negatives, 0 field mismatches, and 0 multiplicity mismatches.

## 4. Source Map: cross-representation position projection

The ordinary non-indexed profile reproduced 685 segments and 1385 bidirectional queries with 0 mismatches. The declared profile is `SUPPORTED`; full ECMA-426 surface coverage remains `PARTIAL` because indexed maps and other declared surfaces are excluded.

## 5. OpenTelemetry: narrower occurrence/execution/causal projection

The formal Q6 rerun reproduced 61367 Core occurrences, 61368 spans, 62557 bindings, 2382 causal Links, and canonical SHA `a0095ed24e3ad6ec58064a1b5803e532b11c85c08a5ad7541b03dac1e064efe8`.

## 6. OpenTelemetry projection through Database

Direct Core→OTel and Core→immutable DatabaseDomainProjection→OTel traversals are isolated and exact on both the small workload and Q6. They share canonical trace formatting, not occurrence/binding/producer/GeneratedOrigin extraction.

## 7. Source Map multistage composition

Core GeneratedOrigin composition and independent native SourceMapConsumer composition agree on 5/5 mappings, with zero false positives, false negatives, broken bridges, ambiguities, cycles, invented mappings, or original→final shortcut bindings. Derived paths are not GenerationBinding entities.

## 8. Strict partiality

Database has 2, OpenTelemetry has 2, and Source Map has 3 rerun counterexamples where complete generation facts differ while the declared projection remains exactly equal. Source Map additionally retains two same-output/different-source-map ambiguity cases.

## 9. Result invariance

Database and OTel CSV/JSON outputs and Source Map generated JavaScript bytes remain identical across capture modes. Control-plane metadata is absent from ordinary output.

## 10. Authority and boundaries

The second-authority count is 0. New Core changes after the Database head are 0; new schema changes and domain-specific Core fields are both zero. The two full scientific runs and normalized test summaries are deterministic (`PASS`); 117 tests passed in the second test run with 0 failures and 6 skips.

This evidence does not establish all provenance as projection, all tracing systems, the full ECMA-426 surface, arbitrary DBMS replacement, distributed causality, a universal projection algebra, existence or uniqueness of all domain projections, or reconstruction of complete generation facts from Source Map/OTel.
