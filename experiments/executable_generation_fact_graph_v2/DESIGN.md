# Executable Generation-Fact Graph v2 — design freeze

## Formal object

For one concrete execution \(e\), v2 is:

\[
G_e=(V_F,V_O,E_I,E_R;\Sigma).
\]

- \(V_F\) contains one `FactNode` per admitted `GenerationBinding`.
- \(V_O\) contains one `OccurrenceNode` per concrete occurrence referenced by
  a fact, primitive relation, registered query contract, or required exact
  execution path.
- \(E_I\) contains exact `OccurrenceNode --realizes_fact--> FactNode`
  incidence derived from the fact occurrence identity.
- \(E_R\) contains validated typed primitive relations and explicitly
  traceable derived relations with their native endpoint domains.
- \(\Sigma\) contains node/relation signatures, evidence and authority rules,
  run scope, capture completeness, and canonical serialization contracts.

The graph is a typed, attributed, directed multigraph. Symmetric relations are
stored once with symmetric semantics.

## Atomic fact boundary

The atomic fact remains:

\[
f=(u,\tau,\bar{\omega},z;\rho).
\]

An `OccurrenceNode` is not a sixth coordinate, not an atomic fact, and not a
member of the atomic fact multiset \(\Gamma\). It is an execution-skeleton
node that can realize zero, one, or multiple facts.

## Fact node

Each admitted `GenerationBinding` produces exactly one node whose instance
identity hashes graph schema version, execution run, snapshot, and binding
identity. It retains:

- the complete five coordinates;
- `generation_occurrence_id` and outcome identity;
- evidence references;
- content and instance hashes;
- validated native aliases where an adapter proves exact coordinate
  correspondence.

Equal content in different bindings or runs is never collapsed. Multiplicity
is list-instance multiplicity, not set membership.

## Occurrence node

Each referenced occurrence produces exactly one run-scoped node. Its source
must be one of:

1. a Core `GenerationOccurrence` in a `ValidatedSnapshot`;
2. a frozen synchronous execution receipt;
3. a relation endpoint catalog whose identity and payload have been validated
   against receipts and relation endpoints.

The node retains concrete occurrence identity, type, stage, stable instance
key, index, transform reference, payload, manifest reference, evidence, and
content/instance hashes. A zero-fact occurrence is valid and has no incidence
edge. A multi-fact occurrence has one independent incidence edge per fact.

Native occurrence aliases and Core occurrence identities may coexist, but a
validated alias map must prove the correspondence. No nearest-event,
timestamp, log-text, or similarity inference is admitted.

## Incidence

For every fact:

```text
OccurrenceNode --realizes_fact--> FactNode
```

is derived exactly once from its occurrence reference. This edge is graph
structure, not a primitive relation and not part of the primitive-sidecar
projection.

## Primitive relation endpoints

Every frozen relation type is registered only after machine endpoint census.
The registry fixes:

- allowed source and target node kinds;
- directed or symmetric semantics;
- primitive or derived status;
- lifting policy;
- capture completeness requirement;
- cycle policy and, when acyclic, the declared acyclic relation family.

Primitive relations retain original relation ID, endpoint kinds, endpoint
identities, payload, establishment source, authority, evidence, rule, input
references, and run identity. v2 never forces `Occurrence -> Occurrence` into
`Fact -> Fact`, never drops zero-fact endpoints, and never expands a
multi-fact occurrence into a Cartesian product.

## Compilation and validation

Compilation order is frozen:

1. construct all FactNodes;
2. construct all referenced OccurrenceNodes;
3. construct exact incidence edges;
4. construct primitive relations with native endpoint domains;
5. construct only contract-authorized derived edges;
6. build canonical indexes;
7. emit an unvalidated graph;
8. independently validate against every admitted authority;
9. issue `ValidatedGenerationFactGraphV2` only after all gates pass.

Validation recompiles expected instances and checks one-to-one coverage,
content, identity, multiplicity, run scope, endpoint signatures, evidence,
authority, no drop/fabrication/lifting/Cartesian expansion, canonical hashing,
reordering invariance, and two-run determinism.

Recompilation validation is a structural mutation check, not an independent
semantic oracle, because it invokes the same compiler. Frozen profile query
claims are therefore compared separately with independent runtime references.

No global transitive closure is persisted. Path and reachability results are
query-local.

## Projections

- \(\pi_\Gamma(G_e)\) selects FactNodes and exactly restores the admitted
  atomic fact instances. OccurrenceNodes do not enter \(\Gamma\).
- \(\pi_{occ}(G_e)\) selects OccurrenceNodes and occurrence-level primitive
  execution relations.
- \(\pi_R(G_e)\) selects primitive RelationEdges and exactly restores the
  complete validated sidecar, excluding incidence and query-local results.
- \(\pi_{fact\_graph}(G_e)\) is an explicitly lossy fact-only view. It reports
  retained fact relations and omitted occurrence relations and never claims
  complete sidecar recovery.
- \(\pi_{signed}(G_e)\) applies only the frozen signed-effect contract to
  FactNodes. OccurrenceNodes and graph relations do not infer sign, and
  `ExplicitDisposition` is not automatically negative.

## Query boundary

The common query engine accepts only a validated graph. It supports fact and
occurrence lookup, typed predecessors/successors, relation enumeration,
formation and execution subgraphs, conflicts, policy-bounded downstream
reachability, projections, and path queries. Domain conclusions such as a
compensation target require a frozen domain query policy and are not inferred
from generic reachability.

`shortest_path` means minimum primitive/allowed-derived edge count under an
explicit relation policy. It is not a claim about computational cost,
causality strength, semantic importance, or optimal execution.

## Scientific release boundary

v2 is supported only if all mandatory gates pass for Signal, Order, Scale,
projections, authority isolation, 48 unique negative controls, protected
paths, Core tests, and the full repository test suite. Any failure produces
`EXECUTABLE_GENERATION_FACT_GRAPH_V2_NOT_SUPPORTED` without reducing the gate
or rewriting frozen evidence.
