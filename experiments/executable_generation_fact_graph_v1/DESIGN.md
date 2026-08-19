# Design freeze

## Final falsification

This file records the frozen v1 hypothesis. Its
`MIN_BINDING_ID_ANCHOR` occurrence-lifting proposal did not satisfy the
required exact native endpoint semantics. Fresh endpoint census found
occurrences with no fact mapping in the order profile and multi-fact
occurrences in the scale profile. Selecting one arbitrary binding would
reattach the native relation; selecting all bindings would create a Cartesian
expansion. Both are prohibited.

The implementation therefore rejects those inputs before complete graph
compilation. The immutable result is
`EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED`; exact counts and relation
IDs are in `artifacts/`.

## Authority

The graph is a compiled view, not a second fact authority. Its only admitted
inputs are:

1. one or more Core v3 `ValidatedSnapshot` documents;
2. validated primitive relation stores;
3. measured capture audits;
4. the frozen graph profile and lifting contract.

Node coordinates are copied from `GenerationBinding`, its referenced
`GenerationOccurrence`, and its referenced outcome. Native fact identities
used by pre-existing experiments are retained as aliases only after their five
coordinates have been checked against the Core node.

## Node identity

`graph_node_id` hashes schema version, execution run, snapshot, and binding
identity. `fact_content_hash` hashes only the five-coordinate content.
`node_instance_hash` hashes content plus run, snapshot, binding, occurrence,
and outcome identities. Consequently equal content in separate instances or
runs is not collapsed.

## Edge identity

Edges are first-class multigraph instances. Symmetric relations use canonical
endpoint order and one edge with `relation_semantics="symmetric"`. Directed
relations retain direction. Primitive relation-store edges retain their native
relation row in `relation_payload`, which permits exact projection without
using a hidden relation table.

Occurrence-level relations are lifted by the frozen `MIN_BINDING_ID_ANCHOR`
rule: one deterministic fact node from each exact occurrence is selected.
This preserves the primitive relation instance without producing an
all-facts Cartesian product. Fact-level relations retain exact fact endpoints.

Adjacent-stage `GeneratedOrigin` edges are compiled from the Core snapshot.
Every producer binding of the exact prior support is connected to every
consumer binding using that exact `GeneratedOrigin`. These paths are required
by the support formation itself; no raw-to-final shortcut or source-union
assignment is introduced.

## Validation and query

Validation recompiles the expected graph from the same authorities, checks
node and primitive-edge coverage, coordinate identity, evidence and authority
closure, run scope, traceability, DAG declarations, capture-completeness
gates, and canonical hashing. Only `ValidatedGenerationFactGraph` is accepted
by `GenerationFactGraphQueryEngine`.

The query engine maintains local adjacency indexes. Reachability and paths are
computed per request and no global transitive closure is persisted.

## Projection

`π_atom` returns every admitted binding instance with its five coordinates,
scope, occurrence and outcome identities, and multiplicity.

`π_rel` returns the native primitive relation rows stored in primitive graph
edges.

`π_signed` applies the frozen signed-effect contract to `π_atom`; graph edges
do not determine signs and ExplicitDisposition is not automatically negative.
