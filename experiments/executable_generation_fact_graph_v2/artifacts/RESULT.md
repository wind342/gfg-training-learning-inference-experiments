# Executable Generation-Fact Graph v2 — result

Final status: **EXECUTABLE_GENERATION_FACT_GRAPH_V2_SUPPORTED**

Scientific SHA-256: `06708547d966a5474512c977daae7ee5670d3dabcb8b39962f1339a554ca8711`

## Preserved v1 falsification

v1 remains `EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED` because exactly
55 mandatory primitive relations could
not be represented when every vertex was required to be a
`GenerationBinding` fact node.

## v2 object

`G_e=(V_F,V_O,E_I,E_R;Sigma)`

- `V_F`: one complete five-coordinate fact node per `GenerationBinding`.
- `V_O`: one execution occurrence node per referenced concrete occurrence.
- `E_I`: exact `OccurrenceNode --realizes_fact--> FactNode` incidence.
- `E_R`: typed native-endpoint execution and generation relations.
- `Sigma`: schemas, endpoint signatures, evidence, capture and canonical rules.

Occurrence nodes are graph execution skeleton nodes. They are not a sixth
coordinate of the atomic generation fact.

## Exact machine results

- Signal: 8420 facts,
  1117 occurrences,
  11078 adjacent-stage relations,
  2880 exact paths and
  197 exact raw sources.
- Order: 83/
  83 primitive relations preserved;
  direct graph queries 56/56 exact;
  compensation targets 4/4 exact;
  projection compatibility
  56/56 exact;
  FP=0,
  FN=0.
- Order zero-fact occurrences:
  44.
- Scale: 10000 occurrences,
  30000 facts,
  30000 incidence edges and
  28900 primitive relation edges.
- Signed projection: 8
  executions exact.
- Negative controls:
  48/48 detected.

## Projection boundary

`pi_Gamma`, `pi_occ`, `pi_R` and `pi_signed` passed their frozen exactness
checks. `pi_fact_graph` is explicitly lossy for occurrence-level primitive
relations and does not claim complete sidecar recovery.

## Limits

`shortest_path` means minimum admitted edge count under a supplied relation
policy. Weighted cost, critical path and algorithmic optimization remain
`NOT_EVALUATED` / `NOT_ESTABLISHED`. Recompilation validation shares the
compiler implementation, so semantic query claims are separately checked
against independent profile references. Scale timings are diagnostic only.

## Test finalization

- v2 focused: `PASS`
- frozen Core: `PASS`
- full repository: `PASS`
- protected paths: `PASS`
- final status: `EXECUTABLE_GENERATION_FACT_GRAPH_V2_SUPPORTED`
