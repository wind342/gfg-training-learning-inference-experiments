# Executable Generation-Fact Graph v1 — final result

Final status: **EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED**

Failure reason:
`PURE_FACT_VERTEX_MODEL_CANNOT_PRESERVE_ALL_NATIVE_PRIMITIVE_RELATION_ENDPOINTS`

## Exact falsification

The frozen v1 vertex set contains only one fact node per
`GenerationBinding`. The fresh real order execution produced
83 mandatory primitive relations.
Only 28 have one and only one fact
mapping at both native endpoints. Exactly
55 cannot be represented without
dropping a relation, fabricating a fact, reattaching an endpoint, or performing
an unsupported Cartesian expansion.

- `commits_version`: total=4, unique=4, unmappable=0
- `conflicts_with`: total=2, unique=2, unmappable=0
- `generated_origin_dependency`: total=10, unique=10, unmappable=0
- `message_send_receive`: total=5, unique=0, unmappable=5
- `program_order`: total=46, unique=5, unmappable=41
- `reads_from`: total=7, unique=7, unmappable=0
- `synchronizes_with`: total=9, unique=0, unmappable=9

All prohibited action counts are zero. The compiler failed closed.

## What this does and does not falsify

This result falsifies only the combination of:

1. graph vertices are restricted to `GenerationBinding` fact nodes; and
2. every native occurrence-level primitive relation must be preserved.

It does not falsify the five-coordinate atomic generation fact, the existing
inter-fact relations, or the possibility of an executable graph with explicit
occurrence nodes.

## Other completed v1 runs

- Signal: 8420 fact nodes,
  11078 adjacent-stage edges,
  2880 exact paths and
  197 exact raw sources.
- Order source: 40 real workflow
  executions and 56/56 exact source queries.
- Scale source: 10000
  occurrences and 30000 facts;
  graph compilation stopped at the ambiguous endpoint precondition.
- Signed projection: 8
  executions exactly matched the frozen Signed Generation Algebra candidate.
- Negative controls: 48/48
  detected once each with unique reason codes.

Scientific SHA-256: `f39221d137f6e8b2dd94c1fec2874d8fc4b4d3cf102b41a7c41688aa4150266b`

## Tests

- v1 focused: `PASS` (8 passed)
- frozen Core: `PASS` (33 passed)
- full repository: `PASS` (135 passed)
