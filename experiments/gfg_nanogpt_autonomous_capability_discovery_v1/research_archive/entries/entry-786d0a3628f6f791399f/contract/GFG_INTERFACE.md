# Participant GFG interface

The evidence database contains append-only, hash-chained `graph_blocks`.
Each zlib-compressed canonical JSON payload contains:

- `objects`: exact tensor/literal identities, content hashes, dtype, shape,
  role, optimizer step and either an object locator or exact replay locator;
- `occurrences`: concrete transformations and occurrence identities;
- `fact_blocks`: a reversible encoding of atomic generation facts;
- `edges`: explicit `GeneratedOrigin` and `program_order` relations.

A fact block contains exactly one outcome and only the ordered
`(source, relation_role)` entries that actually participated in forming that
outcome. It is not a source-by-outcome Cartesian template. The atomic identity
is the canonical hash of scope, source object, occurrence, outcome object and
relation role. `realizes_fact`,
`origin_incidence`, `outcome_incidence` and `reads_from` follow directly.
Expansion preserves multiplicity; it never joins by step, value or time.

`gfg_client.GFG` supplies summary, evaluation, object, occurrence, fact-block,
edge and materialized-tensor access. It logs helper calls, but helper usage is
not a success criterion. Nonmaterialized objects retain their exact content
hash and deterministic replay locator. Evaluation-grid, boundary and batch
objects are directly materialized.
