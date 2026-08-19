# GFG machine semantics orientation

This orientation uses an unrelated toy graph. The target nanoGPT graph is
mounted but permission-locked and unreadable during this gate.

A `generation_occurrence` identifies one concrete execution. A
`generation_fact` identifies one atomic relation established by that
execution:

`(source information, realized transformation, concrete occurrence, formed
outcome; relation role)`.

An occurrence can establish several facts. Those facts remain separate and
must not be broken into source and result sets and recombined by Cartesian
product.

`realizes_fact` connects an occurrence node to a fact it established.
`reads_from` records an actual data dependency. `GeneratedOrigin` allows an
outcome formed earlier to participate as a source later. `program_order`
records declared execution order; it does not by itself establish data
dependency. Primitive relations are generation-time records. Derived
relations must state the traversal rule from primitive records.

Equal numeric values do not identify the same fact, occurrence or result.
Equal outputs do not establish equal formation. The absence of a concurrency
edge does not prove concurrency. A missing relation can mean that the
relation was not established; it does not automatically prove that the
relation is false.

For a training graph, follow exact source and result identities:

training sample -> batch occurrence -> forward occurrence -> loss -> backward
occurrence -> gradient -> optimizer occurrence -> new parameter version.

The reverse direction starts from an evaluation or parameter result and
follows its actual `reads_from`, `GeneratedOrigin`, `realizes_fact` and
incidence relations. Timestamp proximity and numeric similarity are not
valid substitutes.

Before the five-minute gate ends, write `orientation_receipt.json` in the
repository root with:

- `schema` equal to `gfg-orientation-receipt-v1`;
- `read_complete` equal to `true`;
- `target_gfg_accessed` equal to `false`;
- non-empty fields named `generation_fact`, `generation_occurrence`,
  `atomic_generation_fact`, `realizes_fact`, `reads_from`,
  `GeneratedOrigin`, `program_order`, `equal_values`,
  `cartesian_recombination`, `missing_relations`, `forward_query` and
  `reverse_query`.

Use those fields to restate the distinctions and query boundaries above.
After writing the receipt, wait for the external runner to validate it and
release the target evidence.
