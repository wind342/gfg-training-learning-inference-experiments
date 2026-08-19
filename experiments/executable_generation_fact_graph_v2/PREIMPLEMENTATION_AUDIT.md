# Executable Generation-Fact Graph v2 — preimplementation audit

Audit date: 2026-07-27

## Repository and branch state

- Current branch: `research/executable-generation-fact-graph-v2`
- Branch point and current HEAD:
  `cc3a4cd19846e7ebb811afa8a8b4602628397779`
- Immutable v1 final branch:
  `research/executable-generation-fact-graph-v1`
- Immutable v1 final commit:
  `cc3a4cd19846e7ebb811afa8a8b4602628397779`
- v1 experiment tree:
  `891ab228e7337c5030c3f5691a6ec3d74092027b`
- v1 final status:
  `EXECUTABLE_GENERATION_FACT_GRAPH_V1_NOT_SUPPORTED`
- v1 failure reason:
  `PURE_FACT_VERTEX_MODEL_CANNOT_PRESERVE_ALL_NATIVE_PRIMITIVE_RELATION_ENDPOINTS`

The v2 branch was created directly from the immutable v1 final commit. No
rebase, squash, amend, force-push, or frozen-source rewrite was used.

## Frozen source heads and ancestry

| Source | Frozen head | Relation to v2 branch point |
|---|---|---|
| Signal multi-stage native execution | `5694f19d5884cb3572f944db10de2812db86b6b8` | ancestor through Signed v1 |
| Inter-fact hardening and scale | `bd5354bc7a91327839b53600349490c621b6804c` | ancestor through Order v1 |
| Order/refund/freeze | `bc672bc8ce53d9947195ce8eb35f42720b18977a` | exact no-ff merge source in v1 history |
| Signed Generation Algebra | `f06f53cb36859d7012f043f8be326ffbf9efc67f` | v1 branch ancestor |
| Pure-fact graph v1 final | `cc3a4cd19846e7ebb811afa8a8b4602628397779` | v2 branch point |

The merge base of Signed v1 and the Order source remains
`1b31256a50481265f824d92392c45e1db4f986d6`. The merge base of v2 and the
v1 final commit is the v1 final commit itself.

## Frozen tree identities

| Protected tree | Git tree |
|---|---|
| `src/generation_relation_core` | `03fbdce13249f84abe9d8fb605da31cdc36eda27` |
| `protocol/core_v3` | `0b4a2608864e771ebca7cdbfad95aabaed2d0723` |
| `tests/core` | `280cb44d592ae48d986719638980c11e57aab1f9` |
| Signal experiment | `9871b14722548d503324762b6dc3a222828168d0` |
| Inter-fact base | `fccb595dfc0a8c7272f3e6e2af6937a57f8168b7` |
| Inter-fact hardening and scale | `587ae72e94102fe4249eb1c38fa5b54ba9e78633` |
| Order/refund/freeze | `68f8db905678f47b5ddc02637b175b7270556e33` |
| Signed Generation Algebra | `8cfc18a206bde2ceecca5ccee29a18f74d7b2ea1` |
| v1 final experiment | `891ab228e7337c5030c3f5691a6ec3d74092027b` |
| `manuscript` | `3652bd5aef4cfc14f1bf5f1ea08e0f42cd523d47` |
| `claims` | `f6b95d5ae0d8fdb77da13f9dbd48ecaed8af27ff` |
| `claim_atlas` | `85203984d514bf6647f41acd8848fced47bb8bff` |

All new implementation and evidence will be confined to
`experiments/executable_generation_fact_graph_v2/`. Finalization will compare
every protected tree to this table and fail closed on any difference.

## Exact v1 falsification carried into v2

Fresh v1 execution measured 27 order facts and 83 mandatory primitive
relations:

- 23 native `fact -> fact` relations;
- 60 native `occurrence -> occurrence` relations;
- 28 relations with exactly one fact mapping at both endpoints;
- 25 relations with one endpoint lacking a fact mapping;
- 30 relations with both endpoints lacking fact mappings;
- 55 relations that cannot legally enter a fact-only vertex graph.

The unmappable set is exactly:

- 41 of 46 `program_order`;
- 9 of 9 `synchronizes_with`;
- 5 of 5 `message_send_receive`.

The Scale profile additionally measured 28,900 primitive relations, of which
15,900 have multi-fact occurrence endpoints and therefore no unique fact
lifting. v1 performed zero relation drops, fact fabrications, endpoint
reattachments, or Cartesian expansions.

These counts establish the need for heterogeneous nodes. They do not change
the five-coordinate atomic fact and do not make an occurrence node a sixth
coordinate.

## Candidate, reference, and comparison authority

The v2 candidate compiler may read only:

1. Core v3 `ValidatedSnapshot` objects;
2. validated primitive relation stores;
3. an occurrence endpoint catalog validated against synchronous receipts,
   snapshots, and relation endpoints;
4. measured capture audits;
5. frozen graph, endpoint, relation, query, and projection contracts.

It may not read raw query answers, reference outputs, source reports, hidden
fact/occurrence/relation tables, or generator variables that were not
delivered by capture.

References remain independent:

- Signal reads frozen input, mathematical operations, receipts, and query
  geometry; it imports no v2 compiler or query engine.
- Order reads canonical SQLite state plus raw SQL, Queue, Barrier/Event, and
  worker receipts; it reads neither candidate graphs nor validated sidecars.
- Scale reads controlled executor receipts and the frozen query contract.

Candidate, reference, and comparison run in distinct processes where the
source experiment already provides that boundary. Comparison reads only
canonical outputs. Static import/file-read audits and runtime input manifests
will verify the boundary.

## Integration conclusion

v2 requires `FactNode` and `OccurrenceNode` because native primitive
relations have both fact and occurrence endpoint domains. The change is a graph
representation repair, not a Core or atomic-fact change. v1 remains immutable
as the falsification witness and will become the explicitly lossy
`fact-only projection` boundary of v2.
