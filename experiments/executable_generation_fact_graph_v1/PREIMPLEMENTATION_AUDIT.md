# Executable Generation-Fact Graph v1 — preimplementation audit

Audit date: 2026-07-27

## Repository state and integration ancestry

- Starting branch: `research/signed-generation-algebra-v1`
- Starting HEAD: `f06f53cb36859d7012f043f8be326ffbf9efc67f`
- Target branch: `research/executable-generation-fact-graph-v1`
- Target branch point: `f06f53cb36859d7012f043f8be326ffbf9efc67f`
- Worktree before the branch was created: clean
- Remote: `origin=https://github.com/wind342/source-information-continuity.git`

Frozen sources:

| Source | Frozen branch or commit | Frozen HEAD | Ancestry |
|---|---|---|---|
| Core v3 and repository base | `research/signed-generation-algebra-v1` | `f06f53cb36859d7012f043f8be326ffbf9efc67f` | branch point |
| Multi-stage Signal | `codex/signal-multistage-generated-origin-v1` | `5694f19d5884cb3572f944db10de2812db86b6b8` | ancestor of the branch point |
| Inter-fact hardening and scale | `maintenance/inter-fact-relations-v0-hardening-scale-v1` | `bd5354bc7a91327839b53600349490c621b6804c` | ancestor of the order source |
| Order/refund/freeze | `experiment/order-refund-freeze-inter-fact-relations-v1` | `bc672bc8ce53d9947195ce8eb35f42720b18977a` | merge source |
| Signed Generation Algebra | `research/signed-generation-algebra-v1` | `f06f53cb36859d7012f043f8be326ffbf9efc67f` | branch point |

The merge base of the branch point and the order source is
`1b31256a50481265f824d92392c45e1db4f986d6`. The inter-fact hardening
commit is an ancestor of the order source, and the Signal commit is an
ancestor of the Signed Generation Algebra source. Therefore one no-fast-forward
merge of the exact order HEAD after this audit commit preserves all four frozen
source histories.

`git merge-tree --write-tree f06f53cb36859d7012f043f8be326ffbf9efc67f
bc672bc8ce53d9947195ce8eb35f42720b18977a` produced the clean prospective
tree `912e3ee6ee06620437060f51a01ed930f2866514`; it reported no conflicts.

## Frozen tree identities

Core trees at the branch point:

- `src/generation_relation_core`: `03fbdce13249f84abe9d8fb605da31cdc36eda27`
- `protocol/core_v3`: `0b4a2608864e771ebca7cdbfad95aabaed2d0723`
- `tests/core`: `280cb44d592ae48d986719638980c11e57aab1f9`

Frozen source experiment trees:

- `experiments/signal_multistage_generated_origin_v1`:
  `9871b14722548d503324762b6dc3a222828168d0`
- Signal tests:
  `0c4e481658ccafffb7efe9a24b009d6aabd58adf`
- `experiments/inter_fact_relations_v0`:
  `fccb595dfc0a8c7272f3e6e2af6937a57f8168b7`
- `experiments/inter_fact_relations_v0_hardening_scale_v1`:
  `587ae72e94102fe4249eb1c38fa5b54ba9e78633`
- `experiments/order_refund_freeze_inter_fact_relations_v1`:
  `68f8db905678f47b5ddc02637b175b7270556e33`
- `experiments/signed_generation_algebra_v1`:
  `8cfc18a206bde2ceecca5ccee29a18f74d7b2ea1`

Protected publication trees at the branch point:

- `manuscript`: `3652bd5aef4cfc14f1bf5f1ea08e0f42cd523d47`
- `claims`: `f6b95d5ae0d8fdb77da13f9dbd48ecaed8af27ff`
- `claim_atlas`: `85203984d514bf6647f41acd8848fced47bb8bff`

## Code-driven semantic audit

The frozen Core represents one atomic generation fact in a
`GenerationBinding`. Its `origin_reference`, occurrence reference,
`outcome_reference`, and `relation_role` are hashed as relation material.
`GenerationOccurrence.transform_reference` distinguishes the realized
transformation from the concrete occurrence identity. Outcomes are explicitly
tagged support or disposition references. A `GeneratedOrigin` is a typed
source-side reference to a prior generated support; it does not create a
cross-stage shortcut.

`ValidatedSnapshot` is the only admitted atomic-fact authority. Snapshot
validation verifies entity hashes, tagged references, relation material,
evidence closure, authorized evidence authority, and exactly one successful
operation-result closure for every binding. `QueryEngine` applies only
registered support-space predicates and returns binding-level answers. These
roles will remain unchanged.

The Signal source performs synchronous capture inside the native FIR,
downsampling, FFT, and SVG loops. Its frozen result has 512 source samples,
1,117 occurrences, 8,420 bindings, 732 GeneratedOrigins, 834 supports, and 392
ExplicitDispositions. Its rectangular query selects 10 SVG cells, reaches 197
raw sources, and has 2,880 multiplicity-preserving complete paths. Only
adjacent-stage GeneratedOrigin bridges are authoritative.

The inter-fact source separates validated primitive relations from query-local
closure. Relations preserve execution-run identity, exact endpoints,
establishment source, authority, evidence, and type-specific lifting rules.
Capture completeness gates `concurrent_with`; the indexed candidate does not
materialize a global transitive closure. Its frozen large scale is 10,000
occurrences and 30,000 facts.

The order source performs real SQLite WAL transactions and real
multiprocessing Queue/Barrier/Event coordination. Atomic facts and the
primitive relation sidecar are separately captured and validated. Its 56
pre-registered scenario/query pairs are compared against a raw-receipt
reference in isolated processes.

Signed Generation Algebra computes `A±(Γ)=(P+,P-)` and its `Z[X]` net view
from frozen signed-effect rules over atomic generation facts. It is a lossy
downstream projection and is not an authority for graph nodes or graph edges.

## Integration method and byte-preservation controls

1. Commit this audit as the first commit on the target branch.
2. Merge the exact order source HEAD with `--no-ff`.
3. Add all new implementation and evidence only under
   `experiments/executable_generation_fact_graph_v1/`.
4. Do not edit Core, manuscript, claims, Claim Atlas, or any frozen experiment
   and test path.
5. Recompute every protected tree hash during finalization and fail closed on
   any mismatch.
6. Treat source reports and Claim Atlas as audit context only. Candidate graph
   compilation and query code may not import or read them.

## Candidate/reference authority boundary

The candidate path may read only validated snapshots, validated primitive
relation stores, measured capture audits, frozen graph/lifting contracts, and
registered query requests. It may not read generator internals, raw receipts,
reference outputs, existing candidate answers, reports, or a second authority
store.

Signal reference code will independently use frozen raw signal inputs and the
FIR/downsampling/FFT/render rules. Order reference code will independently use
canonical SQLite state and raw SQL, Queue, Barrier/Event, and worker receipts.
Candidate, reference, and comparison stages will execute in distinct
processes. Comparison will read only normalized candidate and reference
answers.

Graph nodes are compiled one-for-one from validated `GenerationBinding`
instances. Graph edges are compiled from validated primitive relations,
adjacent-stage `GeneratedOrigin` dependencies, or explicitly declared
traceable lifting rules. Derived paths and concurrency results remain
query-local unless a frozen rule explicitly defines a derived edge.

## Scope freeze

`G_e=(V_e,E_e)` is an executable, validated multigraph over complete atomic
generation-fact instances. It is not a sixth coordinate, a replacement Core,
or a claim that generic graphs are novel. Version 1 is limited to the frozen
Signal, order/refund/freeze, signed-effect, and 10,000-occurrence scale
profiles. It does not claim arbitrary-system concurrency completeness or
global scheduler reconstruction.
