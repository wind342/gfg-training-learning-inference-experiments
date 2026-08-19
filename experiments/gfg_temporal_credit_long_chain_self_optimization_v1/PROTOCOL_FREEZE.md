# Long-chain temporal-credit discovery and self-optimization protocol v1

## Frozen scientific questions

This experiment asks two questions in sequence.

1. Can a validated Generation-Fact Graph retrieve a compact set of historical
   action occurrences when every relevant early action reaches a terminal-only
   consequence through at least 32 real state transformations, without treating
   formation ancestry as causal credit?
2. Can the validated formation structure of the credit-discovery computation
   reduce exact causal-adjudication work while preserving the exact reference
   credit, sign, interaction and downstream policy conclusions?

The second question is evaluated only after the first question passes its
structural gates.

## Frozen environment

- Horizon: 64 binary decisions.
- Nine opaque event occurrences enter nine versioned state-component chains.
- Six actions are functionally relevant and three are genuine formation
  ancestors but scalar-consequence passengers.
- Every event occurs by step 27. Its state contribution is transformed at every
  subsequent step and therefore crosses at least 36 native state transitions
  before the terminal consequence.
- All other actions produce isolated audit outcomes and are not ancestors of
  the terminal consequence.
- The terminal evaluator reads only the final versioned state. It does not read
  any early action or slot directly.
- The terminal scalar is the exact sum of opaque generated terms containing
  necessity, backup/substitution, pair synergy, a pure three-way interaction
  pressure term and a global closure term. A passenger term is actually read
  but is algebraically zero.
- No intermediate reward is emitted.

The participant-visible graph may contain actual source identities,
transformations, occurrences, outcomes and typed generation relations. It may
not contain `causal_credit`, `necessary`, `backup`, `substitution`, `synergy`,
`functional_action`, hidden target actions or oracle labels.

## Formation capture and validation

The authoritative execution is captured as atomic facts and adapted into the
canonical Core v3 schema. The Core must produce a `ValidatedSnapshot`. That
snapshot and its exact generated-origin relation records are then compiled by
the repository GFG v2 compiler and independently validated as a
`ValidatedGenerationFactGraphV2`.

Candidate retrieval begins at the terminal scalar fact and follows only
`generated_origin_dependency` edges. `program_order` is unavailable to the
retriever. Retrieval is expected to reduce 64 chronological actions to nine
formation ancestors. Causal replay must then assign zero credit to the three
passengers and recover the signed effects and interactions of the other six.

## Exact reference authority

For a retrieved candidate set of size k, the reference enumerates all 2^k
coalitions. Coalition members retain their observed action; non-members receive
the matched binary alternative. Every replay restores the exact episode
initial state and executes all 64 native transitions. No direct terminal
projection is permitted.

Exact Shapley credit over the scalar consequence and exact pair interactions
are the scientific authority. Optimized methods may not redefine this target.

## Credit-discovery GFG and optimizers

The reference discovery computation itself records checkpoint restoration,
coalition assignment, native transitions, opaque terminal terms, scalar
consequence and credit aggregation. Its derived generation facts form a second
validated GFG linked to the base execution by source identities and hashes.

All optimization conditions receive the same candidates, behavior actions,
initial states, replay semantics and correctness target.

- `exact_naive`: all scalar coalitions, executed from the initial state.
- `trace_profile`: chronological event payloads and measured costs, with shared
  prefix memoization but no formation edges.
- `hand_engineered`: a frozen generic prefix/checkpoint cache; it has no hidden
  term-to-candidate map.
- `dependency_dag`: a conventional value-dependency DAG with the same native
  endpoints but without Core coordinates, evidence or graph validation.
- `gfg_guided`: discovers the exact additive terminal decomposition and each
  term's action ancestry from validated GFG paths, enumerates the exact game for
  each term, and sums term Shapley values by linearity.
- `rewired_gfg`: formation edges are deterministically rewired. It must either
  fail equality and be rejected or fall back to the exact method; it may not
  report changed credit as an optimization success.

The dependency-DAG condition is a falsification control. If it matches GFG,
the experiment supports structure-guided optimization but not a GFG-exclusive
advantage.

## Frozen gates and metrics

Correctness precedes cost. For deterministic formal runs:

- optimized credit must equal exact credit within 1e-12;
- credit sign accuracy, passenger-zero accuracy, pair-interaction recovery and
  downstream held-out policy outcomes must match the exact authority;
- canonical Core validation and GFG v2 validation must pass;
- rewired relations may not silently pass;
- the pure three-way term must be retained, preventing single/pair-only pruning.

Cost reports include scalar/term coalition evaluations, full native replay
count, native transition count, unique cached state count, checkpoint restores,
wall-clock time, peak resident memory where available, graph construction cost
and total optimization overhead.

A strong computational result requires at least 75% fewer native transitions
than `exact_naive` and at least 2x end-to-end wall-clock improvement including
graph overhead. Failure to reach these thresholds is reported unchanged.

## Claim boundary

Success can establish only that, in the executed observable, deterministic and
exactly replayable long-chain system, validated generation relations support
compact candidate retrieval and can guide an exact, cheaper credit-discovery
procedure. It cannot establish that arbitrary environments are replayable, that
worst-case interaction complexity disappears, or that ordinary dependency
graphs cannot provide the same optimization when they preserve equivalent
structure.
