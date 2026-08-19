# RL-E03 long-chain temporal-credit discovery and self-optimization

Final status: **PASS**

## Long formation chain

- Formal episodes: 192
- GFG candidate precision/recall/F1: 1.000000 / 1.000000 / 1.000000
- Chronological history reduction: 85.94%
- Every retained early action crossed at least 36 native state transitions before the terminal-only consequence.

## Exact causal credit

- Maximum GFG-guided vs exact credit error: 6.106e-16
- Maximum pair-interaction error: 2.220e-16
- Passenger-zero accuracy: 100.00%
- Mean absolute pure three-way finite difference: 0.200000

## Computation

- Native-transition reduction vs exact: 90.64%
- End-to-end speedup including the one-time credit-discovery GFG: 2.38x
- Exact and GFG-guided credit remained equal within the frozen 1e-12 tolerance.

## Downstream learning

- Held-out terminal success: 98.63%
- Held-out functional-action accuracy: 98.99%

## Critical falsification result

An ordinary value-dependency DAG supplied with the same term-to-action dependency partition matched the GFG-guided exact decomposition. Therefore this execution establishes that validated generation relations can guide exact cheaper credit discovery, but it does **not** establish that the computational saving is exclusive to GFG rather than any representation preserving equivalent dependency structure.

Formation ancestry remained distinct from causal credit: the GFG retained three real passenger ancestors, and matched forks assigned all three zero scalar credit.
