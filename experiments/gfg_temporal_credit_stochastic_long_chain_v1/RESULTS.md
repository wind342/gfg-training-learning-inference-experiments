# RL-E04 stochastic long-chain temporal credit

Final status: **PASS**

## Formation retrieval and conditional credit

- Formal episodes: 192
- Stochastic realizations per episode: 8
- GFG candidate precision/recall/F1: 1.000000 / 1.000000 / 1.000000
- Maximum conditional credit error: 4.108e-15
- Maximum conditional pair-interaction error: 4.441e-16
- Passenger-zero accuracy: 100.00%

## Expected stochastic credit

- Expected-credit MAE against disjoint reference tapes: 0.006054
- Expected-credit sign accuracy: 100.00%
- 95% interval coverage of disjoint reference means: 95.66%
- Conditional sign stability: 100.00%

## Matched stochastic replay and occurrence binding

- Matched/independent effect-variance ratio: 0.8965
- Correct-binding factual reproduction: 100.00%
- Permuted binding changed the realized result: 100.00%
- Permuted binding changed conditional credit: 100.00%

## Computation

- Native-transition reduction vs complete matched reference: 90.63%
- Replay-only speedup: 6.35x
- End-to-end speedup including native capture, canonical base GFG validation and the representative credit-discovery GFG: 1.02x

## Downstream learning

- Held-out terminal success: 94.82%
- Held-out functional-action accuracy: 98.62%
- Mean consequence regret to hidden evaluator oracle: 0.066461

## Interpretation

The stochastic-binding control is a realized-execution test.  Because the
exogenous inputs are exchangeable, the protocol does not require permutation
to change the population expectation.  The supplied-equivalent dependency DAG
control receives the already established term/action partition and therefore
does not claim independent structure discovery.
