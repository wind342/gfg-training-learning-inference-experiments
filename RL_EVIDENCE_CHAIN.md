# Reinforcement-learning evidence chain

The four reinforcement-learning experiments form one cumulative test of the
GFG-based recursive scientific process. The sequence begins by establishing
which relations enter an actual training action, removes the predeclared
relation and discovers it from execution history, recompiles the discovery
computation to reduce its causal-adjudication cost, and finally tests the same
method under occurrence-addressed stochastic variation.

## RL-E01 — consequence binding and temporal credit

RL-E01 established the relation to be discovered. With the receiving
parameter–optimizer state, actions and physical consequences held identical,
changing only consequence binding or temporal credit changed the actual
parameter update and post-update logits in all 48 matched one-step causal
forks. Correct binding and credit re-formed the reversed policy in all twelve
formal seeds; neither disrupted condition reached the frozen 90% threshold in
any seed.

- [Frozen protocol](experiments/gfg_rl_feedback_closure_v1/PROTOCOL_FREEZE.md)
- [Formal results](experiments/gfg_rl_feedback_closure_v1/RESULTS.md)
- [Machine-readable summary](experiments/gfg_rl_feedback_closure_v1/FORMAL_RESULT_SUMMARY.json)
- [Independent checker](experiments/gfg_rl_feedback_closure_v1/independent_checker.py)

## RL-E02 — GFG-guided temporal-credit discovery

RL-E02 removed the true credit relation from the method. Starting from a
terminal consequence in a 64-action history, GFG formation-path retrieval
returned nine candidates while retaining all six functional actions and three
formation ancestors without scalar causal effect. Matched causal replay
assigned zero credit to those passengers and recovered credit signs and
pairwise interactions exactly. Training with the discovered credit matched the
hidden-oracle held-out result, including 96.48% terminal success and 99.08%
functional-action accuracy.

- [Frozen protocol](experiments/gfg_temporal_credit_discovery_v1/PROTOCOL_FREEZE.md)
- [Formal results](experiments/gfg_temporal_credit_discovery_v1/RESULTS.md)
- [Machine-readable summary](experiments/gfg_temporal_credit_discovery_v1/FORMAL_RESULT_SUMMARY.json)
- [Independent checker](experiments/gfg_temporal_credit_discovery_v1/independent_checker.py)

## RL-E03 — recursive optimization of credit discovery

RL-E03 replaced the direct terminal dependency with a genuine 64-step
versioned state-formation chain. Every retained early action passed through at
least 36 native state transformations before the terminal-only consequence.
The credit-discovery computation itself was captured as canonical Core v3
facts and compiled into a validated GFG. Recompiling its term-specific
formation paths preserved exact credit within the frozen `1e-12` tolerance,
reduced native replay transitions by 90.64% and produced a 2.38-fold
end-to-end speedup including the one-time credit-discovery GFG construction.
The held-out policy reached 98.63% terminal success and 98.99%
functional-action accuracy.

- [Frozen protocol](experiments/gfg_temporal_credit_long_chain_self_optimization_v1/PROTOCOL_FREEZE.md)
- [Formal results](experiments/gfg_temporal_credit_long_chain_self_optimization_v1/RESULTS.md)
- [Machine-readable summary](experiments/gfg_temporal_credit_long_chain_self_optimization_v1/artifacts/FORMAL_RESULT_SUMMARY.json)
- [Independent checker](experiments/gfg_temporal_credit_long_chain_self_optimization_v1/independent_checker.py)

An ordinary dependency DAG supplied with the already established
term-to-action partition reproduced the optimized adjudication. The measured
saving therefore follows from the validated formation structure exposed by
the recursive process; the experiment does not claim that an equivalent
supplied dependency representation could not execute the same factorization.

## RL-E04 — stochastic long-chain temporal credit

RL-E04 introduced an occurrence-addressed exogenous stochastic input into
every native transition of the long formation chain. Matched replay held the
realized stochastic tape fixed while changing candidate actions, separating
action-contingent credit from realized environmental variation. Conditional
credit signs were recovered with 100% accuracy, passenger-zero accuracy was
100%, and permuting stochastic occurrence binding changed both the realized
result and conditional credit in every formal case. Training with the
discovered credit achieved 94.82% terminal success and 98.62%
functional-action accuracy on held-out policies.

- [Frozen protocol](experiments/gfg_temporal_credit_stochastic_long_chain_v1/PROTOCOL_FREEZE.md)
- [Formal results](experiments/gfg_temporal_credit_stochastic_long_chain_v1/RESULTS.md)
- [Machine-readable summary](experiments/gfg_temporal_credit_stochastic_long_chain_v1/artifacts/FORMAL_RESULT_SUMMARY.json)
- [Independent checker](experiments/gfg_temporal_credit_stochastic_long_chain_v1/independent_checker.py)

## Cumulative result

The sequence tests four successive operations of the recursive scientific
process:

1. establish a formation relation through matched causal intervention;
2. discover the relation when it is not supplied;
3. compile the discovery computation itself and use its formation structure to
   reorganize subsequent experimentation; and
4. preserve the distinction between action credit and realized environmental
   variation in stochastic long-chain execution.

Formation ancestry is treated only as candidate evidence throughout the
sequence. Causal credit is established separately by matched replay and
coalition intervention.

