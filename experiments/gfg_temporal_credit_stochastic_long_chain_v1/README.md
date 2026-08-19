# RL-E04 — stochastic long-chain temporal credit

This experiment changes one variable from RL-E03: every native transition also
receives an occurrence-addressed exogenous stochastic input.  The realized
input is captured as a generation source, propagated through the 64-step state
chain, and held fixed during matched counterfactual replay.

The protocol is developed and frozen in [`PROTOCOL_FREEZE.md`](PROTOCOL_FREEZE.md).
Machine-readable parameters and executable hashes are in
[`EXPERIMENT_CONTRACT.json`](EXPERIMENT_CONTRACT.json).

## Reproduce

```bash
python -m experiments.gfg_temporal_credit_stochastic_long_chain_v1.runner --development
python -m experiments.gfg_temporal_credit_stochastic_long_chain_v1.runner --formal
python -m experiments.gfg_temporal_credit_stochastic_long_chain_v1.independent_checker
python -m pytest experiments/gfg_temporal_credit_stochastic_long_chain_v1/test_runtime.py -q
```

RL-E03 is imported only for stable utility definitions and is not modified.
