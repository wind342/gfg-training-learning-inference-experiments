# RL-E03 — long-chain temporal credit and self-optimization

This experiment extends RL-E02 in two ways. First, each early action reaches a
terminal-only consequence through a real 64-step versioned state-formation
chain rather than a terminal read of an early slot. Second, the exact
credit-discovery computation is itself captured as canonical Core v3 facts and
compiled into a validated GFG, whose term-specific formation paths guide an
exact factorized Shapley calculation.

The protocol was frozen before formal execution in
[`PROTOCOL_FREEZE.md`](PROTOCOL_FREEZE.md). Machine-readable parameters are in
[`EXPERIMENT_CONTRACT.json`](EXPERIMENT_CONTRACT.json), and the scientific
result is in [`RESULTS.md`](RESULTS.md).

## Reproduce

From the repository root, with the repository and `src/` on `PYTHONPATH`:

```bash
python -m experiments.gfg_temporal_credit_long_chain_self_optimization_v1.runner
python -m experiments.gfg_temporal_credit_long_chain_self_optimization_v1.independent_checker
python -m pytest experiments/gfg_temporal_credit_long_chain_self_optimization_v1/test_runtime.py -q
```

The formal run executes 192 episodes and 6,291,456 exact-reference native
transitions. Artifacts and their SHA-256 identities are under `artifacts/`.

## Claim boundary

The GFG-guided method reproduced exact credit while using 90.64% fewer native
transitions and was 2.38x faster after including its one-time meta-GFG cost.
However, a conventional dependency DAG given the same term-to-action
dependency partition matched that optimized result. The experiment therefore
supports validated formation-structure-guided optimization, but does not claim
that equivalent computational savings are exclusive to GFG.
