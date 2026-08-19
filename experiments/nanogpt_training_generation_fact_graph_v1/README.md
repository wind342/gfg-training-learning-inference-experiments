# nanoGPT Training Generation-Fact Graph v1

This additive experiment runs a pinned, unmodified nanoGPT model and its
PyTorch training operations on a real CUDA device. A write-only dispatch
collector receives each completed tensor-producing ATen call synchronously.
Explicit receipts cover gradient accumulation, clipping, fused AdamW
parameter/state updates and `zero_grad`.

The frozen capture semantics are in `capture_contract.json`. The complete
machine graph is `artifacts/complete_generation_fact_graph.json`; the SVG
files are validated aggregate views, not replacements for the atomic facts.

The experiment intentionally stops after graph construction and validation.
It does not propose or test an optimization.

## Reproduction

Clone nanoGPT at the pinned commit outside the tracked repository, prepare
`data/shakespeare_char`, and create a CUDA-capable Python environment with
PyTorch. From the repository root:

```text
python -m experiments.nanogpt_training_generation_fact_graph_v1.run_experiment \
  --trainer-root data_private/nanogpt_training_gfg_v1/nanoGPT
```

The machine checks require all input references to resolve, one producer per
runtime outcome, all five coordinates on every atomic fact, complete primary
evidence resolution in Core v3, both micro-step gradient receipts for every
parameter, and one optimizer boundary per training step.

## Source optimization comparison

The graph-supported source optimization is preserved as
`nanoGPT_optimized.patch`. It vectorizes batch construction and caches
position indices without changing the frozen training result. After ten real
updates, every model tensor and AdamW state value was bitwise equal.

`OPTIMIZATION_REPORT.md` gives the result summary and
`optimization_comparison.json` contains the measured values. The conservative
full-process median improvement is 3.49%; the synchronized steady-state loop
improvement is 9.42%.
