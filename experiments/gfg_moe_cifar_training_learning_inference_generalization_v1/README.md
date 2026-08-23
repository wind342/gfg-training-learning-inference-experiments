# MoE/CIFAR-100 cross-system validation

This experiment moves the frozen cross-system programme to a deterministic top-2 routed
Vision Mixture-of-Experts classifier trained from random initialization with AdamW.

- `TL-G03` retests receiving-state conditioning, finite-amplitude nonlinear response,
  three-coordinate prediction and persistent reorganization of distributed expert support.
- `INF-G02` retests frozen, query-conditioned, non-additive inference and exact
  pre-learning expert/router rollback.

The protocol and thresholds are frozen in `PROTOCOL_FREEZE.md` and
`MODEL_CONTRACT.json`. A supported result is bounded to this executed system.

## Commands

```powershell
python -m experiments.gfg_moe_cifar_training_learning_inference_generalization_v1.runner --mode smoke --data-root D:\codex\datasets\cifar100
python -m experiments.gfg_moe_cifar_training_learning_inference_generalization_v1.runner --mode formal --data-root D:\codex\datasets\cifar100
```

Aggregation and the independent checker are run only after all formal seeds finish.

## Frozen result

The three-seed formal run produced the joint verdict
`MOE_UNIFIED_DYNAMICS_SUPPORTED`. The exact aggregate, runtime-file manifest,
independent-check result and bounded scientific interpretation are retained in:

- `FORMAL_RESULTS.json`
- `FORMAL_RUNTIME_MANIFEST.json`
- `INDEPENDENT_CHECK_RESULT.json`
- `RESULTS.md`
- `SCIENTIFIC_ASSESSMENT.md`

The implementation evaluates all four expert branches and assigns exactly zero
weight to experts outside the deterministic top-2 selection. It therefore tests
the exact routed function and expert organization, but makes no sparse-kernel or
systems-throughput claim.
