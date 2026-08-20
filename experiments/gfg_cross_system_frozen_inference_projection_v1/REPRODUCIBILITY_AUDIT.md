# Reproducibility audit

The native formal execution and a second execution initiated through
`REPRODUCE_RUN.py` independently returned
`CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED`. All six runs passed the independent
checker in both executions.

The generated files were byte-identical across the two executions:

| File | SHA-256 |
|---|---|
| `FORMAL_RESULTS.json` | `a25f549a414eecb921a5d9a3cdfffbc54eb30626c6759cff1032ba4d4be5312b` |
| `FORMAL_GFG.json` | `76483536511fa5c6d1a08d227a4dbed59773edcc32f407db286aa7ac9d358e5c` |
| `RUN_RESULTS.json` | `217765dfe460ef83cdda6e10011815b011305f377c937d9aeac380ead27ec67c` |

The formal Python environment used PyTorch `2.11.0+cu128` and torchvision
`0.26.0+cu128` with CUDA available. Unit tests include rejection of a tampered
formal verdict and a tampered query-conditioning metric.
