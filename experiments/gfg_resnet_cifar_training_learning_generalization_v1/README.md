# ResNet/CIFAR-100 cross-system training--learning generalization

This experiment changes all four major system choices used in the original
training--learning study:

| Dimension | Original system | Cross-system test |
|---|---|---|
| architecture | nanoGPT Transformer | ResNet-18 |
| modality | tokens/text | images |
| task | next-token prediction | 100-class classification |
| optimizer memory | Adam moments | SGD momentum |

The frozen protocol tests receiving-state conditioning, finite-amplitude
nonlinear response, the adapted F1/F3/F5 coordinate families, distributed
residual-stage support and target-boundary outcomes. It records only registered
scientific occurrences in a compact GFG.

Smoke test:

```powershell
python -m experiments.gfg_resnet_cifar_training_learning_generalization_v1.runner `
  --mode smoke --download `
  --output-root runtime/gfg_resnet_cifar_generalization_v1_smoke
```

By default CIFAR-100 is stored under `runtime/datasets/cifar100`. Set
`GFG_CIFAR100_ROOT` or pass `--data-root` to use another location.

Formal execution:

```powershell
python -m experiments.gfg_resnet_cifar_training_learning_generalization_v1.runner `
  --mode formal --download `
  --output-root runtime/gfg_resnet_cifar_generalization_v1_formal
python -m experiments.gfg_resnet_cifar_training_learning_generalization_v1.aggregate `
  runtime/gfg_resnet_cifar_generalization_v1_formal
python -m experiments.gfg_resnet_cifar_training_learning_generalization_v1.INDEPENDENT_CHECKER `
  runtime/gfg_resnet_cifar_generalization_v1_formal
```

The frozen formal outcome is reported in [RESULTS.md](RESULTS.md), the concise
machine-readable summary is [FORMAL_RESULTS.json](FORMAL_RESULTS.json), and the
claim boundary is assessed in
[SCIENTIFIC_ASSESSMENT.md](SCIENTIFIC_ASSESSMENT.md).
