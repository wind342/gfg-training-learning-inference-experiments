# TL-G02 — DDPM/CIFAR-10 cross-system generalization

This falsification-first experiment changes the system under study from
autoregressive language modelling and supervised image classification to a
generative diffusion objective. A compact time-conditioned U-Net predicts the
concrete noise occurrence used to form a noisy CIFAR-10 image.

The evaluation identity is fixed by `(image, timestep, noise occurrence)`. Its
readout boundary compares the squared error of the realized noise residual with
seven frozen nearby residual competitors. No class label or post-update output
is available to the predictive model.

The frozen protocol is in `PROTOCOL_FREEZE.md`; exact configuration and
prohibited inputs are in `MODEL_CONTRACT.json`.

Smoke execution:

```bash
python -m experiments.gfg_ddpm_cifar_training_learning_generalization_v1.REPRODUCE_RUN \
  --seed 20260830 --smoke --download \
  --output-root runtime/gfg_ddpm_cifar_generalization_v1/smoke/seed_20260830
```

Formal execution and verification:

```powershell
foreach ($seed in 20260830, 20260831, 20260832) {
  python -m experiments.gfg_ddpm_cifar_training_learning_generalization_v1.REPRODUCE_RUN `
    --seed $seed --download `
    --output-root "runtime/gfg_ddpm_cifar_generalization_v1/formal/seed_$seed"
}
python -m experiments.gfg_ddpm_cifar_training_learning_generalization_v1.aggregate `
  runtime/gfg_ddpm_cifar_generalization_v1/formal
python -m experiments.gfg_ddpm_cifar_training_learning_generalization_v1.INDEPENDENT_CHECKER `
  runtime/gfg_ddpm_cifar_generalization_v1/formal
```

The frozen formal outcome is reported in [RESULTS.md](RESULTS.md), the concise
machine-readable summary is [FORMAL_RESULTS.json](FORMAL_RESULTS.json), and the
claim boundary is assessed in
[SCIENTIFIC_ASSESSMENT.md](SCIENTIFIC_ASSESSMENT.md).
