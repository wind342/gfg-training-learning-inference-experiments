# INF-G01 — cross-system frozen-inference projection

This experiment extends the frozen-inference test from nanoGPT to both prior
cross-system training--learning falsification systems:

- ResNet-18 on CIFAR-100 with SGD momentum; and
- a compact time-conditioned diffusion U-Net on CIFAR-10 with AdamW.

Each system has three independently trained frozen checkpoints. Every seed must
pass the same four causal requirements: exact learned-version identity,
component recruitment during a state-frozen execution, query-conditioned and
non-additive combination of distributed support, and output dependence on the
learned component versions under exact rollback and restoration.

For diffusion, the noisy image, timestep and transient sampling state are query
or execution state. They are not treated as persistent learned state.

The frozen protocol is in `PROTOCOL_FREEZE.md`, formal results are in
`FORMAL_RESULTS.json`, and the independent recomputation entry point is:

```bash
python -m experiments.gfg_cross_system_frozen_inference_projection_v1.INDEPENDENT_CHECKER
```

Native re-execution requires the original CIFAR datasets, the frozen TL-G01 and
TL-G02 checkpoint directories, PyTorch with CUDA and torchvision. Use:

```bash
python -m experiments.gfg_cross_system_frozen_inference_projection_v1.REPRODUCE_RUN \
  --resnet-data-root <cifar100-root> \
  --diffusion-data-root <cifar10-root> \
  --output-root <new-empty-output-path>
```
