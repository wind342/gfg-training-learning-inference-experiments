# Formal results

The frozen formal execution returned:

`CROSS_SYSTEM_FROZEN_PROJECTION_SUPPORTED`

All three ResNet seeds and all three diffusion seeds passed every frozen test.

| System | Passing seeds | Maximum query-profile L1 | Maximum pair interaction | Maximum rollback effect | Restoration error |
|---|---:|---:|---:|---:|---:|
| ResNet | 3/3 | 2.000000 | 32.554981 | 32.011292 | 0 |
| Diffusion U-Net | 3/3 | 2.000000 | 0.000654496 | 3.179395 | 0 |

For every seed, the exact source checkpoint and reconstructed pre-learning
version were identified; repeated inference left the model and optimizer state
unchanged and reproduced the same outputs exactly; every declared component
was called; component gates changed the complete output; support profiles varied
with the query and combined non-additively; replacing a trained component with
its exact pre-learning version changed the result; and restoring the trained
version restored the result with zero measured error.

The diffusion rollback changed both one-step epsilon predictions and complete
100-step deterministic samples. The transient sample state was not counted as
a learned-state update.
