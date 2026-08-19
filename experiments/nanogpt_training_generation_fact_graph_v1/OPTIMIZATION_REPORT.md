# nanoGPT optimization comparison

The original checkout remains unchanged at
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. The candidate is a separate
checkout. `nanoGPT_optimized.patch` is the complete source change.

## Result

The candidate passed strict correctness:

- all 28 model checkpoint tensors were bitwise equal after 10 updates;
- all AdamW optimizer state tensors and scalars were bitwise equal;
- iteration, validation loss, model arguments and configuration (except the
  intentionally different output directory) were equal;
- forward outputs were bitwise equal at sequence lengths 1, 17, 64 and 128;
- `crop_block_size(64)` remained bitwise equal and the cached position buffer
  did not enter the checkpoint state.

Performance moved in the same direction under two measurements:

- synchronized steady-state step median: 16.0667 ms to 14.553 ms, 9.42% faster;
- independent 501-update process median: 11.9193 s to 11.5030 s, 3.49% faster.

The independent process runs suffered late system-wide slowdown, so the
conservative end-to-end claim is the 3.49% median improvement. The 9.42%
number describes the synchronized steady-state loop, not the full process.

The experiment did not merge gradient-accumulation steps or remove gradient
clipping because those changes failed the strict semantic conditions.
Machine-readable measurements are in `optimization_comparison.json`.
