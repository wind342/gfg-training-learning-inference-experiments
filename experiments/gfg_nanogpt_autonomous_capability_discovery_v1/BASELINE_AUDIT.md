# Baseline audit

## Frozen starting point

- Repository: `wind342/source-information-continuity`
- Branch: `codex/datafusion-nanogpt-optimization-experiments-v1`
- Commit: `f8a9df00dcd4b240d2a911d431553d1b9cc84b7a`
- Parent: `a918f8b8dd5b9fb33616841f3751a2c2751f17b0`
- Commit tree: `7fb65bf134b65ec2375bafdb40fd515da6d95fbb`
- Audit mode: read-only inspection before creation of this branch

## Existing nanoGPT experiment

The frozen experiment is
`experiments/nanogpt_training_generation_fact_graph_v1/`, tree
`7d5cbb4a8a8880d5b6d0ce8296ac288fdadd0d48`.

It uses the unmodified nanoGPT checkout at
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. Its actual CUDA training
profile has four layers, four heads, 128 embedding dimensions, two gradient
accumulation micro-steps and three optimizer updates.

`capture_runtime.py` supplies a write-only `TorchDispatchMode`. Each completed
tensor-producing or tensor-mutating ATen dispatch records the exact runtime
input references, output references, operator schema, occurrence identity,
step, micro-step and phase. Manual synchronous receipts additionally bind:

- exact dataset windows and input/target tensors;
- initialized parameter versions;
- each parameter gradient at both accumulation boundaries;
- global gradient clipping;
- each fused AdamW parameter and optimizer-state update;
- `zero_grad` completion.

`core_snapshot.py` maps these records into the frozen Core v3 entities,
relation evidence, operation closure and one validated Snapshot. Earlier
tensor results enter later relations through `GeneratedOrigin`. It builds
atomic facts with `u`, `tau`, `omega_bar`, `z` and `rho`, preserving the
input role from the actual operation boundary.

The committed validation reports:

- 3,630 concrete events;
- 4,359 tensor outcomes;
- 13,010 atomic generation facts;
- 197 registered sources;
- 27 parameter tensors;
- complete input-reference resolution;
- one producer per runtime outcome;
- primary relation evidence for every binding;
- complete gradient and optimizer receipts for every parameter;
- zero heuristic or similarity links.

## Existing source-optimization experiment

The same tree contains `nanoGPT_optimized.patch`,
`optimization_comparison.json` and `OPTIMIZATION_REPORT.md`. The candidate:

1. caches the position-index tensor as a non-persistent model buffer; and
2. vectorizes batch-window construction.

The change was applied to a separate checkout. After ten real updates, all
28 model checkpoint tensors and all AdamW state tensors/scalars were bitwise
equal. Variable-length forward and `crop_block_size` checks were also
bitwise equal. The conservative full-process median improvement was 3.49%;
the synchronized steady-state-loop median improvement was 9.42%.

This establishes that the existing training GFG can expose repeated
executions and semantic boundaries sufficient to form and reject source-code
optimizations. It does not discover or validate a capability-formation
mechanism. The present experiment does not repeat that optimization.

## Reusable components

Directly reused without modifying their frozen files:

- `TrainingFactRecorder`, tensor identity and synchronous receipt patterns
  from the existing nanoGPT experiment;
- Core v3 canonical entities, evidence, manifests, operation closure and
  Snapshot validation;
- Executable GFG v2 canonical graph, relation registry, validation and query
  semantics where their frozen interfaces apply;
- candidate isolation, sealing, deterministic replay and protected-path
  audit patterns from `gfg_vs_trace_ai_optimization_v1`.

The three-step nanoGPT artifact writer is not directly reusable for a long
training run because it materializes complete JSON objects in memory and
commits large run artifacts. This experiment therefore adds a local,
append-only, content-addressed training adapter under its own directory. It
does not change Core, Executable GFG, or the earlier experiment.

## Existing DataFusion experiment

`experiments/datafusion_union_all_optimization_v1/` is frozen at tree
`e1d45a033483d9a828a645c99b536254a07a2519f0`. It is outside the scientific
and modification scope of this experiment.

## Frozen experiment tree hashes

| Experiment entry | Git object |
|---|---|
| `audio_pcm` | `4153d455674ca754a313234ecdccaf8fdacb3c44` |
| `core_v3_collection.py` | `cb7597205294f44ee87ed9f80e1fe6c5b369abd8` |
| `cross_projection_relations_v1` | `01d2f25713cc135f14a53ed1b8393625dc58ee12` |
| `database_lineage` | `64b5365d9a828a645c99b536254a07a2519f0cc0` |
| `datafusion_union_all_optimization_v1` | `e1d45a033483d9a828a645c99b536254a07a2519f0cc0` |
| `docx` | `8d90094975229baac5a97373f1026a238b29e560` |
| `executable_generation_fact_graph_v1` | `891ab228e7337c5030c3f5691a6ec3d74092027b` |
| `executable_generation_fact_graph_v2` | `75b2ff3a6b0ee6ff77e9537c3182eb3c7a57da67` |
| `five_generation_fact_irredundancy_v1` | `c3f61a6970cdd6792d4bee1af9cbf48708fae0de` |
| `five_profile_unified_projection_proof` | `1b375ce37f43d98b753034c68268582c366381c5` |
| `four_domain_problem_resolution` | `d394948cb2427c9c18dbe01d80e710038416c8c4` |
| `four_generation_fact_irredundancy_v1` | `669e88401f282bbd23436304d7938a7212810d49` |
| `gfg_vs_trace_ai_optimization_v1` | `a597c33508f674d6a07dafb7ee6b72b04ad58bd4` |
| `html` | `6c6385613a687b2df78801797a26397d34fc868b` |
| `hyperspectral` | `a658d67eb941328c95053f16211d6ef06abcccfa` |
| `inter_fact_relations_v0` | `fccb595dfc0a8c7272f3e6e2af6937a57f8168b7` |
| `inter_fact_relations_v0_hardening_scale_v1` | `587ae72e94102fe4249eb1c38fa5b54ba9e78633` |
| `latex` | `da3bc0f0bab7f09a2e6c91dec6f66b7897f0f6d5` |
| `nanogpt_training_generation_fact_graph_v1` | `7d5cbb4a8a8880d5b6d0ce8296ac288fdadd0d48` |
| `opentelemetry_projection` | `884af1ac48fa115e332cfbc026c82cba90f3805b` |
| `operational_projection_proof` | `88f98c574821bfb7c61d94b11301118d7cbe866a` |
| `operational_projection_proof_v2` | `3631d29601d7ed2cf8bd65214b8b04775c833b83` |
| `order_refund_freeze_inter_fact_relations_v1` | `68f8db905678f47b5ddc02637b175b7270556e33` |
| `pptx` | `d02f3023e6528fecfde7da0bf9955ed5b48bc8a1` |
| `pytorch_autograd_training_lineage_v1` | `75eac6fb2f707f6a4f2ad7e57399acd9f24b1b10` |
| `recursive_generation` | `6e57da67c8931bea28eae396df6657baf3e2d2a7` |
| `signal` | `d2aeb9ae50b0ef06a146bd43cbe11a44f99de16b` |
| `signal_multistage_generated_origin_v1` | `9871b14722548d503324762b6dc3a222828168d0` |
| `signed_generation_algebra_v1` | `8cfc18a206bde2ceecca5ccee29a18f74d7b2ea1` |
| `source_map_projection` | `503885689013d2502e438fc65a54a85b9e1b1407` |
| `svg_pixel` | `4a76786dba51a63b61a68687b27fd9c9c069b8c4` |
| `w3c_prov_projection_v1` | `8662fa487df26f32f25cedf1280d97f22e0002e0` |

All these objects are protected. The allowed tracked change set for the new
experiment is limited to:

- `experiments/gfg_nanogpt_autonomous_capability_discovery_v1/**`
- `tests/experiments/gfg_nanogpt_autonomous_capability_discovery_v1/**`
