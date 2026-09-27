# Safe screening v3: less arithmetic, no demonstrated wall-time benefit

**Correctness: PASS. Net acceleration: NOT ACHIEVED.**

Independent task10 engineering continuation from
`2f5e45beb5bdd2b4031ddb8461d56beb34f0ee3f`.
No old source/results, main, release tag, model, optimizer, LR, K,
Hildreth solver, projection reserve or training policy changes. No Banking77.

## Implemented changes

- Replace 72 per-relation head anchors and their 72 x 80 x 1537 FP64 displacement
  tensors by a single previous candidate head and 80 class-row displacement
  norm upper bounds. Triangle inequality transports per-row cached margins
  through cumulative class/max-class displacement paths. Every actual candidate
  transition, including returns/rollbacks, is accounted for.
- Relation-floor-aware checking: first prove as many positives as possible;
  batch only up to the remaining K deficit from unproven controls. Safe relations
  stop. Still-unresolved relations get complete remaining frontier checks, so
  deficits and closest-wrong selections remain exact. Selected Hildreth rows
  always get exact endpoint evaluation.
- Delete the 25%-frontier/100-update refresh rule. Exact frontier evaluations
  update their own cached lower bounds; no extra refresh forward and no FP64
  feature/head matrix product.
- Vectorized floor masks, one precomputed group-row map, shared z/m scratch
  buffers and batched cross-relation gathers. The original fixed 256-row FP32
  affine primitive is deliberately retained; its allocation and small-GEMM
  costs have NOT been claimed solved.

Safe-relation counts in controller records are **lower bounds**, not fabricated
full counts. Full observer counts are stored separately. Unknown, unneeded rows
are not labelled truly incorrect. Deficient relations are completely resolved.

## Fixed experiment

Same FewRel V2-210 task9 checkpoint, seed2021, 210 task10 updates, LR5e-5,
inherited AdamW moments 1890 -> 2100, frozen cached BERT features, 10072 old
controls, 80 diagnostic-only new controls. Test is read only after training.
No replay or old CE backward.

All 2152 candidate checks passed an independent full fixed-arithmetic observer.
All 210 executed heads, optimizer moments, selection decisions, active pairs,
primal/dual sweep identities and final predictions match the full reference.
Separate full-reference replay and v2/v3 standalone replays also match.

| Check | Result |
|---|---:|
| False-safe certificates |0|
| Relation floor violations |0 / 15120 executed relation-step checks|
| Deficit / closest-wrong selection mismatches |0|
| Exact-row FP32 margin difference |0|
| Executed head matches |210 / 210|
| AdamW moment matches |210 / 210|
| Hildreth sweeps |1522|
| Skips |0|
| Old72 / new8 / all80 macro |72.8881% / 70.6899% / 72.6683%|
| v2 final test predictions |identical|

## Work and standalone wall

The times below are **fresh standalone replays**, not the earlier v2 55.79s /
47.10s paired timings. Same machine, data and exact state trajectory. One
measurement per path, order full -> v3 -> v2, not best-of runs; no timing
confidence interval. No independent full observer or CUDA profiling events in
these timed paths. Hash verification is outside timing. Required safety checks,
refresh, projection and actual commit/recheck remain inside timing.

| Version | Exact old rows | Additional refresh forward rows | Screening FP64 work | Control wall | Total algorithm wall | False-safe | Floor violations |
|---|---:|---:|---|---:|---:|---:|---:|
| Full fixed reference |21,674,944|0|No screening certificate|58.8005s|59.6486s|N/A (no screening)|0|
| v2 |6,288,145|1,062,142|72 x 80 x 1537 displacement; FP64 anchor GEMMs|57.3785s|58.0165s|0|0|
| v3 |5,873,707|0|80 x 1537 class displacement; row-vector cache operations|63.1572s|64.0391s|0|0|

v3 is 7.36% slower than full and 10.38% slower than v2 in these single timing
samples. The initial profiled/observer-interleaved paired pass also did not
show acceleration: v3 57.4125s vs full 55.3588s. Different timing modes are
reported separately, never combined to manufacture a speedup.

v3 refreshes 5,873,707 cached row margins **using the same exact forward rows**;
these are not additional matrix-multiplication rows. Cache update operations
do cost time and are included. Shared Hildreth CPU FP64 work is unchanged and
not included in the table's *screening* FP64 column.

### Why arithmetic really decreased

- Peak displacement tensor: 8,853,120 -> 122,960 FP64 elements (72x smaller).
  This is tensor size, not a claim that all certificate FLOPs fell 72x.
- Additional FP64 refresh GEMMs: approximately 261.202 billion multiply/add
  FLOPs -> zero.
- Actual padded FP32 candidate rows: 6,714,112 -> 6,600,192 (only 1.70% less
  than v2, though 70.05% less than full). This explicitly counts padding and
  both early-stopping batches, not just selected logical rows.
- v3 still launches **25,782 fixed-tile GEMMs**, versus v2's 26,227.

The FLOP figures are analytical dense-matrix multiply/add counts, not GPU
hardware counters. They exclude bias/reduction, vector operations, transfers,
padding initialization, and unchanged Hildreth operations. See WORK_ANALYSIS.
They establish real arithmetic elimination, **not a measured GPU kernel-time
or end-to-end speedup**.

### Located bottlenecks

1. **Little floor slack.** Of 154,944 relation-candidate pairs, 122,756 (79.23%)
   are exactly at K and only 8,418 (5.43%) are above K. The early-stop path omits
   81,784 uncertified rows; it cannot generally stop far before resolving the
   controls needed to maintain a tight floor. This is an observed omission,
   not a separately run ablation effect.
2. **CPU geometry work remains substantial.** A separate, correctness-checked
   first-30-update cProfile records 6.5418s in screened_projection: 1.3467s
   cumulative in logit_roundoff_budgets, 0.7473s in column_stack, 0.6265s in
   ndarray.astype, and 0.5795s in Tensor.cpu across profiled calls. These times
   can overlap through nesting and must not simply be added. Each projected
   scheduled update still rebuilds fixed H and lengths on CPU and computes
   the unchanged full-bank FP64 reserve matrix. These costs were not changed
   by speeding up the screening certificate.
3. **Scheduling/packing costs.** Whole-run CUDA timeline spans: certificate
   3.125s, floor masks 2.214s, gather/pack 3.011s, exact affine 2.321s, cache/
   scatter 2.299s. These spans include host launch gaps; they are not pure
   GPU SM execution measurements. Small forwards, dynamic gathers and mask/
   cache bookkeeping remain; fewer rows alone does not amortize them.

The next engineering target, if authorised, is reuse of immutable CPU feature
geometry/buffers and reduction of transfer/scheduling overhead, without changing
the reserve formula or controller decisions. No additional mechanism, parameter
sweep, new task or automatic policy revision was run here.

## Artifacts / reproduction

- `core.py`: new screen, bounds, floor-aware evaluation, comparison of exact
  decisions while respecting count-lower-bound semantics.
- `run.py`: preflight tests, one task10 run, read-only full candidate observers,
  paired full reference, sealed final test evaluation.
- `diagnostics/verify_benchmark.py`: independent full replay; standalone full,
  v3 and v2 replays; separate 30-update profile. No timing-based policy changes.
- `diagnostics/package.py`: checked compact public records and hash manifest.
- `results/`: exact machine-readable counts, decisions, hashes, work analysis,
  timings, profile and CHECKS. `ARTIFACT_MANIFEST.json`: artifact SHA-256.

Local authority: `E:/gfg-evidence/final-v2-safe-screening-v3`.
The dataset, BERT features and task9 checkpoint are reused read-only from the
existing baseline-comparison authority identified in PREFLIGHT. They are not
duplicated or published here. The only external new checkpoint is the small
final head/moments checkpoint identified by the manifest.

Python runtime used: `E:/gfg-python/governor-v4-bench/Scripts/python.exe`.
Run modules with `python -B -m experiments.gfg_final_architecture_v2_safe_screening_v3.run`
and `...diagnostics.verify_benchmark`, then `...diagnostics.package`.
The create-only training runner requires BASE as HEAD with these new v3 files
present (a separate BASE worktree with v3 sources copied in); it refuses to
overwrite an existing authority. The verifier needs no HEAD reset, only intact
old files and the declared input authorities. Use another machine/location with
the same mapped input paths for a fresh reproduction; do not erase old runs.

Old evidence remains intact. The floating-point certificate retains explicit
error-model assumptions; observed zero false-safe is not a hardware-level
formal proof. This engineering result is preserved as **correct but not faster**.
