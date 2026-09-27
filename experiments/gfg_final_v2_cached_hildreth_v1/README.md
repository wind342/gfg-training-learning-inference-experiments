# Cached Hildreth: 3.21x faster; transition screening adds no further speedup

Independent continuation of `8f00816a6b8c6d61ee87688b242497f95a00d306`.
All original v1/v2/v3 sources and evidence are preserved. No Banking77.

**Both correctness gates PASS. Cached-only is the fastest measured method.**
FewRel task10, 210 updates, seed2021, LR5e-5, inherited task9 V2-210 head and
AdamW moments, 10,072 old controls and 80 diagnostic-only new controls.
Moments advance from step1890 to step2100. Frozen BERT features, control bank,
relation floors, update order, loss and fixed 256-row FP32 endpoint arithmetic
are unchanged. No old-data training/replay, threshold search or solver change.

## Standalone wall time

One fresh replay per method, in order full -> cached -> transition, with no
observer, component timers or per-component CUDA synchronization. Hash checks
are outside timing. Setup, actual commit/recheck, and complete executed-anchor
refresh are inside timing. Feature loading, final test and serialization are
outside, matching the earlier algorithm-wall scope. No best-of selection or
timing confidence interval; these are measurements on one RTX 4060 Ti machine.

| Method | Exact old rows | FP32 GEMM calls | Sweeps | Control wall (s) | Total algorithm wall (s) |
|---|---:|---:|---:|---:|---:|
| Full fixed reference |21,674,944|86,080|1,522|57.6020|58.4595|
| Cached Hildreth |21,674,944|86,080|1,522|17.5959|18.2312|
| Cached + transition |7,041,245|29,730|1,522|20.5184|21.4546|

Cached-only reduces total wall by 68.81% (3.21x speedup), comfortably below the
historical full-reference 59.6486s. Transition is 17.68% slower than cached-only.
Its improvement over full reference must not be attributed to screening: the
cache optimization already supplies a larger improvement with no row reduction.

## Separate component profile

These are additional full 210-update replays with wall timers and synchronized
GPU region boundaries. All replay identities were checked again. The table's
component and total times belong to the same **profiled** replay; do not combine
them with standalone wall times above. Endpoint/materialization intervals
include Python/host scheduling and synchronization, not just GPU kernel time.

| Method | CPU geometry (s) | Budget (s) | Exact old rows | FP32 GEMM calls | Sweeps | Control wall (s) | Total wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full fixed reference |21.7320|9.2055|21,674,944|86,080|1,522|56.8713|57.6989|
| Cached Hildreth |0.1315|2.3960|21,674,944|86,080|1,522|17.8380|18.4507|
| Cached + transition |0.1379|2.4032|7,041,245|29,730|1,522|22.1994|23.1362|

| Method | Hildreth coordinate (s) | Materialization (s) | Endpoint forward (s) |
|---|---:|---:|---:|
| Full fixed reference |3.9143|1.5203|13.6910|
| Cached Hildreth |3.6611|0.6590|7.0909|
| Cached + transition |3.7000|1.2460|3.6536|

Geometry includes the once-per-task cache setup for optimized methods and the
repeated augmented native head/H/length construction in the original reference.
Budget includes envelope construction and its dynamic head transfers. Endpoint
includes the actual executed-anchor full forward, where used. Residual control
time includes margins, counts, selection, pair finding and existing reports.
Identical GEMM counts do not guarantee identical elapsed affine time in different
execution contexts; these measurements are not a causal attribution to GPU FLOPs.

## Stage 1: cache fixed geometry, construct only active-pair reserves

At task entry, create immutable CPU FP64 `H=[h,1]`, `abs(H)`,
`lengths=2*sum(H*H)`, labels and relation-to-row indices. Across 210 updates no
feature/label `.cpu()`, `.astype()` or `column_stack` reconstruction occurs in
the optimized controller. Dynamic head and margin transfers remain necessary.

Every update caches the before/native absolute class-row envelope. New active
pairs request their two budget entries once; all subsequent Hildreth sweeps
reuse those exact values. The original gamma formula, EPS=1e-6, reserve meaning,
coordinate function, ordering, closest-wrong selection, 100-sweep limit and
first-safe stopping are unchanged. No complete `10072 x 80` budget is built
in either optimized path. The unmodified reference still builds it for audit.

### Floating-point implementation detail

Naively replacing a large GEMM by a vector dot/small GEMM changes FP64 reduction
order on this NumPy/OpenBLAS build. The diagnostic shape probes found differences
up to about 6.5e-19 in budgets. They are retained in `results/REDUCTION_SHAPES.json`
and `PAIR_PADDING.json`; no changed reserve was silently accepted.

Instead, each small batch contains at most eight active pairs, their eight
absolute feature rows and only their required target/competitor envelope rows.
Repeat rows/columns to an `8 x 80` output tile to preserve the reference BLAS
reduction path. Unused cross-products and repeated padding are discarded. This
is an empirical backend-specific arithmetic contract, not a claim about all BLAS
implementations. Every one of 90,311 pair reserves was compared bit-for-bit with
the full matrix throughout the 210-update validation.

| Budget work | Full | Cached / transition |
|---|---:|---:|
| Useful requested target/competitor entries |180,622|180,622|
| Computed entries including padding |169,209,600|7,685,120|
| FP64 budget GEMM calls |210|12,008|
| Largest budget output matrix |10,072 x 80|8 x 80|

The FP64 budget GEMMs are separate from the FP32 endpoint GEMM counts above.
Padding is disclosed: this implementation is not literally two scalar dot
products per pair. It reduces computed budget entries by 95.46% while preserving
even the reference solver's internal FP64 primal and dual bytes. The obsolete
full-matrix budget checksum is replaced by an active-pair checksum; the
independent audit compares both source budget entries exactly.

## Stage 2: correctness transitions from a real executed anchor

The anchor is the previous actual executed head and its actual full-bank FP32
margins. It remains fixed across the entire next native/projection sweep path;
candidate observations never update it. A full FP32 commit evaluation refreshes
all margins after each executed endpoint, including a rollback if one occurs.

For each old row, bound its current margin on both sides using its augmented
feature norm times target/max-class displacement norm, plus FP32 roundoff at
both the anchor and candidate and conservative FP64 arithmetic allowances.
Positive lower bounds certify correct; nonpositive upper bounds certify wrong.
Neither test asks whether at least K controls can be certified correct.

Batch-exact-evaluate all ambiguous controls, every selected/active control and
the 80 new diagnostics. Both stable signs plus exact ambiguous signs yield
**complete counts**, not count lower bounds. For every deficient relation, exact
evaluate all remaining rows before ranking wrong controls, so stale margins
never influence selection. Full observers verify counts, deficits, choices,
selected exactness and both signs independently; their values never feed caches.

There were 211 anchor initializations/refreshes (initial plus 210 commits),
including 2,115,120 old-row commit refresh evaluations, already counted in the
7,041,245 exact rows. Cached receiving endpoints avoid 209 duplicate forwards.
Each of 1,522 deficient candidates requires a second exact batch. Endpoint calls
therefore rise from 2,152 to 3,465 despite the 67.51% reduction in exact rows.

Transition-specific profiled costs were bounds 2.4831s, masks 0.9638s,
gather/packing 2.1224s, scatter 1.2647s, and anchor bookkeeping 0.2030s.
The first four total 6.8339s, exceeding the reduction in endpoint-forward time.
The full executed-anchor forward is counted under endpoint, not hidden as free
refresh. Hildreth still requires 1,522 sweeps (7.2476/update) and about 3.7s in
coordinates. This identifies why fewer rows failed to improve net wall time.

No further screening threshold changes were tried. A future independent
active-set/hot-start QP experiment could address repeated solves/endpoints;
this experiment does not replace the solver or assume that a different solver
would preserve its exact trajectory.

## Correctness and evidence

| Check | Cached | Transition |
|---|---:|---:|
| Executed-head byte identity |210/210|210/210|
| AdamW moment byte identity |210/210|210/210|
| Deficits, selections and active pairs identical |PASS|PASS|
| Entire sweep sequence and FP64 primal/dual identical |PASS|PASS|
| Active-pair reserves byte-identical |90,311/90,311|90,311/90,311|
| Final test predictions identical |PASS|PASS|
| Floor violations |0|0|
| Independent candidate observer calls |N/A: full endpoint|2,152|
| False-safe / false-wrong |N/A: no screening|0 / 0|

The full reference is replayed from each receiving/native state in validation,
and all executed heads/moments also match the previously sealed v3 evidence.
The standalone full, cached and transition arms each start independently from
task9 and recheck historical identities. The same checks run in component
profiles. Test predictions are read/evaluated only after each training replay
ends. No failure or metric was used to tune the scientific policy.

Code: `core.py` (cache, pair reserves, controller, instrumented reference),
`transition.py` (two-sided screen and independent observer), `tests.py` (zero
margins, both signs, selected rows, returns and executed-only refresh), `run.py`
(gated validation/timing), `check.py` (independent cross-check/package).
`probe.py` preserves the initial reduction-shape diagnostics.

Compact per-step/per-candidate results, timing records and checks live in
`results/`, hashed by `ARTIFACT_MANIFEST.json`. Large frozen inputs and the two
new final checkpoints stay external. Local authority:
`E:/gfg-evidence/final-v2-cached-hildreth-v1`.

## Reproduction

Runtime: `E:/gfg-python/governor-v4-bench/Scripts/python.exe` (same environment as
v3). Existing inputs/old authority paths are reused read-only. For a fresh run,
set `GFG_CACHED_HILDRETH_ROOT` to a **new, nonexistent** evidence directory; do
not erase or overwrite old runs. From the repository root run:

```text
python -B -m experiments.gfg_final_v2_cached_hildreth_v1.run validate
python -B -m experiments.gfg_final_v2_cached_hildreth_v1.run benchmark
python -B -m experiments.gfg_final_v2_cached_hildreth_v1.run transition
python -B -m experiments.gfg_final_v2_cached_hildreth_v1.run transition-benchmark
python -B -m experiments.gfg_final_v2_cached_hildreth_v1.run profile
python -B -m experiments.gfg_final_v2_cached_hildreth_v1.run transition-profile
python -B -m experiments.gfg_final_v2_cached_hildreth_v1.check --verify-only
```

The entrypoints refuse existing training/timing outputs. `check` without
`--verify-only` packages a new `results/` directory and refuses an existing one.
The base guard allows only this new experiment directory to differ from BASE.
The error-bound assumptions are inherited from prior work; zero observed
false-safe/false-wrong is empirical validation, not a formal GPU hardware proof.
