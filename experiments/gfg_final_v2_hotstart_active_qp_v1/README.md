# B: warm-start active-set QP reduces checks but is not faster

Base `f357dfdb9d13759dc07a5703d6d11301a3adb946`. Separate new actuator; cached Hildreth and all previous evidence remain unchanged. FewRel task10, 210 updates only. No Banking77, old-loss replay or floor relaxation. Original fixed256 endpoint throughout.

**Safety/solver validation PASS; net speedup NOT ACHIEVED. Stop at this result.**

## Continuity before implementation

| Cached Hildreth adjacent updates (209 transitions) | Median | Mean |
|---|---:|---:|
|Final discovered-set Jaccard|0.868726|0.847699|
|Previous constraints retained|0.930288|0.916835|
|Added constraints|31.000000|31.459330|
|Deleted constraints|30.000000|30.617225|
|Positive-dual support Jaccard|0.769231|0.752174|
|Matched dual Pearson correlation|0.834403|0.789895|
|Matched dual symmetric relative L2 change|0.228901|0.254579|

These measurements justified a bounded actuator test. Final_active includes discovered zero-dual constraints; it is not synonymous with positive/binding support. CONTINUITY preserves every transition, the source hash, and both set definitions.

## Standalone result

Single independent timing per method; no best-of selection or confidence interval. Setup, state commit and actual floor recheck are included. External cold-QP audit, hashes, serialization and final test are excluded. The baseline is the fresh original fixed256 arm, not the buffer/batched implementations in A.

| Method | Solver iterations / sweeps | Mean discovered constraints | Endpoint calls | Control wall (s) | Total algorithm wall (s) | Old / new / all macro (%) | Floor violations |
|---|---|---:|---:|---:|---:|---|---:|
|Cached Hildreth|1522 sweeps|430.05|2152|17.4202|18.2998|72.8881 / 70.6899 / 72.6683|0|
|Warm active-set QP|1916 pivots / 629 solves|436.25|1259|18.5711|19.5252|72.8284 / 71.4954 / 72.6951|0|

Warm QP is 6.70% slower in standalone algorithm wall. Solve-followed endpoint checks fall by 58.67% (1522 -> 629); all endpoint calls fall by 41.50% (2152 -> 1259). Pivots and Hildreth sweeps are different operations and must not be compared as equal work units.

All 210 scheduled updates execute; no failed solves, skips or rollbacks. Final predictions differ at 81 test examples. This is a new actuator and trajectory, not byte-identical acceleration. Final macro changes versus baseline are:

| Cohort | Percentage-point change |
|---|---:|
|Old|-0.059668|
|New|+0.805498|
|All|+0.026849|

## Does warm-start use adjacent-update continuity?

Yes, in the measured iteration sense. For the first QP of each update after step1, inherited duals/support require **681 pivots versus 1186 from zero**, on the identical 209 QPs. There are on average 219.35 positive inherited multipliers in those starts. These first-QP results isolate cross-update reuse from within-update constraint expansion.

Across all 629 QPs, warm uses 1916 pivots versus 3645 cold; 617 cases take fewer pivots and one takes more. Independent cold solutions agree in objective and dual values; maximum dual difference is 0. Cold outcomes never feed the actuator. This supports useful warm starts, not a net wall-time gain.

## What is cached and what changes?

A persistent constraint registry and FP64 Gram cache survive all updates. For pair (i,c), its normal has +H_i in target class and -H_i in competitor class; Gram entries use the signed class-incidence factor times H_i dot H_j. Only missing requested entries are computed; new pairs incrementally expand the registry. Working submatrices are gathered from cached entries; principal-system factorizations are still recomputed during pivots. No claim of factorization hot updates is made.

Observed cache: 1432 unique constraints, 47,400 newly computed nonzero upper-triangle entries, 120,129,680 cached entry accesses over 629 QPs. Synthetic signed-normal Gram checks verify incremental reuse.

At each update, rebuild only the current RHS from the native head and unchanged reserve. Match the previous final pairs/duals into currently selected/discovered constraints. Previously known competitor pairs on newly selected controls can seed the current QP; rows no longer selected do not acquire extra protection merely from history. This avoids changing the current problem into permanent protection of every formerly selected control.

On average 406.61 previous pairs are inherited and 28.84 filtered per update after the first. Each active-set solve minimizes 0.5||P-P_native||^2 for its current discovered constraints. The 0.5 factor does not change the requested minimizer. EPS, reserve and relation-floor selection remain unchanged. A block principal-pivot solve checks primal/dual/complementarity residuals, followed by a direct primal check using H and the actual FP32 endpoint.

## Separate component profile and remaining cost

The profile is a separate full replay with region synchronization. Core solver wall compares Hildreth coordinate work with QP pivot/linear-solve work; Gram/RHS and reconstruction are explicitly separate. Profile times are not mixed into standalone totals.

| Profiled method | Core solver wall (s) | Gram/RHS (s) | Primal reconstruction (s) | Endpoint wall (s) | Control wall (s) | Total wall (s) |
|---|---:|---:|---:|---:|---:|---:|
|Cached Hildreth|3.6720|N/A|in coordinates|6.9926|17.5710|18.3952|
|Warm active-set QP|1.6647|3.0540|1.2337|7.0352|18.5448|19.4763|

The new solver reduces repeated endpoint checks but introduces principal-system solves, working-matrix handling and explicit primal reconstruction. Those costs prevent a net speedup here. No additional tuning, solver replacement, task expansion or A+B combination was attempted after the negative timing result.

## Verification and reproduction

32 independent SciPy NNLS synthetic cases pass before task10. All 629 actual QPs pass KKT and direct-primal checks; maximum direct violation is 1.56e-15. Cold solutions independently agree. Timed and profiled replays reproduce every validated head/moment/report, and final predictions. Every executed relation floor passes the unchanged original fixed256 endpoint.

Authority: `E:/gfg-evidence/final-v2-hotstart-active-qp-v1`. Compact results and per-QP audits are in `results/`; inputs and checkpoint remain external, identified by hashes. `ARTIFACT_MANIFEST.json` hashes source and evidence files.

With `E:/gfg-python/governor-v4-bench/Scripts/python.exe`, set `GFG_HOTSTART_ROOT` to a new nonexistent authority, then run this package modules `analysis`, `tests`, `run validate`, `run timed`, `run profile`, `report` using `python -B -m experiments.gfg_final_v2_hotstart_active_qp_v1.<module>`. Report also reads the declared fresh unoptimized baseline from A reference256; it never imports an optimized endpoint into B training. Existing output directories and packaged results are refused.
