# Banking77 cutter repeated-computation experiment

Base `534130d8995fbf9f6cec176afb915f1a56845702`. Only task2 210/630. No changes to the predictor, floor, selection rule, solver tolerance/caps, trust-region rule, reserve, AdamW proposal or schedule. Historical evidence is preserved. See [PROTOCOL.md](PROTOCOL.md).

Stage1 removes duplicate post-ball G@t, skips the first zero-direction apply, and reuses already-checked executed endpoints. Stage2 adds unique-control directional GEMM separately. Independent replay always computes committed-head endpoints again.

| Budget | Version | Strict solver/report identity | Sweeps | Exact endpoints | Post-ball G@t calls | Apply calls | Solver wall | Endpoint wall | Control wall | Total wall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|210|baseline|PASS|48686|728|97372|614|2.7886|0.4280|9.0191|9.8267|
|210|stage1|PASS|48686|518|48686|406|2.6903|0.3327|8.4874|9.3327|
|210|stage2|FAIL|48686|518|48686|406|not promoted|not promoted|not promoted|not promoted|
|630|baseline|PASS|64285|1879|128570|1236|2.8070|1.0693|17.6467|20.1471|
|630|stage1|PASS|64285|1249|64285|618|2.7897|0.7357|15.9693|18.5787|
|630|stage2|FAIL|64285|1249|64285|618|not promoted|not promoted|not promoted|not promoted|

Times are seconds, medians of exactly three prespecified rotated full replays; every timed replay is checked against authority. These are instrumented algorithm timings, not bare deployment latency. Historical validation times are retained separately; no best-of runs. Existing internal cutter hashes/diagnostics are inside all variants’ control timers; outer commit hashes, logs, train evaluation and independent audit are excluded.

| Budget | Version | Predictor s | Parameter geometry s | Jacobian/Gram/guard s | Reconstruction s | Materialization s | Peak GPU bytes |
|---|---|---:|---:|---:|---:|---:|---:|
|210|baseline|0.9483|0.6922|0.7570|0.8246|0.8723|119967744.0000|
|210|stage1|0.9694|0.6961|0.4952|0.6317|0.8790|119967744.0000|
|210|stage2|not promoted|not promoted|not promoted|not promoted|not promoted|not promoted|
|630|baseline|2.9168|2.0718|1.4747|1.4714|1.8013|119967744.0000|
|630|stage1|2.9585|2.0604|0.8072|1.0140|1.7393|119967744.0000|
|630|stage2|not promoted|not promoted|not promoted|not promoted|not promoted|not promoted|

| Budget | Version | Geometry pairs | Geometry unique controls | Actual apply pair rows | Actual apply unique rows | All Gram GEMVs | Skipped zero applies | Eliminated duplicate G@t |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|210|baseline|9002|4751|18004|9502|194744|0|0|
|210|stage1|9002|4751|11954|6258|146058|208|48686|
|210|stage2|9002|4751|11954|6258|146058|208|48686|
|630|baseline|11698|7545|23396|15090|257140|0|0|
|630|stage1|11698|7545|11698|7545|192855|618|64285|
|630|stage2|11698|7545|11698|7545|192855|618|64285|

| Budget | Version | Predictor GFLOPs | Endpoint GFLOPs | Directional apply estimated GFLOPs | Removed duplicate GEMV GFLOPs | Total covered GFLOPs |
|---|---|---:|---:|---:|---:|---:|
|210|baseline|236.826132|80.630252|7.107115|0.000000|336.398537|
|210|stage1|236.826132|57.371525|4.718865|0.097651|310.653911|
|210|stage2|236.826132|57.371525|2.479107|0.097651|308.414152|
|630|baseline|710.478397|208.110223|9.235618|0.000000|949.064298|
|630|stage1|710.478397|138.334044|4.617809|0.050226|874.620084|
|630|stage2|710.478397|138.334044|2.984783|0.050226|872.987058|

Dense multiply-add FLOPs are lower bounds; the inherited analytic/solver estimates exclude many elementwise, GELU and optimizer operations. Known eliminated work is explicitly subtracted. Post-ball G@t counters exclude the other two Gram-vector products for ball/cycle norms. Raw timing repetitions/ranges, all checks, checkpoint identities and FLOP ledgers are in `results/`.

All accepted variants have zero floor violations and zero old CE backward. Task1 state, frozen BERT/cache files and source hashes are checked unchanged. No tasks3–11 or test access.

The sweep counts remain fixed. This experiment measures redundant computation only; it does not establish how early a candidate would have become functionally safe.

External create-only authority: `E:\gfg-evidence\banking77-cutter-dedup-task2-v1`. Full trajectories and checkpoints remain external.

## D: rejected strict numerical identity, unchanged executed FP32 heads

D is not adopted. The first recorded numerical differences occur at step11 (210 arm) and step147 (630 arm). Same-state directional comparisons reproduce nonzero differences when using unique-row GEMM: maximum differences in those first-divergence probes are 9.486769009248164e-20 and 3.469446951953614e-18. This supports a shape-dependent floating-point cause; no tolerance was changed.

Across the complete stored records, the largest scalar differences are 4.551914400963142e-15 and 2.220446049250313e-16. These are maxima over recorded scalar fields, not unrecorded intermediate tensors. Five updates in the 210 arm change FP64 dual bytes. All 840 receiving/native/executed heads, moments, predictor outputs, selections, active pairs, sweep counts, cap decisions, actual endpoints and final checkpoint bytes remain identical. Thus strict solver/report identity fails even though the executed training trajectory matches in this task2 experiment.

D full reference-solver replay was not promoted after failing the identity prerequisite. Its executed-head hashes/counts match every independently forwarded baseline audit step; that verifies committed endpoint safety without claiming identical FP64 solver results. See `stage2__AUDIT.json`, `D_CLASSIFICATION.json` and `D_DIAGNOSTIC.json`.

## Single validation replay timings for all three variants

These measurements are separate from the three-repetition benchmark above. D has only this validation timing and was excluded from accepted-implementation benchmarking.

| Budget | Version | Strict solver/report identity | Executed head identity | Sweeps | Endpoints | Post-ball G@t | Apply calls | Solver s | Endpoint s | Control s | Total s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|210|baseline|PASS|210/210|48686|728|97372|614|2.8293|0.2964|8.6594|9.5164|
|210|stage1|PASS|210/210|48686|518|48686|406|2.7374|0.2259|8.3312|9.1405|
|210|stage2|FAIL|210/210|48686|518|48686|406|2.7447|0.1982|7.5171|8.3603|
|630|baseline|PASS|630/630|64285|1879|128570|1236|2.8268|0.9548|17.2190|19.6021|
|630|stage1|PASS|630/630|64285|1249|64285|618|2.8689|0.8002|16.8021|19.4138|
|630|stage2|FAIL|630/630|64285|1249|64285|618|2.6842|0.7276|15.8205|18.3627|

## Accepted optimization outcome

- 210: total median 9.8267 -> 9.3327s (5.03% reduction); control median 9.0191 -> 8.4874s (5.90%). Total ranges 9.3665-10.1527s vs 9.2892-9.9544s.
- 630: total median 20.1471 -> 18.5787s (7.78% reduction); control median 17.6467 -> 15.9693s (9.51%). Total ranges 19.1733-21.4387s vs 18.4452-19.5098s.

The observed gain is modest, with overlapping timing ranges; three repetitions are not a claim of statistical significance. The 48,686/64,285 sweeps remain. Any attempt to reduce those sweeps requires a separate stopping-policy/solver experiment. No such change was made here.
