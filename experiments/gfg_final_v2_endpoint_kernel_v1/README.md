# A: endpoint engineering, independent of QP actuator B

Base `f357dfdb9d13759dc07a5703d6d11301a3adb946`. FewRel task10, 210 updates only; no Banking77. All previous experiments remain unchanged.

**Preallocated loop256 preserves the complete trajectory. Batched endpoints are a new fixed numerical implementation.**

One standalone replay per arm, fixed order shown, on the same RTX 4060 Ti. No best-of selection or confidence interval. Setup, padding and actual commit/recheck are included. External observers, hashes, serialization and final test are excluded.

| Implementation | Tile | GEMM API calls | Tile matrix products | Padded rows | Control wall (s) | Total algorithm wall (s) |
|---|---:|---:|---:|---:|---:|---:|
|reference256|256|86,080|86,080|22,036,480|17.4202|18.2998|
|buffer256|256|86,080|86,080|22,036,480|15.7962|16.4649|
|batch256|256|2,151|86,040|22,026,240|15.3207|16.0017|
|batch512|512|2,151|43,020|22,026,240|15.3640|16.0422|
|batch1024|1024|2,151|21,510|22,026,240|15.3294|16.0182|
|batch2048|2048|2,151|10,755|22,026,240|15.1710|15.8349|

GEMM API calls count Python addmm/baddbmm invocations, not physical GPU kernel launches. A baddbmm call processes multiple constituent tile matrices. No claim of one hardware launch per API call is made.

The optimized variants pad/copy the immutable feature bank once and preallocate the output. There is no per-tile new_zeros/copy/cat in their Python forward. Library-internal workspaces are not claimed eliminated. Hildreth, reserves, EPS, K, selection and stop rules are unchanged.

## Numerical and capability results

Sign differences compare each candidate head with original fixed256 on the same features, including receiving, native, sweep and commit checks. They count observations, not distinct controls. Floor columns distinguish the arm arithmetic from the original arithmetic.

| Implementation | Sign differences | Executed old sign differences | Own floor violations | Fixed256 observed floor violations | Identical heads / 210 | Old / new / all macro (%) |
|---|---:|---:|---:|---:|---:|---|
|reference256|0|0|0|0|210|72.8881 / 70.6899 / 72.6683|
|buffer256|0|0|0|0|210|72.8881 / 70.6899 / 72.6683|
|batch256|57|3|0|1|29|72.9378 / 70.6892 / 72.7130|
|batch512|57|3|0|1|29|72.9378 / 70.6892 / 72.7130|
|batch1024|57|3|0|1|29|72.9378 / 70.6892 / 72.7130|
|batch2048|57|3|0|1|29|72.9378 / 70.6892 / 72.7130|

All batched sizes share the same observed trajectory: 1,521 sweeps, 28 changed final test predictions, and head identity only through step29. Their own floor checks pass, but the original fixed256 observer detects one relation-step floor violation at step198. This cannot be reported as zero violations under both numerical contracts or as a byte-identical speedup. No floor was relaxed to hide it.

Buffer256 matches all 210 heads, moments, deficits, selections, active pairs, primal/dual sweep hashes and final predictions. Both floor checks and all same-head sign comparisons are zero.

| Batched control comparison to cached baseline | Result |
|---|---:|
|Mean selected-control Jaccard|0.917063|
|Mean active-pair Jaccard|0.912712|
|Per-update sweep-count change range|-6 to 5|

Small aggregate metric changes do not prove equivalent control behavior. Exact mismatches and overlaps are preserved in SUMMARY and per-step logs. No screening thresholds, floor adjustments or numerical-path workarounds were tried.

## Separate instrumented endpoint timing

These are independent profiled 210-update replays with synchronized region timers; their endpoint wall is not subtracted from the standalone total above. Each replay matches its own validated trajectory.

| Implementation | Endpoint wall (s) | Profiled control wall (s) | Profiled algorithm wall (s) |
|---|---:|---:|---:|
|reference256|6.9926|17.5710|18.3952|
|buffer256|5.4376|16.3557|17.0401|
|batch256|4.8552|15.7717|16.4579|
|batch512|4.9692|16.1916|16.8785|
|batch1024|4.9448|15.9613|16.6222|
|batch2048|4.9193|15.8174|16.4974|

## Evidence and reproduction

Create-only authority: `E:/gfg-evidence/final-v2-endpoint-kernel-v1`. Compact evidence is in `results/`; checkpoint hashes and artifact/source hashes are in `ARTIFACT_MANIFEST.json`. Inputs/features remain external and read-only.

Use `E:/gfg-python/governor-v4-bench/Scripts/python.exe`, set `GFG_ENDPOINT_ROOT` to a new nonexistent directory, then run this package with `run validate`, `run timed`, `run profile`, and `report` via `python -B -m experiments.gfg_final_v2_endpoint_kernel_v1.<module>`. Existing output directories are refused. The report package also refuses an existing results directory.

Bookkeeping correction: the first validation pass inherited a helper comparing historical control fields with v3 lower-bound-count reports. This produced misleading sweep/selection identity flags for the unchanged baseline. The final checker recomputes identities from stored raw reports against the correct cached full-count log; standalone/profile runs use that corrected source. No training result was rerun or altered to obtain a better outcome. Raw validation evidence is preserved.

A and B are reported separately. No optimized A endpoint is used in the B actuator.
