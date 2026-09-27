# Safe screening v2: numerical repair, complete task10 validation

**Correctness gate: PASS. Acceleration: NOT ACHIEVED.**

User authorised an engineering repair after v1's preserved frontier-GEMM sign
mismatch. The original failed source/results remain unchanged in the sibling
v1 directory and separate local authority. No main/frozen tag/old experiment
changes. This is Phase A only; Banking77 has not been started.

## Repair

Both full and sparse evaluation now use a fixed **256-row FP32 Linear primitive**,
zero-padding the last tile. Padding is counted, not hidden. This eliminates the
variable GEMM batch-shape mismatch in the tested full/sparse paths. It is a new
numerical authority; it is NOT claimed identical to historical variable-shape
cuBLAS rounding.

The unchanged production Hildreth function is called through a local dependency
injection of this numerical primitive. Its bytecode, selection policy, reserve,
EPS, 100-sweep cap, first-safe stopping, optimizer, LR, moments and data remain
unchanged. No global monkey patch and no tolerance relaxation. Unit checks
verify the function bytecode and absence of imported-module mutation. Initial
task10 control counts match original K exactly, so K was not changed either.

The certificate and 25%/100-update anchor refresh rule are unchanged from v1.
Previously selected rows still receive actual endpoint evaluation. Every
candidate is checked by a read-only full fixed-tile observer. The historical
full-batch output is separately recorded as a numerical diagnostic and never
used to steer the controller.

## Complete task10 results

Same verified V2-210 task9 checkpoint;210 updates;seed2021;LR5e-5;10072 old
controls;original batches and moments;no old loss/replay or BERT changes.

| Method | Old72 macro | New8 macro | All80 macro | Skip |
|---|---:|---:|---:|---:|
| Historical Final V2-210, original full-batch arithmetic |72.9178%|70.3308%|72.6591%|0|
| Fixed-tile full reference / screened v2 |72.8881%|70.6899%|72.6683%|0|

The fixed-tile full controller and screened controller have identical decisions,
selections, active constraints, primal/dual sweep hashes and executed heads for
**all210 updates**. Final endpoint predictions are consequently the same between
these two implementations. All210 updates are also independently reexecuted
using only the full fixed-tile production reference (no screening), including
gradient/proposal/moment and final classifier checks; see CHECKS.json.

| Safety / consistency | Result |
|---|---:|
| Candidate full-observer calls |2152|
| False-safe certificates observed |0|
| Frontier vs full fixed-tile margin max difference |0|
| Relation-count mismatches |0|
| Executed relation-floor violations |0 /15120 relation-step checks|
| Independent full-reference head/moment matches |210 /210|
| Hildreth sweeps |1522|
| Skips |0|

Historical head-byte identity holds only for the first7 steps, moments for8
steps. There were31 same-state sign differences across the fixed vs historical
numerical diagnostics. Historical final predictions are NOT identical. These
differences are disclosed; this result does not rewrite old evidence or claim
an exact acceleration of the historical floating-point trajectory.

## Cost

| Quantity | Result |
|---|---:|
| Full logical old candidate rows on the new trajectory |21674944|
| Screened exact old candidate rows |6288145|
| Useful exact candidate-row reduction |70.99%|
| Additional FP64 anchor refresh rows |1062142|
| Relation refreshes including initialization |7592|
| Executed padded candidate rows (including current audit rows) |6714112|
| Full fixed-reference padded rows (same call count) |22036480|
| Frontier ratio median /p90 /max |28.78% /31.58% /36.51%|
| Screened algorithm wall including setup/refresh |55.7943 s|
| Paired full fixed-reference algorithm wall |47.0980 s|
| Speedup |0.8441x;screened18.46% slower|
| Independent per-candidate checker wall |15.7088 s,excluded above|

Timing uses CUDA synchronization. The reference shares the measured native
current-batch update cost and runs the complete original controller using the
fixed numerical primitive. Required screening, refresh, Hildreth and transfers
are included; independent observers/replay/serialization/test are separate.
Interleaved paired timings are diagnostic, not standalone device throughput.
Logical rows are not FLOPs or bytes, and FP64 refresh work is additional.

Screening overhead exceeded the work it saved in this implementation. This
establishes numerical consistency, not an engineering speedup. No post-result
threshold or controller tuning was performed.

## Evidence and use

`PROTOCOL.md` freezes the repair before the formal run. `results/` contains
preflight source/input hashes, every candidate certificate/observer record,
every step projection/moment identity, anchor refreshes, final metrics and
independent replay. `ARTIFACT_MANIFEST.json` records public byte hashes and
the external small final checkpoint. No raw examples or BERT weights are added.

Local authority: `E:/gfg-evidence/final-v2-safe-screening-v2`.
The underlying cached BERT feature/data/checkpoint authorities are those of
`gfg_frozen_head_baseline_comparison_v1`, verified by hash. The BERT backbone is
never instantiated or updated in this head-only run.

`run.py` is create-only and requires the declared parent HEAD. It does not
overwrite/resume an existing authority. `diagnostics/verify.py` independently
replays all210 steps using the full production controller with the fixed affine
primitive. These diagnostic files were added after formal source freeze without
changing the frozen implementation. They write only new records.

## Scope

Safety is the original fixed per-relation control-count contract, not a guarantee
of identical historical test predictions or no population-level forgetting.
The certificate uses explicit floating-point error-model assumptions; the
observations are not a formal verification of GPU hardware. Both the failure
and repair are preserved. Further performance work is separate from the
completed correctness repair; no Banking77 or larger training was run.
