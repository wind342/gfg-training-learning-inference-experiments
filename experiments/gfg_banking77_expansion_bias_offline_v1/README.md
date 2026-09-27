# Banking77: two-boundary offline new-class bias correction

Both severe expansion boundaries fully recover the exact pre-expansion correct
control masks, not just the counts. One prescribed beta per boundary, no retries.
Independent scalar verification: PASS (`results/CHECKS.json`).

Base: `ce8124de3bd9549520d2327d6949c6522fe3ee45`.
Uses the 210/task6 and 630/task5 end checkpoints from the expansion audit.
Recomputed fixed256 FP32 full77 logits match the saved boundary arrays byte for
byte. No model parameters, optimizer states, K, predictor or cutter are changed.
No training, backward, optimizer step, test read or 11-task replay is performed.

## Results

| Arm / boundary | Controls | Before | Raw expansion | Corrected | Recovered | Exact pre-expansion mask restored |
|---|---:|---:|---:|---:|---:|---|
| 210 / T6 -> T7 |420|158|19|158|139|Yes|
| 630 / T5 -> T6 |350|153|0|153|153|Yes|

| Arm | gap min | gap median | gap max | beta = max gap + 1e-6 | Applied FP32 beta |
|---|---:|---:|---:|---:|---:|
|210|0.011108636856079102|0.33437061309814453|0.7144219875335693|0.7144229875335694|0.7144230008125305|
|630|0.7881120443344116|3.302193343639374|4.690316140651703|4.690317140651703|4.690317153930664|

Gaps use only expansion-induced flips. They are computed from the represented
FP32 logits in FP64; the resulting beta is cast once to FP32. Correction itself
is FP32 subtraction on a separate logits copy, exclusively for the seven new
columns. Old columns and still-future columns remain byte-identical. The minimum
corrected margin among rescued controls is `1.0132789611816406e-6` in both cases.

## Per-relation and error accounting

| Metric |210|630|
|---|---:|---:|
| Relations with zero correct before expansion |11|3|
| Newly zero relations after raw expansion |22|32|
| Newly zero relations after correction |0|0|
| Total zero relations after correction |11|3|
| Raw -> corrected correct-to-wrong |0|0|
| Raw -> corrected wrong-to-correct |139|153|
| Before -> corrected correct-to-wrong |0|0|
| Before -> corrected wrong-to-correct |0|0|
| Existing old-competitor errors remaining |262|197|
| Newly introduced old-competitor errors |0|0|

All per-relation counts and all per-control margins, labels, IDs and correctness
are in `results/RESULTS.json`. The remaining 262/197 old-competitor errors already
existed before expansion; the correction neither introduces nor repairs them.
The original 11/3 zero-count relations similarly remain. All expansion-induced
relation-to-zero events are eliminated.

## Interpretation boundary

Yes: one uniform downward correction to newly admitted logits is sufficient to
reverse every catastrophic expansion flip at these two fixed boundaries on the
old algorithmic controls. This tests an algebraic consequence with real archived
checkpoints and actual FP32 arithmetic. Beta is calculated from those same old
controls; this is not held-out validation of beta, an evaluation of new-class
accuracy, or a demonstration of successful subsequent training. No controller
integration or repair is part of this experiment.

## Reproduction and integrity

`PROTOCOL.md` specifies the single correction before execution. `run.py` loads
only allowed original artifacts, checks input/checkpoint hashes, recomputes
logits without gradients, checks model/optimizer identity before and after, and
creates new results. It never constructs an optimizer or calls the controller.
`verify.py` independently recomputes the correction and strict comparisons with
scalar loops using saved logits; it imports neither the model nor the runner.
Both numerical result and verification outputs are create-only.

```text
python -B -m experiments.gfg_banking77_expansion_bias_offline_v1.run
python -B -m experiments.gfg_banking77_expansion_bias_offline_v1.verify
```

Runtime used: `E:/gfg-python/governor-v4-bench/Scripts/python.exe`.
Input paths and SHA-256 identities are in `results/RESULTS.json`; source hashes
cover `PROTOCOL.md`, `run.py`, and `verify.py`. Original evidence authorities,
main and the paper frozen tag are unchanged. Use a fresh output location for
reproduction; do not overwrite these results.
