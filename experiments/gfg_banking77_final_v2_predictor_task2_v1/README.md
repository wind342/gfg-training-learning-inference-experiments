# Banking77 Final V2 nonlinear predictor: task2 only

Every scheduled update uses the historical actual-update JVP/nested-JVP predictor. The existing nonlinear cutter, trust-region rule, reserve, numerical tolerance, inner/outer caps and cap endpoint acceptance remain unchanged. Initial selection uses predicted deficits; Jacobian geometry uses actual logits. Predicted-safe native candidates also receive exact rechecks. See [PROTOCOL.md](PROTOCOL.md).

Final status **PASS**. Independent native-only and controlled replay **PASS**. Stopped after task2; no tasks3–11, test or new baselines. Historical evidence remains unchanged.

| Controlled task2 | 210 | 630 |
|---|---:|---:|
|Predictor calls|210|630|
|Predicted projection triggers|208|618|
|Actual unsafe native|208|618|
|Actual native-safe|2|12|
|Ordinary PASS + safe|125|497|
|INNER_LIMIT events|100|121|
|Cap-safe accepts|83|121|
|Cap-unsafe continuations|17|0|
|Rollback|0|0|
|Floor violations|0|0|
|Outer iterations|307|618|
|Inner sweeps|48686|64285|
|Jacobian pairs|9002|11698|
|Correct → wrong|5|0|
|Wrong → correct|5|0|
|Cap-safe acceptance rate|83.00%|100.00%|
|Executed/native norm median|0.884034|0.909822|
|Cosine median|0.882970|0.909885|
|old train macro (%)|51.4938|66.8580|
|current train macro (%)|58.8442|76.5460|
|all train macro (%)|55.1690|71.7020|
|Current train CE entry → final|2.686831 → 2.139460|3.042068 → 1.483108|
|Actual endpoint calls|728|1879|
|Predictor wall (s)|1.0187|2.9612|
|Control wall incl predictor (s)|9.7258|19.1797|
|Total algorithm wall (s)|10.6770|21.8851|
|Predictor GFLOPs|236.826132|710.478397|
|Exact endpoints GFLOPs|80.630252|208.110223|
|Cutter estimates GFLOPs|15.902348|21.356261|
|Current CE dense GFLOPs|3.039805|9.119416|
|Total covered GFLOPs|336.398537|949.064298|
|Old CE backward examples / FLOPs|0 / 0|0 / 0|
|Control-bank logical row reads|65660|175630|

| Predictor | Native210 | Native630 | Controlled210 | Controlled630 |
|---|---:|---:|---:|---:|
|Calls|210|630|210|630|
|Margin MAE|1.5378809299598747e-07|3.188382200643319e-07|9.152658131657814e-08|2.607953251083962e-07|
|Margin max error|9.5367431640625e-07|2.384185791015625e-06|9.5367431640625e-07|1.9073486328125e-06|
|Correctness agreement|1.0|1.0|1.0|1.0|
|Argmax agreement|1.0|1.0|1.0|1.0|
|Balanced accuracy|1.0|1.0|1.0|1.0|
|Floor-safe agreement|1.0|1.0|1.0|1.0|
|False-safe updates|0|0|0|0|
|False-unsafe updates|0|0|0|0|
|Predicted-safe updates|2|12|2|12|
|False-safe / all updates|0.0|0.0|0.0|0.0|
|False-safe / predicted-safe|0.0|0.0|0.0|0.0|
|wrong→wrong recall|1.0|1.0|1.0|1.0|
|wrong→correct recall|1.0|N/A (no events)|1.0|N/A (no events)|
|correct→wrong recall|1.0|1.0|1.0|1.0|
|correct→correct recall|1.0|1.0|1.0|1.0|

The same fixed gates are measured on both native-only and controlled trajectories. Missing transitions have undefined recall, never imputed as 1. Prediction does not certify safety; exact FP32 counts are the final authority.

Wall times include synchronization/instrumentation and have one execution per arm; no best-of runs. Hashing, logs, accuracy observers, train-set evaluation and independent audits are excluded. Dense matmul FLOPs exclude GELU transcendental operations, optimizer operations and other elementwise kernels; cutter costs are the existing estimates, not full hardware instruction counts. Predictor cost is measured with FlopCounterMode.

The independent audit recomputes every native proposal, optimizer moment/counter, prediction component, initial selection, local solver coefficient/dual, materialized candidate, cap acceptance/continuation and final head. Capped candidates additionally receive explicit dense-Jacobian reconstruction checks. BERT/cache/input hashes are verified unchanged. Python access logs cover experiment file reads; native library/OS internals are outside the hook.

External authority: `E:\gfg-evidence\banking77-final-v2-predictor-task2-v1`. All compact checks are in `results/`, with full artifact identities in `ARTIFACT_MANIFEST.json`. No automatic 11-task continuation exists.

## Post-training comparison and guard checks

A read-only comparison with the previous cap-recheck task2 authority found all 840 receiving/native/executed head hashes, moments, endpoint hashes, selected controls/active pairs and solver sequences identical. The predictor is accurate on these trajectories but does not change task2 learning outcomes. This observation does not establish what happens at later task boundaries. Comparison inputs and hashes are in `results/PREVIOUS_CAP_COMPARISON.json`; this comparison was not used to choose training settings.

Separate synthetic checks force false-safe and false-unsafe predictions: actual unsafe routes to the cutter with actual selection; actual safe commits the native head unchanged. Both pass. These injected cases are not counted as empirical predictor errors.
