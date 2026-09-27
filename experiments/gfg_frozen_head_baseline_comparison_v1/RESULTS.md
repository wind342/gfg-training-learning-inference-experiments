# Fixed FewRel comparison: results

Status: **PASS**. One seed (2021), fixed order, frozen task1 BERT, all-seen evaluation. No outcome-driven reruns or parameter changes.

## Capability results (%)

| Method | T10 old | T10 new | T10 all | Mean old | Mean new | Mean all |
|---|---:|---:|---:|---:|---:|---:|
| Native-210 | 11.81 | 92.59 | 19.89 | 21.61 | 93.25 | 35.55 |
| ER-210 | 62.23 | 88.66 | 64.87 | 75.34 | 90.30 | 77.33 |
| ER-630 | 71.08 | 91.07 | 73.08 | 81.45 | 91.75 | 82.81 |
| A-GEM-AdamW-210 | 16.31 | 92.86 | 23.96 | 32.03 | 93.30 | 43.04 |
| A-GEM-AdamW-630 | 20.00 | 95.80 | 27.58 | 34.63 | 94.95 | 46.08 |
| GEM-AdamW-210 | 31.80 | 92.50 | 37.87 | 43.18 | 93.17 | 52.40 |
| GEM-AdamW-630 | 47.00 | 95.36 | 51.84 | 52.36 | 94.95 | 61.08 |
| Final v2-210 | 72.92 | 70.33 | 72.66 | 80.45 | 80.70 | 80.53 |
| Final v2-630 | 75.40 | 79.00 | 75.76 | 81.45 | 87.55 | 82.85 |
| Accumulated Ridge (no RP) | 81.78 | 70.43 | 80.65 | 88.58 | 83.12 | 87.65 |
| RanPAC classifier-head RP-10000 | 85.43 | 79.18 | 84.80 | 90.29 | 86.75 | 89.66 |

Means cover task2-task10, not task1. Full per-task, relation and cohort metrics and predictions are in `results/EVALUATION_TASK{1..10}.json`.

## Main observations

- At 630 updates/task, Final v2 vs ER: T10 old **+4.32 pp**, new **-12.07 pp**, all **+2.68 pp**. Mean all is **82.85% vs 82.81%**, essentially tied; do not claim uniform superiority throughout training.
- Final v2 has much higher final retention than the two specified gradient-constraint AdamW adaptations; those methods retain stronger current-task accuracy. This does not establish that every GEM/A-GEM variant or tuned configuration performs this way.
- The RP classifier-head adaptation has the highest T10 and mean all accuracy here. Against v2-630 its T10 old is **+10.03 pp**, with nearly equal T10 new (**79.18% vs 79.00%**). The no-RP statistical classifier is also a strong baseline; its T10 new is lower than v2-630.
- Statistical methods retain all arrived training contributions in sufficient statistics and use no AdamW head or old-loss backward. This stored-information difference and the RP method's larger feature dimension are explicit parts of the comparison, not hidden implementation details.
- Both v2 arms have **zero executed relation-floor violations and zero skips**. Their 18 task-end heads, optimizer states, and all evaluation predictions match historical v2 exactly. Other methods were observed without imposing this contract. Floor safety concerns algorithmic controls, not a guarantee that every old test sample remains correct.

## Cached-feature algorithm cost

| Method | Task2-10 algorithm seconds | Logical old reads | Old CE backward example-equivalents |
|---|---:|---:|---:|
| Native-210 | 4.13 | 0 | 0 |
| ER-210 | 4.25 | 30,240 | 30,240 |
| ER-630 | 14.17 | 90,720 | 90,720 |
| A-GEM-AdamW-210 | 9.47 | 30,240 | 30,240 |
| A-GEM-AdamW-630 | 28.16 | 90,720 | 90,720 |
| GEM-AdamW-210 | 30.40 | 151,200 | 151,200 |
| GEM-AdamW-630 | 87.71 | 453,600 | 453,600 |
| Final v2-210 | 276.49 | 96,481,979 | 0 |
| Final v2-630 | 809.41 | 259,617,654 | 0 |
| Accumulated Ridge (no RP) | 4.07 | 10,072 | 0 |
| RanPAC classifier-head RP-10000 | 46.92 | 10,072 | 0 |

Shared training-feature extraction: **197.44 s** for tasks1-10. Common test-feature extraction: **65.58 s**. Statistical task1 setup/fitting is recorded separately in each arm's task1 record. The table excludes this shared preprocessing, evaluator, common observer and artifact/hash I/O; it includes each method's required replay backward, constraint processing and solve. It is not full end-to-end wall time. In particular v2's production implementation is substantially more expensive here than the cached replay and statistical baselines.

V2 reads count row-logit evaluations; other gradient methods count sampled old rows, and statistics count newly incorporated old controls. These are NOT equivalent FLOPs or hardware memory traffic. CUDA peaks in machine records are shared-process peaks, not isolated per-method memory requirements.

## Validation and retention

- 32,130 formal gradient updates audited for sampling, class access, moment step counters, checkpoint continuity and recorded counts.
- All 99 task2-10 saved classifier predictions and metrics independently recomputed; gradient-arm final control counts also recomputed.
- Task2 of all nine gradient arms reexecuted: **3,570 updates** with bit-identical step head/moment hashes and counts.
- GEM unit KKT checks, A-GEM analytic projection, ER gradient equivalence, ridge primal/dual equivalence, and inherited V2 numerical tests PASS.
- No failure/retry occurred in the formal run. Every statistical full solve used Cholesky; maximum reported relative residual was below 1.9e-14.
- Later task updates were NOT all independently replayed; statistical G/Q were NOT independently reconstructed from every input row by a second implementation. The exact achieved scope is in `results/CHECKS.json`.
- Complete local authority: approximately **1.451 GiB**, including shared features, small task checkpoints, final statistics and task2 replay records. Public compact records: approximately **44.68 MiB**; no raw examples, original BERT weights or feature tensors are uploaded.
- Old experiments, main and the paper's frozen annotated tag remain unchanged.

See [README](README.md), [frozen protocol](PROTOCOL.md), [machine results](results/RESULTS.json) and [checks](results/CHECKS.json). This is a fixed-regime controlled comparison, not a cross-benchmark or universal method ranking.
