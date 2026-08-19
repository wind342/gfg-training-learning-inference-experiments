# nanoGPT transition offset and forecast-gate audit

## Scope

This is a post-run, read-only diagnostic of two completed formal nanoGPT
forecast validations. It does not modify either sealed candidate and does not
retroactively change either formal result.

The question is why two independent validations placed the observed
transition exactly one 100-step evaluation-grid point above the submitted
interval upper bound, and whether that pattern indicates a nanoGPT or evaluator
defect.

## Executive finding

No 100-step counter or time-alignment defect was found. The repeated one-grid
miss is explained by two factors:

1. validation accuracy is observed only every 100 optimizer steps, so the
   reported transition is quantized to that grid; and
2. both candidates predicted an internal formation precursor earlier than the
   first observed threshold-crossing evaluation. In the dual-interval run, the
   candidate accurately forecast the future gradient-burst locations but
   equated the final burst with immediate observable capability, omitting a
   subsequent consolidation delay.

The dual-interval run did **not** fail overall because of the 100-step
high-precision diagnostic miss. Its 500-step primary interval contained the
actual transition. The failing primary gate was full-horizon normalized RMSE:
the candidate predicted an unsupported late accuracy collapse from step 7600
onward.

## Two observed one-grid misses

| Run | Prediction cut | Submitted interval | Actual transition | Distance beyond upper bound | Full-curve nRMSE |
| --- | ---: | --- | ---: | ---: | ---: |
| original single-interval run | 1100 | 1800--2000 | 2100 | 100 | 0.0990248547 |
| dual-interval run, 200-step diagnostic | 800 | 1400--1600 | 1700 | 100 | 0.2074473148 |

The two runs used different task commitments, model seeds and data-order seeds.
The repeated value is therefore suggestive, but two observations are not enough
to establish a universal 100-step bias.

## Time-coordinate audit

The native loop performs `optimizer.step()`, records the generated parameter
version as `step + 1`, and evaluates at `evaluation_step = step + 1`. Evaluation
occurs at step 1, every configured 100-step boundary, and the final step. See
`nanogpt_adapter.py`, `train_plain`, lines 363--380.

The transition detector scans three-point windows and returns the first point
of the first window for which train accuracy is at least 0.99 and validation
accuracy is at least 0.90, provided an earlier validation point was at most
0.30. See `nanogpt_adapter.py`, `detect_transition`, lines 474--495.

Consequently, the implementation has ordinary one-step parameter-version
alignment, not a 100-step off-by-one. The 100-step resolution comes from the
frozen evaluation interval. A transition reported at 1700 may have crossed the
underlying threshold anywhere after the 1600 observation and by the 1700
observation.

## Dual-interval candidate execution

The hidden prefix ended at optimizer step 800. The candidate extracted three
pre-cut gradient-burst peaks:

```text
187, 434, 742
```

Its general extrapolator requires at least four peaks. With only three, the
candidate used its disclosed fallback of a fixed 350-step spacing and predicted
future centers:

```text
1150, 1500
```

The actual future burst peaks recovered from the sealed baseline GFG were:

```text
1094, 1445
```

Thus the structural burst forecast was close: each predicted center was about
55 steps later than the corresponding observed peak. The main timing error came
after the burst. The candidate rounded the second predicted center to the
evaluation grid and used 1500 as the predicted transition center. Actual
validation accuracy was:

| Step | Validation accuracy |
| ---: | ---: |
| 1300 | 0.7075471878 |
| 1400 | 0.7971698046 |
| 1500 | 0.8443396091 |
| 1600 | 0.8679245114 |
| 1700 | 0.9103773832 |
| 1800 | 0.9386792183 |
| 1900 | 0.9292452931 |

The candidate therefore discovered a relevant internal rewrite schedule but
treated its last predicted rewrite as immediate rule-level expression. This
hidden seed required additional consolidation before validation crossed 0.90.

## Why the formal dual-interval run failed

The dual-interval result was:

- actual transition: 1700;
- 200-step diagnostic: 1400--1600, miss;
- 500-step primary interval: 1300--1700, pass;
- causal intervention: pass;
- full-future normalized RMSE: 0.2074473148, fail against the 0.20 gate.

The RMSE decomposition is:

| Evaluation region | Point count | RMSE |
| --- | ---: | ---: |
| 900--2000 | 12 | 0.083900 |
| 2100--7500 | 55 | 0.042275 |
| 7600--10000 | 25 | 0.388659 |
| complete future | 92 | 0.207447 |

The candidate hard-coded `late_instability = transition + 6100` and emitted
validation accuracy 0.60 thereafter. For the hidden run this began at step
7600. The actual 25 late validation values had minimum 0.8773584962, maximum
1.0 and mean 0.9875471711. The late-collapse hypothesis, not the one-grid
diagnostic miss, dominated the failed full-horizon RMSE.

The causal claim was independently successful. From the exact shared
checkpoint, baseline transition occurred at 1700 and intervention transition
at 3300. The actual `DELAY` was 1600 steps, inside the candidate's predicted
600--1800 interval; all causal gates passed.

## Scientific interpretation

Under the frozen protocol the dual-interval opportunity remains an answer
failure: the sealed candidate explicitly predicted a late collapse that did not
occur. Nothing in this audit changes that result.

However, the audit identifies a possible mismatch between the intended claim
and one primary gate. If the scientific target is discovery of capability
formation, a single RMSE over every future point through step 10000 also makes
seed-specific late stability part of the required answer. In this run that late
regime alone overturned otherwise strong transition-local and causal results.

Before consuming another formal opportunity, an independent reviewer should
decide whether the intended claim includes exact long-horizon post-formation
stability. If it does, the existing gate is appropriate. If it does not, any
revised protocol must be frozen prospectively and must keep this completed run
as a failure under its original rules; it cannot be reclassified post hoc.

## Minimal falsification checks recommended before another opportunity

1. Re-run the already frozen hidden seed without AI at a denser evaluation
   interval, while confirming that its 100-step samples remain bitwise or
   numerically identical. This tests whether the apparent one-grid miss is only
   observation quantization.
2. Across several no-AI seeds, measure the delay from the last relevant
   gradient burst to the first sustained validation transition. This tests
   whether the consolidation lag is stable, state-dependent or seed-specific.
3. Across the same seeds, measure whether late post-formation collapse occurs
   and whether its timing is predictable from the prefix. This tests whether
   the full-horizon RMSE gate evaluates the intended mechanism or an additional
   unstable regime.

## Evidence integrity

The accompanying JSON records the numerical observations, code symbols and
SHA-256 hashes of the private run artifacts used for this audit. Large GFG
databases, checkpoints, credentials and private training artifacts are not
included in the repository.
