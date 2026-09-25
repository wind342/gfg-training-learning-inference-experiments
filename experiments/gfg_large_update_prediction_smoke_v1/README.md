# Large-update prediction smoke: results

## Outcome

No large-update capability jump was observed on the six tested historical
receiving states. All prospective choices were .25 or .5; each choice matched a
best observed actual control-bank endpoint among the tested nonzero amplitudes.
Adding 2/4/8/16 did not improve on the restricted .25/.5/1 selector in these cases.
This is **not** a proof that large-step training is impossible.

We restored **100,352-parameter** nanoGPT states, their recorded Adam moments and
steps, encoded batches and seeds. This is NOT a 70M/160M test. The inherited
CausalTrainingRuntime uses LR .003, betas (.9,.98), clipping 1, and parameter-group
weight decay 1/0. These are explicitly recorded in each PREDICTION_SEAL; the recent
fresh-optimizer scale suite instead used weight decay 0. Do not pool them.

There were three initially preselected cases. Two already had 100% validation
accuracy. They test preservation, not initial formation. An explicitly
post-initial exploratory supplement selected the three lowest previously recorded
baseline accuracies among the remaining entries. All six outcomes are retained.
Historical low-accuracy states can reflect decline/recovery, not only formation.
They are not six independently trained seeds and do not represent random starts.

## Four low-baseline cases: actual control-bank accuracy

Each case has 212 control targets; table entries are equal-case means. The two
perfect-baseline preservation cases are excluded from this descriptive table,
not removed from the evidence. See verification JSON for every case/subset.

| Update multiplier | Actual accuracy |
|---|---:|
| 0 (receiving state) | 33.37% |
| .25 | 35.50% |
| .5 | 37.62% |
| 1 (native proposal) | 32.31% |
| 2 | 22.41% |
| 4 | 13.21% |
| 8 | 6.37% |
| 16 | 5.07% |
| Prospective selected multiplier per state | 38.44% |

Prospective selection improved on receiving accuracy by **5.07 percentage points**
on this small group, compared with -1.06 points for alpha 1. This is a one-step
accuracy comparison, not a training-step or wall-time reduction measurement.

| Historical entry suffix | Baseline | Selected alpha | Actual selected | Actual alpha 1 |
|---|---:|---:|---:|---:|
| 481b86f81d58d496a687 | 33.49% | .5 | 37.26% | 30.66% |
| 8fa6576fc7128f93a228 | 30.19% | .5 | 33.96% | 32.55% |
| f34e7e61444c90976b36 | 31.60% | .5 | 40.57% | 39.62% |
| 786d0a3628f6f791399f | 38.21% | .25 | 41.98% | 26.42% |

The selected alphas were fixed using predicted counts/median BEFORE viewing any
actual endpoint. There was no selection using the additional 317 original
training queries. Those are selection-disjoint, NOT held out from historical
training. Their full results are reported separately.

## Prediction and interpretation

On the four low-baseline control sets, alpha .5 boundary agreement averaged
98.58%, compared with 92.69% at alpha 1. At alpha 16, raw agreement was 94.69%,
but balanced correct/incorrect recall averaged only 51.75%: almost everything
was predicted wrong and almost everything actually was wrong. Thus high raw
agreement there is **not** reliable discrimination of remaining capability.

Both issues matter: large actual updates damaged capability along these specific
directions, and second-order endpoint predictions became less discriminating.
The observed failure is not merely an inaccurate prediction concealing a useful
large endpoint. None of the tested 2/4/8/16 endpoints beat alpha 1 on the control
bank in any of the six cases. This does not exclude other directions, other
receiving states, freshly initialized models or alternative parameterizations.

No full training race, 70M/160M continuation, energy benchmark or true
time-to-capability measurement was performed. Per-step prediction plus huge
updates remains an unestablished training strategy, rather than one disproved
universally by this smoke test.

## Verification and preserved interruptions

Independent NumPy recomputation: **PASS** for endpoint arithmetic, selections,
classification metrics and raw hashes. Alpha 0 restoration and alpha 1 native
endpoint were exact; parameter/optimizer snapshots were unchanged by forward
evaluation; original receiving states were restored. Prediction seals precede
all corresponding post-update evaluations. All preexisting tracked source hashes
and the frozen paper annotated tag/object/commit remained unchanged.

Two early setup attempts are preserved, not silently replaced:

1. Attempt-1 input-identity assertion caught a wrong assumption about historical
   token layout/permutation before any optimizer execution. Corrected by directly
   reading original inputs and labels, not token arithmetic.
2. Attempt-2 required byte equality between adapter full-position readout and
   native last-position forward and stopped before endpoint evaluation. Attempt-3
   explicitly anchors g0 to native baseline logits, preserves both raw arrays and
   records their difference; it does not relax the equality assertion by tolerance.
   This is an explicit numerical anchoring difference from the old controller,
   not a claim of bit-identical old-runtime decisions.

Original attempt source snapshots and all raw arrays remain under
`E:/gfg-evidence/large-update-prediction-smoke-v1/`. Raw arrays are not committed.
Small receipts, source seals, interruption reports and independent results are
in `verification/`. Original experiment sources/results were not edited.

## Read-only recheck

From the repository root, using the existing Python environment:

```powershell
E:/gfg-python/torch-cu130/Scripts/python.exe -m experiments.gfg_large_update_prediction_smoke_v1.verify
E:/gfg-python/torch-cu130/Scripts/python.exe -c "from experiments.gfg_large_update_prediction_smoke_v1 import verify as v; v.OUT=v.OUT.parent/'supplement-low-baseline'; print(v.verify()['integrity'])"
```

Run scripts use exclusive output creation. Do not rerun on their existing output
directories. This work is on its own branch; no remote push was performed.
