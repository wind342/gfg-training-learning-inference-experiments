# FewRel sequential relation learning: governor v4 transfer

This is a native AdamW versus governor ablation on a public continual FewRel
split. It is not a reproduction of the CRL algorithm or a claim against its
reported scores. Source code was written here; external data, weights, source
inspection snapshots and checkpoints stay outside Git.

## Public authorities and fixed choices

- FewRel: https://github.com/thunlp/FewRel at
  `278a2315d2138810a379cd8d5718914dc56e2582`, MIT.
- Continual split reference: https://github.com/thuiar/CRL at
  `bcc06732b2ebcc3c8d4adaa5c838b7ce03e5df20`. `run_continual.py` sets 8
  relations/task; `dataloaders/sampler.py` uses 80 relations, seeded relation
  order and per-relation 420/140/140 train/validation/test split.
- CRL's inspected tree has no root LICENSE. Its code was inspected locally,
  not executed or redistributed. Its marker-augmented data are used locally;
  removing markers reproduces the official FewRel token multiset in all 80
  relations. No third-party dataset or pretrained weights are committed here.
- BERT: `google-bert/bert-base-uncased`, revision
  `86b5e0934494bd15c9632b12f734a8a67f723594`, Apache 2.0. Four entity marker
  embeddings are newly initialized, as in the reference representation setup.

Seed 2021; first three consecutive tasks for smoke; one epoch of all 420 training
examples per relation; batch 16; max length 256; AdamW LR 5e-6, betas (.9,.999),
decay 0, clip 1. No sweep or post-result tuning. The one-epoch run is deliberately
smaller than the CRL reference training schedule, and uses a plain biased linear
classifier rather than CRL's contrastive/prototype method. Both arms have exactly
the same training data, order, seed-per-update, dropout, optimizer and head.

Each relation has 10 fixed validation/control probes; final evaluation uses all 140
test examples per seen relation. The class set contains all seen relations, not
just the current task, and no task identifier is supplied at evaluation. Future
relation inputs/labels never enter the controller. Ten probes allow 90% to differ
from 100%, but remain a small, noisy estimate; final claims use the test set.
Native retains no old-data feedback; governor retains old validation examples for
monitoring/prediction. This is disclosed, not called a memory-matched replay study.

An additional content audit, performed during the ten-task run, found one exact
probe/test duplicate in the inherited public split: relation P1923, source records
390 (probe) and 548 (test), encoded indices 7429 and 7561. These are distinct source
sample IDs but identical original tokens/entity placement and model inputs. Thus
sample-ID disjointness does NOT imply fully content-disjoint evaluation. The
original split and primary results are preserved. Reports additionally exclude
the matching test record as a clearly labeled post-hoc integrity sensitivity
check for BOTH arms. No data, control rule, checkpoint or training run was altered.

## Instrument and task boundaries

`model.py` provides BERT entity-marker features and a trainable affine relation
head. Embedding/encoder stages propagate the original directional first/second
derivatives; all competitors are evaluated with the existing scalarized boundary
kernel. Bias is included via homogeneous coordinates `[h,1]`, `[h',0]`, `[h'',0]`;
its update is not silently dropped. Measurements are chunked by 4 examples.

The model's classification head is allocated once. Only seen class rows enter
loss, instrument and evaluation. At a task boundary, `extend_bank` verifies the
entire old probe prefix, preserves old group state/anchors/latches, appends new
groups as FORMING and forces a full current-scope measurement. Ordinary task-local
bank replacement/reordering still fails closed. Task-boundary evaluation before
training records the effect of the enlarged competitor set separately.

An extra audit forward after a full decision records the actually executed endpoint.
It never changes the selected alpha. Its time is separately reported and included
in end-to-end wall time. It tests selected endpoints only, not counterfactual
performance of unexecuted candidates.

## Resource and interface checks

RTX 4060 Ti 8 GiB, FP32, 109,017,680 parameters. A one-update profile measured about
0.535 s forward/backward, 0.109 s proposal, 0.486 s instrument for 4 probes, peak
4,447,360,512 allocated bytes. This is a profile, not a training speed claim.
Four adapter tests cover affine bias, small-displacement agreement with actual
native execution, actual CUDA full/micro paths, and explicit bank extension
preserving old group states while rejecting changed old probes.

The benchmark environment is `E:/gfg-python/governor-v4-bench`; it adds
Transformers 4.57.6, hub 0.36.0, datasets 4.4.2, PEFT 0.18.0 without changing the
original CUDA environment. A local `.pth` imports the existing torch-cu130 install;
this avoids accidentally using the system CPU-only torch. BERT was fetched with
the Hugging Face CLI and source data via pinned public URLs.

## Reproduction and reports

Install `requirements-benchmark.txt` in a separate environment with CUDA PyTorch.
The tested Python version is 3.12.10 and torch is 2.13.0+cu130. Before preparation,
download the pinned model (or supply the same exact local files):

```text
hf download google-bert/bert-base-uncased --revision 86b5e0934494bd15c9632b12f734a8a67f723594 --include "*.json" "*.txt" "model.safetensors" "README.md" "LICENSE" --local-dir E:/gfg-models/bert-base-uncased-v4
```

```text
python -m experiments.gfg_governor_fewrel_v1.prepare
python -m experiments.gfg_governor_fewrel_v1.tests
python -m experiments.gfg_governor_fewrel_v1.profile
python -m experiments.gfg_governor_fewrel_v1.run native --tasks 3
python -m experiments.gfg_governor_fewrel_v1.run governor --tasks 3
python -m experiments.gfg_governor_fewrel_v1.report --tasks 3
```

Artifacts live at `E:/gfg-evidence/governor-fewrel-v1`. Writers are create-only;
existing attempts are never overwritten or silently selected. Task checkpoints
include model/optimizer and controller history. Data manifests retain exact sample
identities and hashes. Reports recompute test accuracy and recorded governor/
scheduler decisions, including selected-endpoint prediction accuracy.

One observer-only endpoint audit was added after native launched and before
governor launched; native training code and both arms' control hyperparameters
were unchanged. Per-arm source hashes preserve that difference. Test evaluation
occurs at fixed epoch/task boundaries, so no precise first-time-to-accuracy claim
is made from this run. End-to-end wall includes evaluation and checkpoint saving;
initial model loading is excluded.

## Three-task gate (fixed run, no retuning)

| After task 3 | Native | Governor |
|---|---:|---:|
| Current 8 relations, test accuracy |98.0357%|98.0357%|
| Previous 16 relations, test accuracy |5.7589%|6.0268%|
| All 24 relations, test accuracy |36.5179%|36.6964%|
| End-to-end wall |267.916 s|626.304 s|
| Updates |630|630|
| Full instrument calls |0|27|
| Peak allocated bytes |5,334,137,856|6,208,527,360|

The old-relation improvement is 0.2679 percentage points, with no current-task
accuracy difference, at 2.338x end-to-end wall. This is small retention benefit,
not evidence of solving forgetting or a speed improvement. Class-scope expansion
alone changed old accuracy by 0 at both transitions; the measured forgetting
occurred during subsequent training.

Governor alpha counts: 1=618, 2=7, 0.5=2, 0.25=2, 0.125=1. All 4320 selected endpoint
correct/incorrect predictions matched native execution. There were 0 full calls
where native was predicted to violate a CURRENT protected group. Accurate local
predictions therefore did not imply useful long-horizon preservation in this run.
The run passed the resource/interface/real-forgetting gate, not an efficacy gate.

The ten-task sequence resumes the exact task 3 model/optimizer checkpoint and
governor history; it does not rerun or select a new three-task prefix. Per-update
seeds and per-task batch permutations remain unchanged. Full scope measurement
is mandatory at task 4, just as in uninterrupted task transitions. End-to-end time
includes the original prefix once and excludes resumed model-loading time.

```text
python -m experiments.gfg_governor_fewrel_v1.run native --tasks 10 --resume-three
python -m experiments.gfg_governor_fewrel_v1.run governor --tasks 10 --resume-three
python -m experiments.gfg_governor_fewrel_v1.report --tasks 10
```

## Completed ten-task sequence

Both arms completed all 2,100 updates, using the unchanged three-task prefix.
Continuation walls were 1,082.626 s native and 3,059.214 s governor, within the
predeclared 3,600 s continuation limit. No sweep, extended budget or selected rerun.

| Final task boundary | Native | Governor v4 |
|---|---:|---:|
| New 8 relations |97.8571% (1096/1120)|97.8571% (1096/1120)|
| Old 72 relations |0.07937% (8/10080)|0.06944% (7/10080)|
| All 80 relations |9.85714% (1104/11200)|9.84821% (1103/11200)|
| End-to-end wall, prefix included once |1350.542 s|3685.518 s|
| Full instrument calls |0|76|
| Micro sensor calls |0|2076|
| Peak allocated bytes, maximum across prefix/continuation |5,334,137,856|6,208,527,360|

This fixed one-seed/one-epoch-per-task run shows NO practical retention advantage:
both arms almost completely forget the earlier relations; governor costs 2.729x
wall. The one-test-example endpoint difference is not evidence of a meaningful
performance difference. Removing the inherited duplicate test item does not change
the old/new macro comparison. Full task-by-task metrics, forgetting/BWT, control
states, raw source identities and secondary duplicate checks are in the JSON report.

Of 2,100 updates, 2,072 used alpha1; counts for alpha2/.5/.25/.125 were11/7/6/4.
Full-call reasons were48 NATIVE_SAFE,11 AMPLIFY_FORMATION,16 FORMATION_PROGRESS
and1 SHRINK_PROTECTED_GROUP. At step873, the predictor reported one current
protected-group violation for native and none for selected alpha.25. The selected
real endpoint was audited; the unselected native counterfactual was NOT executed,
so this is one predicted immediate preservation conflict, not a matched causal fork.

Selected-endpoint boundary agreement was30,238/30,240 (99.9934%). Actual selected
endpoint transitions were23 correct-to-wrong and56 wrong-to-correct. These are
measured full-call endpoints only, not all training transitions. There were71 calls
with recovery debt;45 offered different debt capability across candidate alphas,
and13 had at least one predicted recovery. Thus history and candidate information
were present, but did not produce sustained retention under this fixed policy.

Timing totals: training577.453 s, instrument2030.167 s, micro forwards122.799 s,
endpoint auditing176.226 s. The final checkpoint re-evaluation is a separate
post-run integrity check, not included in benchmark wall. Test evaluation and
checkpoint saving during the benchmark ARE included in wall.

Engineering boundary: highly accurate local endpoint calculation is not enough to
make this sparse scalar-update controller a useful continual learner. The present
observation/selection/action policy is the main unresolved effectiveness problem;
directional instrument cost is the main computational cost. This run does not
identify a uniquely necessary replacement policy or test alternative directions.

```text
python -m experiments.gfg_governor_fewrel_v1.checkpoint_verify
python -m experiments.gfg_governor_fewrel_v1.export
```
