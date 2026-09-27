# Fixed relation-capability floor: task8-only smoke

Base115534a09cd554fcd75fd46dde711fdb98988519. One new arm from the original
Final v1 task7 endpoint,210 scheduled updates,seed2021,LR5e-5. No changes to old
experiments,main,frozen tag,BERT,head,optimizer,batch order or protection bank.
No replay,feature expansion,Router,sweep,extra seed or task9/10.

## Result

Relation-macro test accuracies (%),same primary test split. Existing Final v1
baseline is referenced,not rerun. Test is read only after training is sealed.

| Method | Old56 | New8 | All64 | Skip |
|---|---:|---:|---:|---:|
| Existing Final v1 |76.4439|65.9821|75.1362|0|
| Relation-floor v2 |76.2139|68.2143|75.2139|0|

New+2.2321pp,old-0.2301pp,all+0.0777pp. This is a small positive new-learning
change,not a large rescue. It is not a multi-seed or statistical-significance
claim. Fixed control floors do not guarantee unchanged old test performance.

## Protection and replacement

- All56 old relation floors maintained at every executed step:0 violations.
- Fixed entry floor sum6835/7833; final correct count6836/7833.
- Correct-to-wrong220 and wrong-to-correct221 are cumulative transition events;
  a control may contribute more than once.
- 17 relations have both loss and gain in the SAME executed update.
- 16 relations have both lost entry-correct and gained entry-wrong members at the
  task endpoint;43 entry-correct samples are wrong at the end,44 entry-wrong
  samples are correct. These are distinct samples,not transition-event counts.
- 135 relation-step occurrences use count slack: count decreases versus receiving
  while staying at/above the fixed entry floor.
- Projection-triggered210/210;active-constraint median371.5;skip0.
- Mean executed/native norm ratio0.640027;mean cosine0.624453.
- Training-loop wall55.4843s,projection/check30.0411s,PyTorch peak746195456 bytes.
  Timing excludes initial loading,post-loop tensor serialization,test and checking.

The observed improvement supports some usefulness of the COMBINED relaxation:
fixed entry floor instead of a ratcheting instance rule,and replaceable correct
instances. It does not isolate migration from release of temporary count gains.
"Migration" here means correctness-set replacement,not directly measured neural
support configurations. The result does not establish that instance locking is
the main long-run bottleneck or that all alternative floor-selection policies
would behave similarly. The greedy deficit selection is not a global optimality
or feasibility certificate.

## Frozen algorithm and checks

See [protocol](PROTOCOL.md). K[r] is fixed from the original task8 entry state.
Native candidates are accepted when all relation counts reach K. Otherwise
choose each deficient relation's d nearest-to-zero wrong controls,ties by dataset
ID,and add target/competitor constraints. The original Hildreth coordinate,
FP32 materialization,reserve formula k1539 and epsilon are imported unchanged.
Active/dual state persists within one update,never across updates. Maximum100
total sweeps;first complete-bank relation-floor-safe endpoint commits. No selected
sample becomes a permanent identity lock. Actual count correctness is margin>0.

Preflight covers native swaps,fixed non-ratcheting floor,deterministic ties,true
deficit recovery,sweep cap/skip and zero-floor behavior. An independent checker
reconstructs all saved-gradient native candidates,selections,dual/primal sweeps,
FP32 endpoints,fixed per-relation floors,optimizer states and reported metrics.
The original task7 head/moments and input/source hashes are verified before use;
BERT remains byte-identical and without gradients. Data-isolation corrections
and original task8 bank7833+80 rows are unchanged.

Machine results:[RESULTS.json](results/RESULTS.json),
[CHECKS.json](results/CHECKS.json),[step summaries](results/COMPACT_STEPS.json).
Full head states,clipped gradients and detailed endpoint/selection ledgers stay
local at E:/gfg-evidence/relation-capability-floor-task8-v1 (about0.52GiB).
Their hashes are recorded in TRAIN_RESULT.json. Generated public JSON records
retain their original bytes; no line-ending normalization of frozen identities.

```text
python -B -m experiments.gfg_relation_capability_floor_task8_v1.run
python -B -m experiments.gfg_relation_capability_floor_task8_v1.check
```

The runner refuses to overwrite existing results. There was one formal run,
with no algorithm changes or additional trials after seeing outcomes.
