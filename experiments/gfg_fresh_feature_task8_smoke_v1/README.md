# Fresh fixed features: isolated FewRel task8 comparison

Base: `115534a09cd554fcd75fd46dde711fdb98988519`. Both new arms fork the same
Final Architecture v1 task7 endpoint. Only task8,210 scheduled updates,seed2021,
LR5e-5. No original code/results,main or frozen tag changed.

Model: frozen BERT with old W,b plus zero-initialized U80x128. Same fixed
R128x1536 from torch CPU generator seed2021, scaled1/sqrt(1536).
Linear control uses Rh; nonlinear arm uses ReLU(Rh). R is not trained.
No residual bias, fitted normalization,replay,old loss,Router or test tuning.
Old W,b AdamW moments/counter1470 are inherited; new U moments/counter start0.
Both use the same original batch order,gradient clip1 and fused AdamW settings.

## Results

All accuracies are relation-macro percentages, measured over the same primary
test split after both training endpoints had been sealed.

| Task8 method | Old56 | New8 | All64 | Skip | Protected correct-to-wrong |
|---|---:|---:|---:|---:|---:|
| Existing Final v1 |76.4439|65.9821|75.1362|0|0|
| Fresh linear Rh control |76.2781|67.1429|75.1362|0|0|
| Fresh nonlinear ReLU(Rh) |76.2143|66.4286|74.9911|0|0|

Nonlinear new accuracy is+0.4464pp versus the original, but-0.7143pp versus the
matched linear control. Its old accuracy is-0.2296pp and all accuracy-0.1451pp
versus the original. There is no marked new-learning rescue in this fixed smoke.
All steps execute safely; none of these differences is caused by skipped updates.

| Method | Projection steps | Mean executed/native norm | Mean cosine | Training loop seconds |
|---|---:|---:|---:|---:|
| Existing Final v1 |210|0.628127|0.626858|57.8556|
| Fresh linear |210|0.657860|0.656371|84.3600|
| Fresh nonlinear |210|0.670361|0.668620|85.0748|

New arms' projection/check times:52.0789s and52.9180s. New arms cache all current
training BERT features before the loop; historical timing includes current-batch
BERT forwards. Thus these loop timings are descriptive, not an end-to-end speed
benchmark. PyTorch peak allocated GPU memory:666227200 and667293184 bytes.
Local full evidence is about1.12GiB; weights/caches/raw ledgers are not committed.

## Interpretation boundary

This particular128-dimensional random ReLU expansion did not outperform the
matched redundant-linear expansion or materially restore new accuracy. It does
not support claiming that simply adding this fresh interface resolves the late
plasticity gap. It also does not disprove a capacity contribution or the utility
of other feature maps: fixed random features,feature geometry and optimization
conditioning remain relevant. A redundant linear branch changes parameterization
and fresh optimizer states without changing the mathematical linear function
class; ReLU adds a different class, but also changes feature distribution.
One task and one seed cannot identify a unique cause or a universal capacity bound.

## Exact implementation and audit

Historical Hildreth bytecode,coordinate updates,eligibility,active-set discovery,
first-safe stopping,max100 sweeps and safety threshold are reused unchanged.
Only actual affine forward is adapted to W h+b+U phi(h). Joint coordinates are
[W,U,b],fixed features[h,phi(h),1]. Same reserve formula,with dimension-dependent
k1667 instead of1539. No continuation-to100 stopping change from the preceding
micro-experiment is used. Native positive-margin fast path is unchanged.

Preflight confirms exact initial FP32 logits for ALL cached current train and
controls,known affine directional derivatives,joint autograd gradients,zero U,
inherited optimizer state,once-only counters,and synthetic safety. Training uses
the unchanged7833 old controls and80 new audit-only controls. Data isolation
exclusions match the original long run. Test forwards occur only after both arms.

An independent checker reconstructs saved-gradient native AdamW candidates and
sparse Hildreth primal/dual sweeps without calling the production projector,
checks all actual endpoints and recomputes test metrics. Machine results,checks,
compact step ledgers and full-artifact hashes are packaged under results/.
See [protocol](PROTOCOL.md),[results](results/RESULTS.json) and
[checks](results/CHECKS.json).

Run in the existing environment from repository root:

```text
python -B -m experiments.gfg_fresh_feature_task8_smoke_v1.run
python -B -m experiments.gfg_fresh_feature_task8_smoke_v1.check
```

The runner refuses to overwrite its output directory. Full local artifacts:
E:/gfg-evidence/fresh-feature-task8-smoke-v1. Inputs/source identities are recorded
in PREFLIGHT.json. Before numerical execution, the first import attempt exposed
the torch.no_grad wrapper's closure requirement; the adapter was corrected to
reuse the undecorated original function bytecode and restore no_grad. No formal
training had started; each new arm was trained exactly once, without tuning.
