# Fixed continuation of relation-floor task8: total630 updates

Base7ae5104125c2838d456c240f60018b57001ee481. Extend only the previously validated
relation-floor task8 trajectory. Reuse its actual step210 head and AdamW moments;
do not reset to task7 or rerun the first210 steps. Exactly420 new updates in two
additional epochs,ending at total task8 steps420 and630. One arm,one execution.

All scientific/engineering policy is imported unchanged: frozen/eval BERT,
original80-row linear head,64 seen classes,LR5e-5,fused AdamW(.9,.999),zero weight
decay,clip1,batch16,seed2021,same original current-task batch order in each epoch.
Scheduled RNG seed2021*100000+(1470+total_task8_step). AdamW moments/counter
continue1680->2100,advance once even on skip. No replay,Router,new features,
predictor,solver,reserve or selection changes. No parameter sweep or extra task.

Protection bank7833 old controls+80 new audit-only rows stays unchanged. ALL56
K[r] values are inherited from the ORIGINAL task8 entrance(task7 endpoint),sum6835.
Do not recompute K from step210,420 or any improved current state. The parent
floor_projection is called directly,including deterministic nearest selection,
same Hildreth primitives/FP32 reserve,100-sweep TOTAL cap and first-floor-safe stop.

Read-only evaluation at the predeclared420 and630 endpoints,after each epoch's
training record is sealed. Same primary test split. Evaluation cannot change
head,moments,BERT or the training schedule. Run all630 total steps regardless of
intermediate results; no tuning,early stopping,extra extension or outcome rerun.

Record per-step count floors,selection certificates,FP32 endpoints,head/native/
projected states,clipped gradients and moment hashes. Record per-epoch test old/
new/all macro,skip,floor violations,correctness transitions,projection diagnostics
and wall. Reference210-step parent metrics,never regenerate the parent results.
Independent replay verifies all420 new native candidates and Hildreth sweeps,
23520 relation-step floors,continuity,unchanged input/source hashes and test metrics.

Interpretation: improvement supports insufficient training budget for this fixed
protected trajectory; it does not prove indefinite improvement or that capacity
is unlimited. Flat results do not establish a universal capacity limit. Additional
data passes and time are explicitly part of this comparison.
