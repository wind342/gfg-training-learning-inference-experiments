# Fixed head-only learning-rate calibration

Base c6490aa26f83d046e0101dc2661ab8dae750cb6a. Independent branch and directory;
no existing experiment, evidence, main or frozen tag changes.

Run exactly three new arms in order:5e-5,5e-4,1e-3. The5e-6 arm is the frozen
head_only_exact_monitor result from c6490aa and MUST NOT be retrained. One seed2021,
task2, original task1 checkpoint, seen16, batch16,210 updates, original shuffled
batch order and encoded-input isolation. No additional arms, seeds or restarts.

BERT completely frozen and always eval. Fresh no_grad BERT forward for each
current batch, no training-feature cache. Train only Linear head weight/bias.
Native fused AdamW with original betas(.9,.999),decay0,clip1. Inherit exact task1
head moments and scalar step, THEN set arm LR. Check actual LR every step.
Do not change scheduler/optimizer/memory or introduce replay, LoRA, adapter,
projection, protection, shrink or new predictor.

Use c6490aa's exact cached160-control monitor and shadow-head proposal/commit
path, every step, observation only. No safety decisions affect updates,alpha1
always. Retain all old groups, baseline/candidate capability,q10 and frozen
receiving-anchor safety classification. Preserve model/RNG/moments during
monitoring; verify native committed margins equal candidate margins. Record
all210 losses, especially1,50,100,150,210. Different batches mean losses need not
decrease monotonically.

No test inference in preflight or any trainer. After all three training result
files are COMPLETED and their hashes frozen in TRAINING_COMPLETE.json, a separate
read-only evaluator computes full task2 training16-class and new-only accuracy
for all four saved heads. New-only is diagnostic, not a valid task score.
It then evaluates primary test2239 and original2240 examples for all heads in
one shared frozen-backbone pass;5e-6 predictions must equal existing recorded
predictions. No subsequent training permitted by the orchestrator. No selection
of LR from test and no success gate. Report every LR, old/new/all macro and
per-relation test, controls, old correct->wrong and safe->unsafe occurrences.

This fixed comparison can support insufficient optimization at5e-6 if higher
LR substantially improves training and new-test capability. High new-only but
low16-class can identify remaining old-class competition. Persistently weak
new-only at these LRs merely raises the possibility of representation limits;
it does NOT rule out a short budget or optimization instability. No extra
training is authorized by any outcome.

Preflight: inherited15 tests plus three-LR override/moment-preservation test,
gradient/update identity versus direct native head AdamW at each LR, and exact
monitor noninterference at each LR. Check original data/checkpoint hashes and
original baseline source/result/audit identities. Freeze all Python/protocol
source hashes before training.2h fixed cap per arm; retain failures.

Save local head snapshots, control features and optimizer states, not full BERT
copies. Separate checker verifies barrier ordering/hashes, LR, batch IDs, BERT
identity, all cached candidate endpoints, safety statistics and reported metrics.
Checks are record/affine re-execution, not a second full training run.
Publish only generated ledgers/results/checks, no original texts/tokens or weights.
All write operations create-only. One final commit and push this branch only.
