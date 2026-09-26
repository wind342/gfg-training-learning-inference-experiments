# FewRel short feedback-direction ablation

Frozen before execution. Base commit: 82fb534123e18bc1077fc2acf6c592a83939e007.
No existing source, frozen tag, result or dataset is changed.

All three arms start from the same completed task-1 governor model, AdamW state,
and (where applicable) scheduler history. Checkpoint SHA-256:
bd41bcbd84f280d4de7d721eeee1d7a7bad1ccbb04ceb7d84d0403a6bdbcb676.
Use the first 60 original task-2 batches, seed2021, LR5e-6 and original clipping,
AdamW and v4 controller settings. No sweep, extended training or selected rerun.

Arms: new_governor (new16); replay_native (new16+old16, equal mean weighting);
replay_governor (identical replay batches, unchanged v4 controller).
Replay memory: first ten training examples per old relation, excluding exact
encoded duplicates of ANY current control/test example. Fail if unavailable.
Old batch: two examples per old relation, deterministic cyclic selection.
Validation/control examples never contribute gradients. New examples match across
arms; replay has extra old-data exposure and computation, explicitly not an
equal-total-example comparison. Accumulate two sequential microbatch gradients
to fit 8 GiB; do not retain two full backward graphs. Same new-forward RNG seed;
old forward has a separate fixed seed. No new optimizer or direction geometry.

Evaluate all 2240 test examples from the first 16 relations at offsets0,20,40,60;
report old/new correctness separately. Never use test results for control/selection.
Report actual selected endpoint agreement on measured steps, full calls, selected
alphas, runtime and peak GPU allocation. Measurements do not select directions:
this ablation tests feedback mixing; v4 only selects scalar amplitude afterward.

Primary question: does mixed feedback improve old-test retention without merely
preventing new-task learning? Replay-native versus replay-governor isolates the
incremental controller effect at matched replay exposure. This is one seed and
a short prevention/early-recovery window, not a full recovery/continual-learning
claim or proof that changing direction is necessary. No forced success verdict.
Keep all failed attempts. Runtime cap900 seconds per arm. Raw artifacts outside
Git; no large new checkpoints. Verify new_governor against the original first60
task-2 alpha decisions and control endpoint margins as an execution regression.
