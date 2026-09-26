# Frozen head-only exact-cut mechanistic smoke

Base3f7a592a69947d944027c4761f99f7544119ed4a. One new arm only:
head_only_exact_cut. No existing file/result, main or frozen tag changes.
Historical LR5e-5 native-head baseline is referenced, never retrained.

Same task1 checkpoint, FewRel task2, seed2021, seen16, batch16,210 updates,
LR5e-5, inherited head AdamW moments/step, betas(.9,.999),decay0,clip1 and current
batch order. Set LR after state load. BERT frozen/eval with fresh current-batch
no_grad features. Cache160 controls once. No replay, Router, q10-based policy,
scalar shrink/alpha search, LoRA, adapter, new predictor or BERT update.

## Exactly specified two-pass cut
Form the native fused-AdamW head candidate once per update; moments advance once.
Compute actual baseline/native logits for all160 cached controls. Eligible old
controls are the first80 whose receiving margin is strictly>0. A candidate
margin<=0 violates (ties are not strictly correct). This includes competitors
that were not the receiving maximum. All16 seen competitors participate.

If safe, commit the original native head byte-for-byte. Otherwise set delta to
native_head minus receiving_head, including weights AND bias. For each pass,
construct all violating (control,target,competitor) pairs from that pass's exact
candidate logits, ordered by ascending control index then competitor ID. For
each pair, g has +h,+1 in target row and -h,-1 in competitor row. Compute
dot=g.delta and norm2=2*(h.h+1) with float64 accumulation. If dot<0:
delta <- delta - (dot/norm2)*g, stored in the original FP32 parameter dtype.
Otherwise leave that pair unchanged. No epsilon/safety threshold is introduced.
This locally removes the negative margin derivative, not a claim of global
minimum-distance projection under all constraints. Later cuts may undo earlier
ones. In exact arithmetic an isolated processed pair preserves its receiving gap.

After pass1 reconstruct receiving+delta and EXACTLY reevaluate all160 controls.
If any originally correct old control remains violated, do one second pass with
the newly enumerated pairs and another full exact check. Never more than2 passes.
If still unsafe, skip the parameter update (CUT_FAILED_SKIP). AdamW moments and
step remain advanced once, including skip; no rollback or second optimizer step.
Future native proposals are formed in the ACTUAL receiving head state, not the
rejected shadow state. No projection gradient enters the optimizer moments.

## Records and metrics
Each step records baseline/native/pass1/pass2/final-candidate/executed margins,
eligible/violating identities, all pair certificates, first-pass safety, pass2
use, skip, native/cut delta norms, norm ratio, cosine and removed norm fraction.
Report candidate-cut and actual-executed geometry/gains separately on skipped
steps. New-control gain=correct-after minus correct-before over80 new controls;
unexecuted native/cut alternatives are exact cached-control counterfactuals,
not separately trained trajectories or held-out predictions.

Record total update/training wall, full filter time, projection arithmetic time
and exact-recheck time separately; audit/hash/I/O remain identifiable. Final
controls old/new counts and per-relation capabilities, full3360 task2 train16
accuracy, primary2239 and original2240 test old/new/all macro/per-relation.
Test inference only in a separate evaluator AFTER training result is complete.
No test-based rule changes, reruns, sweeps or extra seeds.2h cap; retain failures.

Preflight verifies frozen backbone/no gradients/eval, moments inheritance and
native update equivalence, analytic margin gradient including bias, zero-tie
trigger, competitor switching, two-pass limit/skip, and no mutation of receiving
or native inputs. Formal checker independently reconstructs recorded projections
from saved head candidates, checks actual endpoints and all correctness masks,
and verifies data/source hashes and final metrics. No independent retraining.
Preserve every failure/skip; no success gate and no automatic next experiment.

One final commit and branch push. Local tensors/head snapshots stay local;
public artifacts contain generated records/hashes, no original token/text data.
