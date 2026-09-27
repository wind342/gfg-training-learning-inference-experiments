# Offline expansion bias check

Base: ce8124de3bd9549520d2327d6949c6522fe3ee45.
Exactly two boundaries: budget210 task7 (158 -> 19) and budget630 task6
(153 -> 0). Read their predecessor checkpoints, not post-task checkpoints.
Recompute the saved full77 logits with the original fixed256 FP32 arithmetic.
Require byte identity with the expansion audit before applying correction.

For expansion-only correct-to-wrong controls, compute gaps from the saved FP32
logits promoted to FP64 (exact subtraction of the represented values).
Set beta = maximum gap + 1e-6 once. Cast beta once to FP32 and subtract it in
FP32 from only the seven newly admitted logit columns, in a separate copy.
Report both formula beta and the applied FP32 scalar. Do not enlarge beta or
retry if FP32 ties or failed recoveries remain. Correctness is strict margin >0,
matching the original audit; argmax correctness is reported separately.

Report complete per-relation and per-control results, old-class-caused errors,
newly-zero versus previously-zero relations, and raw-to-corrected as well as
before-to-corrected flips. No new-class accuracy is claimed or measured.

No training, backward, optimizer construction/step, predictor/cutter call,
parameter correction, K change, test reads, new mechanism or full-run replay.
Hash input files and loaded model/optimizer tensors before and after. The only
GPU workload is read-only boundary forward evaluation. Independently verify
the offline calculation with scalar comparisons against the saved logits.
All new outputs live in this directory; old evidence remains read-only.
