# Fixed task8 fresh-feature smoke

Base115534a09cd554fcd75fd46dde711fdb98988519. New isolated experiment only.
Fork the Final v1 task7 ending head and AdamW states; BERT from the unchanged
task1 checkpoint. Exactly two new arms, task8 only,210 steps,seed2021,LR5e-5,
batch16,original batch order,one epoch,80 head rows,64 seen classes.

Both arms: z=W h+b+U phi(h), U zeros80x128, no residual bias.
R is128x1536 CPU torch.randn with a private generator seed2021, divided by
sqrt(1536), then fixed. Linear control phi=Rh; nonlinear arm phi=ReLU(Rh).
No normalization or fitted feature transform. Same R and parameter counts.
W,b inherit task7 AdamW moments and step1470; U gets fresh zero moments and
step0. Fused AdamW,betas(.9,.999),weight_decay0,gradient clip1 over all plastic
parameters, no scheduler. Counter updates once per scheduled step including skips.

BERT frozen/eval, no gradient. Cache task8 current-training features once in the
original batch16 order; verify first cached batch against a second exact forward
before training. The fixed validation cache and same task8 bank are inherited:
7833 old plus80 new audit-only controls. No replay/old loss/Router/adapters.
No test forward or test predictions until BOTH training endpoints are sealed.

Use a joint coordinate [W,U,b], with fixed feature[h,phi(h),1]. Reuse original
Hildreth function bytecode, coordinate updates, active discovery,first-safe stop,
100-sweep cap and eligibility rule. Adapt ONLY affine evaluation to actual
split FP32 forward W h+b+U phi. Existing reserve formula uses joint dimension
k=1667 (1536+128+1+2), not the old numerical budget. No reserve tuning.
Native safe path requires old receiving-correct margins>0; triggered projection
requires their actual FP32 margins>=1e-6. Projection never changes moments.

Initial zero-U outputs must be exactly identical to original task7 head for all
cached controls and current-training samples before any optimization. Joint
projection is on both W,U. The two new arms differ only by the fixed ReLU.
Compare with existing Final v1 task8 results; do not rerun original baseline.

Preflight: initial equality, affine directional identity in FP64, autograd joint
gradient checks, optimizer inheritance and advancement, zero-residual reduction
to original forward, safety/replay consistency and original source identities.
Post-run independently reconstruct all420 AdamW candidates and Hildreth sweeps,
check endpoint safety and unchanged optimizer moments/BERT/input hashes.

Report old/new/all test macro; skip,protected flips,projection count,norm ratio,
cosine,wall. Preserve all failures; no outcome-driven rerun/extra seed/task/tuning.
New test labels are not used to choose parameters or run additional arms.

Interpretation: nonlinear improvement over BOTH controls supports usefulness of
these additional fixed features under constraints. It does not establish a unique
cause, global capacity bound or a universal architecture. Null results do not
exclude other feature maps. Random ReLU also changes geometry/scaling; matching
parameter count and fresh optimizer state does not isolate all geometry effects.
