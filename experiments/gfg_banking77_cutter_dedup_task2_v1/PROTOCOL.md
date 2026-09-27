# Banking77 task2: remove repeated computation, keep the trajectory

Base/reference 534130d8995fbf9f6cec176afb915f1a56845702. New directory and create-only
evidence root. Preserve every historical source/artifact. Task2 only, independent
210/630 arms from their corresponding frozen task1 native checkpoints. No test access,
tasks3-11, baseline method training, tuning, early endpoint checks, warm start or changes
to predictor, selection, solver tolerance, rho, reserve, EPS, inner200/outer8 or moments.

Stage1: A caches exactly the post-ball G@t shared by z and norm; other vectors are not
substituted. B first outer uses d0=None, ad0=zeros, d0norm=0 and reconstructs d=corr.
C returns native/projected/receiving endpoint with the matching executed parameters.
Commit parameter bytes are checked outside algorithm timing. Independent audit always
recomputes actual committed-head forward, including margins/signs/counts.

Stage2 is allowed only if both Stage1 arms match every required reference field and
pass independent replay. It adds D only: unique control first-layer directional multiply,
inverse map to pair rows; competitor-specific terms unchanged. GEMM shape may change
roundoff. Any mismatch is a negative result, not an equivalent optimization. Record the
first divergence and numerical magnitude/cause using a same-state diagnostic; never
change tolerance or silently switch arithmetic to conceal it. Stop Stage2 promotion
on mismatch, retain full requested trajectories if safe, and do not benchmark it as
an accepted equivalent implementation. No result-driven training reruns.

Freeze all code before training. Run one full validation replay per arm/variant.
Compare receiving/native/executed bytes, moments/counters, predictor components/outputs,
triggers, choices, active pairs, solver coefficient/duals/status/sweeps, cap decisions,
endpoint counts/hash and final checkpoint tensors to original authority. Report serialized
checkpoint hash separately from tensor/state identity. Independently reimplement the
reference solver/geometry and predictor replay to validate accepted variants. C synthetic
native-safe/projected-safe/rollback branches are checked separately from real runs.

Timing: after validation, use exactly three complete replay repetitions per accepted
variant and budget, from the same checkpoint each time (never training extensions).
Orders: baseline,stage1,stage2; stage1,stage2,baseline; stage2,baseline,stage1.
Remove rejected variants from those fixed lists. Warm up each process on synthetic
head/predictor work only, reset seeds before training. No best-of or optional reruns.
Report all repetitions, median/range. Full validation timings remain separately recorded.
Current forward/backward, predictor, endpoint, solver, geometry, reconstruction and
materialization timing boundaries are common. Existing internal cutter diagnostics remain
inside control timing for all variants; external hashing/serialization/comparison,
train metrics and independent audits remain outside. Region timers synchronize as before.

Counters distinguish two post-ball G@t calls per reference sweep from all four Gram
matrix-vector products per sweep; Stage1 has one and three respectively. Count applies
by actual outer position, including reconstruction apply. D reports both geometry
pair/unique sums and actual apply pair/unique row sums. Dense matmul lower-bound FLOPs
and inherited solver/analytic estimates are separate, with the eliminated work explicitly
subtracted, not advertised as hardware instruction counts. Peak GPU allocation reported.

No subsequent task is authorized by any gate. Report and stop after this experiment.
