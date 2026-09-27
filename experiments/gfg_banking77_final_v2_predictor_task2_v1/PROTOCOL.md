# Final V2 nonlinear predictor + existing cutter: task2 only

Frozen before execution. Base bd7d298; separate directory and create-only evidence root.
Only task2 210 and 630, each from its original native task1 checkpoint with matching
optimizer history. No tasks3-11, test, ER/RanPAC runs, tuning, scalar cuts or warm start.
The historical results are known; this run is not a blind hypothesis test.

Frozen BERT train feature cache, MLP 768-256-GELU-77, current batch16, AdamW and
schedule unchanged. Controls never enter CE backward. Moments advance once from
the native proposal, including rollback. K is set after seen expansion, fixed in task.

Every update computes delta=native-before and f0+f1+0.5*f2 using the historical
JVP/nested-JVP definition on the small head only, fixed256 numerical tiles.
Persistent feature/result buffers; no repeated CPU feature conversions.
First audit native-only trajectories for both budgets. Then run controlled trajectories
with prediction accuracy measured again on every actual proposal.

Predicted deficits enter the existing nearest-boundary selection at the first outer
iteration. The Jacobian/reserve always use the true FP32 native endpoint as their
linearization anchor. Subsequent selection uses true projected endpoints. Predicted-safe
native candidates also receive an exact check; false-safe enters the unchanged cutter
using actual deficits. An actual-safe native candidate commits immediately, including
false-unsafe predictions (no unnecessary movement). Record predicted triggers separately
from actual unsafe/cutter-repair steps. Exact native output is reused, not recomputed.

Keep the existing nonlinear cutter: analytic geometry, reserve, trust ball/rho rule,
EPS1e-6, solver PASS1e-8, inner200, outer8, cumulative selected/pairs, and cap
materialize/recheck/accept-or-continue. Endpoint checks remain after local solver returns,
as explicitly directed by the user. No per-inner-sweep endpoint retrofit.

Predictor gates from historical smoke: correctness agreement>=.99, argmax>=.98,
margin MAE<=.001, balanced accuracy>=.95, observed CW/WC recall>=.9,
false-safe updates / all updates<=.01. Also report false-safe / predicted-safe (null
when denominator zero). Native-only exposure requires >=20 distinct CW controls.
No thresholds are tuned after outcomes. Controller diagnostics retain prior task2 gates:
floor0, >=20 unsafe native proposals, rollback<=10%, repair>=90%, new train macro>=50%,
final current train CE<=.9*entry CE, triggered norm median>=.1, old CE backward0,
state/moment checks pass. These gates do NOT authorize continuation: stop at task2.

Independent audit replays native-only and controlled proposals, moments, JVP components,
prediction decisions, selections, solver coef/duals, capped materialization, continuations,
all executed heads and exact floor counts. Independently derived analytic directional
derivatives verify JVP numerics. Train/model sources hashed before and after; Python open
allowlist prevents accessing test evidence. Source and input freeze precedes runs.

Timing excludes serialization, hashes, accuracy metrics and training-set evaluation;
includes predictor, native exact check, cutter and commit recheck. Dense FLOPs are measured
ATen matmul lower bounds, with existing analytic cutter estimates separately labeled.
No claim of complete hardware FLOPs or a globally nearest nonlinear feasible projection.
