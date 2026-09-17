# Functional online amplitude controller v1

Prospective protocol. Scientific baseline and frozen research tag remain unchanged.
This is a new online experiment, not a reinterpretation of smoke v1 and not a
new direction search, blockwise optimizer, probe, fitted predictor or CSRG controller.

## Sources and units

Twelve existing fully materialized smoke runs are all included. Per run select
the section with minimum receiving optimizer step, breaking ties by section ID,
among its six previously frozen smoke sections. No new outcome is used. The
exact twelve checkpoint paths, parameter/moment/step commitments, source bundle
identities, batch objects and evaluation identities are in RUN_ASSIGNMENT.json.
The optimizer steps, in entry-ID order, are 7869, 2986, 7793, 4097, 2261, 2667,
8288, 2177, 1478, 7729, 2075 and 1804.

All sources are checked against SOURCE_MANIFEST.json. These are restored
stepwise-training states, not claims of reconstructing unavailable original-run
RNG payloads. Each selected stepwise transition does retain a seed and CPU,
CUDA, NumPy and Python RNG commitments. Re-seeding has been verified to reproduce
all these commitments exactly. Full current RNG payloads are saved in each new
branch's INITIAL.json and content-addressed tensor store.

The run is the outer comparison unit. The same 212 identity-bound validation
targets per run serve both control and evaluation. There is NO independent new
test set. Results must not be described as improved held-out generalization.
Each source already belongs to previously studied historical trajectories;
this experiment's online trajectories, not its source history, are prospective.

## Training and controls

Each run has exactly three formal branches: native alpha=1, fixed alpha=.25,
and controller alpha in [.25,.5,.75,1]. All begin with byte-identical parameters,
Adam first/second moments and steps, RNG and batch position. Horizon is exactly
200 optimizer opportunities. No early stopping, exclusions or horizon changes.

Use the pinned nanoGPT model 3adf61e154c3fe3fca428ad6bc3818b27a3b8291,
24-token vocabulary, block size 3, 2 layers, 4 heads, embedding 64, dropout=0,
bias=False. Use the existing CausalTrainingRuntime.train_step unchanged:
float32 CUDA, fused AdamW, learning rate .003, betas (.9,.98), eps 1e-8,
weight decay 1 for dimension>=2 tensors and 0 otherwise, gradient clipping 1.
No autocast, TF32 disabled, deterministic algorithms enabled. Model and
dependency identities and runtime versions are recorded before formal execution.

At offset k=0..199 load the original source batch at source_step+k, exactly as
the existing stepwise continuation does. Ordered input and target bytes are
authoritative; absence of a separate batch_selection_order is explicitly
recorded, never synthesized. The selector cannot request a future batch.

Each branch forms its own gradient and native optimizer proposal from its own
current state. m, v and Adam step are updated by the native recurrence in full.
The proposed parameter displacement is the float32 native endpoint minus the
float32 pre-state, as in the existing smoke evidence. For alpha<1, compute
theta_pre + alpha*delta in float32 and install ONLY these parameters; native
moments and steps are not interpolated, transplanted or scaled. For alpha=1,
retain the exact native endpoint bytes: subtraction and readdition in float32
are not an identity. This explicit endpoint convention is inherited from the
frozen amplitude-path infrastructure. Applied displacement is recorded as its
actual post-minus-pre bytes, not asserted to be an exact real-arithmetic product.

At offset 0 use the exactly reconstructed recorded RNG seed. Later opportunities
use the existing CausalTrainingRuntime.derive_seed(protocol_file_sha256, entry_id,
source_step+k, 1). The schedule is identical across branches. Every forward and
backward still executes independently on that branch. Dropout is zero; any
unexpected RNG consumption fails the native runtime check.

## Controller and temporal boundary

The selector directly imports smoke v1 `endpoints` and `choose`. Do not fit,
retune, replace or rescue this rule. On every controller step, use its pre-state
parameters, own actual proposed delta and fixed evaluation inputs/labels.
Use the existing HiddenStateModel and nested JVP implementation to compute h,
h' and h'' under the math SDPA backend. Boundary vectors b and b' are computed
from the current head weights and their actual proposed displacement. The
pre-state logits determine g0 and competitor order. All 23 non-target tokens
are included, including the extra vocabulary token. Existing derivative terms
are computed in float64 from float32 native states/JVPs, stored through float32
and promoted to float64, preserving the smoke/TL-P01 arithmetic convention.

For each alpha, g_hat=((g0+alpha*first_gap)+alpha^2*joint_rotation)
+alpha^2*hidden_curvature. Predicted correctness is strictly min(g_hat)>0.
Choose one shared alpha per step by lexicographic maximum of predicted correct
count, median predicted minimum margin, then -abs(alpha-1). No target-specific
choice. No candidate positive-alpha forward evaluations occur before selection.

Native optimizer.step temporarily forms the proposal endpoint, but no forward
reads this endpoint before selection. Functional calls explicitly substitute
every parameter with the pre-state. Synthetic preflight tests change every
ambient model parameter and require unchanged coordinates. The post proposal
is used only to obtain delta and the native optimizer recurrence, not to obtain
functional outcomes. Selection receives no other branch objects or outcomes.
An audit hook denies evidence/data_private reads inside the selector boundary.
Selection is durably written with fsync before any selected scaled endpoint is
applied or evaluated. GFG stage order and hashes record this separation. This
is auditable process-level prevention, not an adversarial OS sandbox claim.

## Evaluation and frozen primary verdict

Evaluate at t=0,10,20,...,200 using the same native forward and fixed target
identities. Correctness is strict target-minus-maximum-competitor margin >0;
ties are incorrect. The primary metric per run is trapezoidal area under the
accuracy curve divided by 200. Its units are accuracy (0..1).

ONLINE_FUNCTIONAL_CONTROLLER_SUPPORTED requires BOTH comparisons separately:
(1) equal-run mean normalized-AUC improvement >= .01 (one percentage point);
(2) strictly positive AUC improvement in at least 9 of 12 runs.
Otherwise the verdict is ONLINE_FUNCTIONAL_CONTROLLER_NOT_SUPPORTED. Exact
numeric zero counts as a tie. There is no outcome-dependent tolerance.
Integrity failure is reported as FAIL and cannot be converted into support.

Report every run's AUC and final-accuracy deltas, win/tie/loss counts, pooled
descriptive accuracy and margin curves, final margins, alpha distributions and
changes, loss, proposal/application norms and measured computational costs.
Target samples and repeated times are not independent replicates.

Count observed correct->wrong and wrong->correct transitions only on the
10-step grid. A recovery requires a previously observed correct state, then
wrong->correct; first observed acquisition means no preceding correct observation
in this horizon. These are descriptive event counts, not a new capability
formation threshold and not exact event timing. Do not claim faster formation
from a comparison that starts after formation. No new threshold is introduced.

## Integrity and outputs

All 36 branches retain every step, failed execution and source identity. Each
step records raw/clipped gradients, native proposed state (parameters,m,v,step),
delta, loss, RNG commitments/payloads, batch identity, selected alpha, coordinates
(controller only), actual applied displacement and post-state, and evaluations.
Native/fixed do not compute controller coordinates or consume its information.
Content-addressed TensorStore, existing SupportGFGWriter, generated-origin
dependencies, native occurrence identities and incidence bindings are reused.
Large new raw execution evidence is external at the context's output_root;
compact ledgers and their SHA-256 manifests accompany the source directory.
Old research artifacts, commits, tags, manifests and Zenodo archives are not edited.

The independent checker hashes all frozen/source files, validates each GFG,
recomputes alpha from stored coordinates using a separate implementation,
recomputes every controller coordinate from current pre-state and own delta,
and re-executes all 7200 branch steps with the unchanged native trainer. For
native it performs 12x200 unmodified native steps, requiring byte-exact parameter,
gradient, moment, step and evaluation reproduction at the same opportunities.
For scaled branches it independently applies only parameter scaling and checks
the full native moment recurrence. All source/batch/RNG alignment, branch
lineage, durable seals, target margins, metrics and verdict are checked.

Success does not imply a universally superior AdamW replacement, other-architecture
validity, large-model compute savings or improved held-out generalization.
Training+selection timing excludes independent validation and source preparation;
wall time including evidence is also reported. This is equal-step, not
equal-wall-budget comparison. Do not reinterpret a failed primary result.
