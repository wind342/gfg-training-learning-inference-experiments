# Frozen-feature FewRel baseline comparison v1

Parent: `0a62264bcbad0a6dae8bd4e755aae96aec205a07`. Independent extension;
no previous experiment, main, or paper release identity is modified.

## Question and fixed arms

Compare historical capability retention and new formation in the SAME frozen
task1 BERT representation, label stream, and all-seen-class decision space.
Arms: native-210; ER, A-GEM-AdamW, GEM-AdamW and unmodified Final v2, each
210 and 630 updates/task; accumulated ridge without RP; RanPAC classifier-head
adaptation with fixed 10,000-dimensional Gaussian projection and ReLU.
These are task-specific adaptations, not reproductions of published vision
scores or full RanPAC first-task PETL. No winner or effect threshold is assumed.

80 relations, ten fixed tasks, seed2021, batch16; task2-10 continuous training.
Gradient arms inherit identical task1 head AND head AdamW moments. LR5e-5,
betas/epsilon/weight decay/fused setting inherited unchanged. 630 repeats the
same frozen 210-batch order three times. No reset at task boundaries.

## Shared information and data separation

Use the existing data.npz and task1 checkpoint, verified against historical
SHA256 authorities. Reuse the clean 11,190-row validation cache and historical
encoded-input train/control/test exclusions. A historical validation row is
released for algorithm use only when its relation becomes old. All gradient
memory arms have the same available old bank (task10:10,072 rows).
Current validation labels are audit-only and cannot change an update.
Test is evaluator-only; no hyperparameter or rule is chosen from test outcomes.
No future-task training examples are used early by an algorithm.

Ridge/RP accumulate each training example once when its task arrives, and each
newly released old validation example once at the next task. Thus the stream of
available labeled information is equal, but storage/retention/use differs:
sufficient statistics retain all past training contributions; gradient arms
retain parameters/moments and the available validation memory. Do not describe
this as identical historical memory. Ridge task1 starts from task1 train
statistics, not an AdamW classifier.

## Concrete algorithms

ER: current16 + old16, uniform sample without replacement from available old
bank using PCG64(2021+9000000); mean CE over all32, backward, global clip1,
AdamW. A-GEM: same sampling/count and current mean CE; one old mean-CE reference
gradient; Euclidean projection when dot<0. GEM: independently sample16 per old
TASK each update with the same seeded generator; one mean-CE reference gradient
per old task; project current gradient onto all reference halfspaces g_ref dot
g>=0. Deterministic active-set enumeration solves the <=9-variable dual, with
KKT verification. No positive dual lower bound or soft-memory margin is added.
This is the zero-margin GEM constraint, not the official implementation's
nonzero memory-strength option. No gradient constraint claims endpoint safety.
For both: references are un-clipped, projection precedes global clip1, projected
gradient feeds inherited AdamW moments exactly once. Current gradients are not
pre-clipped. V2 remains current gradient -> clip1 -> real AdamW candidate ->
unchanged relation floor/Hildreth/FP32 reserve -> actual FP32 recheck/commit.
No baseline is given V2 corrections or skips.

Ridge: no intercept or normalization; one-hot80 targets. RP: H=ReLU(F R), R
standard Gaussian FP32 with CPU seed2021, dimension10000, no normalization.
No-RP: H=F. G=sum H^T H, Q=sum H^T Y. Every task chooses ridge from
10**[-8,...,8], using FIRST80% / LAST20% of that task's frozen shuffled training
order, temporary statistics from that task only, minimum validation MSE
(first/smallest lambda wins ties). This is the official train-only ridge rule;
not a new outcome-driven LR sweep. Refit cumulative G/Q including all arrived
training and released controls. FP64 accumulation and solving are explicit
numerical adaptations. CV uses primal or dual eigendecomposition of the smaller
Gram matrix, algebraically the same ridge solution. Roundoff-negative
eigenvalues are clamped to zero and counted. Full solve uses FP64 Cholesky;
if Cholesky fails, a recorded eigensolve uses the same lambda, not added jitter.
No test-based retries. Save solver residuals.

## Timing, cache and compact retention

All arms share cached frozen features. Training features are computed in the
exact historical current-batch order/shape; a sampled raw-forward equality
check is required. V2 is rerun through the shared cache for comparable timing;
historical V2 accuracy/identity is separately compared, not silently replaced.
Report shared feature extraction cost; per-arm task1 initialization/statistics
cost; task2-10 algorithm compute; read-only observer; serialization/audit; test.
Do not compare cached baselines against historical BERT-forward wall time.
Algorithm compute includes all mandatory memory backwards, projections,
solves and safety scans. Common independent diagnostics are separate.

Per-step floor diagnostics use each arm's fixed task-entry correct counts
after seen-class expansion. They never control baselines. Record counts,
deficits, flips, sampling identities, gradient projection certificates,
head/moment hashes and costs in compressed JSONL; no per-step large tensors.
Save per-task head/moments, test predictions and small summaries. Ridge saves
per-task classifier weights, final G/Q and RP matrix, not ten Gram copies.
Source/data/cache/protocol/checkpoint hashes and fixed RNG regenerate omitted
intermediates. Preserve failures. Never claim every step independently replayed
unless actually verified; an independent deterministic task replay and metric
recalculation are required. Do not publish raw examples or pretrained weights.

## Reporting

Task2-10 old/new/all macro; T10 and mean old/new/all; per-relation/cohort results;
old feature reads and old-loss backward example-equivalents; floor violations;
wall components and peak GPU memory. V2 mandatory monitor reads are counted
separately from common observer reads; sufficient-statistics reads counted
once per incorporated row plus any explicit diagnostic reads.
Here V2 old-feature reads count row-logit evaluations (receiving, native,
projected and committed scans), not every low-level repeated coordinate/vector
access or device-memory byte. Other gradient arms count sampled rows; these
logical counts must not be presented as equivalent FLOPs or hardware traffic.
No new seed, parameter search, strategy modification or outcome-driven rerun.

## Official implementation authorities

- GEM: facebookresearch/GradientEpisodicMemory, model/gem.py (zero-margin
  mathematical variant; original SGD/task masks are not imported).
- A-GEM: facebookresearch/agem, commit45421499483b28935491251e9e821c55e8b3c089,
  model/model.py, create_stochastic_gem_ops.
- RanPAC: McDonnell-Research-Lab/RanPAC,
  cf4b301d18b0c27db030f4371b72b768005ae58a, RanPAC.py replace_fc,
  optimise_ridge_parameter, setup_RP. No-RP means use_RP with M<=0, NOT NCM.

New equations-based implementations only; no third-party source copied.
