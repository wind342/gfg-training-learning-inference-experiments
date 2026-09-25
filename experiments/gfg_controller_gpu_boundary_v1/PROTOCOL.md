# GPU boundary and standalone runtime gate

Base 992ab35645b60d60f5eb4ab4590b7faaa79ff2df. New directory only; existing
runtime, evidence, commits and frozen tag remain unchanged. No 70M/160M training.

GPU implementation retains explicit target-minus-competitor weight differences,
FP64 dot-products, explicit FP32 coordinate rounding followed by FP64 endpoint
arithmetic, all competitors and strict positive target margin. Class index order
replaces argsort order; minima are order-independent for finite values. The target
itself is masked with positive infinity. Even-N median averages both middle values.
All four alpha candidates and cadence 5 remain unchanged. No approximate pruning.

New CUDA reduction order may differ from NumPy/BLAS. Record exact coordinate,
minimum, score and choice equality separately. Predeclared numerical diagnostic:
rtol=1e-6, atol=1e-8; this does NOT confer byte identity. Record max absolute errors
and mismatched element counts. Never round away discrepancies or silently choose
a CPU fallback. Any changed selected alpha or resulting state must be reported.

Runtime transfers a 9-double boundary summary (72 bytes) per refresh: four correct
counts, four medians, one finite-data flag. Existing scalar loss/evaluation accuracy
transfers remain. AUDIT explicitly permits full diagnostic transfer, not performance
claims. GPU still computes full pre-update logits; tiled differences do not imply
elimination of every three-dimensional temporary or all full-vocabulary tensors.

The standalone entry has no oracle, prior ledger, per-step model hash, or checkpoint
CPU transfer. A local trusted factory supplies actual model/optimizer/data/schedule.
Backend is explicit. Scope is the existing nanoGPT-compatible adapter and eager
FP32 CUDA; no arbitrary decoder, mixed precision or distributed claim.

Before outcomes, seal this directory's Python/protocol files and all old tracked
file SHA256s. Refuse existing output directories. Preserve every failed attempt.

Checks:
1. Synthetic random/tied boundary data, odd/even N, vocabulary 2 to 4097, hidden
   width 3 to 65, FP32/FP64 inputs. Tiles (32,8), (64,64), (256,128). Reconstruct
   coordinate comparison via reference competitor indices. Test zero/tie choice.
2. All 12 existing historical starts x 3 LRs: audited GPU trajectory with old
   same-state CPU boundary checks at every decision, raw outputs/hash chains,
   comparison to old persisted trajectory and final optimizer identity.
3. All 36 cases through the actual standalone runtime entry (AUDIT off), then
   outside its timed region hash final model/optimizer and compare every returned
   alpha/accuracy/score and final logits. No full-model hashes inside entry.
4. Fixed first assignment, LR .003: 3 fresh-state timing repetitions after one
   excluded warmup per backend/tile. Compare CPU blockwise reference and all three
   GPU tile choices using identical native batches/seeds, first99 or cap200.
   Include evaluation in wall time; compare outcomes before speed claims.
   Report median wall/peak allocation. This is one small-model engineering case,
   not a general throughput or large-model result. Default tile remains 256x128
   irrespective of which timing case is fastest; results are not a universal tune.
5. Independent checker reopens records/raw logits, recomputes gaps, decisions,
   trajectory identities, baselines and timing medians. Old files/tag unchanged.

Production source must not depend on old benchmark optimizers or old private
records. Validation may reuse their existing loading/environment helpers.
No extra training regimes, target reduction, candidate reduction or science change.
