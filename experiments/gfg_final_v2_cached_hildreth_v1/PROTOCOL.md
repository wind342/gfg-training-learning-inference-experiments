# Cached Hildreth and executed-anchor transition screening

Base: `8f00816a6b8c6d61ee87688b242497f95a00d306`. Independent directory and
branch. All previous code/results, main and release tag remain unchanged.
FewRel task10 only, seed2021, 210 updates, LR5e-5, task9 V2-210 head and
AdamW moments. Reuse read-only frozen features and inputs. No Banking77.

Stage 1 caches CPU FP64 augmented features, absolute features, squared lengths,
labels and relation row indices once at task entry. Compute the unchanged
roundoff reserve only after active pairs are known; cache within each update.
Keep EPS, coordinate arithmetic, selection, pair ordering, maximum sweeps,
first-safe stopping and fixed 256-row FP32 endpoint primitive unchanged.
Changing reduction shape can change FP64 summation: preflight and full replay
must disclose differences; never silently replace the numerical authority.

Stage 2 is gated on stage 1's complete 210-update identity check. Anchor every
update on the previous actual executed FP32 endpoint. Bound both signs of
margin change using feature norm, target/max class displacement and roundoff.
Exact-evaluate ambiguous controls, all controls in potentially deficient
relations and all previously selected/active controls. Refresh all anchor
margins from an actual full FP32 executed endpoint; include this cost in timing.
No K-positive certificate early stop and no threshold search.

Gate: per-update byte-identical executed heads and AdamW moments; identical
deficits, selected controls, active pairs and sweep sequence; final predictions
identical; zero false-safe and relation-floor violations. Compare primal/dual
hashes and pair reserve values too. Independent observers must never feed the
controller. A failed gate stops dependent work and preserves the failure.

Separate untimed correctness validation, component profiling and standalone
timing. Report budget, CPU geometry, coordinates, materialization, endpoint,
control wall, total algorithm wall, logical exact rows and padded GEMM calls.
Include setup, commit/recheck, and anchor refresh in algorithm timing. Exclude
external observers, hashes used only for validation, test and serialization.
Run each timed method once in a disclosed order, never select best timings.
No active-set/hot-start solver replacement in this experiment.
