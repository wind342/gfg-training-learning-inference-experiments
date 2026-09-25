# Optimizer capability benchmark v1

New extension of 0dc78f0c509ed9ce09b09b585151e41ecd9a6f71. No existing
experiment, frozen identity or result is modified. This is a small historical
continuation benchmark, not a large-model or from-initialization claim.

## Receiving state and comparison

Reuse the twelve immutable RUN_ASSIGNMENT entries, weights, 212 labeled targets
and ordered training batches of the online-controller experiment. Optimizer
statistics start fresh FOR ALL FIVE METHODS: Adam states cannot be interpreted
as Sophia/Shampoo states. Weight decay is zero for all arms to avoid inventing
a decoupled-decay SLS variant. AdamW/controller use betas (.9,.98), epsilon 1e-8;
AdamW has a 200-step cosine schedule ending at 10% of initial LR; controller
uses constant LR and the UNCHANGED quadratic selector, refreshed every 5 steps.
Only alpha is cached. A fresh gradient, moment update and displacement is formed
every step. No candidate endpoint forward participates in controller selection.
The 212 control labels are also the evaluation labels, just as in the source
experiment: do not claim new held-out-target generalization.

Use native official implementations, pinned commits and file hashes for Sophia-G,
Distributed Shampoo and SLS. No copied or reimplemented optimizer equations.
Sophia-G: betas (.965,.99), rho .04, sampled-label GNB curvature every 10 steps
including step 1 BEFORE the true-label update. Sample only valid target positions;
the bs multiplier is the actual number of valid tokens in mean cross entropy.
Shampoo: classic inverse-root (not SOAP), block size 64, preconditioning every
10 steps starting at step 10, Adam grafting (.98,1e-8), betas (.9,.98).
SLS: official Armijo, c=.1, beta_b=.9, gamma=2, reset_option=1,
n_batches_per_epoch=1 (these are full training-set batches). No gradient clipping
for SLS, preserving its native Armijo test. Others clip true-label gradients to
1.0; Sophia curvature gradients are not clipped. CUDA FP32, no mixed precision.

## Equal finite tuning budget

Four configurations per method; ONLY starting LR is tuned:
- AdamW, Shampoo, controller: .0003, .001, .003, .01.
- Sophia-G: .00003, .0001, .0003, .001.
- SLS initial step: .01, .1, 1, 10.

Execute every configuration on each of the twelve histories. For each evaluated
history choose each method's configuration using ONLY the other eleven histories
(leave-one-history-out). Rank by mean restricted first-99% steps, then mean
restricted first-95% steps, then configuration index. Already-at-target histories
are excluded for that threshold, not credited as successes. This is cross-validated
configuration selection on existing histories, not a pristine unseen benchmark.
All settings and this selection rule are sealed before reading new outcomes.
No retuning or dropping unsuccessful arms. Four trials are a bounded comparison,
not a claim to optimally tune any method. Report selected configurations.

## Outcomes and stopping

Evaluate strict target-minus-largest-competitor gap > 0 at step 0 and after EVERY
update. Stop as soon as >=99% is observed, or at 200 updates. Thus no unnecessary
training after the primary capability target. Initial >=99% runs do zero updates
and are reported separately. Null crossings remain failures censored at 200.
First-95% and first-99% steps, success counts and restricted steps are primary;
no loss AUC or post-saturation endpoint replaces these outcomes. No claim of
sustained capability follows from this first-hit experiment.

After selection, rerun each selected arm three times, rotate method order by
history and repeat, require exact parameter/evaluation/decision hashes against
the search trajectory. Time synchronized update compute (including controller
coordinates, SLS trial forwards, Sophia curvature, Shampoo preconditioning) and
optimizer construction, plus separately report evaluation and total wall time.
Do not include evidence serialization in compute, but include it in wall time.
Capture peak CUDA allocated/reserved and process RSS. No energy claims.
Time to threshold includes optimizer construction; source loading is separately
identified as setup and excluded from comparative training time.

## Integrity

Synthetic smoke tests precede freezing. Also replay a source controller_5 run
using unchanged original runtime, moments, decay and selector against archived
cadence hashes: this compatibility test is not a benchmark arm.
Save per-step parameter hashes, batch identities, decisions, evaluation logits,
controller coordinates, final checkpoint and hash-linked trajectory records.
Inputs and source code are hash-bound. This is a compact execution ledger, NOT
an assertion that every raw intermediate state is persisted as a complete GFG.
All selected trajectories are natively replayed three times. Independently
recompute target correctness, threshold crossings, LOO selection and summaries.
Preserve failures. No overwrite of completed results. Raw outputs stay on E:;
dependency clones are ignored local runtime material, not vendored for release.
