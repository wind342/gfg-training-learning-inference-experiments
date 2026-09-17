# Functional step-size smoke v1: frozen before outcome access

Status: exploratory reuse of historically existing outcomes; not a new
prospective confirmation. No outcome-dependent changes to this protocol.

## Population and candidates

All 72 existing native realized-update response sections, 212 concrete targets
per section, 15,264 records, 12 runs. No target/section exclusion or new training,
inference, CSRG, derivative execution, direction search, or blockwise scaling.
Candidates are exactly [0.25, 0.5, 0.75, 1.0]. Alpha 0 is descriptive only.
The two-receiver B-update transplant protocol is not the selection population.

## Predictor and selector

For every target and its 23 competitors:

    g_hat(a) = g0 + a * first_gap + a^2 * joint_rotation + a^2 * hidden_curvature
    m_hat(a) = min_c g_hat_c(a)
    predicted_correct(a) = m_hat(a) > 0

This is Q = joint_rotation + hidden_curvature, with second-order scaling.
Evaluate the four summands left-to-right in float64 to preserve the exact
TL-P01 arithmetic at alpha 1. No coefficients are fitted. Use the original
float64 competitor gap cache, not its float32 copy in the joint cache.

Select one shared alpha per section by lexicographic maximum of:
(predicted correct target count, median predicted minimum margin, -abs(a-1)).
No per-target choice. The actual target labels are the already known task
identities; post-update correctness is never available to this selection.

## Mandatory temporal separation

Stage A loads allowlisted pre-outcome coordinates and source identity metadata
only. It does not import the outcome compiler or load positive-alpha logits,
predictions, margins, endpoint labels, or support probes. Source manifests may
be read for file hashes, not outcome values. It writes all scores, target
identities and source commitments to SELECTION_LEDGER.jsonl, then seals its
SHA-256, source hashes, rule hash and source-code hashes. The process exits.

Stage B verifies that seal before any outcome access. It validates section
hashes, target order, original input hashes, exact parameter/update identities,
and that stored alpha 1 is the native endpoint. It then reads only existing
ungated baseline outcomes at 0, 0.25, 0.5, 0.75, 1. No new forward is run.

## Outcomes and comparison

Correctness is strict positive minimum target-versus-competitor gap, matching
TL-P01 (ties are not correct). Report argmax ties separately, if present.
For controller, every fixed candidate, alpha-0 reference, and oracle report:
correct count/rate; actual mean/median minimum margin; relative to alpha 1,
prevented correct-to-wrong, additional wrong-to-correct, and newly caused
correct-to-wrong. Also report lost wrong-to-correct to complete the accounting.
Here initial correct/wrong always refers to alpha 0.

Oracle chooses one shared alpha per SECTION using actual correct count, then
actual median minimum margin, then closeness to 1. It is an unattainable
hindsight bound for correctness in this candidate set, not a per-target oracle
and not necessarily an upper bound on pooled mean/median margin.

Pooled targets are descriptive, not independent replicates. Report each run's
controller-native difference and run win/tie/loss; compare all fixed amplitudes
runwise too. Reuse the original 8/4 development/confirmation run identities only
for descriptive consistency checks, not fresh confirmation or training.

## Integrity and interpretation

All 15,264 alpha-1 predictions and minima must agree with the archived TL-P01
algorithm (exact booleans, margin tolerance 1e-7). Identity mismatch or missing
source fails the entire experiment. Float32 copies of g0 must match the original
float64 cache cast to float32 exactly. No outcome-driven tolerance adjustment.

An improvement over alpha 1 alone supports only better one-step functional
outcomes on this existing counterfactual corpus. Superiority to the best fixed
candidate is assessed separately; no claim of a better optimizer, better
long-term training/generalization, better support organization, or universality.
All negative results are retained. No rule or candidate rescue is permitted.
